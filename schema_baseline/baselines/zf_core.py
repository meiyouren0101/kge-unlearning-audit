"""Row-restricted ZeroFisher adaptation of paper Eqs. 6, 8–11 and 21."""
from igpu import bootstrap
import torch
import io_utils as io
import resource_bridge as bridge
import legacy_runtime as rt

EPSILON = 1e-5
GAMMA = 1.0
ITERATIONS = 100
DAMP = 0.0
SCALES = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)


def edit_rows(request, retained):
    endpoints = {e for h, r, t in request for e in (h, t)}
    neighbors = set()
    for h, r, t in retained:
        if h in endpoints:
            neighbors.add(t)
        if t in endpoints:
            neighbors.add(h)
    relations = {j for _, r, _ in request for j in (r, r + 1)}
    return sorted(endpoints | neighbors), sorted(relations)


def batch_gradient(model, batch, rows, noise, exact):
    parameters = model.embedding()
    if exact:
        gradients = torch.autograd.grad(model.loss(*batch), parameters)
        return tuple(g.index_select(0, ids) for g, ids in zip(gradients, rows))
    with torch.no_grad():
        base = tuple(p.index_select(0, ids) for p, ids in zip(parameters, rows))
        directions = tuple(torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=noise) for v in base)
        try:
            for p, ids, v, u in zip(parameters, rows, base, directions):
                p.index_copy_(0, ids, v + EPSILON * u)
            plus = model.loss(*batch)
            for p, ids, v, u in zip(parameters, rows, base, directions):
                p.index_copy_(0, ids, v - EPSILON * u)
            minus = model.loss(*batch)
        finally:
            for p, ids, v in zip(parameters, rows, base):
                p.index_copy_(0, ids, v)
        return tuple((plus - minus) / (2 * EPSILON) * u for u in directions)


def cohort_gradient(model, loader, rows, noise, exact, label):
    total = [p.new_zeros((len(ids), p.size(1))) for p, ids in zip(model.embedding(), rows)]
    count = 0
    for index, batch in enumerate(loader, start=1):
        batch = tuple(t.to(model.args.device) for t in batch)
        weight = batch[0].numel()
        gradients = batch_gradient(model, batch, rows, noise, exact)
        for out, g in zip(total, gradients):
            out.add_(g, alpha=weight)
        count += weight
        if bridge.ACTIVE is not None and (index & (index - 1)) == 0:
            bridge.ACTIVE.anchor(f'{label}_gradient_batch/{index}', rt.tensor_tree_hash((total, count)))
    io.require(count > 0, 'Empty gradient cohort')
    return tuple(g / count for g in total), count


def inner(left, right):
    return sum((a * b).sum() for a, b in zip(left, right))


def ihvp(v, retained_gradient):
    """Apply rank-one Fisher, never materializing a quadratic parameter matrix."""
    g = retained_gradient
    norm2 = inner(g, g)
    dot = inner(g, v)
    x = tuple(b / GAMMA - a * dot / (GAMMA * (GAMMA + norm2)) for a, b in zip(g, v))
    # Eq. 8 with residual step alpha <= 1/lambda_max(H), keeping the same fixed point.
    alpha = 1 / (GAMMA + norm2)
    for _ in range(ITERATIONS):
        projection = inner(g, x)
        x = tuple((1 - DAMP) * u + alpha * (b - GAMMA * u - a * projection)
                  for a, b, u in zip(g, v, x))
    return x


def correction(up, model, data, phase, seed, exact):
    import graph_data as gd
    from types import SimpleNamespace
    from torch.utils.data import DataLoader

    d = gd.at(data, phase)
    ids = edit_rows(data['phases'][phase], d['retained'])
    rows = tuple(torch.tensor(v, dtype=torch.long, device=model.args.device) for v in ids)
    def loader(facts):
        ds = up.TrainDataset(model.args, SimpleNamespace(train_data=facts, ent_num=data['counts']['entities']))
        return DataLoader(ds, batch_size=1024, shuffle=False, collate_fn=ds.collate_fn,
                          generator=torch.Generator().manual_seed(seed))

    rt.seed_rng(seed)
    noise = torch.Generator(device=model.args.device).manual_seed(seed ^ 0x5A1D0F31)
    v, n_forget = cohort_gradient(model, loader(data['phases'][phase]), rows, noise, exact, 'forget')
    g, n_retain = cohort_gradient(model, loader(d['retained']), rows, noise, exact, 'retain')
    io.require(n_forget == 22 * len(data['phases'][phase]) and n_retain == 22 * len(d['retained']),
               'Both directions and all gradient batches required')
    step = ihvp(v, g)
    diag = dict(entity_rows=len(rows[0]), relation_rows=len(rows[1]), deleted_loss_rows=n_forget,
                retained_loss_rows=n_retain, v_l2=float(inner(v,v).sqrt()),
                retained_gradient_l2=float(inner(g,g).sqrt()), ihvp_l2=float(inner(step,step).sqrt()))
    return rows, step, diag


@torch.no_grad()
def apply(model, rows, step, scale, base=None):
    if base is None:
        base = tuple(p.index_select(0, ids) for p, ids in zip(model.embedding(), rows))
    for p, ids, delta, original in zip(model.embedding(), rows, step, base):
        p.index_copy_(0, ids, original + delta / scale)
