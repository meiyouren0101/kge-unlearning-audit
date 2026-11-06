"""Verbatim numerical/RNG/checkpoint functions; no server entry point."""
import hashlib,math,os,random,types
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def seed_rng(seed, include_cuda=True):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def schedule_seed(seed, phase):
    raw = hashlib.sha256(("graphdpo-pair-schedule-v1/" + str(seed) + "/" + str(phase)).encode()).digest()
    return int.from_bytes(raw[:4], "little")

def capture_rng(include_cuda=True):
    import numpy as np
    import torch
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": str(state[0]), "keys": state[1].tolist(), "position": int(state[2]),
                  "has_gauss": int(state[3]), "cached_gaussian": float(state[4])},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone().cpu() for value in torch.cuda.get_rng_state_all()]
        if include_cuda and torch.cuda.is_available() else [],
    }

def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        require(torch.cuda.is_available(), "Saved CUDA RNG requires CUDA.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])

def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    require(value is None or type(value) in (str, bool, int, float), "Checkpoint contains an unsupported object.")
    require(type(value) is not float or math.isfinite(value), "Nonfinite checkpoint primitive.")
    return value

def tensor_tree_hash(value):
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(("tensor/" + str(tensor.dtype) + "/" + repr(tuple(tensor.shape)) + "/").encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda key: (str(type(key)), repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update((type(item).__name__ + "[").encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            require(item is None or type(item) in (str, bool, int, float), "Hash input has unsupported objects.")
            digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())

    visit(value)
    return digest.hexdigest()

def snapshot_checkpoint(model, optimizer, metadata, loader_generator=None):
    cuda_model = any(parameter.device.type == "cuda" for parameter in model.parameters())
    return cpu_tree({
        "schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": capture_rng(include_cuda=cuda_model), "metadata": metadata,
        "loader_generator": loader_generator.get_state() if loader_generator is not None else None,
    })

def save_checkpoint_new(path, payload):
    import torch
    with Path(path).open("xb") as stream:
        torch.save(cpu_tree(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)

def load_checkpoint_checked(path, expected_sha256, trust):
    import torch
    require(trust, "Explicit --trust-checkpoint is required even for weights_only loading.")
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_sha256), "Expected checkpoint SHA256 is malformed.")
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Checkpoint is missing/symlinked.")
    require(sha256_file(path) == expected_sha256, "Checkpoint SHA256 mismatch before loading.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and payload.get("schema_version") == 1, "Unknown checkpoint schema.")
    require(set(payload) == {"schema_version", "model", "optimizer", "rng", "metadata", "loader_generator"}, "Checkpoint fields differ.")
    require(sha256_file(path) == expected_sha256, "Checkpoint changed while loading.")
    return payload

def build_args(config, data_root, data_name, seed, device):
    return types.SimpleNamespace(
        data_path=str(Path(data_root).resolve()), data_name=data_name, seed=seed, device=device,
        emb_dim=config["embedding_dim"], margin=config["margin"], neg_ratio=config["negative_ratio"],
        batch_size=config["batch_size"], snapshot_num=config["snapshot_num"], snapshot=0,
        snapshot_test=0, snapshot_valid=0, valid=False, random_policy="far", unlearning_method="GraphDPO",
        begin_pretrain=False, begin_unleanring=True, wo_dpo=False, wo_distill=False, wo_reply=False,
    )

def load_model_state(model, state):
    reference_keys = [name for name in state if name.startswith("old_data_")]
    if reference_keys and not any(name.startswith("old_data_") for name in model.state_dict()):
        model.save_embeddings()
    model.load_state_dict(state, strict=True)

def new_optimizer(model, learning_rate):
    import torch
    return torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=False)

def _batch_to_device(batch, device):
    return tuple(value.to(device) if value is not None else None for value in batch)

def train_step(model, optimizer, batch, group, device):
    import torch
    model.train()
    batch = _batch_to_device(batch, device)
    optimizer.zero_grad()
    if group == "pretrain":
        head, relation, tail, label = batch
        loss = model.loss(head, relation, tail, label).float()
        components = {"margin": float(loss.detach().cpu())}
    else:
        head, relation, tail, label, positive, negative = batch
        loss = model.loss(head, relation, tail, label).float()
        dpo = model.dpo_loss(positive, negative)
        distill = model.distill_loss()
        components = {"margin": float(loss.detach().cpu()), "dpo": float(dpo.detach().cpu()),
                      "distill": float(distill.detach().cpu())}
        # The upstream processor adds both float64 auxiliary terms in-place to
        # a float32-cast margin loss. Keep that casting behavior in both arms.
        loss += dpo
        loss += distill
    require(bool(torch.isfinite(loss).item()), "Nonfinite training loss; no repair/retry.")
    loss.backward()
    for parameter in model.parameters():
        require(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()), "Nonfinite gradient.")
    optimizer.step()
    return {"total": float(loss.detach().cpu()), **components}

def replay_step(model, optimizer, batch, device):
    import torch
    model.train()
    head, relation, tail, label = _batch_to_device(batch, device)
    optimizer.zero_grad()
    loss = model.loss(head, relation, tail, label).float()
    require(bool(torch.isfinite(loss).item()), "Nonfinite replay loss.")
    loss.backward()
    for parameter in model.parameters():
        require(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()), "Nonfinite replay gradient.")
    optimizer.step()
    return float(loss.detach().cpu())

def _loader(dataset, batch_size, seed):
    import torch
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=True, collate_fn=dataset.collate_fn, generator=generator,
    )
    return loader, generator
