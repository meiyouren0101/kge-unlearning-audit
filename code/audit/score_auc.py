from igpu import bootstrap
"""Fixed primary increasing AUC, cumulative-deletion population, no fitted score."""
from collections import Counter, defaultdict
import math
import numpy as np
import io_utils as io
from proxy_core import RetainedFeatures

SEEDS = (0, 42, 123)
OLD_FEATURES = ('S', 'head_degree', 'tail_degree', 'same_query_retained_answers')
FEATURES = ('S', 'f1', 'f2', 'f3', 'f4', 'f5',
            'head_degree', 'tail_degree', 'same_query_retained_answers')
LABELS = dict(S='可推断性分数 S', f1='f1 共同邻居数', f2='f2 其他关系直接相连',
              f3='f3 查询头在该关系的保留事实数（原始递增）',
              f4='f4 答案实体在该关系的保留事实数', f5='f5 长度1规则最高置信度',
              head_degree='查询头实体度', tail_degree='查询尾实体度',
              same_query_retained_answers='同一查询的保留答案数')


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    io.require(values.ndim == 1 and len(values) and np.isfinite(values).all(), 'Finite percentile population')
    _, ids, counts = np.unique(values, return_inverse=True, return_counts=True)
    return ((np.cumsum(counts) - counts / 2) / len(values))[ids]


def score_from_raw(raw):
    raw = np.asarray(raw, dtype=np.float64)
    io.require(raw.ndim == 2 and raw.shape[1] == 5, 'Five original structural features required')
    columns = [percentiles(raw[:, j]) for j in range(5)]
    columns[2] = 1 - columns[2]
    return np.column_stack(columns).mean(1)


def auc(labels, values):
    labels = np.asarray(labels, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    io.require(values.ndim == labels.ndim == 1 and len(values) == len(labels), 'AUC dimensions')
    io.require(np.isfinite(values).all(), 'Nonfinite AUC values')
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def query_array(facts):
    facts = np.asarray(facts, dtype='<u4').reshape(-1, 3)
    queries = np.empty((2 * len(facts), 3), dtype='<u4')
    queries[::2] = facts
    queries[1::2] = facts[:, [2, 1, 0]]
    queries[1::2, 1] += 1
    return queries


def align(source, target):
    dtype = np.dtype([('h', '<u4'), ('r', '<u4'), ('t', '<u4')])
    source = np.ascontiguousarray(source, dtype='<u4').view(dtype).reshape(-1)
    target = np.ascontiguousarray(target, dtype='<u4').view(dtype).reshape(-1)
    order = np.argsort(source, kind='mergesort')
    keys = source[order]
    io.require(np.all(keys[1:] != keys[:-1]), 'Duplicate structural query identity')
    ids = np.searchsorted(keys, target)
    io.require(np.all(ids < len(keys)), 'Cached query outside cumulative feature population')
    io.require(np.array_equal(keys[ids], target), 'Cached query identity differs from feature population')
    return order[ids]


def prepare_scores(retained, deleted, target, saved_s, progress=None):
    """Retained/deleted are full forward facts; target may be a fixed evaluation sample."""
    deleted = sorted(map(tuple, deleted))
    queries = query_array(deleted)
    saved_s = np.asarray(saved_s, dtype=np.float64)
    io.require(saved_s.shape == (len(queries),) and np.isfinite(saved_s).all(), 'Saved S cumulative dimensions')
    extractor = RetainedFeatures(retained)
    raw = np.empty((len(queries), 5), dtype=np.float64)
    for i, q in enumerate(queries):
        raw[i] = extractor.row(tuple(map(int, q)))
        if progress and i and i % 40000 == 0:
            progress(i, len(queries))
    computed = score_from_raw(raw)
    max_error = float(np.max(np.abs(saved_s - computed)))
    io.require(np.array_equal(computed, saved_s),
               'S exact feature bridge failed; max_abs=' + repr(max_error) + '; do not change definitions')
    ids = align(queries, target)
    degree = Counter()
    answers = Counter()
    for h, r, t in retained:
        degree[h] += 1
        if h != t:
            degree[t] += 1
        answers[h, r] += 1
        answers[t, r + 1] += 1
    scores = dict(S=saved_s[ids])
    for j, name in enumerate(('f1', 'f2', 'f3', 'f4', 'f5')):
        scores[name] = raw[ids, j]
    scores.update(head_degree=np.asarray([degree[int(h)] for h, r, t in target], dtype=float),
                  tail_degree=np.asarray([degree[int(t)] for h, r, t in target], dtype=float),
                  same_query_retained_answers=np.asarray([answers[int(h), int(r)] for h, r, t in target], dtype=float))
    io.require(np.array_equal(scores['f3'], scores['same_query_retained_answers']),
               'f3/query answer comparator identity differs')
    return scores, dict(S_exact=True, S_max_absolute_error=max_error,
                        percentile_population_queries=len(queries), retained_forward_facts=len(retained),
                        evaluated_deleted_queries=len(target), labels_used_for_score=False)


def evaluate(graph, phase, queries, scores, refs, names=FEATURES, sampled=False):
    rows = []
    for seed in SEEDS:
        ranks = refs[seed]
        io.require(len(ranks) == len(queries) and np.all(ranks >= 1), 'R labels/query dimensions')
        for direction in ('forward', 'inverse'):
            population = queries[:, 1] % 2 == int(direction == 'inverse')
            selected = population & ((ranks <= 10) | (ranks > 100))
            labels = ranks[selected] <= 10
            counts = dict(positive_count=int(labels.sum()), negative_count=int((~labels).sum()),
                          middle_excluded=int((population & (ranks > 10) & (ranks <= 100)).sum()),
                          cumulative_direction_queries=int(population.sum()))
            for name in names:
                value = auc(labels, scores[name][selected])
                rows.append(dict(dataset=graph, request=phase + 1, phase=phase, direction=direction,
                    R_seed=seed, feature=name, AUC=value,
                    AUC_S_component_direction=(1 - value if name == 'f3' and value is not None else value),
                    primary_orientation='increasing', S_component_orientation='decreasing' if name == 'f3' else 'increasing',
                    population='cumulative_deleted_fixed_sample' if sampled else 'all_cumulative_deleted',
                    sampled=sampled, **counts))
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['request'], row['phase'], row['direction'], row['feature']].append(row)
    out = []
    for key, block in sorted(groups.items()):
        io.require(len(block) == 3 and {r['R_seed'] for r in block} == set(SEEDS), 'Require exactly three R seeds')
        result = dict(zip(('dataset', 'request', 'phase', 'direction', 'feature'), key))
        result.update(seeds=3, population=block[0]['population'], sampled=block[0]['sampled'])
        for field in ('AUC', 'AUC_S_component_direction', 'positive_count', 'negative_count',
                      'middle_excluded', 'cumulative_direction_queries'):
            values = [r[field] for r in block]
            defined = all(v is not None for v in values)
            result[field + '_mean'] = math.fsum(values) / 3 if defined else None
            result[field + '_min'] = min(values) if defined else None
            result[field + '_max'] = max(values) if defined else None
        out.append(result)
    return out


