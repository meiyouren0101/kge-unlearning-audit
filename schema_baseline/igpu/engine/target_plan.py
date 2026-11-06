"""Deterministic retained-target plans; standard library, no data IO or ML."""
from collections import Counter, defaultdict
import hashlib
import json
import random

ARMS = ('B_original', 'H_exclude_deleted', 'T_retained_shuffled', 'Q_retained_aligned')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def local_seed(seed, tag):
    return int.from_bytes(hashlib.sha256(f'graphdpo-retained-target-v1/20261021/{seed}/{tag}'.encode()).digest()[:8], 'little')


def triple(x):
    require(type(x) in (list, tuple) and len(x) == 3
            and all(type(v) is int and v >= 0 for v in x), 'Integer triple required')
    h, r, t = x
    return (h, r, t) if r % 2 == 0 else (t, r-1, h)


def inverse(x):
    h, r, t = x
    return t, r ^ 1, h


def context(fact, is_head):
    h, r, t = fact
    return (t, r+1) if is_head else (h, r)


def aligned(fact, request, is_head):
    return fact[1] == request[1] and (fact[2] == request[2] if is_head else fact[0] == request[0])


def augmented(rows):
    out = []
    for fact in rows:
        out.extend((fact, inverse(fact)))
    return out


def counter_digest(rows):
    return digest([(list(k), v) for k, v in sorted(Counter(rows).items())])


class TargetPlan:
    def __init__(self, data, reconstructed, seed):
        require(type(seed) is int, 'Integer seed')
        self.seed = seed
        requests = [triple(x) for x in data['phases'][2]]
        heads = data['is_heads'][2]
        n, entities = len(requests), data['counts']['entities']
        require(n > 0 and len(heads) == n and all(type(v) is bool for v in heads), 'Request/side roster')
        retained = {triple(x) for x in data['retained']}
        deleted = {triple(x) for x in data['cumulative']}
        require(not retained & deleted and set(requests) <= deleted, 'Retained/deleted partition')
        require(len(reconstructed['positive']) == len(reconstructed['negative']) == len(reconstructed['facts']) == 2*n,
                'Original augmented roster')
        old = []
        for i, req in enumerate(requests):
            pos = tuple(reconstructed['positive'][2*i])
            require(pos[1] % 2 == 0 and aligned(pos, req, heads[i]), 'Original designated context differs')
            require(reconstructed['positive'][2*i+1] == inverse(pos)
                    and tuple(reconstructed['negative'][2*i]) == req
                    and reconstructed['negative'][2*i+1] == inverse(req), 'Original inverse block differs')
            require(all(reconstructed['facts'][j] == {'fact': reconstructed['positive'][j], 'label': 1}
                        for j in (2*i, 2*i+1)), 'Original margin/DPO mismatch')
            old.append(pos)
        replay = set()
        for row in reconstructed['boundary']:
            require(row['label'] == 1 and triple(row['fact']) in retained, 'Replay not retained')
            replay.add(triple(row['fact']))
        unreplayed = retained - replay
        answers = defaultdict(set)
        for fact in unreplayed:
            h, r, t = fact
            answers[h, r].add(fact)
            answers[t, r+1].add(fact)
        answers = {key: sorted(value) for key, value in answers.items()}
        hygienic = list(old)
        hrng = random.Random(local_seed(seed, 'hygiene'))
        qrng = random.Random(local_seed(seed, 'aligned'))
        trng = random.Random(local_seed(seed, 'shuffle'))
        allowed, hygiene_positions = {}, []
        for i, fact in enumerate(old):
            if fact not in deleted:
                continue
            req, is_head = requests[i], heads[i]
            key = context(req, is_head)
            if key not in allowed:
                allowed[key] = [triple((key[0], key[1], e)) for e in range(entities)
                                if triple((key[0], key[1], e)) not in deleted]
            require(bool(allowed[key]), 'No legal hygiene target; no fallback repair')
            hygienic[i] = hrng.choice(allowed[key])
            hygiene_positions.append(i)
        q = list(hygienic)
        eligible = []
        strata = defaultdict(list)
        for i, req in enumerate(requests):
            candidates = answers.get(context(req, heads[i]), ())
            if candidates:
                q[i] = qrng.choice(candidates)
                eligible.append(i)
                strata[(req[1], heads[i])].append(i)
        t = list(q)
        records = []
        for (relation, is_head), positions in sorted(strata.items()):
            donors = list(positions)
            trng.shuffle(donors)  # Exactly one permutation, including fixed points.
            for recipient, donor in zip(positions, donors):
                t[recipient] = q[donor]
            records.append({'relation': relation, 'designated_head': is_head, 'n': len(positions),
                            'distinct_contexts': len({context(requests[i], heads[i]) for i in positions}),
                            'same_context_after_shuffle': sum(aligned(t[i], requests[i], heads[i]) for i in positions),
                            'same_fact_after_shuffle': sum(t[i] == q[i] for i in positions)})
        self.positives = dict(zip(ARMS, (old, hygienic, t, q)))
        self.eligible = eligible
        self.requests = requests
        self.heads = list(heads)
        self.replay = replay
        es = set(eligible)
        require(Counter(t) == Counter(q), 'T/Q full positive exposure differs')
        require(Counter(t[i] for i in eligible) == Counter(q[i] for i in eligible), 'T/Q eligible exposure differs')
        for i, req in enumerate(requests):
            require(hygienic[i] not in deleted and aligned(hygienic[i], req, heads[i]), 'Invalid H target')
            if i in es:
                require(q[i] in unreplayed and t[i] in unreplayed
                        and aligned(q[i], req, heads[i]) and t[i][1] == req[1], 'Invalid retained target')
            else:
                require(t[i] == q[i] == hygienic[i], 'Fallback differs')
        matched = sum(aligned(t[i], requests[i], heads[i]) for i in eligible)
        self.receipt = {
            'seed': seed, 'forward_requests': n, 'augmented_rows': 2*n,
            'eligible_requests': len(eligible), 'fallback_requests': n-len(eligible),
            'hygiene_changed_requests': len(hygiene_positions),
            'eligible_positions_sha256': digest(eligible),
            'hygiene_positions_sha256': digest(hygiene_positions),
            'target_array_sha256': {a: digest(v) for a, v in self.positives.items()},
            'positive_multiset_sha256': {a: counter_digest(v) for a, v in self.positives.items()},
            'T_Q_positive_multisets_equal': True,
            'T_same_context_eligible': matched, 'T_context_different_eligible': len(eligible)-matched,
            'T_same_context_fraction': matched/len(eligible) if eligible else None,
            'strata': records, 'legacy_dataset_sha256': reconstructed['legacy_dataset_sha256'],
            'sampler_seeds': {tag: local_seed(seed, tag) for tag in ('hygiene', 'aligned', 'shuffle')},
            'raw_target_ids_exported': False,
        }

    def require_contrast(self):
        require(self.receipt['eligible_requests'] > 0, 'No eligible retained-answer candidate')
        require(self.receipt['T_context_different_eligible'] > 0, 'No realized context contrast; no retry shuffle')

    def apply(self, dataset, arm):
        require(arm in ARMS and dataset.pos_facts == augmented(self.positives[ARMS[0]]), 'Original runtime positives differ')
        require(dataset.neg_facts == augmented(self.requests), 'Runtime rejected requests differ')
        if arm != ARMS[0]:
            dataset.pos_facts = augmented(self.positives[arm])
            dataset.facts = [{'fact': x, 'label': 1} for x in dataset.pos_facts]
        require(all(row == {'fact': pos, 'label': 1} for row, pos in zip(dataset.facts, dataset.pos_facts)),
                'Margin and DPO targets must be the same')
        return digest(dataset.pos_facts)
