# kge-unlearning-audit

Code, configurations and pre-registered predictions for the paper

**Rethinking knowledge graph embedding unlearning: Retraining-referenced per-fact evaluation and inferability-gated forgetting** (under review).

The repository contains

1. a **retraining-referenced, per-fact audit** for continual KGE unlearning: cross-fitted recoverability strata defined by an independent retrained model, per-query signed and absolute log-rank deviations against a second retrained reference, the inter-retrain gap as reference level, and a probe for the substitute answers used by preference-based unlearning;
2. **IGPU** (inferability-gated preference unlearning): observed retained targets, a label-free inferability gate computed from the retained graph, and per-request repair;
3. adapters for all compared methods (fine-tuning, random labels, negative gradient, Fisher forgetting, boundary shrink, ADV-IMP, SSD, GraphDPO in its paper and public-code forms, ZeroFisher, Schema-based unlearning) and all IGPU ablations and controls;
4. the experiment pipeline used for the paper, including the predictions that were written down and hashed before the corresponding runs.

## Layout

| Path | Content |
|---|---|
| `code/igpu/` | IGPU training, repair, inferability features and gate, audit, probes, bootstrap, figures |
| `code/igpu/engine/` | Shared kernels: graph data, ranking, audit statistics, gate features, Fisher streaming, legacy GraphDPO runtime bridge |
| `code/audit/` | Audit metrics, rebound probe, deletion-inference AUC (appendix) |
| `code/baselines/` | GraphDPO (paper objective and public-code objective), native early stopping, ZeroFisher (paper formulas, autograd variant, official-code wrapper) |
| `code/configs/` | One JSON file per method / ablation / control; `configs/RotatE/`; dataset settings in `configs/datasets/` (FB15k-237, WN18RR, CoDEx-L, YAGO3-10) |
| `code/predictions/` | Pre-registered predictions with `SHA256SUMS` (main results, controls, matched deletion, gate inputs, ZeroFisher, YAGO3-10, ...) |
| `code/job20.py`, `worker20.py`, `matrix20.py` | Pipeline for the FB15k-237 / WN18RR 20% deletion runs (all methods, three seeds, four requests) |
| `code/scripts/reproduce.py` | Serial runner that emits one final-request CSV per paper table/setting |
| `schema_baseline/` | Adapter and pipeline for Schema-based unlearning (upstream code not redistributed) |

## Environment

Python 3.10, PyTorch 2.1.1 (CUDA 12.1), and the packages in `code/requirements.txt`.
The experiments in the paper ran on NVIDIA RTX 4090 and A100 GPUs with float64 parameters, deterministic algorithms enabled and TF32 disabled.

```bash
pip install -r code/requirements.txt
python code/scripts/fetch_upstream.py GraphDPO ZOWFKGIF advertorch   # pinned, hash-verified, into code/.upstream/
cd code
python -m unittest discover -s engine_tests -v                       # CPU unit tests
python -m unittest -v test_matrix20 test_scheduler20 test_seed_chains test_safe_stop
```

## Data

FB15k-237 and WN18RR use the four-request deletion splits released with GraphDPO (10% and 20% deletion ratios). CoDEx-L is rebuilt by concatenating the official train/valid/test files of https://github.com/tsafavi/codex in order and using the first four of its five public requests; YAGO3-10 is rebuilt in the same way. Expected SHA-256 values of all data archives are listed in `code/DATA_SHA256.json` and `code/igpu/assets/`.

## Running

Single setting (one dataset, backbone, method and seed, four requests):

```bash
cd code
python -m igpu.run --help
```

All settings of a paper table, serially:

```bash
cd code
python scripts/reproduce.py --help
```

The 20% deletion experiment (all methods, three seeds, six chains in parallel):

```bash
cd code
python job20.py run --help
```

`code/gpu_wait.py` implements the admission rules we used on a shared GPU server (a file lock, a free-memory check before each launch, retry rules for out-of-memory errors). The lock path is a constant in that file and should be changed for other machines.

## Pre-registration

Files in `code/predictions/` were written before the corresponding runs; their hashes are listed in `code/predictions/SHA256SUMS` and Appendix E of the paper. Predictions that did not hold are reported in the paper as well.

## Provenance

`code/PACKAGE_SHA256.json` lists every file of the experiment package together with the fetched upstream files. After running `fetch_upstream.py`, `python code/job20.py verify` checks that the code is byte-identical to the version used for the paper's runs.

## License

MIT for our code (`LICENSE`). Third-party code is fetched separately and keeps its own license (`THIRD_PARTY.md`).
