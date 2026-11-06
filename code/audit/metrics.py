"""Public API for per-query cross-fitted metrics and rank-cache analysis."""
from igpu.audit import cache, run
from igpu.probes import execute as alternative_answer_probe
from igpu.deletion_attack import main as deletion_inference
from .rebound import leak_stat
from .score_auc import auc

