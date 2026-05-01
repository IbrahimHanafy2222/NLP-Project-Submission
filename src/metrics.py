"""Shared evaluation metrics: SARI, BLEU, FKGL."""

from __future__ import annotations

from typing import List

import textstat
from easse.sari import corpus_sari
from sacrebleu.metrics import BLEU


def compute_sari(sources: List[str], predictions: List[str], references: List[str]) -> float:
    return float(corpus_sari(sources, predictions, [references]))


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(predictions, [references])
    return float(result.score)


def compute_fkgl(texts: List[str]) -> float:
    scores = [textstat.flesch_kincaid_grade(t) for t in texts]
    return float(sum(scores) / len(scores)) if scores else 0.0
