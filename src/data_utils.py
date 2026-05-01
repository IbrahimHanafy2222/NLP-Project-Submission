"""
Data pipeline utilities for medical text simplification.
All functions are stateless and importable standalone.
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd
import textstat


# ---------------------------------------------------------------------------
# T005: Sentence splitting
# ---------------------------------------------------------------------------

def split_into_sentences(text: str, nlp) -> List[str]:
    """Split text into sentences using SpaCy sentencizer.

    Returns list of stripped, non-empty sentences.
    """
    if not text or not text.strip():
        return []
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ---------------------------------------------------------------------------
# T006: Sentence pair alignment
# ---------------------------------------------------------------------------

def align_pairs(
    sources: List[str],
    targets: List[str],
    source_name: str,
) -> List[dict]:
    """Positionally align source and target sentence lists.

    Discards remainder when lists have different lengths.
    Returns list of {source, target, source_dataset} dicts.
    """
    pairs = []
    for src, tgt in zip(sources, targets):
        src = src.strip()
        tgt = tgt.strip()
        if src and tgt:
            pairs.append({
                "source": src,
                "target": tgt,
                "source_dataset": source_name,
            })
    return pairs


# ---------------------------------------------------------------------------
# T007: Statistics computation
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    return len(text.split())


def _safe_fkgl(text: str) -> float:
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return 0.0


def compute_stats(df: pd.DataFrame, stage: str, prev_count: int) -> dict:
    """Compute DatasetStatistics for a pipeline stage.

    Args:
        df: Current DataFrame after stage filter applied.
        stage: Stage name string.
        prev_count: Total pair count before this stage (0 for first stage).

    Returns:
        Dict matching DatasetStatistics entity schema.
    """
    if df.empty:
        return {
            "stage": stage,
            "total_pairs": 0,
            "plaba_pairs": 0,
            "cochrane_pairs": 0,
            "pairs_removed": prev_count,
            "avg_source_tokens": 0.0,
            "avg_target_tokens": 0.0,
            "avg_source_fkgl": 0.0,
            "avg_target_fkgl": 0.0,
        }

    src_tokens = df["source"].apply(_token_count)
    tgt_tokens = df["target"].apply(_token_count)
    src_fkgl = df["source"].apply(_safe_fkgl)
    tgt_fkgl = df["target"].apply(_safe_fkgl)

    dataset_counts = df["source_dataset"].value_counts()

    return {
        "stage": stage,
        "total_pairs": len(df),
        "plaba_pairs": int(dataset_counts.get("plaba", 0)),
        "cochrane_pairs": int(dataset_counts.get("cochrane", 0)),
        "pairs_removed": max(0, prev_count - len(df)),
        "avg_source_tokens": round(float(src_tokens.mean()), 2),
        "avg_target_tokens": round(float(tgt_tokens.mean()), 2),
        "avg_source_fkgl": round(float(src_fkgl.mean()), 2),
        "avg_target_fkgl": round(float(tgt_fkgl.mean()), 2),
    }


# ---------------------------------------------------------------------------
# T008: Length ratio filter
# ---------------------------------------------------------------------------

def apply_length_ratio_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep rows where length_ratio_min <= src_tokens/tgt_tokens <= length_ratio_max."""
    src_tokens = df["source"].apply(_token_count)
    tgt_tokens = df["target"].apply(_token_count).replace(0, 1)  # avoid div/0
    ratio = src_tokens / tgt_tokens
    mask = (ratio >= config["length_ratio_min"]) & (ratio <= config["length_ratio_max"])
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# T009: FKGL difference filter
# ---------------------------------------------------------------------------

def apply_fkgl_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep rows where FKGL(source) - FKGL(target) >= fkgl_diff_min."""
    src_fkgl = df["source"].apply(_safe_fkgl)
    tgt_fkgl = df["target"].apply(_safe_fkgl)
    diff = src_fkgl - tgt_fkgl
    mask = diff >= config["fkgl_diff_min"]
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# T010: Token count filter
# ---------------------------------------------------------------------------

def apply_token_count_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep rows within source and target token count bounds (whitespace split)."""
    src_tokens = df["source"].apply(_token_count)
    tgt_tokens = df["target"].apply(_token_count)
    mask = (
        (src_tokens >= config["min_source_tokens"])
        & (src_tokens <= config["max_source_tokens"])
        & (tgt_tokens >= config["min_target_tokens"])
        & (tgt_tokens <= config["max_target_tokens"])
    )
    return df[mask].reset_index(drop=True)
