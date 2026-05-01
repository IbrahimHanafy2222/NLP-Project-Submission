"""Hybrid simplification pipeline: rule-based jargon substitution followed by T5-small generation."""

from __future__ import annotations

from typing import List

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.rule_based import RuleBasedSimplifier

PREFIX = "simplify: "
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4


class HybridPipeline:
    def __init__(self, dict_path: str, checkpoint_path: str, device: str | None = None) -> None:
        """Load jargon substituter and T5-small checkpoint."""
        self.substituter = RuleBasedSimplifier(dict_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(checkpoint_path)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def simplify(self, text: str) -> str:
        """Substitute jargon then generate simplified text with T5-small."""
        substituted = self.substituter.substitute_jargon(text)
        input_text = PREFIX + substituted
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                max_new_tokens=MAX_NEW_TOKENS,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def batch_simplify(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Run simplify over texts in batches; return results in original order."""
        results: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            substituted = [self.substituter.substitute_jargon(t) for t in batch]
            input_texts = [PREFIX + t for t in substituted]
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=MAX_INPUT_TOKENS,
                truncation=True,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    num_beams=NUM_BEAMS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    early_stopping=True,
                )
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)
        return results
