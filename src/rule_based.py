"""Rule-based medical text simplification: jargon substitution, sentence splitting, parenthetical removal."""

from __future__ import annotations

import json
import re
from typing import List


LATIN_MARKERS = [
    "in vivo", "in vitro", "in situ", "ex vivo", "et al", "i.e.", "e.g.",
    "per os", "ad libitum", "vs.", "cf.", "ibid.",
]

_LATIN_MARKER_PATTERN = re.compile(
    r"\(" +
    r"(?:[^()]*\b(?:" +
    r"|".join(re.escape(m) for m in LATIN_MARKERS) +
    r")\b[^()]*)" +
    r"\)",
    re.IGNORECASE,
)

_LATIN_SUFFIX = re.compile(r"\b[a-z]{4,}(?:us|um|ae|is)\b", re.IGNORECASE)

_FANBOYS = re.compile(r",\s+(for|and|nor|but|or|yet|so)\s+", re.IGNORECASE)


class RuleBasedSimplifier:
    def __init__(self, dict_path: str, split_threshold: int = 30, min_fragment: int = 5):
        self.dictionary = self.load_dict(dict_path)
        self.split_threshold = split_threshold
        self.min_fragment = min_fragment
        # Pre-compile substitution patterns, longest term first to avoid partial overlap
        self._patterns = [
            (re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), synonym)
            for term, synonym in sorted(self.dictionary.items(), key=lambda x: -len(x[0]))
        ]

    @staticmethod
    def load_dict(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def remove_parentheticals(self, text: str) -> str:
        def _has_latin(match: re.Match) -> bool:
            inner = match.group(0)[1:-1]
            if _LATIN_MARKER_PATTERN.search(match.group(0)):
                return True
            return bool(_LATIN_SUFFIX.search(inner))

        def _replacer(match: re.Match) -> str:
            return "" if _has_latin(match) else match.group(0)

        result = re.sub(r"\([^()]*\)", _replacer, text)
        return re.sub(r"\s{2,}", " ", result).strip()

    def substitute_jargon(self, text: str) -> str:
        for pattern, synonym in self._patterns:
            text = pattern.sub(synonym, text)
        return text

    def split_long_sentences(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: List[str] = []
        for sent in sentences:
            tokens = sent.split()
            if len(tokens) <= self.split_threshold:
                result.append(sent)
                continue
            split_done = False
            # Try semicolon first
            parts = sent.split(";", 1)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                if len(a.split()) >= self.min_fragment and len(b.split()) >= self.min_fragment:
                    result.extend([a + ".", b])
                    split_done = True
            # Try FANBOYS comma
            if not split_done:
                m = _FANBOYS.search(sent)
                if m:
                    a = sent[: m.start()].strip()
                    b = (m.group(1).capitalize() + " " + sent[m.end():]).strip()
                    if len(a.split()) >= self.min_fragment and len(b.split()) >= self.min_fragment:
                        result.extend([a + ".", b])
                        split_done = True
            if not split_done:
                result.append(sent)
        return " ".join(result)

    def simplify(self, text: str) -> str:
        text = self.remove_parentheticals(text)
        text = self.substitute_jargon(text)
        text = self.split_long_sentences(text)
        return text
