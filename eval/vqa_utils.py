"""Shared VQA answer normalization and soft scoring (VQAv2 convention).

soft score = min(#annotators giving that answer / 3, 1) -- used by both
TextVQA and A-OKVQA (direct answer setting).
"""
from __future__ import annotations

import re

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-<>@`?,!.']")
_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
_CONTRACTIONS = {"aint": "ain't", "isnt": "isn't", "arent": "aren't",
                 "cant": "can't", "dont": "don't", "wont": "won't",
                 "wouldnt": "wouldn't", "couldnt": "couldn't",
                 "shouldnt": "shouldn't", "wasnt": "wasn't",
                 "werent": "weren't", "hasnt": "hasn't", "havent": "haven't"}


def normalize_answer(ans: str) -> str:
    ans = ans.strip().lower().replace("\n", " ")
    ans = _PUNCT.sub("", ans)
    ans = ans.replace(":", " ")
    words = []
    for w in ans.split():
        w = _NUMBERS.get(w, w)
        w = _CONTRACTIONS.get(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(words)


def soft_score(pred: str, gold_answers: list[str]) -> float:
    """VQA accuracy for one sample: min(matches/3, 1)."""
    p = normalize_answer(pred)
    matches = sum(normalize_answer(g) == p for g in gold_answers)
    return min(matches / 3.0, 1.0)


def exact_or_contained(pred: str, gold_answers: list[str]) -> float:
    """Lenient fallback: 1 if any normalized gold answer equals or is
    contained in the prediction (useful for chatty models)."""
    p = normalize_answer(pred)
    for g in gold_answers:
        g = normalize_answer(g)
        if g and (g == p or f" {g} " in f" {p} "):
            return 1.0
    return 0.0
