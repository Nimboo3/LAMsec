"""Defense layers: InputSanitizer, SemanticAnalyzer, ActionValidator.
Simplified initial implementations for MVP.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

INJECTION_PATTERNS = [
    r"ignore (previous|prior) (instructions|goal)",
    r"delete (everything|all)",
    r"format (?:the )?(?:directory|drive)",
    r"use admin", r"token=", r"escalate privileges",
]

BENIGN_TEMPLATES = [
    "List files in a directory",
    "Read a file",
    "Show current directory",
]

_model_cache = None

def _load_embed_model():
    global _model_cache
    if _model_cache is None and SentenceTransformer:
        _model_cache = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _model_cache

class InputSanitizer:
    def __init__(self, patterns=None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or INJECTION_PATTERNS)]

    def scan(self, text: str) -> Dict:
        matches = [p.pattern for p in self.patterns if p.search(text)]
        return {"is_injection": bool(matches), "patterns": matches}

class SemanticAnalyzer:
    def __init__(self, benign_templates: List[str] | None = None):
        self.templates = benign_templates or BENIGN_TEMPLATES
        self.model = _load_embed_model()

    def score(self, text: str) -> float:
        if not self.model:
            return 0.0  # fallback
        emb_text = self.model.encode([text], normalize_embeddings=True)
        emb_temp = self.model.encode(self.templates, normalize_embeddings=True)
        import numpy as np
        sims = (emb_text @ emb_temp.T)[0]
        # Suspicion = 1 - max similarity
        return float(1.0 - sims.max())

FORBIDDEN_ACTIONS = {"delete", "format"}

class ActionValidator:
    def __init__(self):
        self.forbidden = FORBIDDEN_ACTIONS

    def validate(self, actions: List[Dict]) -> Tuple[bool, List[str]]:
        violations = []
        for a in actions:
            if a["action"].lower() in self.forbidden:
                violations.append(f"Forbidden action: {a['action']}")
        return (len(violations) == 0, violations)

__all__ = ["InputSanitizer", "SemanticAnalyzer", "ActionValidator"]
