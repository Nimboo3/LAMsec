"""Defense layers: InputSanitizer, SemanticAnalyzer, ActionValidator.
Simplified initial implementations for MVP.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Callable, Optional
import re

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

INJECTION_PATTERNS = [
    r"ignore (previous|prior) (instructions|goal)",
    r"disregard (all )?(instructions|above)",
    r"delete (everything|all)",
    r"remove all files",
    r"format (?:the )?(?:directory|drive)",
    r"use admin|admin override|with admin rights",
    r"token=|api_key=",
    r"escalate privileges|gain elevated rights|elevated context",
    r"system override",
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

__all__ = []

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
PROTECTED_PATHS = {"/home/root", "/home/user/Protected"}
HIDDEN_FILE_PREFIX = "."
PRIVILEGE_TOKENS = ["root", "admin", "sudo"]

class ActionValidator:
    def __init__(self):
        self.forbidden = FORBIDDEN_ACTIONS
        self.protected = PROTECTED_PATHS

    def validate(self, actions: List[Dict]) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        safe = True
        for a in actions:
            act = a["action"].lower()
            if act in self.forbidden:
                safe = False
                violations.append(f"forbidden:{act}")
            if act == "cd" and "path" in a.get("args", {}):
                path = a["args"]["path"]
                if any(path.startswith(p) for p in self.protected):
                    safe = False
                    violations.append("protected_path")
            if act == "delete" and "file" in a.get("args", {}):
                file = a["args"]["file"]
                if file in {"config.json"}:
                    safe = False
                    violations.append("protected_file")
            if act == "read" and "file" in a.get("args", {}):
                f = a["args"]["file"]
                if f.startswith(HIDDEN_FILE_PREFIX):
                    safe = False
                    violations.append("hidden_file_access")
            # multi-file inflation (parser may have arg1/arg2 tokens)
            extra_args = [k for k in a.get("args", {}) if k.startswith("arg")]
            if act == "read" and extra_args:
                safe = False
                violations.append("param_inflation")
            # privilege escalation phrases in raw
            raw = a.get("raw", "").lower()
            if any(tok in raw for tok in PRIVILEGE_TOKENS):
                violations.append("privilege_phrase")
        return safe, violations


class SafeAlternativeGenerator:
    """Produces a conservative fallback action list when an attack is detected."""
    def generate(self, intended_goal: str) -> list[dict]:
        base = [
            {"step": 1, "action": "pwd", "args": {}, "raw": "pwd"},
            {"step": 2, "action": "ls", "args": {}, "raw": "ls"},
        ]
        return base


def policy_decide(prompt: str, intended_prompt: str, generated_actions: list[dict],
                  sanitizer: InputSanitizer, semantic: SemanticAnalyzer,
                  validator: ActionValidator, safe_gen: SafeAlternativeGenerator,
                  suspicion_threshold: float = 0.65,
                  regenerate_fn: Optional[Callable[[str], Tuple[List[Dict], str]]] = None,
                  intended_goal: Optional[str] = None) -> dict:
    """Apply multi-layer policy and produce possibly substituted safe actions.

    Returns dict with keys: actions, blocked(bool), reasons(list)
    """
    reasons = []
    san = sanitizer.scan(prompt)
    suspicion = semantic.score(prompt) if semantic else 0.0
    is_safe, violations = validator.validate(generated_actions)
    blocked = False
    if san["is_injection"] and any(x for x in san["patterns"] if any(k in x for k in ["delete", "format", "wipe"])):
        blocked = True
        reasons.append("sanitizer_high_confidence")
    if suspicion > suspicion_threshold:
        reasons.append("high_suspicion")
    if not is_safe:
        blocked = True
        reasons.extend(violations)
    # Constrained re-generation path: if high suspicion but not already blocked by high-confidence sanitizer
    used_constrained = False
    if (not blocked) and (suspicion > suspicion_threshold) and regenerate_fn:
        constraint_goal = intended_goal or intended_prompt
        constrained_prompt = (
            f"{intended_prompt}\nSafety Constraints:\n"
            "- Avoid destructive actions (delete, format, wipe).\n"
            "- Only use navigation (cd, ls), read, or pwd if needed.\n"
            f"- Focus strictly on achieving: {constraint_goal}\n"
            "Actions:\n1."
        )
        try:
            alt_actions, _raw = regenerate_fn(constrained_prompt)
            alt_safe, alt_viol = validator.validate(alt_actions)
            # Prefer alternative if it's safe and original had violations or alt reduces violations
            if alt_safe and (not is_safe or len(alt_viol) <= len(violations)):
                generated_actions = alt_actions
                is_safe = alt_safe
                violations = alt_viol
                reasons.append("constrained_regen")
                used_constrained = True
        except Exception:  # pragma: no cover - resilience
            pass
    if blocked:
        alt = safe_gen.generate(intended_prompt)
        return {"actions": alt, "blocked": True, "reasons": reasons, "suspicion": suspicion, "san_patterns": san["patterns"], "constrained": used_constrained}
    return {"actions": generated_actions, "blocked": False, "reasons": reasons, "suspicion": suspicion, "san_patterns": san["patterns"], "constrained": used_constrained}

__all__.extend([
    "InputSanitizer", "SemanticAnalyzer", "ActionValidator",
    "SafeAlternativeGenerator", "policy_decide"
])
