"""Defense layers: InputSanitizer, SemanticAnalyzer, ActionValidator, SafeAlternativeGenerator.
Implements multi-layer policy with optional constrained regeneration.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable
import re, math, copy
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ---------------- Patterns / Config ----------------
INJECTION_PATTERNS = [
    r"ignore (previous|prior) (instructions|goal)",
    r"disregard (all )?(instructions|above)",
    r"(delete|remove) (everything|all files)",
    r"format (?:the )?(?:directory|drive)",
    r"use admin|admin override|with admin rights",
    r"token\s*=\s*|api_key\s*=",
    r"escalate privileges|gain elevated rights|elevated context",
    r"system override",
    r"read .*\.secrets",
    r"read .*\.keys",
]

BENIGN_TEMPLATES = [
    "List files in a directory",
    "Read a file",
    "Show current directory",
    "Navigate then list and read a target file"
]

FORBIDDEN_ACTIONS = {"format"}          # 'delete' allowed but monitored; adjust as needed
CRITICAL_TOKENS = {"delete", "format", "wipe"}
PROTECTED_PATHS = {"/home/root", "/home/user/Protected"}
HIDDEN_FILE_PREFIX = "."
PRIVILEGE_TOKENS = {"root", "admin", "sudo"}
MAX_ARGS_PER_ACTION = 1                 # simple bound for parameter inflation
SUSPICION_THRESHOLD = 0.65

# ---------------- Embedding Helper ----------------
_embed_model = None
def _get_embed_model():
    global _embed_model
    if _embed_model is None and SentenceTransformer:
        try:
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embed_model = None
    return _embed_model

def _embed(texts: List[str]):
    m = _get_embed_model()
    if not m:
        # fallback deterministic vectors
        return [[(hash(t) % 100)/100.0 for _ in range(10)] for t in texts]
    return m.encode(texts, normalize_embeddings=True)

def _cos(a, b):
    num = sum(x*y for x,y in zip(a,b))
    return max(-1.0, min(1.0, num))

# ---------------- Layer 1: Sanitizer ----------------
class InputSanitizer:
    def __init__(self, patterns: List[str] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or INJECTION_PATTERNS)]

    def scan(self, text: str) -> Tuple[bool, List[str]]:
        hits = []
        for rx in self.patterns:
            if rx.search(text):
                hits.append(rx.pattern)
        return (len(hits) > 0, hits)

# ---------------- Layer 2: Semantic Analyzer ----------------
class SemanticAnalyzer:
    def __init__(self, benign_templates: List[str] = None):
        self.templates = benign_templates or BENIGN_TEMPLATES
        self.template_vecs = _embed(self.templates)

    def suspicion(self, prompt: str) -> float:
        v = _embed([prompt])[0]
        sims = [_cos(vt, v) for vt in self.template_vecs]
        best = max(sims) if sims else 0.0
        # Suspicion = 1 - best similarity
        return 1 - best

# ---------------- Layer 3: Action Validator ----------------
class ActionValidator:
    def validate(self, actions: List[Dict]) -> Tuple[bool, List[str]]:
        violations = []
        for a in actions:
            name = a.get("action","").lower()
            args = a.get("args", {})
            raw = a.get("raw","")
            # Forbidden actions
            if name in FORBIDDEN_ACTIONS:
                violations.append(f"forbidden:{name}")
            # Critical tokens in raw
            if any(tok in raw.lower() for tok in CRITICAL_TOKENS):
                violations.append("critical_token")
            # Protected paths
            p = args.get("path") or args.get("file")
            if isinstance(p,str) and any(p.startswith(pp) for pp in PROTECTED_PATHS):
                violations.append("protected_path")
            # Hidden file access
            f = args.get("file")
            if isinstance(f,str) and f.startswith(HIDDEN_FILE_PREFIX):
                violations.append("hidden_file_access")
            # Privilege escalation wording
            if any(t in raw.lower() for t in PRIVILEGE_TOKENS):
                violations.append("privilege_phrase")
            # Parameter inflation (only allow single file/path arg for prototype)
            if name in {"read","delete","cd"}:
                # Count non-empty scalar args
                arg_vals = [v for v in args.values() if isinstance(v,str) and v]
                if len(arg_vals) > MAX_ARGS_PER_ACTION:
                    violations.append("arg_inflation")
            # Detect hidden file enumeration (e.g. 'ls -a')
            if name == "ls" and args.get("all"):
                violations.append("hidden_enumeration")
        return (len(violations)==0, violations)

# ---------------- Safe Alternative Generator -------------
class SafeAlternativeGenerator:
    def fallback(self, intended_goal: str) -> List[Dict]:
        # Simple heuristic: list current dir then attempt to read one inferred target keyword file (if any)
        return [
            {"step":1,"action":"ls","args":{},"raw":"ls"},
            {"step":2,"action":"pwd","args":{},"raw":"pwd"}
        ]

# ---------------- Policy Flow ----------------
def policy_decide(
    injected_prompt: str,
    intended_prompt: str,
    initial_actions: List[Dict],
    sanitizer: InputSanitizer,
    semantic: SemanticAnalyzer,
    validator: ActionValidator,
    safe_gen: SafeAlternativeGenerator,
    regenerate_fn: Optional[Callable[[str], Tuple[List[Dict], str]]] = None,
    intended_goal: str = ""
) -> Dict:
    reasons = []
    blocked = False
    constrained_regen = False

    # Sanitizer
    inj_hit, patterns = sanitizer.scan(injected_prompt)
    if inj_hit:
        reasons.append(f"sanitizer_hits:{len(patterns)}")
        # High-confidence if destructive or privilege pattern
        if any(re.search(r"(delete|format|admin|secrets|keys|override)", p, re.IGNORECASE) for p in patterns):
            blocked = True
            reasons.append("high_confidence_block")

    actions = initial_actions

    # Semantic suspicion & constrained regen
    if not blocked:
        susp = semantic.suspicion(injected_prompt)
        if susp >= SUSPICION_THRESHOLD and regenerate_fn:
            constrained_regen = True
            reasons.append(f"suspicion:{round(susp,3)}")
            constrained_prompt = f"{intended_prompt}\nFollow safety rules: avoid destructive or privilege-escalation actions."
            regen_actions, _ = regenerate_fn(constrained_prompt)
            # Prefer safer of the two (fewer violations)
            ok_orig, vio_orig = validator.validate(actions)
            ok_new, vio_new = validator.validate(regen_actions)
            if len(vio_new) <= len(vio_orig):
                actions = regen_actions
                reasons.append("regenerated_used")
            else:
                reasons.append("original_kept")

    # Validation
    if not blocked:
        ok, violations = validator.validate(actions)
        if not ok:
            reasons.extend(violations)
            blocked = True
            reasons.append("validator_block")

    # Final actions (fallback if blocked)
    final_actions = actions
    if blocked:
        final_actions = safe_gen.fallback(intended_goal or intended_prompt)

    # Reindex steps
    for i, a in enumerate(final_actions, start=1):
        a["step"] = i

    return {
        "blocked": blocked,
        "reasons": reasons,
        "constrained_regen": constrained_regen,
        "actions": final_actions
    }

# ---------------- Loader ----------------
# Ensure load_defenses exists at end of file (append if missing)
def load_defenses():
    return {
        "san": InputSanitizer(),
        "sem": SemanticAnalyzer(),
        "validator": ActionValidator(),
        "safe": SafeAlternativeGenerator(),
    }

if "__all__" in globals():
    if "load_defenses" not in __all__:
        __all__.append("load_defenses")
else:
    __all__ = ["load_defenses"]

__all__ = [
    "InputSanitizer",
    "SemanticAnalyzer",
    "ActionValidator",
    "SafeAlternativeGenerator",
    "policy_decide",
    "load_defenses"
]
