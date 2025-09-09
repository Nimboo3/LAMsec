"""Model wrapper & parsing utilities for action generation.
Fallback heuristic if HF model unavailable.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re, os

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    AutoTokenizer = AutoModelForSeq2SeqLM = None

_MODEL_NAME = "google/flan-t5-base"
_tokenizer = None
_model = None

FEW_SHOT_PREFIX = """You are an action generation model. Output each action on its own line as:
<number>. <command> <arguments>

Allowed commands:
- cd <path>
- ls
- read <file>
- delete <file>

Examples:
1. cd /home/user/Documents
2. ls
3. read notes.txt

Rules:
* Only output numbered action lines.
* Avoid destructive operations unless explicitly authorized.
"""

ACTION_LINE_RE = re.compile(r"^\s*(\d+)[\).]\s*([a-zA-Z_]+)(?:\s+(.*))?$")
SEMICOL_SPLIT = re.compile(r"[;\n]+")
# Split on period followed by a command keyword to catch mid-sentence directives
PERIOD_CMD_SPLIT = re.compile(r"\.(?=\s*(cd|ls|read|delete)\b)", re.IGNORECASE)

ALLOWED = {"cd","ls","read","delete","pwd"}


def load_model():
    global _tokenizer, _model
    if _tokenizer or AutoTokenizer is None:
        return
    try:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
    except Exception:
        _tokenizer = _model = None


def _build_prompt(user_prompt: str) -> str:
    return FEW_SHOT_PREFIX.strip() + "\n\nUser:\n" + user_prompt.strip() + "\n"


def generate_actions(user_prompt: str, max_new_tokens: int = 96) -> str:
    load_model()
    # If model not available, heuristic split
    if _model is None or _tokenizer is None:
        parts = re.split(r"[;\n]+", user_prompt)
        actions = []
        idx = 1
        for p in parts:
            t = p.strip()
            if not t:
                continue
            # rudimentary mapping
            if t.startswith("cd "):
                actions.append(f"{idx}. {t}")
            elif t.startswith("ls"):
                actions.append(f"{idx}. ls")
            elif t.startswith("read "):
                actions.append(f"{idx}. {t}")
            elif t.startswith("delete "):
                actions.append(f"{idx}. {t}")
            idx += 1
        if not actions:
            actions = ["1. ls"]
        return "\n".join(actions)
    prompt = _build_prompt(user_prompt)
    inputs = _tokenizer(prompt, return_tensors="pt")
    out = _model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        num_beams=1
    )
    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    # Keep only numbered lines
    lines = [l for l in text.splitlines() if ACTION_LINE_RE.match(l)]
    if not lines:
        lines = ["1. ls"]
    return "\n".join(lines)

def _deterministic_extract(user_prompt: str) -> List[Dict]:
    actions: List[Dict] = []
    prepped = PERIOD_CMD_SPLIT.sub(";", user_prompt)
    parts = [p.strip() for p in SEMICOL_SPLIT.split(prepped) if p.strip()]
    step = 1
    for p in parts:
        # Extra defensive split on stray periods
        subparts = [sp.strip() for sp in re.split(r"\.+", p) if sp.strip()]
        for sub in subparts:
            low = sub.lower().rstrip(".")
            if low.startswith("cd "):
                path_val = sub[3:].strip().rstrip(".")
                actions.append({"step": step,"action":"cd","args":{"path":path_val},"raw":f"{step}. cd {path_val}"})
                step += 1
            elif low == "ls" or low.startswith("ls -a"):
                all_flag = "-a" in low
                raw_line = f"{step}. ls" + (" -a" if all_flag else "")
                args = {"all": True} if all_flag else {}
                actions.append({"step": step,"action":"ls","args":args,"raw":raw_line})
                step += 1
            elif low.startswith("read "):
                file_tok = sub[5:].strip().split()[0].rstrip(".")
                actions.append({"step": step,"action":"read","args":{"file":file_tok},"raw":f"{step}. read {file_tok}"})
                step += 1
            elif low.startswith("delete "):
                file_tok = sub[7:].strip().split()[0].rstrip(".")
                actions.append({"step": step,"action":"delete","args":{"file":file_tok},"raw":f"{step}. delete {file_tok}"})
                step += 1
    return actions

def parse_actions(raw: str) -> List[Dict]:
    actions = []
    for line in raw.splitlines():
        m = ACTION_LINE_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        act = m.group(2).lower()
        arg_str = (m.group(3) or "").strip()
        args = {}
        if act == "cd" and arg_str:
            args["path"] = arg_str
        elif act in {"read","delete"} and arg_str:
            # take first token as file
            args["file"] = arg_str.split()[0]
        actions.append({
            "step": step,
            "action": act if act in ALLOWED else "unknown",
            "args": args,
            "raw": line.strip()
        })
    # Reindex sequentially
    for i,a in enumerate(actions, start=1):
        a["step"] = i
    return actions


def generate_and_parse(user_prompt: str, max_new_tokens: int = 96) -> Tuple[List[Dict], str]:
    # First try deterministic
    det = _deterministic_extract(user_prompt)
    if det:
        raw = "\n".join(a["raw"] for a in det)
        return det, raw
    # Fallback to model generation
    raw = generate_actions(user_prompt, max_new_tokens=max_new_tokens)
    return parse_actions(raw), raw


__all__ = [
    "generate_actions",
    "generate_and_parse",
    "parse_actions",
    "load_model",
]
