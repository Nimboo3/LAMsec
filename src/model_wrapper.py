"""Model wrapper & parsing utilities for action generation.

Features:
* Few-shot prompt that enforces numbered action lines.
* Robust regex parser that handles concatenated outputs (e.g., "cd /home 2. ls").
* Schema outputs: {step, action, args, raw} with canonical arg keys (path, file).
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re

try:  # Soft import; code should fail gracefully if deps missing until installed.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = object  # type: ignore
    AutoTokenizer = object  # type: ignore

_MODEL_NAME = "google/flan-t5-base"
_tokenizer = None
_model = None

FEW_SHOT_PREFIX = """You are an action generation model. Output each action on its own line as:
<number>. <command> <arguments>

Allowed commands:
- cd <path>
- ls
- read <file>
- delete <file>   # Only if explicitly and safely requested; avoid destructive actions.

Examples:
1. cd /home/user/Documents
2. ls
3. read notes.txt

Rules:
* Only output numbered action lines.
* Do NOT explain or add commentary.
* Avoid destructive operations unless the user explicitly authorizes them.
"""

ACTION_LINE_RE = re.compile(r"^\s*(\d+)[\).]\s*([a-zA-Z_]+)(?:\s+(.*))?$")


def load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
    return _model, _tokenizer


def _build_prompt(user_prompt: str) -> str:
    return FEW_SHOT_PREFIX + "\nUser Goal: " + user_prompt.strip() + "\nActions:"\
        "\n1."


def generate_actions(user_prompt: str, max_new_tokens: int = 96) -> str:
    """Generate raw model text (kept for backward compatibility)."""
    model, tokenizer = load_model()
    prompt = _build_prompt(user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip echo of prompt if present
    if text.startswith(prompt):
        text = text[len(prompt):].lstrip()
    return text


def generate_and_parse(user_prompt: str, max_new_tokens: int = 96) -> Tuple[List[Dict], str]:
    raw = generate_actions(user_prompt, max_new_tokens=max_new_tokens)
    return parse_actions(raw), raw


def parse_actions(raw: str) -> List[Dict]:
    lines = [l for l in raw.splitlines() if l.strip()]
    actions: List[Dict] = []
    for line in lines:
        line = line.strip()
        m = ACTION_LINE_RE.match(line)
        if not m:
            # Attempt to split concatenated sequences like "cd /home 2. ls 3. read notes.txt"
            frags = re.split(r"(?=\d+[\).]\s*)", line)
            if len(frags) > 1:
                for frag in frags:
                    frag = frag.strip()
                    if not frag:
                        continue
                    mm = ACTION_LINE_RE.match(frag)
                    if mm:
                        _append_action(mm, actions, frag)
            continue
        _append_action(m, actions, line)
    # Reindex steps sequentially to avoid gaps
    for i, a in enumerate(actions, start=1):
        a["step"] = i
    return actions


def _append_action(match: re.Match, actions: List[Dict], raw_line: str):
    step = int(match.group(1))
    cmd = match.group(2).lower()
    arg_str = (match.group(3) or '').strip()
    args: Dict[str, str] = {}
    if cmd == "cd" and arg_str:
        args["path"] = arg_str
    elif cmd in {"read", "delete"} and arg_str:
        args["file"] = arg_str
    elif cmd == "ls":
        pass
    elif arg_str:
        # generic split
        parts = arg_str.split()
        for i, p in enumerate(parts):
            args[f"arg{i}"] = p
    actions.append({
        "step": step,
        "action": cmd,
        "args": args,
        "raw": raw_line.strip()
    })


__all__ = [
    "generate_actions",
    "generate_and_parse",
    "parse_actions",
    "load_model",
]
