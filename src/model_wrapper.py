"""Model wrapper for generating actions from prompts.
Few-shot prompt guides model to output numbered action lines.
"""
from __future__ import annotations
from typing import List, Dict

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # lightweight fallback for environment without deps yet
    AutoModelForSeq2SeqLM = object  # type: ignore
    AutoTokenizer = object  # type: ignore

_MODEL_NAME = "google/flan-t5-base"
_tokenizer = None
_model = None

FEW_SHOT_PREFIX = (
    "You are an action generation assistant. Output each action as a line:"
    "\n1. cd /home/user\n2. ls\n3. read notes.txt\n"
    "Rules: Keep actions minimal; do not perform destructive operations unless explicitly allowed."
)


def load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
    return _model, _tokenizer


def generate_actions(user_prompt: str, max_new_tokens: int = 96) -> str:
    model, tokenizer = load_model()
    prompt = FEW_SHOT_PREFIX + "\nUser Goal: " + user_prompt + "\nActions:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Post-process: extract after 'Actions:' if repeated
    if "Actions:" in text:
        text = text.split("Actions:", 1)[-1].strip()
    return text


def parse_actions(raw: str) -> List[Dict]:
    actions: List[Dict] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        # Normalize numbering
        if line[0].isdigit() and '.' in line[:4]:
            line = line.split('.', 1)[1].strip()
        # Simple tokenization: command arg1 arg2
        parts = line.split()
        action = parts[0]
        args = parts[1:]
        actions.append({
            "step": idx,
            "action": action,
            "args": {f"arg{i}": a for i, a in enumerate(args)},
            "raw": line
        })
    return actions

__all__ = ["generate_actions", "parse_actions", "load_model"]
