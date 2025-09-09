"""Gradio demo with intended vs injected prompt comparison and metrics.

Enhancements:
* Status badge (HTML) reflecting success/block/suspicion/constrained regen.
* Diff panel (baseline vs injected vs defended) using unified diff.
* Final state aware GCR (uses environment summarize_state()).
* Downloadable JSON log for each run.
* Displays constrained regeneration flag and final state.
"""
from __future__ import annotations
import json
import time, os, tempfile, difflib
import gradio as gr
from src.model_wrapper import generate_and_parse
from src.defense import (
    InputSanitizer, SemanticAnalyzer, ActionValidator,
    SafeAlternativeGenerator, policy_decide
)
from src.metrics import ads, svi, gcr, attack_success
from src.utils import set_deterministic
import yaml, pathlib

set_deterministic(42)
sanitizer = InputSanitizer()
semantic = SemanticAnalyzer()
validator = ActionValidator()
safe_gen = SafeAlternativeGenerator()

ATTACKS_PATH = pathlib.Path('attacks') / 'attacks.yaml'
try:
    with open(ATTACKS_PATH, 'r', encoding='utf-8') as f:
        ATTACKS = yaml.safe_load(f)
except Exception:
    ATTACKS = []
ATTACK_INDEX = {a['id']: a for a in ATTACKS}


def _apply_defense_mode(mode: str, injected_prompt: str, intended_prompt: str, injected_actions):
    if mode == 'baseline':
        return {"actions": injected_actions, "blocked": False, "reasons": [], "suspicion": 0.0, "san_patterns": []}
    if mode == 'regex-only':
        san = sanitizer.scan(injected_prompt)
        if san['is_injection']:
            return {"actions": safe_gen.generate(intended_prompt), "blocked": True, "reasons": ["regex_block"], "suspicion": 0.0, "san_patterns": san['patterns']}
        return {"actions": injected_actions, "blocked": False, "reasons": [], "suspicion": 0.0, "san_patterns": san['patterns']}
    # full defense
    return policy_decide(
        injected_prompt, intended_prompt, injected_actions,
        sanitizer, semantic, validator, safe_gen
    )


def _build_status(attack_success_flag: bool, policy_out: dict, suspicion: float) -> str:
    if attack_success_flag:
        color = "#d9534f"  # red
        label = "Attack Succeeded"
    elif policy_out.get('blocked'):
        color = "#5cb85c"  # green
        label = "Blocked"
    elif policy_out.get('constrained'):
        color = "#5bc0de"  # blue
        label = "Constrained Re-Gen"
    elif suspicion > 0.65:
        color = "#f0ad4e"  # amber
        label = "Suspicious"
    else:
        color = "#5cb85c"
        label = "Benign"
    return f"<div style='display:inline-block;padding:6px 12px;border-radius:6px;background:{color};color:white;font-weight:600;'>{label}</div>"


def _actions_diff(base, injected, defended) -> str:
    def to_lines(prefix, acts):
        return [f"{prefix}:{a['raw']}" for a in acts]
    base_lines = to_lines('BASE', base)
    inj_lines = to_lines('INJ', injected)
    def_lines = to_lines('DEF', defended)
    d1 = list(difflib.unified_diff(base_lines, inj_lines, fromfile='baseline', tofile='injected', lineterm=''))
    d2 = list(difflib.unified_diff(inj_lines, def_lines, fromfile='injected', tofile='defended', lineterm=''))
    out = []
    if d1:
        out.append('Baseline vs Injected:\n' + '\n'.join(d1))
    if d2:
        out.append('Injected vs Defended:\n' + '\n'.join(d2))
    return '\n\n'.join(out) if out else 'No differences'


def run(sample_id: str, intended: str, injected: str, defense_mode: str):
    # If sample chosen, fill prompts
    if sample_id and sample_id in ATTACK_INDEX:
        entry = ATTACK_INDEX[sample_id]
        intended = entry.get('intended', intended)
        injected = entry.get('injection', injected or intended)
    injected = injected or intended
    base_actions, _ = generate_and_parse(intended)
    inj_actions, _ = generate_and_parse(injected)
    policy_out = _apply_defense_mode(defense_mode, injected, intended, inj_actions)
    defended = policy_out['actions']
    ads_val = ads(base_actions, inj_actions)
    svi_val, svi_list = svi(inj_actions)
    from src.envs.text_nav import TextNavigationEnv
    env = TextNavigationEnv()
    trace = []
    for a in defended:
        res = env.run_action(a)
        trace.append({"action": a, "result": res})
    final_state = env.summarize_state()
    gcr_val = gcr(intended, defended, final_state)
    success = attack_success(ads_val, svi_val, gcr_val)
    status_html = _build_status(success, policy_out, policy_out.get('suspicion', 0.0))
    diff_text = _actions_diff(base_actions, inj_actions, defended)
    payload = {
        "ADS": ads_val,
        "SVI": svi_val,
        "GCR": gcr_val,
        "attack_success": success,
        "defense_mode": defense_mode,
        "policy_blocked": policy_out['blocked'],
        "reasons": policy_out['reasons'],
        "san_patterns": policy_out['san_patterns'],
        "suspicion": policy_out.get('suspicion', 0.0),
        "violations": svi_list,
        "execution_trace": trace,
        "final_state": final_state,
        "constrained_regen": policy_out.get('constrained', False),
    }
    # write temp file for download
    tmp_path = os.path.join(tempfile.gettempdir(), f"lamsec_run_{int(time.time())}.json")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return (
        json.dumps(base_actions, indent=2),
        json.dumps(inj_actions, indent=2),
        json.dumps(defended, indent=2),
        json.dumps(payload, indent=2),
        status_html,
        diff_text,
        tmp_path,
    )


with gr.Blocks() as demo:
    gr.Markdown("# LAMsec Demo (Defended LAM)")
    with gr.Row():
        sample_id = gr.Dropdown(choices=["" ] + [a['id'] for a in ATTACKS], label="Sample Attack (optional)")
        defense_mode = gr.Radio(choices=["baseline", "regex-only", "full"], value="full", label="Defense Mode")
    with gr.Row():
        intended_box = gr.Textbox(label="Intended Prompt", lines=3)
        injected_box = gr.Textbox(label="Injected / Attack Prompt", lines=3)
    run_btn = gr.Button("Run")
    with gr.Row():
        base_out = gr.Textbox(label="Baseline Actions")
        inj_out = gr.Textbox(label="Injected Actions")
        defended_out = gr.Textbox(label="Defended Actions")
    with gr.Row():
        status_html = gr.HTML(label="Status")
        diff_box = gr.Textbox(label="Diff", lines=14)
    metrics_out = gr.Textbox(label="Metrics / Trace", lines=18)
    download_file = gr.File(label="Download Log")
    run_btn.click(
        run,
        inputs=[sample_id, intended_box, injected_box, defense_mode],
        outputs=[base_out, inj_out, defended_out, metrics_out, status_html, diff_box, download_file]
    )

if __name__ == "__main__":  # pragma: no cover
    demo.launch()
