"""Gradio demo scaffold."""
from __future__ import annotations
import gradio as gr
from src.model_wrapper import generate_actions, parse_actions
from src.defense import InputSanitizer, SemanticAnalyzer, ActionValidator
from src.metrics import ADS, SVI, GCR

sanitizer = InputSanitizer()
semantic = SemanticAnalyzer()
validator = ActionValidator()

def run(prompt: str):
    raw = generate_actions(prompt)
    actions = parse_actions(raw)
    san = sanitizer.scan(prompt)
    suspicion = semantic.score(prompt)
    is_safe, violations = validator.validate(actions)

    ads = 0.0  # need intended baseline; placeholder
    svi = SVI(actions)
    gcr = 0.0  # placeholder without goal context

    status = "safe" if is_safe and not san['is_injection'] else "flagged"

    return raw, str(actions), (
        f"Sanitized: {san}; Suspicion: {suspicion:.2f}; Violations: {violations}; "
        f"ADS:{ads:.2f} SVI:{svi:.2f} GCR:{gcr:.2f}"
    )

with gr.Blocks() as demo:
    gr.Markdown("# LAMsec Demo (MVP)")
    prompt = gr.Textbox(label="Prompt", value="List files in /home/user/Documents")
    run_btn = gr.Button("Run")
    raw_out = gr.Textbox(label="Raw Model Output")
    actions_out = gr.Textbox(label="Parsed Actions")
    meta_out = gr.Textbox(label="Defense & Metrics")

    run_btn.click(run, inputs=[prompt], outputs=[raw_out, actions_out, meta_out])

if __name__ == "__main__":
    demo.launch()
