# LAMsec

Prototype demonstrating prompt-injection threats & defenses for action-oriented models (Large Action Models / LAMs).

## Overview
Implements a minimal, reproducible stack:
* Toy action model wrapper (Flan-T5 based) producing line-oriented actions.
* Defense layers: regex InputSanitizer, embedding SemanticAnalyzer, rule-based ActionValidator.
* In-memory text navigation environment for safe simulation.
* Metrics (initial simplified forms): ADS, SVI, GCR.
* Attack dataset (seed 10 examples) spanning categories (goal_subversion, parameter_manipulation, context_poisoning, privilege_escalation, chained_injection).
* Gradio demo scaffold.

Full design & roadmap: see `context.MD` (authoritative).

## Repository Layout
```
attacks/attacks.yaml        # minimal attack dataset
src/                        # core modules
	model_wrapper.py          # load model + generate/parse actions
	defense.py                # defense components
	envs/text_nav.py          # text navigation environment
	metrics.py                # metrics implementations (simplified)
	benchmark_runner.py       # batch run across attacks
gradio_app.py               # demo UI scaffold
requirements.txt            # dependencies
results/                    # (output artifacts)
```

## Installation
Python 3.10+ recommended.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Benchmark Run
```
python -m src.benchmark_runner
```
Outputs JSON list of per-attack results to stdout (redirect as needed).

## Demo UI
```
python gradio_app.py
```

## Roadmap (Immediate Enhancements)
1. Improve ADS: integrate embedding semantic distance (SentenceTransformers) when available.
2. (Done) Add proper intended baseline to Gradio app (dual prompt fields).
3. Persist run logs to `results/` with timestamp.
4. Add Jupyter / Colab notebooks (`notebooks/demo_colab.ipynb`).
5. Expand environment safety rules & validator coverage.

### Recent Updates
* Enhanced action parser (handles concatenated numbering).
* Added embedding-backed ADS / GCR (with graceful fallback).
* Dual-run benchmark (intended vs injected) with per-case timing and JSON persistence.
* Gradio UI: dataset sample selection, defense modes (baseline/regex-only/full), policy block & safe alternative, execution trace.
* Policy flow with SafeAlternativeGenerator and environment simulation in benchmark + UI.

## License
TBD (add an OSS license appropriate for research reproducibility).

## Disclaimer
Research prototype; not production hardened. Avoid connecting to real systems without additional safeguards.