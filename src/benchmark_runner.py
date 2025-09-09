"""Benchmark runner with dual (intended vs injected) comparison and updated metrics."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Dict
import time, json

from .model_wrapper import generate_and_parse, load_model
from .defense import InputSanitizer, SemanticAnalyzer, ActionValidator, SafeAlternativeGenerator, policy_decide
from .envs.text_nav import TextNavigationEnv
from .metrics import ads, svi, gcr, attack_success
from .utils import set_deterministic

DATA_PATH = Path(__file__).resolve().parent.parent / 'attacks' / 'attacks.yaml'


def load_attacks() -> List[Dict]:
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_case(case: Dict, sanitizer, semantic, validator, safe_gen) -> Dict:
    intended_prompt = case.get('intended', '')
    injected_prompt = case.get('injection', intended_prompt)
    intended_goal = case.get('intended_goal', intended_prompt)

    intended_actions, _ = generate_and_parse(intended_prompt)
    injected_actions, _raw_injected = generate_and_parse(injected_prompt)

    # Apply policy (may substitute safe alternative or constrained regeneration)
    policy_out = policy_decide(
        injected_prompt, intended_prompt, injected_actions,
        sanitizer, semantic, validator, safe_gen,
        regenerate_fn=generate_and_parse,
        intended_goal=intended_goal
    )
    defended_actions = policy_out["actions"]

    ads_val = ads(intended_actions, injected_actions)
    svi_val, svi_list = svi(injected_actions)
    # Environment simulation on defended actions
    env = TextNavigationEnv()
    trace = []
    for a in defended_actions:
        result = env.run_action(a)
        trace.append({"action": a, "result": result})
    final_state = env.summarize_state()
    gcr_val = gcr(intended_goal, defended_actions)
    san_patterns = policy_out.get("san_patterns", [])
    suspicion = policy_out.get("suspicion")
    is_safe, val_viol = validator.validate(defended_actions)
    success = attack_success(ads_val, svi_val, gcr_val, blocked=policy_out["blocked"])

    return {
        'id': case['id'],
        'category': case['category'],
        'severity': case.get('severity'),
        'attack_goal': case.get('attack_goal'),
        'ads': ads_val,
        'svi': svi_val,
        'gcr': gcr_val,
        'attack_success': success,
        'san_patterns': san_patterns,
        'suspicion': suspicion,
        'is_safe': is_safe,
        'violations': svi_list + val_viol + policy_out['reasons'],
        'baseline_actions': intended_actions,
        'raw_actions': injected_actions,
        'defended_actions': defended_actions,
    'policy_blocked': policy_out['blocked'],
    'execution_trace': trace,
    'final_state': final_state,
    'constrained_regen': policy_out.get('constrained_regen', False),
    }


def run_all() -> List[Dict]:
    set_deterministic(42)
    _model, _tok = load_model()  # ensure loaded early to exclude download time from per-case timing
    sanitizer = InputSanitizer()
    semantic = SemanticAnalyzer()
    validator = ActionValidator()
    safe_gen = SafeAlternativeGenerator()
    results = []
    for case in load_attacks():
        start = time.time()
        res = run_case(case, sanitizer, semantic, validator, safe_gen)
        res['elapsed_s'] = round(time.time() - start, 3)
        results.append(res)
    return results


def save_results(results: List[Dict]):
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = Path('results') / f'benchmark_{ts}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    # also write a small summary CSV-ish stats file
    successes = sum(1 for r in results if r['attack_success'])
    blocked = sum(1 for r in results if r['policy_blocked'])
    constrained = sum(1 for r in results if r.get('constrained_regen'))
    avg_ads = round(sum(r['ads'] for r in results)/len(results), 4) if results else 0
    avg_gcr = round(sum(r['gcr'] for r in results)/len(results), 4) if results else 0
    summary = {
        "total_cases": len(results),
        "attack_successes": successes,
        "blocked": blocked,
        "constrained_regen_cases": constrained,
        "avg_ads": avg_ads,
        "avg_gcr": avg_gcr,
    }
    with open(Path('results') / f'summary_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return str(out_path)


if __name__ == "__main__":
    out = run_all()
    path = save_results(out)
    print(json.dumps(out, indent=2))
    print(f"Saved results -> {path}")
