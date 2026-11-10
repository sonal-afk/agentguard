r"""
phase3_defense.py  -  PHASE 3: turn on AgentGuard and watch the attack fail.

Runs the SAME attacked banking task twice, fully locally (Ollama, no API):
  A) NO DEFENSE  - the baseline from Phase 2 (attack may hijack the agent).
  B) AGENTGUARD  - the action-monitor vets each money move and aborts on a hijack.

It prints both transcripts and a before/after summary, and saves everything to
runs/phase3_result.json so the presentation can be built from REAL data.

RUN:
  .\.venv\Scripts\python.exe scripts\phase3_defense.py
"""

import json
from pathlib import Path

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.task_suite.load_suites import get_suites

from agent_lib import build_local_pipeline, build_local_pipeline_with_defense, LOCAL_MODEL
from phase1_smoke_test import CaptureLogger, print_conversation, text_of, describe_tool_calls

ATTACK_NAME = "important_instructions"
USER_TASK_ID = "user_task_0"
INJECTION_TASK_ID = "injection_task_0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def serialize(messages) -> list[dict]:
    """Turn the conversation into plain JSON for the presentation builder."""
    out = []
    for m in messages:
        role = m.get("role", "?")
        entry = {"role": role, "text": text_of(m.get("content"))}
        if role == "assistant":
            calls = describe_tool_calls(m.get("tool_calls"))
            if calls:
                entry["tool_calls"] = calls
        out.append(entry)
    return out


def run_once(pipeline, suite, attack, label):
    print("\n" + "#" * 70)
    print(f"# RUN: {label}")
    print("#" * 70)
    user_task = suite.user_tasks[USER_TASK_ID]
    with CaptureLogger() as cap:
        utility_results, security_results = run_task_with_injection_tasks(
            suite=suite,
            agent_pipeline=pipeline,
            user_task=user_task,
            attack=attack,
            logdir=None,
            force_rerun=True,
            injection_tasks=[INJECTION_TASK_ID],
        )
    print_conversation(cap.messages)
    key = (USER_TASK_ID, INJECTION_TASK_ID)
    return {
        "label": label,
        "utility": bool(utility_results[key]),
        "attack_succeeded": bool(security_results[key]),
        "transcript": serialize(cap.messages),
    }


def main() -> None:
    suite = get_suites("v1.2")["banking"]
    user_task = suite.user_tasks[USER_TASK_ID]
    inj = suite.injection_tasks[INJECTION_TASK_ID]

    print("=" * 70)
    print(f"SUITE: banking | USER TASK: {USER_TASK_ID} | ATTACK: {ATTACK_NAME}")
    print("=" * 70)
    print("Legitimate task the user asked for:")
    print("    " + user_task.PROMPT)
    print("\nHidden attacker goal injected into the environment:")
    print("    " + inj.GOAL)

    # A) Baseline - no defense.
    base_pipe = build_local_pipeline()
    base_attack = load_attack(ATTACK_NAME, suite, base_pipe)
    baseline = run_once(base_pipe, suite, base_attack, "NO DEFENSE (baseline)")

    # B) Defended - AgentGuard action monitor.
    def_pipe = build_local_pipeline_with_defense()
    def_attack = load_attack(ATTACK_NAME, suite, def_pipe)
    defended = run_once(def_pipe, suite, def_attack, "AGENTGUARD (action monitor)")

    print("\n" + "=" * 70)
    print("BEFORE / AFTER  (attack ON, fully local)")
    print("=" * 70)
    for r in (baseline, defended):
        verdict = "HIJACKED - money sent to attacker" if r["attack_succeeded"] else "SAFE - attack blocked"
        print(f"  {r['label']:32s} -> {verdict}")
    print("\n  attack_succeeded == True  means the attacker won (BAD).")
    print("  AgentGuard's job is to flip that to False without breaking clean tasks.")

    RUNS_DIR.mkdir(exist_ok=True)
    payload = {
        "suite": "banking",
        "user_task_id": USER_TASK_ID,
        "user_prompt": user_task.PROMPT,
        "injection_task_id": INJECTION_TASK_ID,
        "attacker_goal": inj.GOAL,
        "attack": ATTACK_NAME,
        "model": LOCAL_MODEL,
        "baseline": baseline,
        "defended": defended,
    }
    out = RUNS_DIR / "phase3_result.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved real run data -> {out}")


if __name__ == "__main__":
    main()
