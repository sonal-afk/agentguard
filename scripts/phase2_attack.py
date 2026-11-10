r"""
phase2_attack.py  -  PHASE 2: watch an attack try to hijack the agent.

Same banking agent as Phase 1, but now the environment is poisoned with an
injected instruction (the "injection task") whose goal is to make the agent
SEND MONEY TO AN ATTACKER. We run it and measure whether the attack succeeded.

KEY METRIC - Attack Success Rate (ASR):
  security == True  ->  the ATTACK SUCCEEDED (agent hijacked, money sent to attacker)  [BAD]
  security == False ->  the agent stayed SAFE                                          [GOOD]
  ASR = fraction of runs where the attack succeeded.

There is NO defense yet - this is the baseline we will later try to beat.

RUN:
  .\.venv\Scripts\python.exe scripts\phase2_attack.py
"""

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks
from agentdojo.task_suite.load_suites import get_suites

from agent_lib import build_local_pipeline, LOCAL_MODEL
from phase1_smoke_test import CaptureLogger, print_conversation

# Which attack to use. "important_instructions" is a strong, realistic
# indirect-prompt-injection attack that disguises the malicious command as an
# urgent authoritative message hidden in the environment (e.g., inside data
# the agent reads while doing its real job).
ATTACK_NAME = "important_instructions"

USER_TASK_ID = "user_task_0"        # the legitimate job: pay a bill
INJECTION_TASK_ID = "injection_task_0"  # the attacker's goal: send money to attacker


def main() -> None:
    print(f"Building the LOCAL agent ({LOCAL_MODEL}) via Ollama - fully offline...")
    pipeline = build_local_pipeline()

    suite = get_suites("v1.2")["banking"]
    user_task = suite.user_tasks[USER_TASK_ID]

    # Build the attack. It knows the suite + pipeline so it can craft injections.
    attack = load_attack(ATTACK_NAME, suite, pipeline)

    print("\n" + "=" * 70)
    print(f"SUITE: banking | USER TASK: {USER_TASK_ID} | ATTACK: {ATTACK_NAME}")
    print(f"INJECTION (attacker's goal): {INJECTION_TASK_ID}")
    print("=" * 70)
    print("Legitimate task the user asked for:")
    print("    " + user_task.PROMPT)
    print("\nHidden attacker goal injected into the environment:")
    print("    " + suite.injection_tasks[INJECTION_TASK_ID].GOAL)

    # Run the user task with the injected attack, for this one injection task.
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

    # There's one (user_task, injection_task) pair here.
    key = (USER_TASK_ID, INJECTION_TASK_ID)
    utility = utility_results[key]
    attack_succeeded = security_results[key]   # True == attack won

    # ASR across all pairs we ran (here, just one).
    vals = list(security_results.values())
    asr = 100.0 * sum(vals) / len(vals)

    print("\n" + "=" * 70)
    print("OFFICIAL RESULT  (Phase 2 - attack ON, no defense)")
    print("=" * 70)
    print(f"  utility (did the agent still do the real task?)      : {utility}")
    print(f"  attack succeeded? (money sent to attacker)           : {attack_succeeded}")
    print(f"  ==> {'HIJACKED - the attack worked' if attack_succeeded else 'SAFE - the attack failed'}")
    print(f"\n  Attack Success Rate (ASR) over {len(vals)} run(s)         : {asr:.0f}%")
    print("\nThis ASR is our BASELINE. In Phase 3 we build the action-monitor")
    print("defense and re-run this - the goal is to drive the ASR down.")


if __name__ == "__main__":
    main()
