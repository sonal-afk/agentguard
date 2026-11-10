r"""
check_local_utility.py  -  run the LOCAL agent on the first few banking tasks
(no attack) and report utility for each. Confirms the end-to-end loop works and
shows how capable qwen2.5 is at completing tasks locally.

RUN:
  .\.venv\Scripts\python.exe scripts\check_local_utility.py
"""

from agentdojo.logging import Logger
from agentdojo.task_suite.load_suites import get_suites

from agent_lib import build_local_pipeline, LOCAL_MODEL


class NullCap(Logger):
    def __init__(self):
        self.messages = []
        self.logdir = None
    def log(self, messages, **k):
        self.messages = messages
    def log_error(self, *a, **k):
        pass


def main():
    print(f"Local model: {LOCAL_MODEL}. Running first banking tasks (no attack)...\n")
    pipeline = build_local_pipeline()
    suite = get_suites("v1.2")["banking"]

    ids = list(suite.user_tasks)[:5]   # first 5 tasks
    passed = 0
    for tid in ids:
        task = suite.user_tasks[tid]
        with NullCap():
            utility, _ = suite.run_task_with_pipeline(
                pipeline, task, injection_task=None, injections={}
            )
        passed += bool(utility)
        mark = "PASS" if utility else "fail"
        print(f"  {tid:14s} [{mark}]  - {task.PROMPT[:70]}")

    print(f"\nUtility: {passed}/{len(ids)} tasks completed by the local model.")
    print("Even a few PASSes prove the full local agent loop works end-to-end.")


if __name__ == "__main__":
    main()
