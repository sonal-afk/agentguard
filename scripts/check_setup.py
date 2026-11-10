r"""
check_setup.py  -  Phase 0 sanity check.

WHAT THIS DOES (and nothing more):
  - Confirms you're running inside the project's .venv
  - Confirms AgentDojo is importable
  - Lists the real task suites, attacks, and defenses that ship with AgentDojo
  - Checks whether any API keys are set (it does NOT need them to run this file)

It makes ZERO network calls and costs ZERO money. Safe to run anytime.

HOW TO RUN (in VSCode's integrated terminal, from the project folder):
  .\.venv\Scripts\python.exe scripts\check_setup.py
"""

import os
import sys


def ok(flag: bool) -> str:
    return "[OK]" if flag else "[X]"


def line(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# --- 1. Are we in the right Python (the .venv)? -------------------------
line("1. PYTHON ENVIRONMENT")
print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
in_venv = ".venv" in sys.executable.replace("\\", "/")
print("Inside project .venv? ->", ok(in_venv),
      "" if in_venv else "(select the .venv interpreter in VSCode)")

# --- 2. Can we import AgentDojo? ----------------------------------------
line("2. AGENTDOJO IMPORT")
try:
    import agentdojo
    print("AgentDojo imported successfully", ok(True))
    print("Installed at:", os.path.dirname(agentdojo.__file__))
except Exception as e:  # noqa: BLE001
    print("Could NOT import agentdojo", ok(False), ":", e)
    sys.exit(1)

# --- 3. What suites / attacks / defenses ship with it? ------------------
line("3. WHAT'S INSIDE AGENTDOJO")

from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.attack_registry import ATTACKS  # name -> attack class
from agentdojo.agent_pipeline.agent_pipeline import DEFENSES  # list of defense names

suites = get_suites("v1.2")
print("\nTask suites (realistic environments):")
for name, suite in suites.items():
    print(f"  - {name:10s} : {len(suite.user_tasks):3d} user tasks, "
          f"{len(suite.injection_tasks):2d} injection (attack) tasks")

print("\nBuilt-in attacks:")
for name in ATTACKS:
    print(f"  - {name}")

print("\nBuilt-in defenses (note: these are all INPUT/PROMPT-stage):")
for name in DEFENSES:
    print(f"  - {name}")

# --- 4. Are any API keys configured? (not required for THIS script) -----
line("4. API KEYS (needed later, for live runs)")
for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
    present = "set" if os.environ.get(var) else "not set  (fine for now)"
    print(f"  {var:20s}: {present}")

line("DONE - Phase 0 environment looks good.")
print("Next step: choose a model provider and add its API key to a .env file.")
