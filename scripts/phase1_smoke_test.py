r"""
phase1_smoke_test.py  -  PHASE 1: watch a Gemini agent do ONE real task.

PURPOSE:
  - Prove the free Gemini key works end-to-end with AgentDojo.
  - Let you WATCH the agent: read the task, plan, call tools, answer.
  - Report the OFFICIAL score (did it complete the task correctly?).
  There is NO attack in this phase - this is the honest, working baseline.

COST: one small task on free-tier Gemini = a handful of API calls (free).

HOW TO RUN (from the project folder, in VSCode's terminal):
  .\.venv\Scripts\python.exe scripts\phase1_smoke_test.py
"""

from agentdojo.logging import Logger
from agentdojo.task_suite.load_suites import get_suites

from agent_lib import build_local_pipeline, LOCAL_MODEL


class CaptureLogger(Logger):
    """A minimal logger that just remembers the latest full message list.
    AgentDojo pipeline elements call Logger.get().log(messages) after each
    step, passing the running conversation - so after the run, self.messages
    holds the whole conversation."""

    def __init__(self) -> None:
        self.messages = []
        self.logdir = None  # AgentDojo may read this attribute; None = don't log to disk

    def log(self, messages, **kwargs) -> None:
        self.messages = messages

    def log_error(self, *args, **kwargs) -> None:
        pass


def text_of(content) -> str:
    """Message 'content' can be a plain string OR a list of content blocks.
    Return a readable string either way."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # list of blocks like {"type": "text", "content": "..."}
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(str(block.get("content") or block.get("text") or ""))
        else:
            parts.append(str(block))
    return " ".join(p for p in parts if p)


def describe_tool_calls(tool_calls) -> list[str]:
    """Turn tool-call objects into readable 'func(args)' strings."""
    lines = []
    for tc in tool_calls or []:
        # tool calls may be objects (with .function/.args) or dicts
        name = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
        args = getattr(tc, "args", None)
        if args is None and isinstance(tc, dict):
            args = tc.get("args")
        lines.append(f"{name}({dict(args) if args else {}})")
    return lines


def print_conversation(messages) -> None:
    print("\n" + "#" * 70)
    print("# WHAT THE AGENT DID, STEP BY STEP")
    print("#" * 70)
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        body = text_of(msg.get("content"))
        tool_calls = msg.get("tool_calls") if role == "assistant" else None

        print(f"\n[{i}] {role.upper()}")
        if body.strip():
            # keep it readable; trim very long tool outputs
            snippet = body if len(body) < 800 else body[:800] + " ...[truncated]"
            print("    " + snippet.replace("\n", "\n    "))
        for call in describe_tool_calls(tool_calls):
            print("    -> TOOL CALL: " + call)


def main() -> None:
    print(f"Building a LOCAL agent ({LOCAL_MODEL}) via Ollama - fully offline...")
    pipeline = build_local_pipeline()

    # Use the smallest suite (banking, 16 tasks) and its first task.
    suite = get_suites("v1.2")["banking"]
    task_id, task = next(iter(suite.user_tasks.items()))

    print("\n" + "=" * 70)
    print(f"SUITE: banking   |   TASK: {task_id}")
    print("=" * 70)
    print("The legitimate job we're asking the agent to do:")
    print("    " + task.PROMPT)

    # Run the task with NO attack. We wrap it in CaptureLogger so we can both
    # (a) get the official (utility, security) score, and
    # (b) grab the conversation to print - all from ONE agent run.
    #
    # injection_task=None and injections={} mean "no attack in this run".
    with CaptureLogger() as cap:
        utility, security = suite.run_task_with_pipeline(
            pipeline,
            task,
            injection_task=None,
            injections={},
        )

    print_conversation(cap.messages)

    print("\n" + "=" * 70)
    print("OFFICIAL RESULT")
    print("=" * 70)
    print(f"  utility  (did the agent complete the real task correctly?) : {utility}")
    print(f"  security (always True here - there was no attack)          : {security}")
    print("\nPhase 1 done: if utility is True, your LOCAL agent works and can")
    print("plan + use tools. Next (Phase 2): turn on an attack and watch it get")
    print("hijacked, then measure the attack success rate.")


if __name__ == "__main__":
    main()
