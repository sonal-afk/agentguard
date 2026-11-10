r"""
local_hf.py  -  run a model IN-PROCESS with Hugging Face transformers (no server,
no HTTP, no API). This is the "no API / work hard" path: we load the weights
ourselves, build the prompt (including the tool definitions), run the neural
network, and PARSE the tool calls out of the raw generated text.

The class below plugs into AgentDojo exactly like its OpenAILLM element does.
"""

import json
import re
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


def _text(msg) -> str:
    """AgentDojo message content is a list of blocks or None -> plain string."""
    c = msg.get("content")
    return get_text_content_as_str(c) if c else ""


def _to_chat(messages) -> list[dict]:
    """Convert AgentDojo messages into the transformers chat-template format."""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": _text(m)})
        elif role == "assistant":
            entry = {"role": "assistant", "content": _text(m)}
            tcs = m.get("tool_calls")
            if tcs:
                # render prior tool calls the way the chat template expects
                entry["tool_calls"] = [
                    {"type": "function", "function": {"name": tc.function, "arguments": tc.args}}
                    for tc in tcs
                ]
            out.append(entry)
        elif role == "tool":
            name = m["tool_call"].function if m.get("tool_call") else ""
            content = m.get("error") or _text(m)
            out.append({"role": "tool", "name": name, "content": content})
    return out


def _tools_schema(runtime: FunctionsRuntime) -> list[dict]:
    """Turn AgentDojo's tools into JSON-schema tool definitions for the prompt."""
    tools = []
    for f in runtime.functions.values():
        tools.append({
            "type": "function",
            "function": {
                "name": f.name,
                "description": f.description,
                "parameters": f.parameters.model_json_schema(),
            },
        })
    return tools


# Qwen-style models emit tool calls as:  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _parse_output(text: str):
    """Split raw model text into (final_text_or_None, [FunctionCall] or None)."""
    calls = []
    for match in _TOOLCALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            calls.append(FunctionCall(
                function=obj["name"],
                args=obj.get("arguments") or {},
                id=uuid.uuid4().hex[:8],
            ))
        except Exception:  # noqa: BLE001 - skip a malformed tool call block
            continue
    content = _TOOLCALL_RE.sub("", text).strip()
    return (content or None), (calls or None)


class LocalHFLLM(BasePipelineElement):
    """AgentDojo pipeline element that runs a local Hugging Face model in-process."""

    def __init__(self, model_name: str, max_new_tokens: int = 512):
        self.name = model_name
        print(f"[local_hf] loading tokenizer + model: {model_name} (first time downloads weights)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.max_new_tokens = max_new_tokens
        print(f"[local_hf] ready on device: {self.model.device}")

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages=[],
        extra_args: dict = {},
    ):
        chat = _to_chat(messages)
        tools = _tools_schema(runtime)
        input_ids = self.tokenizer.apply_chat_template(
            chat, tools=tools, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic (temperature 0)
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        content, tool_calls = _parse_output(text)
        assistant = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(content)] if content else None,
            tool_calls=tool_calls,
        )
        return query, runtime, env, [*messages, assistant], extra_args
