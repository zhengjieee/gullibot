"""The agent's tool-calling loop.

A manual loop rather than the SDK's tool runner: the experiment needs a hard
cap on tool calls, timing and token counts for every API call, and the full
transcript, all of which are simplest to get by owning the loop.
"""

import time

from agent.models import MODELS, cost_usd

MAX_STEPS = 25  # tool calls per session
MAX_TOKENS = 16000


def block_dict(block):
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}


def usage_dict(usage):
    return {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def run_agent(client, model, system, tools, user_message, execute, max_steps=MAX_STEPS):
    """Run one shopping conversation.

    execute(name, args) -> (page_text, is_error) runs a tool against the store.
    Returns a dict with end_reason ("end_turn", "step_cap", "max_tokens",
    "refusal", ...), the final report text, steps used, the transcript,
    token usage, cost and the model version the API reported.
    """
    params = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "tools": tools,
        "cache_control": {"type": "ephemeral"},  # cache the growing conversation prefix
    }
    if MODELS[model]["thinking"]:
        params["thinking"] = MODELS[model]["thinking"]

    messages = [{"role": "user", "content": user_message}]
    transcript, steps, report, model_version = [], 0, None, None
    totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}

    while True:
        started = time.monotonic()
        response = client.messages.create(messages=messages, **params)
        model_version = model_version or response.model
        usage = usage_dict(response.usage)
        for k in totals:
            totals[k] += usage[k]
        turn = {
            "stop_reason": response.stop_reason,
            "latency_s": round(time.monotonic() - started, 2),
            "usage": usage,
            "content": [block_dict(b) for b in response.content],
            "tool_results": [],
        }
        transcript.append(turn)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            end_reason = response.stop_reason
            if end_reason == "end_turn":
                report = "\n".join(b.text for b in response.content if b.type == "text").strip()
            break

        results, capped = [], False
        for block in (b for b in response.content if b.type == "tool_use"):
            if steps >= max_steps:
                capped = True
                break
            steps += 1
            text, is_error = execute(block.name, block.input)
            result = {"type": "tool_result", "tool_use_id": block.id, "content": text}
            if is_error:
                result["is_error"] = True
            results.append(result)
            turn["tool_results"].append({"id": block.id, "content": text, "is_error": is_error})
        if capped:
            end_reason = "step_cap"
            break
        messages.append({"role": "user", "content": results})

    return {
        "end_reason": end_reason,
        "report": report,
        "steps": steps,
        "api_calls": len(transcript),
        "model_version": model_version,
        "usage": totals,
        "cost_usd": round(cost_usd(model, totals), 5),
        "transcript": transcript,
    }
