"""IH-Challenge data / grader / async-LLM utilities.

Moved from scripts/ih_challenge/utils.py — the data-pipeline side of that
original mixed module (the training-side helpers are in
`ctx_to_lora.training.sft_utils`).
"""

import asyncio
import json
import logging
import os

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_FILES = {
    "single-constraint": "single-constraint.jsonl",
    "multi-constraint": "multi-constraint.jsonl",
    "input-conditioned": "input-conditioned.jsonl",
    "anti-overrefusal": "anti-overrefusal.jsonl",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_instruction(defender_template: list[dict]) -> tuple[str, str]:
    """Extract the high-priority instruction and the attack placeholder role."""
    instruction_parts = []
    attack_role = None
    for msg in defender_template:
        if "<<<REPLACE THIS WITH YOUR ATTACK>>>" in msg["content"]:
            attack_role = msg["role"]
        else:
            instruction_parts.append(msg["content"])
    return "\n\n".join(instruction_parts), attack_role


def load_completed_ids(output_path: str) -> set[str]:
    """Read existing output JSONL and return set of completed IDs (for resume)."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def run_grader(grader_code: str, prompt: str, response: str) -> bool:
    """Execute IH-Challenge Python grader, return True if defender passed."""
    try:
        namespace = {}
        exec(grader_code, namespace)
        return bool(namespace["grade_output_correct"](prompt, response))
    except Exception as e:
        log.warning(f"Grader error: {e}")
        return False


# ---------------------------------------------------------------------------
# Async LLM
# ---------------------------------------------------------------------------

async def async_llm_call(
    client,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Call OpenAI-compatible API with semaphore and exponential backoff retry.

    Returns the raw response text. For scripts that need JSON parsing with
    parse-error retry, use your own retry loop instead.
    """
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(
                    f"API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                log.error(f"API call failed after {max_retries} attempts: {e}")
                raise
