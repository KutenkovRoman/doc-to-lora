"""
SFT dataset for D2L instruction-following training.

Loads (context, prompt, response) triplets from JSONL files.
Supports mode filtering (conflict, knowledge, neutral, compatible).
Each sample is tokenized into:
  - ctx_ids: instruction tokenized for the hypernetwork (→ LoRA), chunked if long
  - input_ids: prompt + response tokenized as chat (→ base model input)
  - labels: -100 for prompt tokens, token ids for response tokens (→ CE loss target)
"""

import json
import logging
from collections import Counter

import torch
from torch.utils.data import Dataset

from ctx_to_kv_cache.data.ctx_chunking import (
    chunk_ctx,
    get_ctx_affixes,
    tokenize_context_chunks,
)
from ctx_to_lora.data.self_gen_template import PRE_CTX, SELF_QA_INTX

log = logging.getLogger(__name__)


# Verbatim copy from SkillStorage/src/skillstorage/evo_eval/solver.py
# (the one passed to D2LBackend.complete_with_skills as `system=`, without the
# per-skill <skill_content> blocks — those go through the hypernet ctx path).
# Keep byte-identical with the eval-side constant; if SkillStorage's solver
# updates this string we must update it here too.
SOLVER_SYSTEM_PROMPT_WITH_SKILL = """\
You are a skilled Python programmer solving a task using a provided skill.

You are given:
- A task instruction.
- One or more skill documents (SKILL.md) wrapped in <skill_content> blocks — \
these describe the libraries, conventions, and helpers you should use.
- A workspace that already contains any task inputs AND the skill's bundled \
files (e.g. `scripts/*.py`, `references/*.md`, `assets/*`). Skill-shipped \
helpers are there for you to use directly — import them, invoke them via \
`subprocess.run`, or read them with `Path(...).read_text()`.

Your job: write a single Python script that solves the task by applying the \
skill's documented approach.
- The script runs in the workspace directory; write any outputs there.
- Import whatever third-party library fits the job — anything the skill \
references (or that the task clearly needs) will be installed automatically \
into an isolated per-task environment. No pre-approved list.
- Prefer the libraries/APIs the skill explicitly documents over reinventing \
them in pure stdlib.

Respond with ONLY a Python code block. No explanation before or after.

```python
# your solution here
```"""


class IHChallengeSFTDataset(Dataset):
    """
    Dataset of (context, prompt, response) triplets for SFT.

    context → tokenized for hypernetwork → LoRA
    prompt + response → tokenized as chat → CE loss on response tokens

    Supports filtering by mode (conflict/knowledge/neutral/compatible).
    Preserves all entry fields as metadata for eval breakdown.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        ctx_tokenizer,
        max_samples: int = 0,
        max_response_tokens: int = 512,
        max_ctx_chunk_len: int = 512,
        max_ctx_chunks: int = 8,
        modes: list[str] | None = None,
        return_teacher: bool = False,
        teacher_max_ctx_tokens: int = 4096,
        system_prompt: str | None = None,
        num_chunk_probs: dict | None = None,
        min_ctx_chunk_len: int = -1,
    ):
        self.tokenizer = tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.max_response_tokens = max_response_tokens
        self.max_ctx_chunk_len = max_ctx_chunk_len
        self.max_ctx_chunks = max_ctx_chunks
        self.return_teacher = return_teacher
        self.teacher_max_ctx_tokens = teacher_max_ctx_tokens
        # system_prompt: if set, prepended as a {"role":"system"} message to every
        # training sample (single-turn and multi-turn). Used to match evo_eval's
        # solver prompt format at training time.
        self.system_prompt = system_prompt
        self.num_chunk_probs = num_chunk_probs
        self.min_ctx_chunk_len = min_ctx_chunk_len
        self.entries = []

        # Resolve ctx affixes for chunking (shared helper — single source of
        # truth with inference-time tokenization, see ctx_chunking.py)
        self.ctx_prefix, self.ctx_suffix = get_ctx_affixes(ctx_tokenizer)

        with open(data_path) as f:
            for line in f:
                entry = json.loads(line)
                # Skip entries without gold response, unless they're behavioral-
                # eval rows (`judge_template` set) that a downstream judge will
                # score offline. __getitem__ handles response=None by emitting
                # a placeholder input/label pair.
                if (
                    entry.get("response") in (None, "", "__GRADER_EVAL__")
                    and not entry.get("judge_template")
                ):
                    continue
                # Filter by mode if specified
                if modes is not None and entry.get("mode") not in modes:
                    continue
                self.entries.append(entry)

        if max_samples > 0:
            self.entries = self.entries[:max_samples]

        mode_counts = Counter(e.get("mode", "unknown") for e in self.entries)
        log.info(f"Loaded {len(self.entries)} SFT samples from {data_path} (modes: {dict(mode_counts)})")

    def _chunk_ctx(self, token_ids: list[int], max_chunks: int | None = None) -> list[list[int]]:
        """Delegate to the shared `ctx_chunking.chunk_ctx` (single source of
        truth with inference). Behaviour is byte-identical to the previous
        in-class implementation."""
        return chunk_ctx(
            token_ids, self.max_ctx_chunk_len, self.max_ctx_chunks,
            self.ctx_prefix, self.ctx_suffix, max_chunks=max_chunks,
        )

    def _tokenize_context(self, context) -> list[list[int]]:
        """Delegate to the shared `ctx_chunking.tokenize_context_chunks`.
        Inference (`ModulatedPretrainedModelX.internalize`) uses the exact
        same function, so the hypernet sees the same input distribution at
        train and eval time."""
        return tokenize_context_chunks(
            self.ctx_tokenizer, context,
            self.max_ctx_chunk_len, self.max_ctx_chunks,
            num_chunk_probs=self.num_chunk_probs,
            min_ctx_chunk_len=self.min_ctx_chunk_len,
        )

    def _truncate_ctx_text(self, ctx_text: str) -> str:
        ids = self.tokenizer.encode(ctx_text, add_special_tokens=False)
        if len(ids) <= self.teacher_max_ctx_tokens:
            return ctx_text
        return self.tokenizer.decode(ids[: self.teacher_max_ctx_tokens], skip_special_tokens=True)

    def _build_teacher(self, context, attack, gold_response):
        """Build teacher input (ctx_in_prompt + prompt + response) with loss labels on assistant tokens.

        For alignment with the student side the response is built via the same
        path on both sides:
          * single-turn: raw `tokenizer.encode(gold_response)` + optional EOS
          * multi-turn : `_tokenize_multiturn(ctx_injected_messages, gold_response)`

        This guarantees teacher and student share the exact response token IDs.
        The only length cap is on ctx_text (`teacher_max_ctx_tokens`) — we never
        tail-truncate input_ids (would silently drop response label positions).
        """
        if isinstance(context, list):
            ctx_text = "\n\n---\n\n".join(d for d in context if d and d.strip())
        else:
            ctx_text = context or ""
        ctx_text = self._truncate_ctx_text(ctx_text)

        ctx_block = (
            f"{PRE_CTX}{ctx_text}\n\n---\n\n{SELF_QA_INTX.strip()}"
        )

        if isinstance(attack, list) and len(attack) > 1:
            # Multi-turn: inject ctx into first user content then reuse
            # _tokenize_multiturn so assistant boundary logic matches student.
            ctx_messages = [dict(m) for m in attack]
            ctx_messages[0]["content"] = f"{ctx_block}\n\n{ctx_messages[0]['content']}"
            if self.system_prompt is not None and (not ctx_messages or ctx_messages[0].get("role") != "system"):
                ctx_messages = [{"role": "system", "content": self.system_prompt}] + ctx_messages
            # Budget: ctx cap + same natural-dialog headroom student gets.
            teacher_max_len = self.teacher_max_ctx_tokens + self.max_response_tokens * 8
            full_ids, labels = self._tokenize_multiturn(
                ctx_messages, gold_response, max_total_len=teacher_max_len,
            )
            return (
                torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )

        if isinstance(attack, list):
            user_content = attack[0]["content"]
        else:
            user_content = attack
        messages = [{"role": "user", "content": f"{ctx_block}\n\n{user_content}"}]
        if self.system_prompt is not None:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        response_ids = self.tokenizer.encode(gold_response, add_special_tokens=False)
        if len(response_ids) > self.max_response_tokens:
            response_ids = response_ids[: self.max_response_tokens]
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return len(self.entries)

    def _tokenize_multiturn(
        self, messages: list[dict], final_response: str, max_total_len: int | None = None
    ) -> tuple[list[int], list[int]]:
        """Tokenize multi-turn conversation with loss on ALL assistant turns.

        Returns (input_ids, labels) where labels=-100 for user/system tokens
        and labels=token_ids for all assistant tokens.

        For each assistant at index j, boundaries are found via:
          start = len(tok(messages[:j],   add_gen=True))   # after gen prompt
          end   = len(tok(messages[:j+1], add_gen=False))  # after assistant content
        Both are valid prefixes of the full tokenized conversation.
        """
        full_messages = list(messages) + [{"role": "assistant", "content": final_response}]

        # Tokenize full conversation (no generation prompt — this is the ground truth)
        input_ids = self.tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
        )

        # Start with all labels masked
        labels = [-100] * len(input_ids)

        # Find and unmask each assistant turn
        for j in range(len(full_messages)):
            if full_messages[j]["role"] != "assistant":
                continue

            # Start: tokenize everything before this assistant + generation prompt
            start = len(self.tokenizer.apply_chat_template(
                full_messages[:j], tokenize=True, add_generation_prompt=True,
            ))
            # End: tokenize through this assistant (no generation prompt)
            end = len(self.tokenizer.apply_chat_template(
                full_messages[:j + 1], tokenize=True, add_generation_prompt=False,
            ))

            # Copy token IDs into labels for this assistant span
            for pos in range(start, min(end, len(input_ids))):
                labels[pos] = input_ids[pos]

        # Add EOS only if template didn't already end with one
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not input_ids or input_ids[-1] != eos):
            input_ids = input_ids + [eos]
            labels.append(eos)

        # Truncate if too long (safety net for unexpected data).
        # Caller can override when building teacher inputs that prepend ctx.
        max_len = max_total_len if max_total_len is not None else self.max_response_tokens * 8
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        return input_ids, labels

    def __getitem__(self, idx) -> dict:
        entry = self.entries[idx]
        instruction = entry["context"]  # str or list[str] (multi-skill)
        attack = entry["prompt"]
        gold_response = entry["response"]

        # 1. Tokenize context for hypernetwork (handles str or list[str])
        chunks = self._tokenize_context(instruction)

        # Pad chunks to same length for stacking
        max_len = max(len(c) for c in chunks)
        pad_id = self.ctx_tokenizer.pad_token_id or 0
        padded = [c + [pad_id] * (max_len - len(c)) for c in chunks]
        ctx_ids = torch.tensor(padded, dtype=torch.long)  # [n_chunks, chunk_len]
        ctx_attn_mask = (ctx_ids != pad_id).long()

        # 2. Tokenize prompt + response
        # Behavioral-eval rows have response=None (offline judging downstream).
        # The eval caller builds its own prompt_ids from entry["prompt"], so
        # we only need ctx_ids/ctx_attn_mask to be correct — emit a 1-token
        # placeholder for input_ids/labels to keep the sample dict shape.
        if gold_response in (None, ""):
            full_ids = [self.tokenizer.eos_token_id or 0]
            labels = [-100]
        # Support both single-turn (str) and multi-turn (list of message dicts)
        elif isinstance(attack, list) and len(attack) > 1:
            # Multi-turn: compute loss on ALL assistant turns
            messages = attack
            if self.system_prompt is not None and (not messages or messages[0].get("role") != "system"):
                messages = [{"role": "system", "content": self.system_prompt}] + list(messages)
            full_ids, labels = self._tokenize_multiturn(messages, gold_response)
        else:
            # Single-turn: loss on response only
            if isinstance(attack, list):
                prompt_messages = list(attack)
            else:
                prompt_messages = [{"role": "user", "content": attack}]
            if self.system_prompt is not None and (not prompt_messages or prompt_messages[0].get("role") != "system"):
                prompt_messages = [{"role": "system", "content": self.system_prompt}] + prompt_messages
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True,
            )

            response_ids = self.tokenizer.encode(
                gold_response, add_special_tokens=False,
            )
            if len(response_ids) > self.max_response_tokens:
                response_ids = response_ids[:self.max_response_tokens]
            if self.tokenizer.eos_token_id is not None:
                response_ids = response_ids + [self.tokenizer.eos_token_id]

            full_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        out = {
            "ctx_ids": ctx_ids,
            "ctx_attn_mask": ctx_attn_mask,
            "mode": entry.get("mode", "unknown"),
            "input_ids": input_ids.unsqueeze(0),      # [1, seq_len] — model expects batch dim
            "attention_mask": attention_mask.unsqueeze(0),
            "labels": labels.unsqueeze(0),
        }

        if self.return_teacher:
            t_ids, t_labels = self._build_teacher(instruction, attack, gold_response)
            out["teacher_input_ids"] = t_ids.unsqueeze(0)
            out["teacher_attention_mask"] = torch.ones_like(t_ids).unsqueeze(0)
            out["teacher_labels"] = t_labels.unsqueeze(0)

        return out
