"""
Shared context tokenization + chunking for the hypernet input.

Single source of truth used by BOTH the training dataset
(`IHChallengeSFTDataset`) and inference (`ModulatedPretrainedModelX
.internalize`). Keeping these byte-identical is a correctness requirement:
the hypernet must see the same input distribution at train and eval time
(chat template + 512-token chunking + per-chunk affixes + chunk count cap).
Any change here affects both paths automatically.
"""

from __future__ import annotations

from math import ceil
import random

import torch

from ctx_to_lora.data.definitions import CTX_AFFIXES  # borrow from ctx_to_lora


def get_ctx_affixes(ctx_tokenizer) -> tuple[list[int], list[int]]:
    """Resolve the chat-template prefix/suffix token ids for the encoder."""
    name = getattr(ctx_tokenizer, "name_or_path", None)
    if name in CTX_AFFIXES:
        return CTX_AFFIXES[name]["prefix"], CTX_AFFIXES[name]["suffix"]
    return [], []


def chunk_ctx(
    token_ids: list[int],
    max_ctx_chunk_len: int,
    max_ctx_chunks: int,
    ctx_prefix: list[int],
    ctx_suffix: list[int],
    max_chunks: int | None = None,
    num_chunk_probs: dict | None = None,
    min_ctx_chunk_len: int = -1,
) -> list[list[int]]:
    """Split context token IDs into chunks of `max_ctx_chunk_len`.

    Deterministic even-split by default. When `num_chunk_probs` is given
    (train-only), the chunk COUNT is sampled from it, filtered to the feasible
    range [ceil(content/max_len), ceil(content/min_len)] — yielding per-chunk
    lengths ≈ [min_ctx_chunk_len, max_ctx_chunk_len] (the trailing chunk may be
    shorter). Requires `min_ctx_chunk_len > 0`; otherwise the feasible range
    collapses to the deterministic minimum (no randomisation). Inference callers
    pass `num_chunk_probs=None` → byte-identical deterministic behaviour.
    """
    chunk_len = max_ctx_chunk_len
    if chunk_len <= 0 or len(token_ids) <= chunk_len:
        return [token_ids]

    cap = max_ctx_chunks if max_chunks is None else max_chunks

    prefix = ctx_prefix
    suffix = ctx_suffix

    has_prefix = token_ids[: len(prefix)] == prefix if prefix else False
    has_suffix = token_ids[-len(suffix):] == suffix if suffix else False

    if has_prefix and has_suffix:
        content = token_ids[len(prefix):-len(suffix)]
    elif has_prefix:
        content = token_ids[len(prefix):]
    elif has_suffix:
        content = token_ids[:-len(suffix)]
    else:
        content = token_ids

    affix_len = len(prefix) + len(suffix)
    content_per_chunk = max(1, chunk_len - affix_len)

    max_content = content_per_chunk * max(1, cap)
    if len(content) > max_content:
        content = content[:max_content]

    min_required = max(1, ceil(len(content) / content_per_chunk))
    if num_chunk_probs is not None:
        if min_ctx_chunk_len and min_ctx_chunk_len > 0:
            min_content_per_chunk = max(1, min_ctx_chunk_len - affix_len)
            max_split = max(1, ceil(len(content) / min_content_per_chunk))
        else:
            max_split = min_required
        filt = {
            int(k): v for k, v in num_chunk_probs.items()
            if min_required <= int(k) <= max_split and int(k) <= cap
        }
        if filt:
            n_chunks = random.choices(
                list(filt.keys()), weights=list(filt.values()), k=1
            )[0]
        else:
            n_chunks = min(min_required, cap)
    else:
        n_chunks = min_required

    n_chunks = max(1, n_chunks)
    avg_content = ceil(len(content) / n_chunks)
    raw_chunks = [
        content[i:i + avg_content] for i in range(0, len(content), avg_content)
    ]

    return [prefix + c + suffix for c in raw_chunks]


def tokenize_context_chunks(
    ctx_tokenizer,
    context,
    max_ctx_chunk_len: int,
    max_ctx_chunks: int,
    num_chunk_probs: dict | None = None,
    min_ctx_chunk_len: int = -1,
) -> list[list[int]]:
    """Tokenize context (str or list[str]) into a list of chunks.

    Single-doc (str): one chat-template tokenize, one chunking.
    Multi-doc (list[str]): chat-template + chunk each doc independently with
    a per-doc cap = max(1, max_ctx_chunks // n_docs), concat in order,
    total capped at max_ctx_chunks.

    Verbatim from the original `IHChallengeSFTDataset._tokenize_context`.
    """
    ctx_prefix, ctx_suffix = get_ctx_affixes(ctx_tokenizer)

    def _tok(text: str) -> list[int]:
        inp = ctx_tokenizer.apply_chat_template(
            [[{"role": "system", "content": ""},
              {"role": "user", "content": text.strip()}]],
            tokenize=True, add_generation_prompt=True,
            return_attention_mask=False, padding=False, truncation=False,
            add_special_tokens=False, return_dict=True,
        )
        return inp["input_ids"][0]

    def _chunk(ids, max_chunks=None):
        return chunk_ctx(
            ids, max_ctx_chunk_len, max_ctx_chunks,
            ctx_prefix, ctx_suffix, max_chunks=max_chunks,
            num_chunk_probs=num_chunk_probs,
            min_ctx_chunk_len=min_ctx_chunk_len,
        )

    if isinstance(context, str):
        return _chunk(_tok(context))

    docs = [d for d in context if d and d.strip()]
    if not docs:
        return _chunk(_tok(""))
    if len(docs) == 1:
        return _chunk(_tok(docs[0]))

    per_doc_cap = max(1, max_ctx_chunks // len(docs))
    chunks: list[list[int]] = []
    for doc in docs:
        chunks.extend(_chunk(_tok(doc), max_chunks=per_doc_cap))
        if len(chunks) >= max_ctx_chunks:
            break
    return chunks[:max_ctx_chunks]


def context_to_padded_tensor(
    ctx_tokenizer,
    context,
    max_ctx_chunk_len: int,
    max_ctx_chunks: int,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full inference-side path: context → (ctx_ids, ctx_attn_mask).

    Pads chunks to equal length and stacks them — mirrors exactly the
    padding done in `IHChallengeSFTDataset.__getitem__`.
    """
    chunks = tokenize_context_chunks(
        ctx_tokenizer, context, max_ctx_chunk_len, max_ctx_chunks
    )
    max_len = max(len(c) for c in chunks)
    pad_id = ctx_tokenizer.pad_token_id or 0
    padded = [c + [pad_id] * (max_len - len(c)) for c in chunks]
    ctx_ids = torch.tensor(padded, dtype=torch.long)
    ctx_attn_mask = (ctx_ids != pad_id).long()
    if device is not None:
        ctx_ids = ctx_ids.to(device)
        ctx_attn_mask = ctx_attn_mask.to(device)
    return ctx_ids, ctx_attn_mask
