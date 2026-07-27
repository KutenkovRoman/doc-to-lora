"""Training-pipeline utilities (moved from scripts/ih_challenge/utils.py).

Only the training-side helpers live here (no_lora_patch,
enable_aggregator_checkpointing, save_checkpoint). The IH-Challenge data /
grader helpers from the same original file are in
`ctx_to_lora.eval.ih_utils`. Named `sft_utils` to avoid colliding with the
existing `ctx_to_lora.utils` module.
"""

import glob
import logging
import os
import re
import shutil
from contextlib import contextmanager

import torch
import torch.utils.checkpoint as torch_ckpt

log = logging.getLogger(__name__)


@contextmanager
def no_lora_patch(base_model):
    """Temporarily restore stock Linear.forward on D2L-patched modules.

    D2L patches target Linear modules in two stages:
      1. `ModulatedPretrainedModel._init_model` → `patch_lora_forward` saves
         `module.forward_orig = module.forward` (bound `nn.Linear.forward`) and
         replaces `module.forward` with
         `partial(lora_forward, self=module, lora_dropout_p=..., scaling=...)` —
         still missing `n_qs, tot_q, A, B`.
      2. Every `forward()` call runs `apply_lora_to_layers` which further wraps
         that partial with the hypernet-generated `A, B` for this batch.

    Running the base model as a plain Qwen (teacher pass) requires restoring
    the original bound `nn.Linear.forward` (= `module.forward_orig`). This
    context manager swaps forward to `forward_orig`, yields, and restores the
    batch-bound partial on exit. The `patched_forward` flag is left alone so
    subsequent student passes still skip re-patching and apply_lora_to_layers
    overwrites forward with the new batch's A, B partial.
    """
    saved = {}
    for module in base_model.modules():
        orig_fwd = getattr(module, "forward_orig", None)
        if orig_fwd is not None and module.forward is not orig_fwd:
            saved[id(module)] = (module, module.forward)
            module.forward = orig_fwd
    try:
        yield
    finally:
        for _mid, (module, patched_fwd) in saved.items():
            module.forward = patched_fwd


def enable_aggregator_checkpointing(model) -> None:
    """Wrap hypernet aggregator forward in torch.utils.checkpoint.checkpoint.

    The Perceiver aggregator's stored activations dominate memory growth with
    `max_ctx_chunks` (~3.5 GB/chunk). Checkpointing trades ~20% backward
    compute for eliminating those on the forward path (they're recomputed
    during backward). Call once per model instance after construction.
    """
    model = getattr(model, "module", model)
    hyper_module = getattr(model, "hypernet", None)
    if hyper_module is None:
        hyper_module = getattr(model, "hyperx", None)
    if hyper_module is None:
        raise AttributeError("model has neither .hypernet nor .hyperx")

    agg = hyper_module.aggregator
    if getattr(agg, "_ckpt_wrapped", False):
        log.info("Aggregator forward already wrapped in checkpoint; skipping.")
        return
    orig_forward = agg.forward

    def ckpt_forward(*args, **kwargs):
        def fn(*tensor_args):
            return orig_forward(*tensor_args, **kwargs)
        return torch_ckpt.checkpoint(fn, *args, use_reentrant=False)

    agg.forward = ckpt_forward
    agg._ckpt_wrapped = True
    log.info("Aggregator forward wrapped in torch.utils.checkpoint (use_reentrant=False).")


def save_checkpoint(model, optimizer, step, output_dir, keep_last_n=0, name=None):
    """Save hypernet checkpoint and optionally remove old ones.

    Args:
        keep_last_n: Keep only last N step-numbered checkpoints. 0 = keep all.
        name: Override directory name (default ``checkpoint-{step}``).
            Non-numeric names (e.g. ``checkpoint-best``) survive rotation
            since the regex only matches ``checkpoint-<int>``.
    """
    ckpt_name = name or f"checkpoint-{step}"
    path = os.path.join(output_dir, ckpt_name)
    os.makedirs(path, exist_ok=True)
    state = model.state_dict()
    torch.save(state, os.path.join(path, "pytorch_model.bin"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    log.info(f"Saved checkpoint to {path}")

    if keep_last_n > 0:
        ckpt_dirs = sorted(
            [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
             if re.search(r"checkpoint-(\d+)$", d)],
            key=lambda d: int(re.search(r"checkpoint-(\d+)$", d).group(1)),
        )
        for old in ckpt_dirs[:-keep_last_n]:
            shutil.rmtree(old)
            log.info(f"Removed old checkpoint: {old}")
