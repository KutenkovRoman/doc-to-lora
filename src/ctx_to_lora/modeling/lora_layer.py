from collections.abc import Iterable
from functools import partial
from operator import attrgetter
import re

import torch
from torch import nn
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from ctx_to_lora.utils import get_layers


def lora_forward(
    x: Float[Tensor, "tot_q seq_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self, *args, **kwargs,
) -> Float[Tensor, "tot_q seq_len d_out"]:
    # A: [n_ctx, r, d_in] -> [tot_q, r, d_in]
    A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    # B: [n_ctx, d_out, r] -> [tot_q, d_out, r]
    B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)

    base_out = nn.Linear.forward(self, x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    delta_x = einsum(A, delta_x, "tot_q r d_in, tot_q s_len d_in -> tot_q s_len r")
    delta_x = einsum(B, delta_x, "tot_q r d_out, tot_q s_len r -> tot_q s_len d_out")
    delta_x = delta_x * scaling
    return (base_out + delta_x).to(base_out.dtype)


def lora_forward_packed(
    x: Float[Tensor, "1 tot_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self, *args, **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = nn.Linear.forward(self, x, *args, **kwargs)

    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)

    repeated_A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    repeated_A = repeated_A.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    repeated_B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    repeated_B = repeated_B.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    delta_x = einsum(
        repeated_A, delta_x, "tot_len r d_in, bs tot_len d_in -> bs tot_len r"
    )
    delta_x = einsum(
        repeated_B, delta_x, "tot_len r d_out, bs tot_len r -> bs tot_len d_out"
    )
    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def apply_lora_to_layers(
    model: nn.Module,
    layer_indices: Iterable[int],
    combined_loras: dict[str, dict[str, Float[Tensor, "n_ctx n_layers r d"]]],
    n_qs: Integer[Tensor, "n_ctx"],
    position_ids: Integer[Tensor, "bs seq_len"] = None,
) -> None:
    layers = get_layers(model)
    if position_ids is not None:
        position_ids = position_ids.squeeze(0)
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1
        tot_len = seq_lens.sum().item()

    tot_q = n_qs.sum().item()
    for layer_idx in layer_indices:
        layer = layers[layer_idx]

        for mname in combined_loras:
            if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
                long_mname = f"self_attn.{mname}"
            elif mname in ["down_proj", "up_proj", "gate_proj"]:
                long_mname = f"mlp.{mname}"
            module = attrgetter(long_mname)(layer)
            A = combined_loras[mname]["A"][:, layer_idx]
            B = combined_loras[mname]["B"][:, layer_idx]
            # here module.forward already refers to one of lora_forward or lora_forward_packed
            module.forward = partial(module.forward, n_qs=n_qs, tot_q=tot_q, A=A, B=B)
            if position_ids is not None:
                module.forward = partial(
                    module.forward, seq_lens=seq_lens, tot_len=tot_len
                )


def _orthog_repr_forward(
    x: Float[Tensor, "1 tot_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
    tot_q: int,
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    coeffs: Float[Tensor, "n_ctx tot_chunks"],
    generator: torch.Generator,
    lora_dropout_p: float, # unused, kept for compatibility
    scaling: float, # unused
    self, *args, **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = nn.Linear.forward(self, x, *args, **kwargs)

    n_ctx, tot_chunks = coeffs.shape

    with torch.no_grad():
        orthog_repr, _ = torch.linalg.qr(
            torch.randn(
                (tot_chunks, self.in_features, self.out_features),
                generator=generator,
                device=coeffs.device,
            ),
            mode="reduced",
        )
        orthog_repr = orthog_repr.to(dtype=coeffs.dtype)  # qr does not work with bf16?

    weighted_repr = einsum(
        coeffs, orthog_repr, "n_ctx tot_chunks, tot_chunks d_in d_out -> n_ctx d_in d_out"
    )

    # repeated_repr = torch.repeat_interleave(
    #     weighted_repr, n_qs, dim=0, output_size=tot_q
    # )
    # repeated_repr = torch.repeat_interleave(
    #     repeated_repr, seq_lens, dim=0, output_size=tot_len
    # )
    # delta_x = einsum(
    #     repeated_repr, x, "tot_len d_in d_out, bs tot_len d_in -> bs tot_len d_out"
    # )

    # suggested by deepseek to save memory, I hope it is correct
    ctx_idx = torch.repeat_interleave(
        torch.arange(n_ctx, device=coeffs.device), n_queries, dim=0, output_size=tot_q
    )
    ctx_idx = torch.repeat_interleave(ctx_idx, seq_lens, dim=0, output_size=tot_len)

    delta_x = torch.zeros(
        (1, tot_len, self.out_features),
        device=x.device,
        dtype=x.dtype,
    )
    for i in range(n_ctx):
        mask = (ctx_idx == i)
        # only materialises the slice of x that belongs to this context
        # [1, sum_len_i, d_in] @ [d_in, d_out] -> [1, sum_len_i, d_out]
        delta_x[:, mask] = x[:, mask] @ weighted_repr[i]

    return base_out + delta_x


def orthog_repr_forward(
    x: Float[Tensor, "1 tot_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    coeffs: Float[Tensor, "tot_chunks"],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    generator: torch.Generator,
    lora_dropout_p: float, # unused, kept for compatibility
    scaling: float, # unused
    self, *args, **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = nn.Linear.forward(self, x, *args, **kwargs)

    delta_x = torch.empty(
        (1, tot_len, self.out_features),
        device=x.device,
        dtype=x.dtype,
    )

    start_q = start_c = start = 0
    for n_q, n_c in zip(n_queries, n_ctx_chunks):
        end_q = start_q + n_q
        end_c = start_c + n_c
        end = start + seq_lens[start_q:end_q].sum().item()

        with torch.no_grad():
            orthog_repr_i, _ = torch.linalg.qr(
                torch.randn(
                    (n_c, self.in_features, self.out_features),
                    generator=generator,
                    device=coeffs.device,
                ),
                mode="reduced",
            )
            orthog_repr_i = orthog_repr_i.to(dtype=coeffs.dtype)

        coeffs_i = coeffs[start_c:end_c]
        weighted_repr_i = einsum(
            coeffs_i, orthog_repr_i, "n_c, n_c d_in d_out -> d_in d_out"
        )

        delta_x[:, start:end, :] = x[:, start:end, :] @ weighted_repr_i

        start_q = end_q
        start_c = end_c
        start = end

    return base_out + delta_x


def random_repr_forward(
    x: Float[Tensor, "1 tot_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
    tot_q: int,
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    coeffs: Float[Tensor, "tot_chunks n_reprs"],  #"n_ctx max_n_ctx_chunks"
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    repr_seeds: Integer[Tensor, "n_ctx"],
    generator: torch.Generator,
    # layer_idx: int,
    lora_dropout_p: float, # unused, kept for compatibility
    scaling: float, # unused, kept for compatibility
    self, *args, **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = nn.Linear.forward(self, x, *args, **kwargs)

    n_reprs = coeffs.shape[1]
    r = 8

    # stat = torch.cuda.memory.memory_allocated(device=coeffs.device)
    # print(f"Starting forward pass for {layer_idx = } with {stat / (1024 ** 3):.1f}Gb allocated", flush=True)

    ctx_idx = torch.repeat_interleave(
        torch.arange(len(n_ctx_chunks), device=x.device),
        n_queries, dim=0, output_size=tot_q
    )
    ctx_idx = torch.repeat_interleave(
        ctx_idx, seq_lens, dim=0, output_size=tot_len
    )

    delta_x = torch.zeros(
        (1, tot_len, self.out_features),
        device=x.device,
        dtype=x.dtype,
    )

    start = 0
    i = 0
    for n_chunks, seed in zip(n_ctx_chunks, repr_seeds):
        # seed = seed.item()
        # generator.manual_seed(seed + layer_idx)
        generator.manual_seed(seed.item())

        end = start + n_chunks.item()
        mask = (ctx_idx == i)

        with torch.no_grad():
            repr_A = torch.randn(
                (n_chunks, n_reprs, r, self.in_features),
                generator=generator,
                device=coeffs.device,
                dtype=coeffs.dtype,
            )
            repr_B = torch.randn(
                (n_chunks, n_reprs, self.out_features, r),
                generator=generator,
                device=coeffs.device,
                dtype=coeffs.dtype,
            )

        coeffs_i = coeffs[start:end]

        delta_x_masked = einsum(
            repr_A, x[:, mask],
            "n_chunks n_reprs r d_in, bs ctx_len d_in -> n_chunks n_reprs bs ctx_len r"
        )
        delta_x_masked = einsum(
            repr_B, delta_x_masked,
            "n_chunks n_reprs d_out r, n_chunks n_reprs bs ctx_len r -> n_chunks n_reprs bs ctx_len d_out"
        )
        delta_x_masked = einsum(
            coeffs_i, delta_x_masked,
            "n_chunks n_reprs, n_chunks n_reprs bs ctx_len d_out -> bs ctx_len d_out"
        )

        delta_x[:, mask] = delta_x_masked
        start = end
        i += 1

    # stat = torch.cuda.memory.memory_allocated(device=coeffs.device)
    # print(f"Finishing forward pass for {layer_idx = } with {stat / (1024 ** 3):.1f}Gb allocated", flush=True)

    return base_out + delta_x


def apply_random_repr(
    model: nn.Module,
    layer_indices: Iterable[int],
    # combined_coeffs: Float[Tensor, "n_layers n_modules tot_chunks"],
    coeffs: Float[Tensor, "tot_chunks n_layers n_modules n_reprs"],
    n_queries: Integer[Tensor, "n_ctx"],
    position_ids: Integer[Tensor, "bs seq_len"],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    repr_seeds: Integer[Tensor, "n_ctx"],
    generator: torch.Generator,
) -> None:
    layers = get_layers(model)

    # removed checking if position_ids is None as they should not be
    position_ids = position_ids.squeeze(0)
    seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
    seq_lens = torch.cat(
        (seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device))
    )
    seq_lens += 1
    tot_len = seq_lens.sum().item()
    tot_q = n_queries.sum().item()

    for layer_idx in layer_indices:
        idx = layer_idx.item()
        layer = layers[idx]

        # FIX! hardcoded for down_proj (should wrap into a loop for all modules)
        long_mname = "mlp.down_proj"
        module = attrgetter(long_mname)(layer)

        # FIX! index 0 refers to the single module down_proj
        # layer_coeffs = combined_coeffs[idx, 0, :]
        layer_coeffs = coeffs[:, idx, 0, :]
        module.forward = partial(
            module.forward,
            n_queries=n_queries,
            tot_q=tot_q,
            seq_lens=seq_lens,
            tot_len=tot_len,
            coeffs=layer_coeffs,
            n_ctx_chunks=n_ctx_chunks,
            repr_seeds=repr_seeds,
            generator=generator,
            # layer_idx=idx,
        )
