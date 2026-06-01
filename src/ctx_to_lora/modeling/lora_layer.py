from collections.abc import Iterable
from functools import partial
from operator import attrgetter
from math import floor

import torch
from torch import nn
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from ctx_to_lora.utils import get_layers


def lora_forward(
    x: Float[Tensor, "tot_q seq_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
    tot_q: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self, *args, **kwargs,
) -> Float[Tensor, "tot_q seq_len d_out"]:
    # A: [n_ctx, r, d_in] -> [tot_q, r, d_in]
    A = A.repeat_interleave(n_queries, dim=0, output_size=tot_q)
    # B: [n_ctx, d_out, r] -> [tot_q, d_out, r]
    B = B.repeat_interleave(n_queries, dim=0, output_size=tot_q)

    base_out = nn.Linear.forward(self, x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    delta_x = einsum(A, delta_x, "tot_q r d_in, tot_q s_len d_in -> tot_q s_len r")
    delta_x = einsum(B, delta_x, "tot_q r d_out, tot_q s_len r -> tot_q s_len d_out")
    delta_x = delta_x * scaling
    return (base_out + delta_x).to(base_out.dtype)


def lora_forward_packed(
    x: Float[Tensor, "1 tot_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
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

    repeated_A = A.repeat_interleave(n_queries, dim=0, output_size=tot_q)
    repeated_A = repeated_A.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    repeated_B = B.repeat_interleave(n_queries, dim=0, output_size=tot_q)
    repeated_B = repeated_B.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    delta_x = einsum(
        repeated_A, delta_x, "tot_len r d_in, bs tot_len d_in -> bs tot_len r"
    )
    delta_x = einsum(
        repeated_B, delta_x, "tot_len r d_out, bs tot_len r -> bs tot_len d_out"
    )
    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def lora_forward_packed_(  # potentially optimized version
    x: Float[Tensor, "1 tot_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
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

    delta_x = torch.zeros(
        (1, tot_len, self.out_features),
        device=x.device,
        dtype=x.dtype,
    )

    start_q = start = 0
    for j, n_q in enumerate(n_queries):
        end_q = start_q + n_q.item()
        end = start + seq_lens[start_q:end_q].sum().item()

        A_ctx = A[j]
        B_ctx = B[j]

        delta_x[:, start:end, :] = (x[:, start:end, :] @ A_ctx.T) @ B_ctx

        start_q = end_q
        start = end

    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def apply_lora_to_layers(
    model: nn.Module,
    layer_indices: Iterable[int],
    combined_loras: dict[str, dict[str, Float[Tensor, "n_ctx n_layers r d"]]],
    n_queries: Integer[Tensor, "n_ctx"],
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

    tot_q = n_queries.sum().item()
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
            module.forward = partial(module.forward, n_queries=n_queries, tot_q=tot_q, A=A, B=B)
            if position_ids is not None:
                module.forward = partial(
                    module.forward, seq_lens=seq_lens, tot_len=tot_len
                )


def orthog_proj_forward(  # maybe needs fixing/optimization
    x: Float[Tensor, "tot_q seq_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
    tot_q: int,
    A: Float[Tensor, "tot_chunks r d_in"],
    B: Float[Tensor, "tot_chunks r d_out"],
    V: Float[Tensor, "d_in k"],
    U: Float[Tensor, "d_out k"],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    repr_seeds: Integer[Tensor, "n_ctx"],
    generator: torch.Generator,
    lora_dropout_p: float,
    scaling: float,
    self, *args, **kwargs,
) -> Float[Tensor, "tot_q seq_len d_out"]:
    base_out = nn.Linear.forward(self, x, *args, **kwargs)

    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)

    max_num_slots = 32
    k_preserve = 128
    k = min(self.in_features, self.out_features)
    cols_per_slot = floor((k - k_preserve) / max_num_slots)

    seq_len = x.shape[1]
    delta_x = torch.zeros(
        (tot_q, seq_len, self.out_features),
        device=x.device,
        dtype=x.dtype,
    )

    start_q = 0
    j = 0
    for n_qs, n_chunks, seed in zip(n_queries, n_ctx_chunks, repr_seeds):
        end_q = start_q + n_qs.item()

        if n_chunks > 1:
            generator.manual_seed(seed.item())

            chunk_perm = torch.randperm(
                max_num_slots, generator=generator, device=x.device
            )[:n_chunks]

            col_start = k_preserve + chunk_perm[0].item()
            indices = torch.arange(col_start, k, cols_per_slot, device=x.device)

            chunk_A = A[j]
            chunk_B = B[j]
            V_proj = V[:, indices].to(A.dtype)
            U_proj = U[:, indices].to(A.dtype)

            # [n_q, seq_len, d_in]
            h = (x[start_q:end_q] @ V_proj) @ V_proj.T
            h = (h @ chunk_A.T) @ chunk_B
            h = (h @ U_proj) @ U_proj.T
            delta_x_masked = h
            j += 1

            for chunk_slot in chunk_perm[1:]:
                col_start = k_preserve + chunk_slot.item()
                indices = torch.arange(col_start, k, cols_per_slot, device=x.device)

                chunk_A = A[j]
                chunk_B = B[j]
                V_proj = V[:, indices].to(A.dtype)
                U_proj = U[:, indices].to(A.dtype)

                # [n_q, seq_len, d_in]
                h = (x[start_q:end_q] @ V_proj) @ V_proj.T
                h = (h @ chunk_A.T) @ chunk_B
                h = (h @ U_proj) @ U_proj.T

                delta_x_masked = delta_x_masked + h
                j += 1

        else:
            chunk_A = A[j]
            chunk_B = B[j]
            delta_x_masked = (x[start_q:end_q] @ chunk_A.T) @ chunk_B
            j += 1

        delta_x[start_q:end_q] = delta_x_masked
        start_q = end_q

    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def orthog_proj_forward_packed(
    x: Float[Tensor, "1 tot_len d_in"],
    n_queries: Integer[Tensor, "n_ctx"],
    tot_q: int, # unused
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    A: Float[Tensor, "tot_chunks r d_in"],
    B: Float[Tensor, "tot_chunks r d_out"],
    V: Float[Tensor, "d_in k"],
    U: Float[Tensor, "d_out k"],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    repr_seeds: Integer[Tensor, "n_ctx"],
    generator: torch.Generator,
    lora_dropout_p: float,
    scaling: float,
    self, *args, **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = nn.Linear.forward(self, x, *args, **kwargs)

    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)

    max_num_slots = 32
    k_preserve = 128
    k = min(self.in_features, self.out_features)
    cols_per_slot = floor((k - k_preserve) / max_num_slots)

    delta_x = torch.zeros(
        (1, tot_len, self.out_features),
        device=x.device,
        dtype=x.dtype,
    )

    start_q = start_c = start = 0
    for n_q, n_chunks, seed in zip(n_queries, n_ctx_chunks, repr_seeds):
        end_q = start_q + n_q.item()
        end_c = start_c + n_chunks.item()
        end = start + seq_lens[start_q:end_q].sum().item()

        if n_chunks > 1:
            generator.manual_seed(seed.item())

            chunk_perm = torch.randperm(
                max_num_slots, generator=generator, device=x.device
            )[:n_chunks]

            indices = []
            for chunk_slot in chunk_perm:
                start_col = k_preserve + chunk_slot.item()
                end_col = k_preserve + cols_per_slot * max_num_slots
                slot_indices = torch.arange(
                    start_col, end_col, cols_per_slot, device=x.device
                )
                indices.append(slot_indices)

            indices = torch.stack(indices, dim=0)  # [n_chunks, cols_per_slot]

            V_ctx = V[:, indices].to(A.dtype)  # [d_in, n_chunks, cols_per_slot]
            U_ctx = U[:, indices].to(A.dtype)  # [d_out, n_chunks, cols_per_slot]

            A_ctx = A[start_c:end_c]  # [n_chunks, r, d_in]
            B_ctx = B[start_c:end_c]  # [n_chunks, r, d_out]

            A_proj = einsum(
                V_ctx, A_ctx,
                "d_in n_chunks cols_per_slot, n_chunks r d_in -> n_chunks cols_per_slot r",
            )
            A_proj = einsum(
                V_ctx, A_proj,
                "d_in n_chunks cols_per_slot, n_chunks cols_per_slot r -> d_in n_chunks r",
            )
            A_proj = A_proj.reshape(self.in_features, -1)

            B_proj = einsum(
                B_ctx, U_ctx,
                "n_chunks r d_out, d_out n_chunks cols_per_slot -> n_chunks r cols_per_slot",
            )
            B_proj = einsum(
                B_proj, U_ctx,
                "n_chunks r cols_per_slot, d_out n_chunks cols_per_slot -> n_chunks r d_out",
            )
            B_proj = B_proj.reshape(-1, self.out_features)

            # [1, seq_len_i, d_in]
            delta_x_masked = (x[:, start:end, :] @ A_proj) @ B_proj
        else:
            A_ctx = A[start_c]
            B_ctx = B[start_c]
            delta_x_masked = (x[:, start:end, :] @ A_ctx.T) @ B_ctx

        delta_x[:, start:end, :] = delta_x_masked

        start_q = end_q
        start_c = end_c
        start = end

    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def apply_orthog_proj(
    model: nn.Module,
    layer_indices: Iterable[int],
    generated_loras: dict[str, dict[str, Tensor]],  # "tot_chunks n_layers r d_in/out"
    svd: dict[str, Tensor],
    n_queries: Integer[Tensor, "n_ctx"],
    position_ids: Integer[Tensor, "bs seq_len"],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    repr_seeds: Integer[Tensor, "n_ctx"],
    generator: torch.Generator,
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

    tot_q = n_queries.sum().item()

    for layer_idx in layer_indices:
        layer = layers[layer_idx]

        for mname in generated_loras:
            if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
                long_mname = f"self_attn.{mname}"
            elif mname in ["down_proj", "up_proj", "gate_proj"]:
                long_mname = f"mlp.{mname}"
            module = attrgetter(long_mname)(layer)

            A = generated_loras[mname]["A"][:, layer_idx]
            B = generated_loras[mname]["B"][:, layer_idx]

            full_name = f"{layer_idx.item()}.{long_mname}"
            V = svd[f"{full_name}.V"]
            U = svd[f"{full_name}.U"]

            module.forward = partial(
                module.forward,
                n_queries=n_queries, tot_q=tot_q,
                A=A, B=B,
                V=V, U=U,
                n_ctx_chunks=n_ctx_chunks,
                repr_seeds=repr_seeds,
                generator=generator,
            )

            if position_ids is not None:
                module.forward = partial(
                    module.forward, seq_lens=seq_lens, tot_len=tot_len
                )
