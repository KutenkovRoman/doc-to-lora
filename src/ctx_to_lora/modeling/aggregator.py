import logging
from dataclasses import dataclass
from enum import Enum

from einops import rearrange, repeat, unpack
from jaxtyping import Float, Integer
from torch import Tensor, Generator, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from ctx_to_lora.configs import (
    AggregatorArguments,
)
from ctx_to_lora.modeling.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig
from ctx_to_lora.pooling import POOL_FN
from ctx_to_lora.utils import get_num_layers

logger = logging.getLogger()


class AGGREGATOR_TYPE(str, Enum):
    POOLER = "pooler"
    PERCEIVER = "perceiver"


@dataclass
class AggregatorConfig:
    aggregator_type: AGGREGATOR_TYPE
    num_layers: int
    num_modules: int
    num_extra_modules: int
    output_size: int
    feature_size: int

    # pooler
    pooling_type: POOL_FN

    # perceiver
    num_latent_factor: int
    lora_r: int
    per_rank_gen: bool

    n_latent_queries: int
    num_blocks: int
    num_self_attn_per_block: int
    shared_weights: bool
    layer_to_layer_ctx_encoder: bool


def get_aggregator_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    layer_to_layer_ctx_encoder: bool,
    output_size: int,
    num_modules: int,
    num_extra_modules: int,
    lora_r: int,
    per_rank_gen: bool,
    aggregator_args: AggregatorArguments,
):
    return AggregatorConfig(
        feature_size=ctx_encoder_model_config.hidden_size,
        output_size=output_size,
        num_layers=get_num_layers(model),
        num_modules=num_modules,
        num_extra_modules=num_extra_modules,
        lora_r=lora_r,
        per_rank_gen=per_rank_gen,
        layer_to_layer_ctx_encoder=layer_to_layer_ctx_encoder,
        **vars(aggregator_args),
    )


class Perceiver(nn.Module):
    """perceiver with bottleneck size = n_modules * n_layers"""

    def __init__(
        self,
        feature_size,
        output_size,
        num_layers,
        num_modules,
        num_extra_modules,
        per_rank_gen,
        lora_r,
        # num_latent_factor,  # unused
        layer_to_layer_ctx_encoder,
        n_latent_queries,
        use_orthog_proj,
        *args, **kwargs,
    ):
        super().__init__()
        assert num_extra_modules == 0
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.num_extra_modules = num_extra_modules
        self.per_rank_gen = per_rank_gen
        self.r = lora_r if self.per_rank_gen else 1
        self.layer_to_layer = layer_to_layer_ctx_encoder

        n_output_queries = (
            (num_modules * self.r + num_extra_modules) if self.layer_to_layer else
            (num_modules * self.r + num_extra_modules) * num_layers
        )

        self.config = Idefics2PerceiverConfig(
            input_size=feature_size,
            num_blocks=kwargs["num_blocks"],
            num_self_attn_per_block=kwargs["num_self_attn_per_block"],
            shared_weights=kwargs["shared_weights"],
            n_latents=n_latent_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.decoder_config = Idefics2PerceiverConfig(
            input_size=output_size,
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.perceiver = Idefics2Perceiver(
            self.config, self.decoder_config,
            use_orthog_proj=use_orthog_proj,
        )
        self.iterative_mode = False

    def enable_iterative_mode(self, flag: bool):
        self.iterative_mode = flag

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"] | Float[Tensor, "bs n_layers seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        repr_seeds: Integer[Tensor, "n_ctx"] | None = None,
        generator: Generator | None = None,
    ):
        if self.layer_to_layer and not self.iterative_mode:
            if attn_mask is not None:  # skipping this
                attn_mask = repeat(
                    attn_mask,
                    "bs seq_len -> (num_layers bs) seq_len",
                    num_layers=self.num_layers,
                )
                features = rearrange(
                    features,
                    "bs num_layers seq_len feature_dim -> (num_layers bs) seq_len feature_dim",
                )
            if position_ids is not None:
                position_ids = repeat(
                    position_ids,
                    "1 seq_len -> 1 (n_layers seq_len)",
                    n_layers=self.num_layers,
                )
                features = rearrange(
                    features,
                    "1 n_layers seq_len feature_dim -> 1 (n_layers seq_len) feature_dim",
                )

        outputs, slot_emb = self.perceiver(
            features,
            attn_mask,
            position_ids,
            n_ctx_chunks,
            repr_seeds,
            generator,
        )

        if self.layer_to_layer:
            if self.iterative_mode:
                lora = rearrange(
                    outputs,
                    "bs (n_modules r) d -> bs n_modules r d",
                    n_modules=self.num_modules,
                    r=self.r,
                )
                # before: return lora, None
                return lora

            per_layer_size = self.num_modules * self.r + self.num_extra_modules
            outputs = rearrange(
                outputs,
                "(n_layers bs) (per_layer_size) d -> bs (n_layers per_layer_size) d",
                n_layers=self.num_layers,
                per_layer_size=per_layer_size,
            )
            slot_emb = rearrange(  # not tested!
                slot_emb,
                "(n_layers bs) d -> bs n_layers d",
                n_layers=self.num_layers,
            )

        lora, _ = unpack(
            outputs,
            [
                [self.num_layers * self.num_modules * self.r],
                [self.num_layers * self.num_extra_modules],
            ],
            "bs * feature_dim",
        )
        lora = rearrange(
            lora,
            "bs (n_layers n_modules r) d -> bs n_layers n_modules r d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
            r=self.r,
        )
        if not self.per_rank_gen:
            lora = lora.squeeze(3)

        return lora, slot_emb


AGGREGATOR_CLS = {
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
}
