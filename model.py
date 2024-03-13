from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers import T5EncoderModel, T5Tokenizer


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm_type="ln",
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.norm_type = norm_type

        if norm_type == "ln":
            self.ln_1 = nn.LayerNorm(d_model)
            self.ln_2 = nn.LayerNorm(d_model)
            self.ln_ff = nn.LayerNorm(d_model)
        elif norm_type in ("ada_norm", "ada_norm_zero"):
            self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
            self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
            self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)
        else:
            raise ValueError(f"invalid {norm_type=}")

        if self.norm_type == "ada_norm_zero":
            self.gate_linear = nn.Linear(
                time_embedding_dim or d_model, 2 * d_model, bias=True
            )
            nn.init.zeros_(self.gate_linear.bias)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        if self.norm_type == "ln":
            latents = latents + self.attention(
                q=self.ln_1(latents),
                kv=torch.cat([self.ln_1(latents), self.ln_2(x)], dim=1),
            )
            latents = latents + self.mlp(self.ln_ff(latents))
        elif self.norm_type == "ada_norm":
            normed_latents = self.ln_1(latents, timestep_embedding)
            latents = latents + self.attention(
                q=normed_latents,
                kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
            )
            latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        elif self.norm_type == "ada_norm_zero":
            gate1, gate2 = (
                self.gate_linear(torch.nn.functional.silu(timestep_embedding))
                .view(len(x), 1, -1)
                .chunk(2, dim=-1)
            )
            normed_latents = self.ln_1(latents, timestep_embedding)
            latents = latents + gate1 * self.attention(
                q=normed_latents,
                kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
            )
            latents = latents + gate2 * self.mlp(
                self.ln_ff(latents, timestep_embedding)
            )
        else:
            raise ValueError(f"invalid {self.norm_type=}")

        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        learnable_query=True,
        output_dim=None,
        input_dim=None,
        norm_type="ln",
        time_embedding_dim: Optional[int] = None,
        time_aware_query_type: Optional[str] = None,
    ):
        super().__init__()
        self.learnable_query = learnable_query
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.time_aware_query_type = time_aware_query_type
        if learnable_query:
            self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))

            if self.time_aware_query_type is not None:
                if self.time_aware_query_type == "add":
                    self.time_aware_linear = nn.Linear(
                        time_embedding_dim or width, width, bias=True
                    )
                    nn.init.zeros_(self.time_aware_linear.bias)
                else:
                    raise ValueError(f"invalid {time_aware_query_type=}")

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width,
                    heads,
                    norm_type=norm_type,
                    time_embedding_dim=time_embedding_dim,
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(
        self, x: torch.Tensor, latents=None, timestep_embedding: torch.Tensor = None
    ):
        learnable_latents = None

        if self.learnable_query:
            learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
            if self.time_aware_query_type == "add":
                learnable_latents = learnable_latents + self.time_aware_linear(
                    torch.nn.functional.silu(timestep_embedding)
                )

        if self.learnable_query and latents is None:
            latents = learnable_latents
        elif self.learnable_query and latents is not None:
            latents = torch.cat([latents, learnable_latents], dim=1)

        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class T5TextEmbedder(nn.Module):
    def __init__(self, pretrained_path="google/flan-t5-xl", max_length=None):
        super().__init__(is_trainable=False, input_keys=["caption"])

        self.model = T5EncoderModel.from_pretrained(pretrained_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        self.max_length = max_length

    def forward(self, caption, text_input_ids=None, attention_mask=None):
        if text_input_ids is None or attention_mask is None:
            if self.max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(
                    caption, return_tensors="pt", add_special_tokens=True
                )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(text_input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings


class ELLA(nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        learnable_query=True,
        input_dim=2048,
        norm_type="ada_norm",
        time_aware_query_type="add",
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            learnable_query=learnable_query,
            input_dim=input_dim,
            norm_type=norm_type,
            time_embedding_dim=time_embed_dim,
            time_aware_query_type=time_aware_query_type,
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states
