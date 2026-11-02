"""
same_s.py - TorchScript & ONNX compatible architecture and weight loader for Stable Audio 3 SAME-S.

Based on stabilityai/stable-audio-3 SAME-S autoencoder and jasper-zheng/streamable-same-s.
Supports direct weight loading from safetensors checkpoints without requiring external heavy dependencies.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- architecture constants (SAME-S) ---
DIM = 768
NUM_HEADS = 12
HEAD_DIM = 64
ROPE_DIMS = 32
NUM_BLOCKS = 6
FF_INNER = 2304  # ff_mult = 3
LATENT_DIM = 256
IO_CHANNELS = 2
PATCH_SIZE = 256
IN_CHANNELS = IO_CHANNELS * PATCH_SIZE  # 512, pretransform output
STRIDE = 16
SUB_CHUNK = STRIDE + 1  # 17 (16 real + 1 query token)
CHUNK_SIZE = 32
EFFECTIVE_CHUNK = CHUNK_SIZE + CHUNK_SIZE // STRIDE  # 34
SHIFT = EFFECTIVE_CHUNK // 2  # 17
ALIGN = CHUNK_SIZE // STRIDE  # 2 latent frames (8192 audio samples)
ROPE_BASE = 10000.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


class DyT(nn.Module):
    """Dynamic Tanh: gamma * tanh(alpha * x) + beta."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class DifferentialAttention(nn.Module):
    """Differential self-attention: SDPA(q,k,v) - SDPA(q_diff,k_diff,v)."""

    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.rope_dims = ROPE_DIMS
        self.dim = DIM
        self.to_qkv = nn.Linear(DIM, 5 * DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)
        self.q_norm = DyT(HEAD_DIM)
        self.k_norm = DyT(HEAD_DIM)
        self.scale = HEAD_DIM ** -0.5
        inv_freq = 1.0 / (
            ROPE_BASE ** (torch.arange(0, ROPE_DIMS, 2).float() / ROPE_DIMS)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # (N, ROPE_DIMS/2)
        freqs = torch.cat((freqs, freqs), dim=-1)  # (N, ROPE_DIMS)
        cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,N,ROPE_DIMS)
        sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rot = x[..., :self.rope_dims]
        passthrough = x[..., self.rope_dims:]
        rot = rot * cos + _rotate_half(rot) * sin
        return torch.cat((rot, passthrough), dim=-1)

    def _heads(self, t: torch.Tensor, b: int, n: int) -> torch.Tensor:
        return t.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        n = x.shape[1]
        qkv = self.to_qkv(x)
        chunks = qkv.chunk(5, dim=-1)
        q = self._heads(chunks[0], b, n)
        k = self._heads(chunks[1], b, n)
        v = self._heads(chunks[2], b, n)
        q_diff = self._heads(chunks[3], b, n)
        k_diff = self._heads(chunks[4], b, n)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q_diff = self.q_norm(q_diff)
        k_diff = self.k_norm(k_diff)

        cos, sin = self._rope_cos_sin(n, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        q_diff = self._apply_rope(q_diff, cos, sin)
        k_diff = self._apply_rope(k_diff, cos, sin)

        out_main = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out_diff = F.scaled_dot_product_attention(q_diff, k_diff, v, scale=self.scale)
        out = out_main - out_diff

        out = out.permute(0, 2, 1, 3).reshape(b, n, self.dim)
        return self.to_out(out)


class GLUFeedForward(nn.Module):
    """SwiGLU feedforward: proj_out(value * silu(gate)), inner = FF_INNER."""

    def __init__(self):
        super().__init__()
        self.glu_proj = nn.Linear(DIM, FF_INNER * 2, bias=True)
        self.proj_out = nn.Linear(FF_INNER, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.glu_proj(x)
        value, gate = x.chunk(2, dim=-1)
        return self.proj_out(value * F.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = DyT(DIM)
        self.attn = DifferentialAttention()
        self.ff_norm = DyT(DIM)
        self.ff = GLUFeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.pre_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class _ChunkedTransformer(nn.Module):
    """6 SAME-S blocks with chunked attention + midpoint shift."""

    def __init__(self):
        super().__init__()
        self.effective_chunk = EFFECTIVE_CHUNK
        self.shift = SHIFT
        self.blocks_first = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS // 2)])
        self.blocks_second = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS - NUM_BLOCKS // 2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        internal_t = x.shape[1]
        d = x.shape[2]
        eff = self.effective_chunk
        shift = self.shift

        # First half: standard chunk boundaries.
        nc1 = internal_t // eff
        x = x.reshape(b * nc1, eff, d)
        for blk in self.blocks_first:
            x = blk(x)
        x = x.reshape(b, internal_t, d)

        # Second half: pad half a chunk on each side, re-chunk with offset boundaries.
        left = x[:, :shift, :]
        right = x[:, internal_t - shift:, :]
        x = torch.cat((left, x, right), dim=1)
        t2 = internal_t + 2 * shift
        nc2 = t2 // eff
        x = x.reshape(b * nc2, eff, d)
        for blk in self.blocks_second:
            x = blk(x)
        x = x.reshape(b, t2, d)
        x = x[:, shift:t2 - shift, :]
        return x


class SAMESCore(nn.Module):
    """
    Scriptable Stable Audio 3 SAME-S Autoencoder Core.
    Encode: [B, 2, T] -> [B, 256, T // 4096]
    Decode: [B, 256, T_lat] -> [B, 2, T_lat * 4096]
    """

    def __init__(self):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.io_channels = IO_CHANNELS
        self.in_channels = IN_CHANNELS
        self.stride = STRIDE
        self.dim = DIM
        self.sub_chunk = SUB_CHUNK
        self.align = ALIGN

        # Encoder
        self.enc_mapping = nn.Linear(IN_CHANNELS, DIM, bias=True)
        self.enc_new_tokens = nn.Parameter(torch.zeros(1, 1, DIM))
        self.enc_transformer = _ChunkedTransformer()
        self.enc_project_out = nn.Linear(DIM, LATENT_DIM, bias=True)

        # Decoder
        self.dec_project_in = nn.Linear(LATENT_DIM, DIM, bias=True)
        self.dec_new_tokens = nn.Parameter(torch.zeros(1, 1, DIM))
        self.dec_transformer = _ChunkedTransformer()
        self.dec_mapping = nn.Conv1d(DIM, IN_CHANNELS, kernel_size=3, padding=1, bias=True)

        # Softnorm bottleneck
        self.scaling_factor = nn.Parameter(torch.ones(1, LATENT_DIM, 1))
        self.sn_bias = nn.Parameter(torch.zeros(1, LATENT_DIM, 1))
        self.running_std = nn.Parameter(torch.ones(1), requires_grad=False)

    def _patch(self, audio: torch.Tensor) -> torch.Tensor:
        b = audio.shape[0]
        t = audio.shape[2]
        lp = t // self.patch_size
        x = audio.reshape(b, self.io_channels, lp, self.patch_size)
        x = x.permute(0, 1, 3, 2)  # (B, 2, 256, L)
        return x.reshape(b, self.in_channels, lp)

    def _unpatch(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        lp = x.shape[2]
        x = x.reshape(b, self.io_channels, self.patch_size, lp)
        x = x.permute(0, 1, 3, 2)  # (B, 2, L, 256)
        return x.reshape(b, self.io_channels, lp * self.patch_size)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        x = self._patch(audio)
        b = x.shape[0]
        t_aud = x.shape[2]

        mod = self.stride * self.align
        pad = (mod - t_aud % mod) % mod
        if pad > 0:
            x = F.pad(x, (0, pad))
            t_aud = t_aud + pad

        x = x.transpose(1, 2)
        x = self.enc_mapping(x)

        t_lat = t_aud // self.stride
        x = x.reshape(b * t_lat, self.stride, self.dim)
        nt = self.enc_new_tokens.expand(b * t_lat, 1, self.dim)
        x = torch.cat((x, nt), dim=1)
        x = x.reshape(b, t_lat * self.sub_chunk, self.dim)

        x = self.enc_transformer(x)

        x = x.reshape(b * t_lat, self.sub_chunk, self.dim)
        x = x[:, self.sub_chunk - 1:self.sub_chunk, :]
        x = x.reshape(b, t_lat, self.dim)

        x = self.enc_project_out(x)
        x = x.transpose(1, 2)

        x = x * self.scaling_factor + self.sn_bias
        x = x / self.running_std
        return x

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents * self.running_std
        b = x.shape[0]
        t_lat = x.shape[2]
        t_lat_real = t_lat
        x = x.transpose(1, 2)
        x = self.dec_project_in(x)

        pad = (self.align - t_lat % self.align) % self.align
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            t_lat = t_lat + pad

        x = x.reshape(b * t_lat, 1, self.dim)
        nt = self.dec_new_tokens.expand(b * t_lat, self.stride, self.dim)
        x = torch.cat((x, nt), dim=1)
        x = x.reshape(b, t_lat * self.sub_chunk, self.dim)

        x = self.dec_transformer(x)

        x = x.reshape(b * t_lat, self.sub_chunk, self.dim)
        x = x[:, 1:self.sub_chunk, :]
        x = x.reshape(b, t_lat * self.stride, self.dim)

        x = x.transpose(1, 2)
        x = self.dec_mapping(x)
        audio = self._unpatch(x)
        return audio[..., :t_lat_real * self.patch_size * self.stride]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        n = audio.shape[-1]
        return self.decode(self.encode(audio))[..., :n]


def load_same_s_from_safetensors(safetensors_path: str, device: str = "cpu") -> SAMESCore:
    """Loads SAMESCore directly from a safetensors checkpoint without external dependencies."""
    import safetensors.torch
    state_dict = safetensors.torch.load_file(safetensors_path, device=device)
    core = SAMESCore().to(device).eval()

    def load_dyt(dst: DyT, prefix: str):
        dst.alpha.data.copy_(state_dict[f"{prefix}.alpha"])
        dst.gamma.data.copy_(state_dict[f"{prefix}.gamma"])
        dst.beta.data.copy_(state_dict[f"{prefix}.beta"])

    def load_block(dst: TransformerBlock, prefix: str):
        load_dyt(dst.pre_norm, f"{prefix}.pre_norm")
        dst.attn.to_qkv.weight.data.copy_(state_dict[f"{prefix}.self_attn.to_qkv.weight"])
        dst.attn.to_out.weight.data.copy_(state_dict[f"{prefix}.self_attn.to_out.weight"])
        load_dyt(dst.attn.q_norm, f"{prefix}.self_attn.q_norm")
        load_dyt(dst.attn.k_norm, f"{prefix}.self_attn.k_norm")
        load_dyt(dst.ff_norm, f"{prefix}.ff_norm")
        dst.ff.glu_proj.weight.data.copy_(state_dict[f"{prefix}.ff.ff.0.proj.weight"])
        dst.ff.glu_proj.bias.data.copy_(state_dict[f"{prefix}.ff.ff.0.proj.bias"])
        dst.ff.proj_out.weight.data.copy_(state_dict[f"{prefix}.ff.ff.2.weight"])
        dst.ff.proj_out.bias.data.copy_(state_dict[f"{prefix}.ff.ff.2.bias"])

    # 1. Encoder Mapping
    enc_wv = state_dict["encoder.layers.0.mapping.weight_v"]
    enc_wg = state_dict["encoder.layers.0.mapping.weight_g"]
    enc_w = enc_wg * (enc_wv / torch.norm(enc_wv, dim=(1, 2), keepdim=True))
    core.enc_mapping.weight.data.copy_(enc_w.squeeze(-1))
    core.enc_mapping.bias.data.copy_(state_dict["encoder.layers.0.mapping.bias"])
    core.enc_new_tokens.data.copy_(state_dict["encoder.layers.0.new_tokens"])

    # 2. Encoder Transformer Blocks
    for i in range(3):
        load_block(core.enc_transformer.blocks_first[i], f"encoder.layers.0.transformers.{i}")
        load_block(core.enc_transformer.blocks_second[i], f"encoder.layers.0.transformers.{3 + i}")

    core.enc_project_out.weight.data.copy_(state_dict["encoder.layers.2.weight"])
    core.enc_project_out.bias.data.copy_(state_dict["encoder.layers.2.bias"])

    # 3. Decoder
    core.dec_project_in.weight.data.copy_(state_dict["decoder.layers.1.weight"])
    core.dec_project_in.bias.data.copy_(state_dict["decoder.layers.1.bias"])
    core.dec_new_tokens.data.copy_(state_dict["decoder.layers.3.new_tokens"])

    for i in range(3):
        load_block(core.dec_transformer.blocks_first[i], f"decoder.layers.3.transformers.{i}")
        load_block(core.dec_transformer.blocks_second[i], f"decoder.layers.3.transformers.{3 + i}")

    dec_wv = state_dict["decoder.layers.3.mapping.weight_v"]
    dec_wg = state_dict["decoder.layers.3.mapping.weight_g"]
    dec_w = dec_wg * (dec_wv / torch.norm(dec_wv, dim=(1, 2), keepdim=True))
    core.dec_mapping.weight.data.copy_(dec_w)
    core.dec_mapping.bias.data.copy_(state_dict["decoder.layers.3.mapping.bias"])

    # 4. Bottleneck Softnorm
    core.scaling_factor.data.copy_(state_dict["bottleneck.scaling_factor"])
    core.sn_bias.data.copy_(state_dict["bottleneck.bias"])
    core.running_std.data.copy_(state_dict["bottleneck.running_std"])

    core.requires_grad_(False)
    return core

