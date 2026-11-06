"""
codicodec_flow_exporter.py - Exporter for CoDiCodec-Flow DiT Prior.

Exports the Flow-Matching Diffusion Transformer (FlowDiT) as an unrolled Euler
ODE companion generator for NetBender ('flow_dit' mode).
Operates directly on CoDiCodec's 64-channel continuous latent space.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Add project root to sys.path so direct script execution works
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from python_tools.onnx_bend_prep import promote_initializers_to_inputs


# --------------------------------------------------------------------------- #
# Factorized RoPE (intra-chunk & inter-chunk)
# --------------------------------------------------------------------------- #

class FactorizedRoPE(nn.Module):
    """Rotary positional embeddings split between intra-chunk and inter-chunk."""

    def __init__(self, head_dim: int, block_size: int = 8, max_chunks: int = 128, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim={head_dim} must be divisible by 4"
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_chunks = max_chunks
        max_seq_len = block_size * max_chunks

        per_half = head_dim // 2
        n_freqs = per_half // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, n_freqs, dtype=torch.float32) / n_freqs))

        idx = torch.arange(max_seq_len, dtype=torch.float32)
        intra_pos = idx % block_size
        inter_pos = torch.div(idx, block_size, rounding_mode="floor")

        ang_intra = intra_pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
        ang_inter = inter_pos.unsqueeze(-1) * inv_freq.unsqueeze(0)

        cos_intra = ang_intra.cos().repeat_interleave(2, dim=-1)
        sin_intra = ang_intra.sin().repeat_interleave(2, dim=-1)
        cos_inter = ang_inter.cos().repeat_interleave(2, dim=-1)
        sin_inter = ang_inter.sin().repeat_interleave(2, dim=-1)

        cos = torch.cat([cos_intra, cos_inter], dim=-1)  # [max_seq_len, head_dim]
        sin = torch.cat([sin_intra, sin_inter], dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, head_dim]
        n = x.shape[-2]
        cos = self.cos[:n].to(x.dtype)
        sin = self.sin[:n].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Layers & Modulations
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embeddings for scalar/per-token time values."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DiTBlock(nn.Module):
    """AdaLN-Zero conditioned Transformer block with QK-norm and RoPE."""

    def __init__(self, dim: int, n_heads: int, head_dim: int, mlp_mult: int, cond_dim: int, rope: FactorizedRoPE):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope = rope

        self.norm1 = RMSNorm(dim)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.qkv = nn.Linear(dim, 3 * n_heads * head_dim, bias=False)
        self.proj_attn = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim=int(dim * mlp_mult * 2 / 3))

        # AdaLN-Zero: scale1, shift1, gate1, scale2, shift2, gate2
        self.mod = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, l, d = x.shape
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.mod(F.silu(c)).chunk(6, dim=-1)

        # Self-attention branch
        h = self.norm1(x) * (1.0 + gamma1) + beta1
        qkv = self.qkv(h).view(b, l, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)  # [B, H, L, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(b, l, -1)
        x = x + alpha1 * self.proj_attn(attn_out)

        # MLP branch
        h2 = self.norm2(x) * (1.0 + gamma2) + beta2
        x = x + alpha2 * self.mlp(h2)
        return x


# --------------------------------------------------------------------------- #
# FlowDiT Architecture
# --------------------------------------------------------------------------- #

class FlowDiT(nn.Module):
    """
    Diffusion Transformer operating on CoDiCodec continuous latents.
    Predicts velocity field v = eps - x_data in normalized space.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        block_size: int = 8,
        dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        head_dim: int = 64,
        mlp_mult: int = 4,
        cond_dim: int = 384,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.dim = dim
        self.cond_dim = cond_dim
        self.max_seq_len = max_seq_len

        self.register_buffer("lat_mean", torch.zeros(latent_dim, dtype=torch.float32))
        self.register_buffer("lat_std", torch.ones(latent_dim, dtype=torch.float32))

        self.proj_in = nn.Linear(latent_dim, dim)
        self.rope = FactorizedRoPE(head_dim=head_dim, block_size=block_size, max_chunks=max_seq_len // block_size)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(dim=dim, n_heads=n_heads, head_dim=head_dim, mlp_mult=mlp_mult, cond_dim=cond_dim, rope=self.rope)
                for _ in range(n_layers)
            ]
        )

        self.norm_out = RMSNorm(dim)
        self.mod_out = nn.Linear(cond_dim, 2 * dim)
        self.proj_out = nn.Linear(dim, latent_dim)

        nn.init.zeros_(self.mod_out.weight)
        nn.init.zeros_(self.mod_out.bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.lat_mean) / self.lat_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lat_std + self.lat_mean

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x_t: [B, L, latent_dim]
        # t: [B, L]
        b, l, _ = x_t.shape
        h = self.proj_in(x_t)

        t_emb = timestep_embedding(t, self.cond_dim)
        c = self.cond_mlp(t_emb)

        for block in self.blocks:
            h = block(h, c, attn_mask=attn_mask)

        scale, shift = self.mod_out(F.silu(c)).chunk(2, dim=-1)
        h = self.norm_out(h) * (1.0 + scale) + shift
        v_pred = self.proj_out(h)
        return v_pred


# --------------------------------------------------------------------------- #
# Unrolled Flow Matching Generator (Single-Pass for NetBender)
# --------------------------------------------------------------------------- #

class UnrolledFlowDiTGenerator(nn.Module):
    """
    Fixed-step unrolled Euler ODE flow sampler for real-time inference in NetBender.
    Converts seed noise (and optional audio conditioning latents) directly into
    synthesizable CoDiCodec continuous latent frames [B, 64, frames].
    """

    def __init__(self, dit: FlowDiT, steps: int = 4, target_frames: int = 8):
        super().__init__()
        self.dit = dit
        self.steps = steps
        self.target_frames = target_frames
        self.dt = 1.0 / steps

        # Register fixed time schedule: e.g. [1.0, 0.75, 0.5, 0.25]
        self.schedule = [float(1.0 - (i * self.dt)) for i in range(steps)]

        # Static block-causal attention mask for target_frames
        mask = torch.zeros(target_frames, target_frames, dtype=torch.float32)
        # Full bidirectional within the block
        self.register_buffer("attn_mask", mask)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Runs unrolled Euler flow matching:
        Input:  [B, 64, target_frames] (seed noise or previous chunk)
        Output: [B, 64, target_frames] (clean continuous latents)
        """
        # Permute to token layout [B, L, 64]
        b = z_in.shape[0]
        x = z_in.permute(0, 2, 1).contiguous()
        x = self.dit.normalize(x)

        # Unrolled ODE integration from t=1.0 down to t=0.0
        for step in range(self.steps):
            t_val = self.schedule[step]
            t_tensor = torch.full((b, self.target_frames), t_val, dtype=torch.float32, device=x.device)
            v = self.dit(x, t_tensor, attn_mask=self.attn_mask)
            x = x - self.dt * v

        clean = self.dit.denormalize(x)
        # Return standard NetBender format: [B, 64, target_frames]
        return clean.permute(0, 2, 1).contiguous()


# --------------------------------------------------------------------------- #
# Exporter Routine
# --------------------------------------------------------------------------- #

def export_codicodec_flow(
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
    format: str = "onnx",
    steps: int = 4,
    target_frames: int = 8,
    latent_dim: int = 64,
    promote_for_bending: bool = True,
    verbose: bool = True,
) -> str:
    """Exports CoDiCodec-Flow DiT to bendable ONNX, TorchScript, or ExecuTorch."""
    if output_path is None:
        base_dir = "models"
        os.makedirs(base_dir, exist_ok=True)
        if format == "onnx":
            output_path = os.path.join(base_dir, "codicodec_bendable_flow_dit.onnx")
        elif format == "pte":
            output_path = os.path.join(base_dir, "codicodec_flow_dit.pte")
        else:
            output_path = os.path.join(base_dir, "codicodec_flow_dit.ts")

    base_stem, ext = os.path.splitext(output_path)

    # Initialize FlowDiT architecture
    dit = FlowDiT(
        latent_dim=latent_dim,
        block_size=8,
        dim=384,
        n_layers=6,
        n_heads=6,
        head_dim=64,
        mlp_mult=4,
        cond_dim=384,
        max_seq_len=256,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        if verbose:
            logging.info(f"Loading weights from {checkpoint_path}...")
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = sd.get("ema", sd.get("model", sd))
        # Filter state dict keys
        clean_sd = {k.replace("module.", ""): v for k, v in weights.items() if k in dit.state_dict()}
        dit.load_state_dict(clean_sd, strict=False)

    generator = UnrolledFlowDiTGenerator(dit, steps=steps, target_frames=target_frames).eval()
    sample_latent = torch.randn(1, latent_dim, target_frames, dtype=torch.float32)

    if format in ("onnx", "both", "all"):
        raw_onnx = f"{base_stem}_raw.onnx"
        if verbose:
            logging.info(f"Exporting FlowDiT to ONNX -> {raw_onnx}...")
        torch.onnx.export(
            generator,
            (sample_latent,),
            raw_onnx,
            input_names=["latent_in"],
            output_names=["latent_out"],
            dynamic_axes={"latent_in": {0: "batch", 2: "frames"}, "latent_out": {0: "batch", 2: "frames"}},
            opset_version=17,
            dynamo=False,
        )

        metadata = {
            "netbender.architecture": "flow_dit",
            "netbender.sample_rate": "48000",
            "netbender.ratio": "4096",
            "netbender.latent_dim": str(latent_dim),
            "netbender.flow_steps": str(steps),
        }

        if promote_for_bending:
            if verbose:
                logging.info(f"Promoting FlowDiT parameters to graph inputs for bending -> {output_path}...")
            promote_initializers_to_inputs(
                input_path=raw_onnx,
                output_path=output_path,
                keep_initializers_as_defaults=True,
                verbose=verbose,
                extra_metadata=metadata,
            )
            if os.path.exists(raw_onnx) and raw_onnx != output_path:
                os.remove(raw_onnx)
        else:
            if raw_onnx != output_path:
                os.replace(raw_onnx, output_path)

        if verbose:
            logging.info(f"Successfully exported Bendable Flow-DiT: {output_path}")

    elif format == "ts":
        if verbose:
            logging.info(f"Exporting FlowDiT to TorchScript -> {output_path}...")
        traced = torch.jit.trace(generator, (sample_latent,), check_trace=False)
        traced.save(output_path)
        if verbose:
            logging.info(f"Successfully exported TorchScript Flow-DiT: {output_path}")

    elif format == "pte":
        if verbose:
            logging.info(f"Exporting FlowDiT to ExecuTorch -> {output_path}...")
        from executorch.exir import to_edge
        with torch.no_grad():
            ep = torch.export.export(generator, (sample_latent,), strict=False)
        et = to_edge(ep).to_executorch()
        with open(output_path, "wb") as pf:
            et.write_to_file(pf)
        if verbose:
            logging.info(f"Successfully exported ExecuTorch Flow-DiT: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export CoDiCodec-Flow DiT Prior for NetBender.")
    parser.add_argument("-i", "--checkpoint", default=None, help="Path to trained .pt checkpoint (optional)")
    parser.add_argument("-o", "--output", default=None, help="Output file path (.onnx, .ts, or .pte)")
    parser.add_argument("--format", choices=["onnx", "ts", "pte"], default="onnx", help="Export format")
    parser.add_argument("--steps", type=int, default=4, help="Unrolled Euler flow steps (default: 4)")
    parser.add_argument("--target-frames", type=int, default=8, help="Latent chunk frames (default: 8)")
    parser.add_argument("--no-bending", action="store_true", help="Do not promote parameters for weight bending")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    export_codicodec_flow(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        format=args.format,
        steps=args.steps,
        target_frames=args.target_frames,
        promote_for_bending=not args.no_bending,
    )


if __name__ == "__main__":
    main()
