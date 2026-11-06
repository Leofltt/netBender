"""
codicodec_exporter.py - Exporter for Sony CSL Paris CoDiCodec.

Converts CoDiCodec (48 kHz / 44.1 kHz Stereo, 64-dim continuous latents, 4096x ratio)
into TorchScript (.ts), Bendable ONNX (.onnx + NBND v2 .bin), and ExecuTorch (.pte)
for real-time weight bending and latent modulation in NetBender.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Add project root to sys.path so direct script execution works
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NumPy 2.x compatibility patch for checkpoint unpickling
import numpy as np
if "f16" not in np.sctypeDict:
    np.sctypeDict["f16"] = np.dtype("float16")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from codicodec.hparams import (
    hop,
    fac,
    alpha_rescale,
    beta_rescale,
    spec_length,
    num_latents,
    bottleneck_channels,
)
from codicodec.inference import sigma_rescale
from codicodec.models import (
    preprocess_parallel_input,
    preprocess_parallel_features,
    postprocess_parallel_input,
    get_sigma_continuous,
    reverse_step,
    sigma_max,
)

from python_tools.exporters.base_exporter import AudioCodecWrapper
from python_tools.onnx_bend_prep import promote_initializers_to_inputs


def safe_atanh(x: torch.Tensor) -> torch.Tensor:
    """ONNX-safe numerical equivalent of atanh."""
    x = x.clamp(-0.9999, 0.9999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


class ConvSTFT(nn.Module):
    """
    Operator-safe STFT frontend implemented using standard Conv1d.
    Converts 48 kHz stereo audio directly into complex STFT patch representations
    with bit-exact numerical parity to codicodec.audio.wv2realimag.
    """

    def __init__(self, hop_size: int = 1024, fac_val: int = 2):
        super().__init__()
        self.hop_size = hop_size
        n_fft = hop_size * fac_val
        num_bins = hop_size * (fac_val // 2)

        window = torch.hann_window(n_fft)
        n = torch.arange(0, n_fft).float().unsqueeze(0)
        k = torch.arange(0, num_bins).float().unsqueeze(1)

        cos_basis = (torch.cos(2 * math.pi * k * n / n_fft) * window).unsqueeze(1)
        sin_basis = (-torch.sin(2 * math.pi * k * n / n_fft) * window).unsqueeze(1)
        basis_mono = torch.cat([cos_basis, sin_basis], dim=0)
        basis_stereo = torch.cat([basis_mono, basis_mono], dim=0)
        self.register_buffer("weight", basis_stereo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, T]
        conv_out = F.conv1d(x, self.weight, stride=self.hop_size, groups=2)
        c0_r, c0_i, c1_r, c1_i = conv_out.chunk(4, dim=1)
        X = torch.stack([c0_r, c0_i, c1_r, c1_i], dim=1)
        return beta_rescale * torch.sign(X) * (X.abs() ** alpha_rescale)


class ConvISTFT(nn.Module):
    """
    Operator-safe iSTFT backend implemented using standard ConvTranspose1d.
    Converts CoDiCodec output spectrograms directly back into stereo audio waveforms
    with overlap-add synthesis matching codicodec.audio.realimag2wv.
    """

    def __init__(self, hop_size: int = 1024, fac_val: int = 2):
        super().__init__()
        self.hop_size = hop_size
        N = hop_size * fac_val

        w = torch.hann_window(N)
        denom = w ** 2
        overlaps = -(-N // hop_size)
        denom = F.pad(denom, (0, overlaps * hop_size - N))
        denom = denom.reshape(overlaps, hop_size).sum(0, keepdim=True).tile(overlaps, 1).reshape(-1)
        w_norm = w / denom[:N]

        n = torch.arange(N).float().unsqueeze(0)
        k = torch.arange(1025).float().unsqueeze(1)

        f_real = torch.ones(1025, 1) * (2.0 / N)
        f_real[0, 0] = 1.0 / N
        f_real[1024, 0] = 1.0 / N
        cos_k = f_real * torch.cos(2 * math.pi * k * n / N) * w_norm

        f_imag = torch.ones(1025, 1) * (-2.0 / N)
        f_imag[0, 0] = 0.0
        f_imag[1024, 0] = 0.0
        sin_k = f_imag * torch.sin(2 * math.pi * k * n / N) * w_norm

        weight_mono = torch.cat([cos_k, sin_k], dim=0).unsqueeze(1)
        weight_stereo = torch.cat([weight_mono, weight_mono], dim=0)
        self.register_buffer("weight", weight_stereo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, 1024, frames]
        x = F.pad(x, (0, 0, 0, 1))
        x = x / beta_rescale
        x = torch.sign(x) * (x.abs() ** (1.0 / alpha_rescale))
        b, c, freq, frames = x.shape
        inp = x.reshape(b, -1, frames)
        out = F.conv_transpose1d(inp, self.weight, stride=self.hop_size, groups=2)
        return out.clamp(-1.0, 1.0)


class CodiCodecEncoderAdapter(nn.Module):
    """
    Adapts CoDiCodec continuous encoding pipeline to pure PyTorch nn.Module:
    Input: [B, C, T] -> Output: [B, latent_dim, T // ratio]
    """

    def __init__(self, gen: nn.Module, latent_dim: int = 64, ratio: int = 4096, use_conv_stft: bool = True):
        super().__init__()
        self.gen = gen
        self.latent_dim = latent_dim
        self.ratio = ratio
        self.c_channels = int(bottleneck_channels)
        self.d_factor = latent_dim // self.c_channels
        self.spec_length = int(spec_length)
        self.sigma_rescale = float(sigma_rescale)
        self.use_conv_stft = use_conv_stft
        if use_conv_stft:
            self.stft = ConvSTFT(hop_size=hop, fac_val=fac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        b, c, t = x.shape
        if self.use_conv_stft:
            repr_encoder = self.stft(x)
        else:
            from codicodec.audio import to_representation_encoder
            repr_encoder = to_representation_encoder(x)

        if repr_encoder.shape[-1] % self.spec_length != 0:
            pad = self.spec_length - (repr_encoder.shape[-1] % self.spec_length)
            repr_encoder = F.pad(repr_encoder, (0, pad))

        # Forward pass through encoder
        latent = self.gen.encoder_forward(repr_encoder, dont_quantize=True)

        # Reshape to desired latent channels: [B, (l d), c] -> [B, l, (d c)]
        z = latent.view(b, -1, self.d_factor, self.c_channels).reshape(b, -1, self.latent_dim)
        z = safe_atanh(z) / self.sigma_rescale

        # NetBender standard format: [B, latent_dim, frames]
        if z.ndim == 4:
            z = z.squeeze(1)
        z = z.permute(0, 2, 1).contiguous()
        return z


class CodiCodecDecoderAdapter(nn.Module):
    """
    Adapts CoDiCodec parallel/autoregressive decoding pipeline to pure PyTorch nn.Module:
    Input: [B, latent_dim, frames] -> Output: [B, C, frames * ratio]
    """

    def __init__(
        self,
        gen: nn.Module,
        latent_dim: int = 64,
        ratio: int = 4096,
        denoising_steps: int = 2,
        mode: str = "parallel",
        use_conv_stft: bool = True,
        deterministic_noise: bool = True,
    ):
        super().__init__()
        self.gen = gen
        self.latent_dim = latent_dim
        self.ratio = ratio
        self.denoising_steps = denoising_steps
        self.mode = mode
        self.c_channels = int(bottleneck_channels)
        self.d_factor = latent_dim // self.c_channels
        self.num_latents = int(num_latents)
        self.spec_length = int(spec_length)
        self.sigma_rescale = float(sigma_rescale)
        self.use_conv_stft = use_conv_stft
        self.deterministic_noise = deterministic_noise
        if use_conv_stft:
            self.istft = ConvISTFT(hop_size=hop, fac_val=fac)
        if deterministic_noise and mode == "parallel":
            # 64 frames = 2 chunks of spec_length (32)
            self.register_buffer("init_noise", torch.randn(1, 4, 1024, 64) * float(sigma_max))
            self.register_buffer("step_noise", torch.randn(1, 4, 1024, 64))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim, frames]
        b, dim, frames = z.shape
        expected_audio_len = frames * self.ratio

        # Permute from NetBender [B, latent_dim, frames] to [B, frames, latent_dim]
        z_perm = z.permute(0, 2, 1).contiguous()
        inv = z_perm * self.sigma_rescale
        inv = torch.tanh(inv)

        # Invert latents2dim -> [B, num_latents_total, bottleneck_channels]
        latents = inv.view(b, -1, self.d_factor, self.c_channels).reshape(b, -1, self.c_channels)

        if self.mode == "parallel":
            # Pad latents to multiple of num_latents * 2 for parallel processing
            pad_needed = (self.num_latents * 2) - (latents.shape[-2] % (self.num_latents * 2))
            if pad_needed != (self.num_latents * 2):
                latents = F.pad(latents, (0, 0, 0, pad_needed))

            if self.deterministic_noise:
                step_size = 1.0 / self.denoising_steps
                latents_ls = torch.split(latents, self.num_latents, dim=-2)
                num_chunks = len(latents_ls)
                features = self.gen.pre_decoder_forward(torch.cat(latents_ls, dim=0))
                sigma = float(sigma_max)
                inp = self.init_noise
                out = inp
                for step in range(self.denoising_steps):
                    inp, num_samples = preprocess_parallel_input(inp, iteration=step, length=self.spec_length, dim=-1)
                    inp_latents, num_samples = preprocess_parallel_input(latents, iteration=step, length=self.num_latents, dim=-2)
                    inp_features = preprocess_parallel_features(features, iteration=step, num_samples=num_chunks, dim=-1)
                    out = self.gen.decoder_forward(inp, inp_latents, inp_features, output="both", sigma_left=sigma, sigma_right=sigma)
                    out = postprocess_parallel_input(out, iteration=step, num_samples=num_samples, length=self.spec_length, dim=-1)
                    sigma = float(get_sigma_continuous(1.0 - (step + 1) * step_size))
                    if step < self.denoising_steps - 1:
                        inp = reverse_step(out, self.step_noise, sigma)
                out_spec = out
            else:
                out_spec = self.gen.decode_single_parallel(
                    latents, denoising_steps=self.denoising_steps
                )
        else:
            out_spec = self.gen.decode_autoregressive(
                latents, denoising_steps=self.denoising_steps
            )

        if self.use_conv_stft:
            wv = self.istft(out_spec)
        else:
            from codicodec.audio import to_waveform
            wv = to_waveform(out_spec)

        # Output shape is [B, channels, time]
        if wv.ndim == 2:
            wv = wv.unsqueeze(0)

        # Truncate any STFT/iSTFT padding to exact buffer length
        if wv.shape[-1] > expected_audio_len:
            wv = wv[..., :expected_audio_len]
        elif wv.shape[-1] < expected_audio_len:
            pad = expected_audio_len - wv.shape[-1]
            wv = F.pad(wv, (0, pad))

        return wv


class CodiCodecTSModule(nn.Module):
    """
    TorchScript-compatible multi-method module for CoDiCodec, exposing
    forward, encode, and decode with registered nn~ metadata buffers.
    """

    def __init__(
        self,
        gen: nn.Module,
        sr: int = 48000,
        latent_dim: int = 64,
        ratio: int = 4096,
        denoising_steps: int = 2,
        mode: str = "parallel",
        use_conv_stft: bool = False,
    ):
        super().__init__()
        self.encoder = CodiCodecEncoderAdapter(gen, latent_dim=latent_dim, ratio=ratio, use_conv_stft=use_conv_stft)
        self.decoder = CodiCodecDecoderAdapter(
            gen, latent_dim=latent_dim, ratio=ratio, denoising_steps=denoising_steps, mode=mode, use_conv_stft=use_conv_stft
        )
        self.register_buffer("forward_params", torch.tensor([2, 1, 2, 1], dtype=torch.int64))
        self.register_buffer("encode_params", torch.tensor([2, 1, latent_dim, ratio], dtype=torch.int64))
        self.register_buffer("decode_params", torch.tensor([latent_dim, ratio, 2, 1], dtype=torch.int64))
        self.register_buffer("sr", torch.tensor(sr, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class CodiCodecWrapper(AudioCodecWrapper):
    """
    nn~ / NetBender wrapper for Sony CSL Paris CoDiCodec
    (Stereo 48 kHz / 44.1 kHz, 64-dim latents, 4096x ratio).
    """

    def __init__(
        self,
        gen: nn.Module,
        sr: int = 48000,
        latent_dim: int = 64,
        ratio: int = 4096,
        denoising_steps: int = 2,
        mode: str = "parallel",
        use_conv_stft: bool = False,
    ):
        super().__init__(
            encoder=CodiCodecEncoderAdapter(gen, latent_dim=latent_dim, ratio=ratio, use_conv_stft=use_conv_stft),
            decoder=CodiCodecDecoderAdapter(
                gen,
                latent_dim=latent_dim,
                ratio=ratio,
                denoising_steps=denoising_steps,
                mode=mode,
                use_conv_stft=use_conv_stft,
            ),
            sr=sr,
            in_channels=2,
            out_channels=2,
            latent_dim=latent_dim,
            ratio=ratio,
            test_method=False,
        )
        self.gen = gen


def export_codicodec(
    output_path: Optional[str] = None,
    format: Optional[str] = None,
    sr: int = 48000,
    ratio: int = 4096,
    latent_dim: int = 64,
    buffer_size: int = 32768,
    denoising_steps: int = 2,
    mode: str = "parallel",
    promote_for_bending: bool = True,
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Union[str, Tuple[str, ...]]:
    """
    Loads CoDiCodec pretrained checkpoint and exports to TorchScript (.ts),
    Bendable ONNX (.onnx + NBND v2 .bin), and/or ExecuTorch (.pte).
    """
    from codicodec import EncoderDecoder

    if output_path is None:
        base_dir = "models"
        os.makedirs(base_dir, exist_ok=True)
        if format == "onnx":
            output_path = os.path.join(base_dir, "codicodec_bendable.onnx")
        elif format == "pte":
            output_path = os.path.join(base_dir, "codicodec_coreml.pte")
        else:
            output_path = os.path.join(base_dir, "codicodec.ts")

    base_stem, ext = os.path.splitext(output_path)
    target_format = format or ("onnx" if ext.lower() == ".onnx" else ("pte" if ext.lower() == ".pte" else "ts"))

    if verbose:
        logging.info("Initializing CoDiCodec EncoderDecoder...")
    device = torch.device("cpu")
    encdec = EncoderDecoder(device=device)

    exported_files: List[str] = []

    # 1. TorchScript Export
    if target_format in ("ts", "both", "all"):
        ts_path = f"{base_stem}.ts" if not output_path.endswith(".ts") else output_path
        if verbose:
            logging.info(f"Exporting CoDiCodec to TorchScript -> {ts_path}...")
        try:
            ts_mod = CodiCodecTSModule(
                gen=encdec.gen,
                sr=sr,
                latent_dim=latent_dim,
                ratio=ratio,
                denoising_steps=denoising_steps,
                mode=mode,
                use_conv_stft=False,
            ).eval()
            sample_audio = torch.zeros(1, 2, buffer_size, dtype=torch.float32)
            sample_latent = torch.zeros(1, latent_dim, buffer_size // ratio, dtype=torch.float32)

            traced = torch.jit.trace_module(
                ts_mod,
                {
                    "forward": (sample_audio,),
                    "encode": (sample_audio,),
                    "decode": (sample_latent,),
                },
                check_trace=False,
            )
            traced.save(ts_path)
            if verbose:
                logging.info(f"Successfully exported TorchScript: {ts_path}")
            exported_files.append(ts_path)
        except Exception as e:
            logging.error(f"Failed to export TorchScript: {e}")

    # 2. Bendable ONNX Export (using operator-safe ConvSTFT and ConvISTFT)
    if target_format in ("onnx", "both", "all"):
        onnx_path = output_path if ext.lower() == ".onnx" else f"{base_stem}_bendable.onnx"
        base_onnx_stem, _ = os.path.splitext(onnx_path)
        # Strip _bendable if already present
        if base_onnx_stem.endswith("_bendable"):
            base_onnx_stem = base_onnx_stem[:-9]

        if verbose:
            logging.info(f"Exporting CoDiCodec to Bendable ONNX -> {onnx_path}...")

        raw_onnx = f"{base_onnx_stem}_raw.onnx"
        sample_audio = torch.zeros(1, 2, buffer_size, dtype=torch.float32)
        sample_latent = torch.zeros(1, latent_dim, buffer_size // ratio, dtype=torch.float32)

        # Full forward module
        class CodiCodecConvForward(nn.Module):
            def __init__(self, gen):
                super().__init__()
                self.encoder = CodiCodecEncoderAdapter(gen, latent_dim=latent_dim, ratio=ratio, use_conv_stft=True)
                self.decoder = CodiCodecDecoderAdapter(
                    gen, latent_dim=latent_dim, ratio=ratio, denoising_steps=denoising_steps, mode=mode, use_conv_stft=True
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.decoder(self.encoder(x))

        conv_fwd = CodiCodecConvForward(encdec.gen).eval()

        try:
            torch.onnx.export(
                conv_fwd,
                (sample_audio,),
                raw_onnx,
                input_names=["audio_in"],
                output_names=["audio_out"],
                dynamic_axes={
                    "audio_in": {0: "batch", 2: "samples"},
                    "audio_out": {0: "batch", 2: "samples"},
                },
                opset_version=17,
                dynamo=False,
            )

            metadata = {
                "netbender.architecture": "codicodec",
                "netbender.sample_rate": str(sr),
                "netbender.ratio": str(ratio),
                "netbender.latent_dim": str(latent_dim),
                "netbender.in_channels": "2",
                "netbender.out_channels": "2",
                "netbender.history_samples": "0",
                "netbender.output_offset": "0",
                "netbender.latent_scale": "1.0",
                "netbender.decode_latent_steps": str(buffer_size // ratio),
            }

            if promote_for_bending:
                if verbose:
                    logging.info(f"Promoting initializers to dynamic inputs for real-time weight bending -> {onnx_path}...")
                promote_initializers_to_inputs(
                    input_path=raw_onnx,
                    output_path=onnx_path,
                    keep_initializers_as_defaults=True,
                    verbose=verbose,
                    extra_metadata=metadata,
                )
                if os.path.exists(raw_onnx) and raw_onnx != onnx_path:
                    os.remove(raw_onnx)
            else:
                if raw_onnx != onnx_path:
                    os.replace(raw_onnx, onnx_path)

            if verbose:
                logging.info(f"Successfully exported Bendable ONNX: {onnx_path}")
            exported_files.append(onnx_path)

            # Companion encode & decode subgraphs
            if methods is None or "all" in methods or "encode" in methods:
                enc_path = f"{base_onnx_stem}_bendable_encode.onnx"
                if verbose:
                    logging.info(f"Exporting companion method 'encode' -> {enc_path}...")
                class CodiCodecConvEncode(nn.Module):
                    def __init__(self, gen):
                        super().__init__()
                        self.encoder = CodiCodecEncoderAdapter(gen, latent_dim=latent_dim, ratio=ratio, use_conv_stft=True)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.encoder(x)

                enc_mod = CodiCodecConvEncode(encdec.gen).eval()
                torch.onnx.export(
                    enc_mod,
                    (sample_audio,),
                    enc_path,
                    input_names=["audio_in"],
                    output_names=["latent_out"],
                    dynamic_axes={"audio_in": {0: "batch", 2: "samples"}, "latent_out": {0: "batch", 2: "frames"}},
                    opset_version=17,
                    dynamo=False,
                )
                exported_files.append(enc_path)

            if methods is None or "all" in methods or "decode" in methods:
                dec_path = f"{base_onnx_stem}_bendable_decode.onnx"
                if verbose:
                    logging.info(f"Exporting companion method 'decode' -> {dec_path}...")
                class CodiCodecConvDecode(nn.Module):
                    def __init__(self, gen):
                        super().__init__()
                        self.decoder = CodiCodecDecoderAdapter(
                            gen, latent_dim=latent_dim, ratio=ratio, denoising_steps=denoising_steps, mode=mode, use_conv_stft=True
                        )

                    def forward(self, z: torch.Tensor) -> torch.Tensor:
                        return self.decoder(z)

                dec_mod = CodiCodecConvDecode(encdec.gen).eval()
                torch.onnx.export(
                    dec_mod,
                    (sample_latent,),
                    dec_path,
                    input_names=["latent_in"],
                    output_names=["audio_out"],
                    dynamic_axes={"latent_in": {0: "batch", 2: "frames"}, "audio_out": {0: "batch", 2: "samples"}},
                    opset_version=17,
                    dynamo=False,
                )
                exported_files.append(dec_path)

        except Exception as oe:
            logging.error(f"Failed to export ONNX: {oe}")
            if os.path.exists(raw_onnx):
                os.remove(raw_onnx)

    # 3. ExecuTorch Export (.pte)
    if target_format in ("pte", "all"):
        pte_path = output_path if ext.lower() == ".pte" else f"{base_stem}_portable.pte"
        base_pte_stem, _ = os.path.splitext(pte_path)
        if base_pte_stem.endswith("_portable"):
            base_pte_stem = base_pte_stem[:-9]

        if verbose:
            logging.info(f"Exporting CoDiCodec to ExecuTorch -> {pte_path}...")

        try:
            from executorch.exir import to_edge

            class CodiCodecPteForward(nn.Module):
                def __init__(self, gen):
                    super().__init__()
                    self.encoder = CodiCodecEncoderAdapter(gen, latent_dim=latent_dim, ratio=ratio, use_conv_stft=True)
                    self.decoder = CodiCodecDecoderAdapter(
                        gen, latent_dim=latent_dim, ratio=ratio, denoising_steps=denoising_steps, mode=mode, use_conv_stft=True, deterministic_noise=True
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.decoder(self.encoder(x))

            pte_fwd = CodiCodecPteForward(encdec.gen).eval()
            sample_audio = torch.zeros(1, 2, buffer_size, dtype=torch.float32)
            sample_latent = torch.zeros(1, latent_dim, buffer_size // ratio, dtype=torch.float32)

            with torch.no_grad():
                ep_fwd = torch.export.export(pte_fwd, (sample_audio,), strict=False)
            et_prog = to_edge(ep_fwd).to_executorch()
            with open(pte_path, "wb") as pf:
                et_prog.write_to_file(pf)
            if verbose:
                logging.info(f"Successfully exported ExecuTorch: {pte_path}")
            exported_files.append(pte_path)

            if methods is None or "all" in methods or "encode" in methods:
                enc_pte_path = f"{base_pte_stem}_portable_encode.pte"
                if verbose:
                    logging.info(f"Exporting ExecuTorch companion 'encode' -> {enc_pte_path}...")
                enc_mod = CodiCodecEncoderAdapter(encdec.gen, latent_dim=latent_dim, ratio=ratio, use_conv_stft=True).eval()
                with torch.no_grad():
                    ep_enc = torch.export.export(enc_mod, (sample_audio,), strict=False)
                et_enc = to_edge(ep_enc).to_executorch()
                with open(enc_pte_path, "wb") as pf:
                    et_enc.write_to_file(pf)
                exported_files.append(enc_pte_path)

            if methods is None or "all" in methods or "decode" in methods:
                dec_pte_path = f"{base_pte_stem}_portable_decode.pte"
                if verbose:
                    logging.info(f"Exporting ExecuTorch companion 'decode' -> {dec_pte_path}...")
                dec_mod = CodiCodecDecoderAdapter(
                    encdec.gen, latent_dim=latent_dim, ratio=ratio, denoising_steps=denoising_steps, mode=mode, use_conv_stft=True, deterministic_noise=True
                ).eval()
                with torch.no_grad():
                    ep_dec = torch.export.export(dec_mod, (sample_latent,), strict=False)
                et_dec = to_edge(ep_dec).to_executorch()
                with open(dec_pte_path, "wb") as pf:
                    et_dec.write_to_file(pf)
                exported_files.append(dec_pte_path)

        except Exception as pe:
            logging.error(f"Failed to export ExecuTorch: {pe}")

    if len(exported_files) == 1:
        return exported_files[0]
    return tuple(exported_files)


def main():
    parser = argparse.ArgumentParser(description="Export CoDiCodec to TorchScript, ONNX, and ExecuTorch for NetBender.")
    parser.add_argument("-o", "--output", default=None, help="Output file path (.ts, .onnx, or .pte)")
    parser.add_argument("--format", choices=["ts", "onnx", "pte", "both", "all"], default=None, help="Target format")
    parser.add_argument("--sr", type=int, default=48000, help="Audio sample rate (default: 48000)")
    parser.add_argument("--ratio", type=int, default=4096, help="Temporal downsampling ratio (default: 4096)")
    parser.add_argument("--latent-dim", type=int, default=64, help="Continuous latent channels (default: 64)")
    parser.add_argument("--buffer-size", type=int, default=32768, help="Buffer size in samples (default: 32768)")
    parser.add_argument("--denoising-steps", type=int, default=2, help="Decoder diffusion denoising steps (default: 2)")
    parser.add_argument("--mode", choices=["parallel", "autoregressive"], default="parallel", help="Decoding mode")
    parser.add_argument("--no-bending", action="store_true", help="Do not promote parameters to graph inputs for bending")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    export_codicodec(
        output_path=args.output,
        format=args.format,
        sr=args.sr,
        ratio=args.ratio,
        latent_dim=args.latent_dim,
        buffer_size=args.buffer_size,
        denoising_steps=args.denoising_steps,
        mode=args.mode,
        promote_for_bending=not args.no_bending,
    )


if __name__ == "__main__":
    main()
