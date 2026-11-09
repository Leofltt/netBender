"""
same_s_exporter.py - Exporter for Stable Audio 3 SAME-S Autoencoder.

Exports the SAME-S DiT Transformer Autoencoder (Stereo 44.1 kHz, 256-dim latents, 4096x ratio)
to TorchScript (.ts), Bendable ONNX (.onnx), and ExecuTorch (.pte) formats for NetBender.
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

import torch
import torch.nn as nn

from python_tools.same_s import load_same_s_from_safetensors, SAMESCore
from python_tools.exporters.base_exporter import AudioCodecWrapper, export_torchscript
from python_tools.onnx_bend_prep import promote_initializers_to_inputs


class SameSEncoderAdapter(nn.Module):
    def __init__(self, core: SAMESCore):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.encode(x)


class SameSDecoderAdapter(nn.Module):
    def __init__(self, core: SAMESCore):
        super().__init__()
        self.core = core

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.core.decode(z)


class SameSWrapper(AudioCodecWrapper):
    """nn~ wrapper for Stable Audio 3 SAME-S (Stereo 44.1 kHz, 256-dim latents, 4096x ratio)."""

    def __init__(
        self,
        core: SAMESCore,
        sr: int = 44100,
        latent_dim: int = 256,
        ratio: int = 4096,
    ):
        super().__init__(
            encoder=SameSEncoderAdapter(core),
            decoder=SameSDecoderAdapter(core),
            sr=sr,
            in_channels=2,
            out_channels=2,
            latent_dim=latent_dim,
            ratio=ratio,
            test_method=False,
        )
        self.core = core


def export_same_s(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    format: Optional[str] = None,
    buffer_size: int = 8192,
    promote_for_bending: bool = True,
    methods: Optional[List[str]] = None,
    delegate: str = "xnnpack",
    verbose: bool = True,
) -> Union[str, Tuple[str, ...]]:
    """Exports Stable Audio 3 SAME-S to TorchScript, ONNX, and/or ExecuTorch."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAME-S checkpoint not found at: {checkpoint_path}")

    if output_path is None:
        base_dir = "models"
        os.makedirs(base_dir, exist_ok=True)
        if format == "onnx":
            output_path = os.path.join(base_dir, "stable_audio_3_same_s_bendable.onnx")
        elif format == "pte":
            suffix = f"_{delegate}" if delegate in ("xnnpack", "coreml", "portable") else ""
            output_path = os.path.join(base_dir, f"stable_audio_3_same_s{suffix}.pte")
        else:
            output_path = os.path.join(base_dir, "stable_audio_3_same_s.ts")

    base_stem, ext = os.path.splitext(output_path)
    target_format = format or ("onnx" if ext.lower() == ".onnx" else ("pte" if ext.lower() == ".pte" else "ts"))

    if verbose:
        logging.info(f"Loading SAME-S weights from '{checkpoint_path}'...")
    core = load_same_s_from_safetensors(checkpoint_path, device="cpu")

    ts_path = f"{base_stem}.ts" if target_format in ("ts", "both", "all") else None
    onnx_path = output_path if ext.lower() == ".onnx" else f"{base_stem}_bendable.onnx"
    pte_path = output_path if ext.lower() == ".pte" else f"{base_stem}_{delegate}.pte"

    exported_ts = None
    exported_onnx = None
    exported_pte = None

    # 1. Export TorchScript
    if target_format in ("ts", "both"):
        wrapper = SameSWrapper(core)
        if verbose:
            logging.info(f"Exporting SAME-S to TorchScript: {ts_path}...")
        exported_ts = export_torchscript(wrapper, ts_path, test_buffer_size=buffer_size)
        if verbose:
            logging.info(f"Exported TorchScript successfully: {exported_ts}")

    # 2. Export ONNX
    if target_format in ("onnx", "both"):
        raw_onnx = f"{base_stem}_raw.onnx"
        if verbose:
            logging.info(f"Exporting standard ONNX: {raw_onnx}...")

        class CoreForward(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.core = c

            def forward(self, x):
                return self.core(x)

        mod = CoreForward(core).eval()
        dummy_audio = torch.zeros(1, 2, buffer_size, dtype=torch.float32)

        torch.onnx.export(
            mod,
            (dummy_audio,),
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
            "netbender.architecture": "same_s",
            "netbender.sample_rate": "44100",
            "netbender.ratio": "4096",
            "netbender.latent_dim": "256",
            "netbender.in_channels": "2",
            "netbender.out_channels": "2",
            "netbender.history_samples": "16384",
            "netbender.output_offset": "8192",
            "netbender.latent_scale": "1.0",
            "netbender.decode_latent_steps": "6",
        }

        if promote_for_bending:
            if verbose:
                logging.info(f"Promoting initializers for real-time bending: {onnx_path}...")
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

        exported_onnx = onnx_path
        if verbose:
            logging.info(f"Exported Bendable ONNX successfully: {exported_onnx}")

        # 3. Slicing/Exporting companion encode & decode subgraphs
        if methods is None or "all" in methods or "encode" in methods:
            enc_path = f"{base_stem}_bendable_encode.onnx"
            if verbose:
                logging.info(f"Exporting companion method 'encode' -> {enc_path}...")
            class CoreEncode(nn.Module):
                def __init__(self, c):
                    super().__init__()
                    self.core = c

                def forward(self, x):
                    return self.core.encode(x)

            enc_mod = CoreEncode(core).eval()
            torch.onnx.export(
                enc_mod,
                (dummy_audio,),
                enc_path,
                input_names=["audio_in"],
                output_names=["latent_out"],
                dynamic_axes={
                    "audio_in": {0: "batch", 2: "samples"},
                    "latent_out": {0: "batch", 2: "frames"},
                },
                opset_version=17,
                dynamo=False,
            )

        if methods is None or "all" in methods or "decode" in methods:
            dec_path = f"{base_stem}_bendable_decode.onnx"
            if verbose:
                logging.info(f"Exporting companion method 'decode' -> {dec_path}...")
            class CoreDecode(nn.Module):
                def __init__(self, c):
                    super().__init__()
                    self.core = c

                def forward(self, z):
                    return self.core.decode(z)

            dec_mod = CoreDecode(core).eval()
            dummy_latent = torch.zeros(1, 256, buffer_size // 4096, dtype=torch.float32)
            torch.onnx.export(
                dec_mod,
                (dummy_latent,),
                dec_path,
                input_names=["latent_in"],
                output_names=["audio_out"],
                dynamic_axes={
                    "latent_in": {0: "batch", 2: "frames"},
                    "audio_out": {0: "batch", 2: "samples"},
                },
                opset_version=17,
                dynamo=False,
            )

    # 3. Export ExecuTorch (.pte)
    if target_format in ("pte", "all"):
        from python_tools.pte_bend_prep import export_bendable_pte

        class CoreForwardPTE(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.core = c

            def forward(self, x):
                return self.core(x)

        class CoreEncodePTE(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.core = c

            def forward(self, x):
                return self.core.encode(x)

        class CoreDecodePTE(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.core = c

            def forward(self, z):
                return self.core.decode(z)

        fwd_mod = CoreForwardPTE(core).eval()
        enc_mod = CoreEncodePTE(core).eval()
        dec_mod = CoreDecodePTE(core).eval()

        dummy_audio = torch.zeros(1, 2, buffer_size, dtype=torch.float32)
        dummy_latent = torch.zeros(1, 256, buffer_size // 4096, dtype=torch.float32)

        companion_modules = {
            "encode": (enc_mod, dummy_audio),
            "decode": (dec_mod, dummy_latent),
        }

        pte_meta = {
            "sr": 44100,
            "sample_rate": 44100,
            "higher_ratio": 4096,
            "ratio": 4096,
            "latent_dim": 256,
            "buffer_size": buffer_size,
            "in_channels": 2,
            "out_channels": 2,
            "history_samples": 0,
            "output_offset": 0,
            "latent_scale": 0.25,
            "decode_latent_steps": 2,
        }

        # For XNNPACK, filter out linear projection biases to prevent ConvertToLinearPass crash
        layer_filter = (
            ["norm", "scale", "snake", "alpha", "beta", "gamma"]
            if (delegate.lower().strip() == "xnnpack" and promote_for_bending)
            else None
        )

        bend_mode = "delegate" if promote_for_bending else "none"

        if verbose:
            logging.info(f"Exporting ExecuTorch .pte ({delegate}): {pte_path}...")
        export_bendable_pte(
            module=fwd_mod,
            sample_audio_input=dummy_audio,
            output_pte_path=pte_path,
            delegate=delegate,
            bend_mode=bend_mode,
            layer_filter=layer_filter,
            companion_modules=companion_modules,
            extra_metadata=pte_meta,
            write_sidecar=True,
            verbose=verbose,
        )
        exported_pte = pte_path
        if verbose:
            logging.info(f"Exported ExecuTorch .pte successfully: {exported_pte}")

    if target_format == "all":
        return (exported_ts, exported_onnx, exported_pte)
    if target_format == "both":
        return (exported_ts, exported_onnx)
    if target_format == "pte":
        return exported_pte
    return exported_onnx if target_format == "onnx" else exported_ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stable Audio 3 SAME-S for NetBender")
    parser.add_argument("-i", "--checkpoint", type=str, default="models/checkpoints/stable_audio_3_same_s.safetensors",
                        help="Path to SAME-S .safetensors checkpoint")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path (.ts, .onnx, or .pte)")
    parser.add_argument("--format", type=str, choices=["ts", "onnx", "pte", "both", "all"], default="both",
                        help="Export format (default: both)")
    parser.add_argument("--delegate", type=str, choices=["xnnpack", "coreml", "portable"], default="xnnpack",
                        help="ExecuTorch delegate for .pte export (default: xnnpack)")
    parser.add_argument("--buffer-size", type=int, default=8192, help="Buffer size in samples (multiple of 8192)")
    parser.add_argument("--no-bending", action="store_true", help="Skip promoting initializers for ONNX/PTE bending")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    export_same_s(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        format=args.format,
        delegate=args.delegate,
        buffer_size=args.buffer_size,
        promote_for_bending=not args.no_bending,
    )

