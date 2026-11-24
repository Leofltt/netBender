"""
sa3_sampler_exporter.py - Exporter & numerical parity verifier for Stable Audio 3 Sampler bundles.

Exports the 4-component SA3 bundle (text_encoder, number_cond, dit, same_s) to ONNX
with NBND v2 bendable sidecars, copies tokenizer.json, and writes `<stem>.nbsampler.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python_tools.onnx_bend_prep import promote_initializers_to_inputs
from python_tools.pte_bend_prep import (
    export_bendable_pte,
    export_multi_input_bendable_pte,
)
from python_tools.sa3_bundle import (
    build_bundle_manifest,
    extract_diffusion_config_from_model_config,
    load_bundle_manifest,
    read_nbnd_v2_sidecar,
    resolve_component_path,
    save_bundle_manifest,
    validate_bundle_manifest,
)
from python_tools.sa3_exportable import (
    ExportableDiTStep,
    ExportableNumberConditioner,
    ExportableTextEncoder,
    PretransformDecodeAdapter,
    PretransformEncodeAdapter,
    PretransformForwardAdapter,
)

CONDITIONING_HEAD_KEYWORDS = ["proj_out", "to_cond_embed", "to_timestep_embed"]
XNNPACK_SAFE_SA3_FILTER = ["norm", "scale", "alpha", "beta", "gamma", "to_scale_shift_gate", "dyt"]
COREML_MAX_PROMOTED_LAYERS = 240


def _export_onnx_with_promotion(
    module: nn.Module,
    dummy_inputs: tuple,
    output_onnx_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    bend_mode: str,
    layer_filter: Optional[List[str]],
    extra_keywords: Optional[List[str]],
    extra_metadata: Dict[str, Any],
    verbose: bool = True,
) -> Optional[str]:
    """Exports `module` to ONNX opset 17 and promotes initializers + writes NBND v2 `.bin` sidecar."""
    raw_path = output_onnx_path + ".raw.onnx"
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    module.eval()

    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            raw_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )

    promote_initializers_to_inputs(
        input_path=raw_path,
        output_path=output_onnx_path,
        layer_filter=layer_filter,
        keep_initializers_as_defaults=True,
        verbose=verbose,
        extra_metadata={k: str(v) for k, v in extra_metadata.items()},
        write_sidecar=True,
        bend_mode=bend_mode,
        extra_keywords=extra_keywords,
    )

    if os.path.exists(raw_path):
        os.remove(raw_path)

    sidecar_path = os.path.splitext(output_onnx_path)[0] + ".bin"
    return sidecar_path if os.path.exists(sidecar_path) else None


def _ensure_writable(path: str) -> None:
    """Makes an existing file at `path` writable (best-effort chmod)."""
    if not os.path.exists(path):
        return
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


def _copy_tokenizer_json(src: str, dest: str) -> None:
    """Copies `src` to `dest` without preserving read-only permission bits."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    _ensure_writable(dest)
    shutil.copyfile(src, dest)
    try:
        os.chmod(dest, 0o644)
    except OSError:
        pass


def _save_tokenizer_json(
    conditioner: Any,
    tokenizer_json_path: Optional[str],
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Saves/copies tokenizer.json into `output_dir` and returns tokenizer metadata fields."""
    dest_tok = os.path.join(output_dir, "tokenizer.json")
    info: Dict[str, Any] = {
        "file": "tokenizer.json",
        "max_length": int(getattr(conditioner, "max_length", 128)),
        "pad_token_id": 0,
        "eos_token_id": 1,
        "add_eos": True,
    }

    tok = getattr(conditioner, "tokenizer", None)
    if tok is not None:
        if getattr(tok, "pad_token_id", None) is not None:
            info["pad_token_id"] = int(tok.pad_token_id)
        if getattr(tok, "eos_token_id", None) is not None:
            info["eos_token_id"] = int(tok.eos_token_id)

    if tokenizer_json_path and os.path.isfile(tokenizer_json_path):
        if os.path.abspath(tokenizer_json_path) != os.path.abspath(dest_tok):
            _copy_tokenizer_json(tokenizer_json_path, dest_tok)
        if verbose:
            logging.info(f"Copied tokenizer.json -> {dest_tok}")
        return info

    if tok is not None:
        backend_tok = getattr(tok, "backend_tokenizer", None)
        if backend_tok is not None and hasattr(backend_tok, "save"):
            _ensure_writable(dest_tok)
            backend_tok.save(dest_tok)
            if verbose:
                logging.info(f"Saved backend tokenizer.json -> {dest_tok}")
            return info
        if hasattr(tok, "save_pretrained"):
            tok.save_pretrained(output_dir)
            _ensure_writable(dest_tok)
            if os.path.isfile(dest_tok):
                if verbose:
                    logging.info(f"Saved pretrained tokenizer.json -> {dest_tok}")
                return info

    # Fallback placeholder if caller didn't supply tokenizer_json_path (e.g. minimal offline test)
    if not os.path.isfile(dest_tok):
        _ensure_writable(dest_tok)
        with open(dest_tok, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "model": {"type": "Unigram", "vocab": []}}, f)
    return info


def export_sa3_sampler(
    *,
    variant: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    model_instance: Any = None,
    model_config: Optional[Dict[str, Any]] = None,
    tokenizer_json_path: Optional[str] = None,
    t5_model_path: Optional[str] = None,
    output_dir: str = "models/sa3_sampler",
    stem: Optional[str] = None,
    format: str = "onnx",
    delegate: str = "coreml",
    compute_precision: str = "float16",
    bend_mode: str = "none",
    promote_conditioning_heads: bool = True,
    layer_filter: Optional[List[str]] = None,
    max_duration_sec: float = 30.0,
    dummy_latent_frames: int = 8,
    verify: bool = False,
    verbose: bool = True,
):
    """
    Exports a Stable Audio 3 model to a `.nbsampler.json` multi-component bundle
    for ExecuTorch (`.pte`: CoreML / XNNPACK / portable), ONNX (`.onnx`), or both.
    Returns the `.nbsampler.json` path (or `(pte_manifest, onnx_manifest)` when `format='both'`).
    """
    target_format = format.lower().strip()
    if target_format not in ("onnx", "pte", "both"):
        raise ValueError(f"Unsupported format {format!r}. Choose from: pte, onnx, both")

    if model_instance is None:
        if checkpoint_path and config_path:
            from stable_audio_3.loading_utils import load_diffusion_cond

            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)

            resolved_t5_dir = t5_model_path
            if not resolved_t5_dir and tokenizer_json_path:
                cand_dir = os.path.dirname(os.path.abspath(tokenizer_json_path))
                if os.path.isfile(os.path.join(cand_dir, "config.json")) and os.path.isfile(
                    os.path.join(cand_dir, "model.safetensors")
                ):
                    resolved_t5_dir = cand_dir
            if resolved_t5_dir:
                for c_info in model_config.get("model", {}).get("conditioning", {}).get("configs", []):
                    if c_info.get("type") == "t5gemma":
                        c_cfg = c_info.setdefault("config", {})
                        c_cfg["model_path"] = resolved_t5_dir
                        c_cfg.pop("repo_id", None)
                        c_cfg.pop("subfolder", None)

            model_instance = load_diffusion_cond(
                model_config, checkpoint_path, device="cpu", model_half=False
            )
            variant = variant or Path(checkpoint_path).stem
        elif variant:
            from stable_audio_3.loading_utils import load_diffusion_cond
            from stable_audio_3.model_configs import models as sa3_models

            if variant not in sa3_models:
                raise ValueError(
                    f"Unknown SA3 variant '{variant}'. Valid: {list(sa3_models.keys())}"
                )
            cfg_obj = sa3_models[variant]
            local_cfg, local_ckpt = cfg_obj.resolve()
            with open(local_cfg, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            model_instance = load_diffusion_cond(
                model_config, local_ckpt, device="cpu", model_half=False
            )
        else:
            raise ValueError(
                "Provide either model_instance, (checkpoint_path & config_path), or variant."
            )

    model_instance.eval().requires_grad_(False).cpu()
    if model_config is None:
        model_config = {}

    extracted = extract_diffusion_config_from_model_config(model_config)
    variant_name = variant or "custom"
    bundle_stem = stem or f"sa3_{variant_name.replace('-', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Extract components from ConditionedDiffusionModelWrapper
    conds = model_instance.conditioner.conditioners
    prompt_cond = conds["prompt"]
    seconds_cond = conds["seconds_total"]
    dit_module = model_instance.model
    pretransform = model_instance.pretransform

    # Infer exact runtime dimensions from actual modules when model_config is minimal
    sample_rate = extracted["sample_rate"]
    io_channels = int(getattr(pretransform, "io_channels", extracted["io_channels"]))
    latent_dim = int(getattr(model_instance, "io_channels", extracted["latent_dim"]))
    ratio = int(getattr(pretransform, "downsampling_ratio", extracted["downsampling_ratio"]))
    latent_align = extracted["latent_align"]
    max_tok_len = int(getattr(prompt_cond, "max_length", extracted["tokenizer"]["max_length"]))

    dit_core = dit_module.model if hasattr(dit_module, "model") and not hasattr(dit_module, "_forward") else dit_module
    cond_dim = int(getattr(dit_core, "cond_token_dim", extracted["cond_dim"]) or extracted["cond_dim"])
    global_cond_dim = int(
        getattr(prompt_cond, "output_dim", extracted["global_cond_dim"])
        if hasattr(dit_core, "to_global_embed")
        else 0
    )
    input_concat_dim = int(getattr(dit_core, "input_concat_dim", 0) or extracted["input_concat_dim"])

    # Shared dummy inputs
    cross_ids = extracted["diffusion"]["cross_attn_cond_ids"]
    cond_seq_len = max_tok_len + (1 if "seconds_total" in cross_ids else 0)
    dummy_ids = torch.zeros(1, max_tok_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_tok_len, dtype=torch.long)
    dummy_sec = torch.tensor([6.0], dtype=torch.float32)
    dummy_x = torch.zeros(1, latent_dim, dummy_latent_frames, dtype=torch.float32)
    dummy_t = torch.tensor([0.5], dtype=torch.float32)
    dummy_ca = torch.zeros(1, cond_seq_len, cond_dim, dtype=torch.float32)
    dummy_gc = torch.zeros(1, max(1, global_cond_dim), dtype=torch.float32)
    dummy_ic = torch.zeros(1, max(1, input_concat_dim), dummy_latent_frames, dtype=torch.float32)
    dummy_pm = torch.ones(1, dummy_latent_frames, dtype=torch.long)

    dummy_samples = dummy_latent_frames * ratio
    dummy_audio = torch.zeros(1, io_channels, dummy_samples, dtype=torch.float32)
    dummy_latent = torch.zeros(1, latent_dim, dummy_latent_frames, dtype=torch.float32)

    ae_meta = {
        "netbender.subsystem": "ae",
        "netbender.architecture": "same_s",
        "netbender.sample_rate": str(sample_rate),
        "netbender.ratio": str(ratio),
        "netbender.latent_dim": str(latent_dim),
        "netbender.in_channels": str(io_channels),
        "netbender.out_channels": str(io_channels),
        "netbender.history_samples": str(ratio * 4),
        "netbender.output_offset": str(ratio * 2),
        "netbender.latent_scale": "1.0",
        "netbender.decode_latent_steps": "6",
    }

    tok_meta = _save_tokenizer_json(prompt_cond, tokenizer_json_path, output_dir, verbose=verbose)
    tok_meta["max_length"] = max_tok_len

    pte_manifest_path: Optional[str] = None
    onnx_manifest_path: Optional[str] = None

    if target_format in ("pte", "both"):
        del_lower = delegate.lower().strip()
        eff_filter = layer_filter or (
            XNNPACK_SAFE_SA3_FILTER if (del_lower == "xnnpack" and bend_mode != "none") else None
        )
        max_layers = COREML_MAX_PROMOTED_LAYERS if del_lower == "coreml" else None
        pte_stem = f"{bundle_stem}_{del_lower}"

        te_wrap_pte = ExportableTextEncoder(prompt_cond).eval()
        te_pte_file = f"{pte_stem}_text_encoder.pte"
        te_pte_path = os.path.join(output_dir, te_pte_file)
        export_multi_input_bendable_pte(
            te_wrap_pte,
            (dummy_ids, dummy_mask),
            te_pte_path,
            delegate=del_lower,
            bend_mode=bend_mode,
            layer_filter=eff_filter,
            max_promoted_layers=max_layers,
            extra_metadata={"netbender.subsystem": "prompt", "netbender.component": "text_encoder"},
            write_sidecar=True,
            compute_precision=compute_precision,
            verbose=verbose,
        )

        nc_wrap_pte = ExportableNumberConditioner(seconds_cond).eval()
        nc_pte_file = f"{pte_stem}_number_cond.pte"
        nc_pte_path = os.path.join(output_dir, nc_pte_file)
        export_multi_input_bendable_pte(
            nc_wrap_pte,
            (dummy_sec,),
            nc_pte_path,
            delegate=del_lower,
            bend_mode=bend_mode,
            layer_filter=eff_filter,
            max_promoted_layers=max_layers,
            extra_metadata={"netbender.subsystem": "prompt", "netbender.component": "number_cond"},
            write_sidecar=True,
            compute_precision=compute_precision,
            verbose=verbose,
        )

        dit_wrap_pte = ExportableDiTStep(dit_module).eval()
        dit_pte_file = f"{pte_stem}_dit.pte"
        dit_pte_path = os.path.join(output_dir, dit_pte_file)
        export_multi_input_bendable_pte(
            dit_wrap_pte,
            (dummy_x, dummy_t, dummy_ca, dummy_gc, dummy_ic, dummy_pm),
            dit_pte_path,
            delegate=del_lower,
            bend_mode=bend_mode,
            layer_filter=eff_filter,
            max_promoted_layers=max_layers,
            extra_metadata={"netbender.subsystem": "dit", "netbender.component": "dit"},
            write_sidecar=True,
            compute_precision=compute_precision,
            verbose=verbose,
        )

        ae_for_pte = pretransform
        if checkpoint_path and os.path.isfile(checkpoint_path) and latent_dim == 256 and ratio == 4096:
            try:
                from python_tools.same_s import load_same_s_from_safetensors
                ae_for_pte = load_same_s_from_safetensors(checkpoint_path, device="cpu")
                if verbose:
                    logging.info("Using pure-tensor SAMESCore for ExecuTorch autoencoder export.")
            except Exception as exc:
                if verbose:
                    logging.warning(f"Falling back to pretransform for AE export: {exc}")

        ae_fwd_pte = PretransformForwardAdapter(ae_for_pte).eval()
        ae_enc_pte = PretransformEncodeAdapter(ae_for_pte).eval()
        ae_dec_pte = PretransformDecodeAdapter(ae_for_pte).eval()
        ae_pte_file = f"{pte_stem}_same_s.pte"
        ae_pte_path = os.path.join(output_dir, ae_pte_file)
        export_bendable_pte(
            module=ae_fwd_pte,
            sample_audio_input=dummy_audio,
            output_pte_path=ae_pte_path,
            delegate=del_lower,
            bend_mode=bend_mode,
            layer_filter=eff_filter,
            companion_modules={"encode": (ae_enc_pte, dummy_audio), "decode": (ae_dec_pte, dummy_latent)},
            extra_metadata=ae_meta,
            write_sidecar=True,
            compute_precision=compute_precision,
            verbose=verbose,
        )

        ae_base_stem = os.path.splitext(ae_pte_file)[0]
        comp_enc_file = f"{ae_base_stem}_encode.pte"
        comp_dec_file = f"{ae_base_stem}_decode.pte"
        ae_comp_entry: Dict[str, str] = {
            "model": ae_pte_file,
            "sidecar": f"{ae_base_stem}.bin",
            "encode_model": comp_enc_file if os.path.isfile(os.path.join(output_dir, comp_enc_file)) else ae_pte_file,
            "decode_model": comp_dec_file if os.path.isfile(os.path.join(output_dir, comp_dec_file)) else ae_pte_file,
        }

        pte_components = {
            "text_encoder": {"model": te_pte_file, "sidecar": f"{os.path.splitext(te_pte_file)[0]}.bin"},
            "number_cond": {"model": nc_pte_file, "sidecar": f"{os.path.splitext(nc_pte_file)[0]}.bin"},
            "dit": {"model": dit_pte_file, "sidecar": f"{os.path.splitext(dit_pte_file)[0]}.bin"},
            "autoencoder": ae_comp_entry,
        }

        pte_manifest = build_bundle_manifest(
            variant=variant_name,
            runtime="executorch",
            delegate=del_lower,
            compiled_latent_frames=dummy_latent_frames,
            sample_rate=sample_rate,
            io_channels=io_channels,
            latent_dim=latent_dim,
            downsampling_ratio=ratio,
            latent_align=latent_align,
            cond_dim=cond_dim,
            global_cond_dim=global_cond_dim,
            input_concat_dim=input_concat_dim,
            max_duration_sec=max_duration_sec,
            duration_padding_sec=6.0,
            diffusion=extracted["diffusion"],
            tokenizer=tok_meta,
            components=pte_components,
        )

        pte_manifest_name = f"{pte_stem}.nbsampler.json" if target_format == "both" else f"{bundle_stem}.nbsampler.json"
        pte_manifest_path = os.path.join(output_dir, pte_manifest_name)
        save_bundle_manifest(pte_manifest, pte_manifest_path)
        pte_errors = validate_bundle_manifest(pte_manifest, bundle_dir=output_dir, check_files=True)
        if pte_errors:
            raise RuntimeError(f"ExecuTorch bundle validation failed: {pte_errors}")
        if verbose:
            logging.info(f"Exported SA3 ExecuTorch ({del_lower}) sampler bundle -> {pte_manifest_path}")

    if target_format in ("onnx", "both"):
        extra_kw = CONDITIONING_HEAD_KEYWORDS if (bend_mode == "delegate" and promote_conditioning_heads) else None

        te_wrap = ExportableTextEncoder(prompt_cond).eval()
        te_file = f"{bundle_stem}_text_encoder.onnx"
        te_path = os.path.join(output_dir, te_file)
        te_sidecar = _export_onnx_with_promotion(
            te_wrap,
            (dummy_ids, dummy_mask),
            te_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["prompt_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "prompt_embeds": {0: "batch"},
            },
            bend_mode=bend_mode,
            layer_filter=layer_filter,
            extra_keywords=extra_kw,
            extra_metadata={"netbender.subsystem": "prompt", "netbender.component": "text_encoder"},
            verbose=verbose,
        )

        nc_wrap = ExportableNumberConditioner(seconds_cond).eval()
        nc_file = f"{bundle_stem}_number_cond.onnx"
        nc_path = os.path.join(output_dir, nc_file)
        nc_sidecar = _export_onnx_with_promotion(
            nc_wrap,
            (dummy_sec,),
            nc_path,
            input_names=["seconds_total"],
            output_names=["time_embed"],
            dynamic_axes={"seconds_total": {0: "batch"}, "time_embed": {0: "batch"}},
            bend_mode=bend_mode,
            layer_filter=layer_filter,
            extra_keywords=extra_kw,
            extra_metadata={"netbender.subsystem": "prompt", "netbender.component": "number_cond"},
            verbose=verbose,
        )

        dit_wrap = ExportableDiTStep(dit_module).eval()
        dit_file = f"{bundle_stem}_dit.onnx"
        dit_path = os.path.join(output_dir, dit_file)
        dit_sidecar = _export_onnx_with_promotion(
            dit_wrap,
            (dummy_x, dummy_t, dummy_ca, dummy_gc, dummy_ic, dummy_pm),
            dit_path,
            input_names=[
                "x",
                "t",
                "cross_attn_cond",
                "global_cond",
                "input_concat_cond",
                "padding_mask",
            ],
            output_names=["v_pred"],
            dynamic_axes={
                "x": {0: "batch", 2: "frames"},
                "t": {0: "batch"},
                "cross_attn_cond": {0: "batch", 1: "cond_seq"},
                "global_cond": {0: "batch"},
                "input_concat_cond": {0: "batch", 2: "frames"},
                "padding_mask": {0: "batch", 1: "frames"},
                "v_pred": {0: "batch", 2: "frames"},
            },
            bend_mode=bend_mode,
            layer_filter=layer_filter,
            extra_keywords=extra_kw,
            extra_metadata={"netbender.subsystem": "dit", "netbender.component": "dit"},
            verbose=verbose,
        )

        ae_file = f"{bundle_stem}_same_s.onnx"
        ae_enc_file = f"{bundle_stem}_same_s_encode.onnx"
        ae_dec_file = f"{bundle_stem}_same_s_decode.onnx"
        ae_path = os.path.join(output_dir, ae_file)
        ae_enc_path = os.path.join(output_dir, ae_enc_file)
        ae_dec_path = os.path.join(output_dir, ae_dec_file)

        ae_fwd = PretransformForwardAdapter(pretransform).eval()
        ae_sidecar = _export_onnx_with_promotion(
            ae_fwd,
            (dummy_audio,),
            ae_path,
            input_names=["audio_in"],
            output_names=["audio_out"],
            dynamic_axes={"audio_in": {0: "batch", 2: "samples"}, "audio_out": {0: "batch", 2: "samples"}},
            bend_mode=bend_mode,
            layer_filter=layer_filter,
            extra_keywords=None,
            extra_metadata=ae_meta,
            verbose=verbose,
        )

        ae_enc = PretransformEncodeAdapter(pretransform).eval()
        with torch.no_grad():
            torch.onnx.export(
                ae_enc,
                (dummy_audio,),
                ae_enc_path,
                input_names=["audio_in"],
                output_names=["latent_out"],
                dynamic_axes={"audio_in": {0: "batch", 2: "samples"}, "latent_out": {0: "batch", 2: "frames"}},
                opset_version=17,
                dynamo=False,
            )

        ae_dec = PretransformDecodeAdapter(pretransform).eval()
        with torch.no_grad():
            torch.onnx.export(
                ae_dec,
                (dummy_latent,),
                ae_dec_path,
                input_names=["latent_in"],
                output_names=["audio_out"],
                dynamic_axes={"latent_in": {0: "batch", 2: "frames"}, "audio_out": {0: "batch", 2: "samples"}},
                opset_version=17,
                dynamo=False,
            )

        onnx_components = {
            "text_encoder": {
                "model": te_file,
                "sidecar": os.path.basename(te_sidecar) if te_sidecar else "",
            },
            "number_cond": {
                "model": nc_file,
                "sidecar": os.path.basename(nc_sidecar) if nc_sidecar else "",
            },
            "dit": {
                "model": dit_file,
                "sidecar": os.path.basename(dit_sidecar) if dit_sidecar else "",
            },
            "autoencoder": {
                "model": ae_file,
                "sidecar": os.path.basename(ae_sidecar) if ae_sidecar else "",
                "encode_model": ae_enc_file,
                "decode_model": ae_dec_file,
            },
        }

        onnx_manifest = build_bundle_manifest(
            variant=variant_name,
            runtime="onnx",
            sample_rate=sample_rate,
            io_channels=io_channels,
            latent_dim=latent_dim,
            downsampling_ratio=ratio,
            latent_align=latent_align,
            cond_dim=cond_dim,
            global_cond_dim=global_cond_dim,
            input_concat_dim=input_concat_dim,
            max_duration_sec=max_duration_sec,
            duration_padding_sec=6.0,
            diffusion=extracted["diffusion"],
            tokenizer=tok_meta,
            components=onnx_components,
        )

        onnx_manifest_name = f"{bundle_stem}_onnx.nbsampler.json" if target_format == "both" else f"{bundle_stem}.nbsampler.json"
        onnx_manifest_path = os.path.join(output_dir, onnx_manifest_name)
        save_bundle_manifest(onnx_manifest, onnx_manifest_path)
        errors = validate_bundle_manifest(onnx_manifest, bundle_dir=output_dir, check_files=True)
        if errors:
            raise RuntimeError(f"ONNX bundle validation failed: {errors}")

        if verbose:
            logging.info(f"Exported SA3 ONNX sampler bundle -> {onnx_manifest_path}")

        if verify:
            metrics = verify_sa3_onnx_parity(model_instance, onnx_manifest_path, verbose=verbose)
            if verbose:
                logging.info(f"Parity verification metrics: {metrics}")

    if target_format == "both":
        return (pte_manifest_path, onnx_manifest_path)
    return pte_manifest_path if target_format == "pte" else onnx_manifest_path


def _build_schedule_np(
    steps: int,
    dist_shift_cfg: Dict[str, Any],
    effective_seq_len: int,
    sigma_max: float = 1.0,
) -> np.ndarray:
    """NumPy equivalent of `stable_audio_3.inference.sampling.build_schedule`."""
    t = np.linspace(sigma_max, 0.0, steps + 1, dtype=np.float32)
    shift_type = dist_shift_cfg.get("type", "logsnr")
    seq_len = max(1, int(effective_seq_len))

    if shift_type == "logsnr":
        anchor_len = float(dist_shift_cfg.get("anchor_length", 2000))
        anchor_logsnr = float(dist_shift_cfg.get("anchor_logsnr", -6.2))
        rate = float(dist_shift_cfg.get("rate", 1.0))
        logsnr_end = float(dist_shift_cfg.get("logsnr_end", 2.0))
        logsnr_start = anchor_logsnr - rate * math.log2(seq_len / anchor_len)
        logsnr = logsnr_end - t * (logsnr_end - logsnr_start)
        t_out = 1.0 / (1.0 + np.exp(logsnr))
        t_out = np.where(t <= 0.0, 0.0, t_out)
        t_out = np.where(t >= 1.0, 1.0, t_out)
        t = t_out.astype(np.float32)
    elif shift_type == "flux":
        min_l = float(dist_shift_cfg.get("min_length", 256))
        max_l = float(dist_shift_cfg.get("max_length", 4096))
        a_min = float(dist_shift_cfg.get("alpha_min", 1.0))
        a_max = float(dist_shift_cfg.get("alpha_max", 1.0))
        clamped = max(min(float(seq_len), max_l), min_l)
        denom = math.log(max_l) - math.log(min_l)
        frac = (math.log(clamped) - math.log(min_l)) / (denom if denom != 0.0 else 1e-8)
        alpha = math.exp(math.log(max(a_min, 1e-8)) + frac * (math.log(max(a_max, 1e-8)) - math.log(max(a_min, 1e-8))))
        t = (alpha * t / (1.0 + (alpha - 1.0) * t)).astype(np.float32)
    elif shift_type == "full":
        min_l = float(dist_shift_cfg.get("min_length", 256))
        max_l = float(dist_shift_cfg.get("max_length", 4096))
        base_s = float(dist_shift_cfg.get("base_shift", 0.5))
        max_s = float(dist_shift_cfg.get("max_shift", 1.15))
        clamped = min(max(float(seq_len), min_l), max_l)
        mu = -(base_s + (max_s - base_s) * (clamped - min_l) / max(1e-8, (max_l - min_l)))
        t_safe = np.clip(t, 1e-7, 1.0 - 1e-7)
        t_out = 1.0 - math.exp(mu) / (math.exp(mu) + (1.0 / (1.0 - t_safe) - 1.0))
        t_out = np.where(t <= 0.0, 0.0, t_out)
        t_out = np.where(t >= 1.0, 1.0, t_out)
        t = t_out.astype(np.float32)

    t[0] = np.float32(sigma_max)
    return t


def run_onnx_sa3_sampling(
    manifest_path: str,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    duration_sec: float = 2.0,
    steps: Optional[int] = None,
    sampler_type: Optional[str] = None,
    noise: Optional[np.ndarray] = None,
    step_noises: Optional[np.ndarray] = None,
    weight_overrides: Optional[Dict[str, np.ndarray]] = None,
    decode: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Pure NumPy + ONNX Runtime reference execution of an exported `.nbsampler.json` bundle.
    Supports deterministic `noise`/`step_noises` and `weight_overrides` (`prompt.*`, `dit.*`)
    for parity and bending verification.
    """
    import onnxruntime as ort

    manifest = load_bundle_manifest(manifest_path)
    bundle_dir = os.path.dirname(os.path.abspath(manifest_path))
    comps = manifest["components"]
    diff_cfg = manifest["diffusion"]

    te_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["text_encoder"]["model"]), providers=["CPUExecutionProvider"])
    nc_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["number_cond"]["model"]), providers=["CPUExecutionProvider"])
    dit_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["dit"]["model"]), providers=["CPUExecutionProvider"])

    overrides = weight_overrides or {}

    def _run_with_overrides(sess, base_feed: Dict[str, np.ndarray], prefix: str) -> List[np.ndarray]:
        feed = dict(base_feed)
        all_inputs = list(sess.get_inputs()) + list(sess.get_overridable_initializers())
        for inp in all_inputs:
            if inp.name in base_feed:
                continue
            pref_key = f"{prefix}.{inp.name}"
            if pref_key in overrides:
                feed[inp.name] = overrides[pref_key].astype(np.float32)
            elif inp.name in overrides:
                feed[inp.name] = overrides[inp.name].astype(np.float32)
        return sess.run(None, feed)

    prompt_embeds = _run_with_overrides(
        te_sess,
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        },
        "prompt",
    )[0]

    sec_arr = np.full((input_ids.shape[0],), float(duration_sec), dtype=np.float32)
    time_embed = _run_with_overrides(nc_sess, {"seconds_total": sec_arr}, "prompt")[0]

    cross_ids = diff_cfg.get("cross_attn_cond_ids", ["prompt", "seconds_total"])
    ca_parts = [prompt_embeds if cid == "prompt" else time_embed for cid in cross_ids]
    cross_attn_cond = np.concatenate(ca_parts, axis=1).astype(np.float32)
    global_cond = time_embed.squeeze(1).astype(np.float32)

    sr = int(manifest["sample_rate"])
    ratio = int(manifest["downsampling_ratio"])
    align = int(manifest["latent_align"])
    latent_dim = int(manifest["latent_dim"])
    pad_sec = float(manifest.get("duration_padding_sec", 6.0))

    if noise is not None:
        x = noise.astype(np.float32).copy()
        latent_frames = x.shape[-1]
    else:
        raw_samples = int((duration_sec + pad_sec) * sr)
        unit = ratio * align
        aligned_samples = ((raw_samples + unit - 1) // unit) * unit
        latent_frames = aligned_samples // ratio
        x = np.random.randn(input_ids.shape[0], latent_dim, latent_frames).astype(np.float32)

    eff_frames = max(1, math.ceil(int(duration_sec * sr) / ratio))
    headroom_frames = int(pad_sec * sr / ratio)
    valid_len = min(latent_frames, eff_frames + headroom_frames)
    padding_mask = np.zeros((input_ids.shape[0], latent_frames), dtype=np.int64)
    padding_mask[:, :valid_len] = 1

    ic_dim = int(manifest.get("input_concat_dim", latent_dim + 1))
    input_concat = np.zeros((input_ids.shape[0], ic_dim, latent_frames), dtype=np.float32)

    n_steps = int(steps if steps is not None else diff_cfg.get("default_steps", 8))
    samp_type = sampler_type or diff_cfg.get("default_sampler", "pingpong")
    sigmas = _build_schedule_np(n_steps, diff_cfg.get("dist_shift", {}), eff_frames, sigma_max=1.0)

    for i in range(n_steps):
        t_curr = np.full((x.shape[0],), sigmas[i], dtype=np.float32)
        t_next = sigmas[i + 1]
        v = _run_with_overrides(
            dit_sess,
            {
                "x": x,
                "t": t_curr,
                "cross_attn_cond": cross_attn_cond,
                "global_cond": global_cond,
                "input_concat_cond": input_concat,
                "padding_mask": padding_mask,
            },
            "dit",
        )[0]

        if samp_type == "pingpong":
            denoised = x - t_curr[:, None, None] * v
            eps = step_noises[i] if step_noises is not None else np.random.randn(*x.shape).astype(np.float32)
            x = (1.0 - t_next) * denoised + t_next * eps
        else:
            x = x + (t_next - sigmas[i]) * v

    result: Dict[str, np.ndarray] = {
        "prompt_embeds": prompt_embeds,
        "time_embed": time_embed,
        "latents": x,
    }

    if decode:
        dec_model = comps["autoencoder"].get("decode_model") or comps["autoencoder"]["model"]
        dec_sess = ort.InferenceSession(resolve_component_path(bundle_dir, dec_model), providers=["CPUExecutionProvider"])
        audio = dec_sess.run(None, {"latent_in": x.astype(np.float32)})[0]
        result["audio"] = audio

    return result


def verify_sa3_onnx_parity(
    model_instance: Any,
    manifest_path: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Runs component-level, dynamic-shape, multi-step pingpong diffusion, and NBND v2
    weight-bending parity tests comparing exported ONNX bundle against PyTorch.
    """
    import onnxruntime as ort

    manifest = load_bundle_manifest(manifest_path)
    bundle_dir = os.path.dirname(os.path.abspath(manifest_path))
    comps = manifest["components"]

    conds = model_instance.conditioner.conditioners
    te_pt = ExportableTextEncoder(conds["prompt"]).eval()
    nc_pt = ExportableNumberConditioner(conds["seconds_total"]).eval()
    dit_pt = ExportableDiTStep(model_instance.model).eval()
    enc_pt = PretransformEncodeAdapter(model_instance.pretransform).eval()
    dec_pt = PretransformDecodeAdapter(model_instance.pretransform).eval()

    max_tok = int(manifest["tokenizer"]["max_length"])
    latent_dim = int(manifest["latent_dim"])
    io_ch = int(manifest["io_channels"])
    ratio = int(manifest["downsampling_ratio"])
    align = int(manifest["latent_align"])
    frames = max(4, align * 4)

    ids = torch.randint(0, 64, (1, max_tok), dtype=torch.long)
    mask = torch.ones(1, max_tok, dtype=torch.long)
    mask[:, max(1, max_tok // 2):] = 0
    sec = torch.tensor([3.5], dtype=torch.float32)

    te_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["text_encoder"]["model"]), providers=["CPUExecutionProvider"])
    nc_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["number_cond"]["model"]), providers=["CPUExecutionProvider"])
    dit_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["dit"]["model"]), providers=["CPUExecutionProvider"])
    enc_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["autoencoder"]["encode_model"]), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(resolve_component_path(bundle_dir, comps["autoencoder"]["decode_model"]), providers=["CPUExecutionProvider"])

    with torch.no_grad():
        pt_pe = te_pt(ids, mask).numpy()
        ort_pe = te_sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
        te_err = float(np.max(np.abs(pt_pe - ort_pe)))

        pt_te = nc_pt(sec).numpy()
        ort_te = nc_sess.run(None, {"seconds_total": sec.numpy()})[0]
        nc_err = float(np.max(np.abs(pt_te - ort_te)))

        x = torch.randn(1, latent_dim, frames)
        t = torch.tensor([0.6], dtype=torch.float32)
        cross_ids = manifest["diffusion"].get("cross_attn_cond_ids", ["prompt", "seconds_total"])
        ca = torch.cat([torch.from_numpy(pt_pe) if cid == "prompt" else torch.from_numpy(pt_te) for cid in cross_ids], dim=1)
        gc = torch.from_numpy(pt_te).squeeze(1)
        ic = torch.zeros(1, int(manifest.get("input_concat_dim", latent_dim + 1)), frames)
        pm = torch.ones(1, frames, dtype=torch.long)

        pt_v = dit_pt(x, t, ca, gc, ic, pm).numpy()
        ort_v = dit_sess.run(
            None,
            {
                "x": x.numpy(),
                "t": t.numpy(),
                "cross_attn_cond": ca.numpy(),
                "global_cond": gc.numpy(),
                "input_concat_cond": ic.numpy(),
                "padding_mask": pm.numpy(),
            },
        )[0]
        dit_err = float(np.max(np.abs(pt_v - ort_v)))

        # Dynamic batch=2 & frames + 4
        dyn_frames = frames + align * 2
        x_d = torch.randn(2, latent_dim, dyn_frames)
        t_d = torch.tensor([0.7, 0.3], dtype=torch.float32)
        ca_d = ca.repeat(2, 1, 1)
        gc_d = gc.repeat(2, 1)
        ic_d = torch.zeros(2, int(manifest.get("input_concat_dim", latent_dim + 1)), dyn_frames)
        pm_d = torch.ones(2, dyn_frames, dtype=torch.long)
        pm_d[:, -2:] = 0
        pt_vd = dit_pt(x_d, t_d, ca_d, gc_d, ic_d, pm_d).numpy()
        ort_vd = dit_sess.run(
            None,
            {
                "x": x_d.numpy(),
                "t": t_d.numpy(),
                "cross_attn_cond": ca_d.numpy(),
                "global_cond": gc_d.numpy(),
                "input_concat_cond": ic_d.numpy(),
                "padding_mask": pm_d.numpy(),
            },
        )[0]
        dit_dyn_err = float(np.max(np.abs(pt_vd - ort_vd)))

        audio_in = torch.randn(1, io_ch, frames * ratio)
        pt_z = enc_pt(audio_in).numpy()
        ort_z = enc_sess.run(None, {"audio_in": audio_in.numpy()})[0]
        enc_err = float(np.max(np.abs(pt_z - ort_z)))

        pt_rec = dec_pt(torch.from_numpy(pt_z)).numpy()
        ort_rec = dec_sess.run(None, {"latent_in": pt_z})[0]
        dec_err = float(np.max(np.abs(pt_rec - ort_rec)))

    # Multi-step pingpong parity (4 steps with fixed noise)
    n_steps = 4
    rng = np.random.default_rng(12345)
    init_noise = rng.standard_normal((1, latent_dim, frames)).astype(np.float32)
    step_noises = rng.standard_normal((n_steps, 1, latent_dim, frames)).astype(np.float32)
    dur_sec = float(frames * ratio) / float(manifest["sample_rate"])

    onnx_res = run_onnx_sa3_sampling(
        manifest_path,
        ids.numpy(),
        mask.numpy(),
        duration_sec=dur_sec,
        steps=n_steps,
        sampler_type="pingpong",
        noise=init_noise,
        step_noises=step_noises,
        decode=False,
    )

    sigmas = _build_schedule_np(n_steps, manifest["diffusion"].get("dist_shift", {}), frames, 1.0)
    with torch.no_grad():
        x_pt = torch.from_numpy(init_noise.copy())
        pe_pt = te_pt(ids, mask)
        te_val = nc_pt(torch.tensor([dur_sec], dtype=torch.float32))
        ca_pt = torch.cat([pe_pt if cid == "prompt" else te_val for cid in cross_ids], dim=1)
        gc_pt = te_val.squeeze(1)
        ic_pt = torch.zeros(1, int(manifest.get("input_concat_dim", latent_dim + 1)), frames)
        pm_pt = torch.ones(1, frames, dtype=torch.long)
        for i in range(n_steps):
            t_c = torch.full((1,), float(sigmas[i]), dtype=torch.float32)
            t_n = float(sigmas[i + 1])
            v_pt = dit_pt(x_pt, t_c, ca_pt, gc_pt, ic_pt, pm_pt)
            den = x_pt - t_c.view(-1, 1, 1) * v_pt
            x_pt = (1.0 - t_n) * den + t_n * torch.from_numpy(step_noises[i])
        pingpong_err = float(np.max(np.abs(x_pt.numpy() - onnx_res["latents"])))

    # Weight bending + restore verification via NBND v2 sidecars
    prompt_bend_diff = 0.0
    prompt_restore_err = 0.0
    te_sidecar_rel = comps["text_encoder"].get("sidecar", "")
    if te_sidecar_rel:
        _, te_tensors = read_nbnd_v2_sidecar(resolve_component_path(bundle_dir, te_sidecar_rel))
        if te_tensors:
            k_p = next(iter(te_tensors.keys()))
            w0_p = te_tensors[k_p]
            bent = run_onnx_sa3_sampling(
                manifest_path,
                ids.numpy(),
                mask.numpy(),
                duration_sec=dur_sec,
                steps=n_steps,
                sampler_type="pingpong",
                noise=init_noise,
                step_noises=step_noises,
                weight_overrides={f"prompt.{k_p}": w0_p * 2.5 + 0.5},
                decode=False,
            )
            prompt_bend_diff = float(np.max(np.abs(bent["prompt_embeds"] - onnx_res["prompt_embeds"])))
            restored = run_onnx_sa3_sampling(
                manifest_path,
                ids.numpy(),
                mask.numpy(),
                duration_sec=dur_sec,
                steps=n_steps,
                sampler_type="pingpong",
                noise=init_noise,
                step_noises=step_noises,
                weight_overrides={f"prompt.{k_p}": w0_p},
                decode=False,
            )
            prompt_restore_err = float(np.max(np.abs(restored["prompt_embeds"] - onnx_res["prompt_embeds"])))

    dit_bend_diff = 0.0
    dit_restore_err = 0.0
    dit_sidecar_rel = comps["dit"].get("sidecar", "")
    if dit_sidecar_rel:
        _, dit_tensors = read_nbnd_v2_sidecar(resolve_component_path(bundle_dir, dit_sidecar_rel))
        if dit_tensors:
            # Prefer a non-zero norm/scale/bias tensor so scaling visibly changes DiT output
            k_d = next((k for k, v in dit_tensors.items() if float(np.max(np.abs(v))) > 1e-4), next(iter(dit_tensors.keys())))
            w0_d = dit_tensors[k_d]
            bent_d = run_onnx_sa3_sampling(
                manifest_path,
                ids.numpy(),
                mask.numpy(),
                duration_sec=dur_sec,
                steps=n_steps,
                sampler_type="pingpong",
                noise=init_noise,
                step_noises=step_noises,
                weight_overrides={f"dit.{k_d}": w0_d * 3.0 + 0.25},
                decode=False,
            )
            dit_bend_diff = float(np.max(np.abs(bent_d["latents"] - onnx_res["latents"])))
            restored_d = run_onnx_sa3_sampling(
                manifest_path,
                ids.numpy(),
                mask.numpy(),
                duration_sec=dur_sec,
                steps=n_steps,
                sampler_type="pingpong",
                noise=init_noise,
                step_noises=step_noises,
                weight_overrides={f"dit.{k_d}": w0_d},
                decode=False,
            )
            dit_restore_err = float(np.max(np.abs(restored_d["latents"] - onnx_res["latents"])))

    metrics = {
        "text_encoder_max_err": te_err,
        "number_cond_max_err": nc_err,
        "dit_max_err": dit_err,
        "dit_dynamic_max_err": dit_dyn_err,
        "encode_max_err": enc_err,
        "decode_max_err": dec_err,
        "pingpong_latent_max_err": pingpong_err,
        "prompt_bend_diff": prompt_bend_diff,
        "dit_bend_diff": dit_bend_diff,
        "prompt_restore_err": prompt_restore_err,
        "dit_restore_err": dit_restore_err,
    }
    if verbose:
        for k, v in metrics.items():
            logging.info(f"  [Parity] {k}: {v:.3e}")
    return metrics


def export_golden_tokenizer_fixture(
    tokenizer: Any,
    prompts: Sequence[str],
    output_path: str,
    max_length: int = 128,
) -> str:
    """
    Exports golden HuggingFace AutoTokenizer cases (`text -> input_ids, attention_mask, tokens`)
    for Phase 1 C++ `TextTokenizer` parity tests.
    """
    cases = []
    for text in prompts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np",
        )
        ids_list = enc["input_ids"][0].astype(int).tolist()
        mask_list = enc["attention_mask"][0].astype(int).tolist()
        pieces = tokenizer.convert_ids_to_tokens(ids_list) if hasattr(tokenizer, "convert_ids_to_tokens") else []
        cases.append(
            {
                "text": text,
                "input_ids": ids_list,
                "attention_mask": mask_list,
                "tokens": pieces,
            }
        )

    payload = {"max_length": int(max_length), "cases": cases}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stable Audio 3 Sampler bundle for NetBender")
    parser.add_argument("--variant", type=str, default=None, help="SA3 HF variant (e.g. small-music, small-sfx)")
    parser.add_argument("-i", "--checkpoint", type=str, default=None, help="Local .safetensors checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Local model_config.json path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Local tokenizer.json path")
    parser.add_argument("-o", "--output", type=str, default="models/sa3_sampler", help="Output bundle directory")
    parser.add_argument("--stem", type=str, default=None, help="Filename stem inside output directory")
    parser.add_argument("--format", choices=["pte", "onnx", "both"], default="pte", help="Export format (default: pte)")
    parser.add_argument("--delegate", choices=["coreml", "xnnpack", "portable"], default="coreml", help="ExecuTorch delegate (default: coreml)")
    parser.add_argument("--latent-frames", type=int, default=128, help="Compiled latent frame bucket for .pte export (default: 128)")
    parser.add_argument("--bend-mode", choices=["delegate", "full", "none"], default="delegate")
    parser.add_argument("--no-conditioning-heads", action="store_true")
    parser.add_argument("--filter", nargs="+", default=None)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    export_sa3_sampler(
        variant=args.variant,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tokenizer_json_path=args.tokenizer,
        output_dir=args.output,
        stem=args.stem,
        format=args.format,
        delegate=args.delegate,
        bend_mode=args.bend_mode,
        promote_conditioning_heads=not args.no_conditioning_heads,
        layer_filter=args.filter,
        max_duration_sec=args.max_duration,
        dummy_latent_frames=args.latent_frames,
        verify=args.verify,
    )
