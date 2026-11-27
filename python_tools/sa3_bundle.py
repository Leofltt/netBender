"""
sa3_bundle.py - Manifest specification, validation, and NBND v2 reader for
NetBender Stable Audio 3 Sampler bundles (`.nbsampler.json`).
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any

import numpy as np

BUNDLE_FORMAT = "netbender.sampler"
BUNDLE_VERSION = 1

REQUIRED_COMPONENTS = ("text_encoder", "number_cond", "dit", "autoencoder")


def extract_diffusion_config_from_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """
    Extracts normalized SA3 sampler metadata from a `stable_audio_3` `model_config.json` dict.
    Falls back cleanly to `small-music` / `SAME-S` defaults when optional fields are omitted.
    """
    model_section = model_config.get("model", {})
    diffusion_cfg = model_section.get("diffusion", {})
    dit_cfg = diffusion_cfg.get("config", {})
    pretransform_cfg = model_section.get("pretransform", {}).get("config", {})
    encoder_cfg = pretransform_cfg.get("encoder", {}).get("config", {})
    cond_cfg = model_section.get("conditioning", {})

    sample_rate = int(model_config.get("sample_rate", 44100))
    io_channels = int(pretransform_cfg.get("io_channels", 2))
    latent_dim = int(model_section.get("io_channels", pretransform_cfg.get("latent_dim", 256)))
    downsampling_ratio = int(pretransform_cfg.get("downsampling_ratio", 4096))

    chunk_size = int(encoder_cfg.get("chunk_size", 32))
    strides = encoder_cfg.get("strides", [16])
    stride = int(strides[0]) if strides else 16
    latent_align = max(1, chunk_size // max(1, stride))

    objective = str(diffusion_cfg.get("diffusion_objective", "rf_denoiser"))
    default_sampler = "pingpong" if objective == "rf_denoiser" else "euler"
    default_steps = 8 if objective == "rf_denoiser" else 50
    default_cfg_scale = 1.0 if objective == "rf_denoiser" else 7.0

    dist_shift = (
        diffusion_cfg.get("sampling_distribution_shift_options")
        or diffusion_cfg.get("distribution_shift_options")
        or {
            "type": "logsnr",
            "anchor_length": 2000,
            "anchor_logsnr": -6.2,
            "rate": 1.0,
            "logsnr_end": 2.0,
        }
    )

    input_concat_ids = list(diffusion_cfg.get("input_concat_ids", []))
    local_add_cond_ids = list(diffusion_cfg.get("local_add_cond_ids", []))
    input_concat_dim = int(dit_cfg.get("input_concat_dim", 0))
    local_add_cond_dim = int(dit_cfg.get("local_add_cond_dim", 0))
    has_inpaint_concat = (
        ("inpaint_mask" in input_concat_ids)
        or ("inpaint_mask" in local_add_cond_ids)
        or (input_concat_dim > 0)
        or (local_add_cond_dim > 0)
    )
    eff_concat_dim = input_concat_dim or local_add_cond_dim or (latent_dim + 1 if has_inpaint_concat else 0)

    cross_attn_cond_ids = list(diffusion_cfg.get("cross_attention_cond_ids", ["prompt", "seconds_total"]))
    global_cond_ids = list(diffusion_cfg.get("global_cond_ids", ["seconds_total"]))

    cond_dim = int(cond_cfg.get("cond_dim", dit_cfg.get("cond_token_dim", 768)))
    global_cond_dim = int(dit_cfg.get("global_cond_dim", cond_dim if global_cond_ids else 0))

    max_length = 128
    for c_info in cond_cfg.get("configs", []):
        if c_info.get("type") == "t5gemma":
            max_length = int(c_info.get("config", {}).get("max_length", 128))
            break

    return {
        "sample_rate": sample_rate,
        "io_channels": io_channels,
        "latent_dim": latent_dim,
        "downsampling_ratio": downsampling_ratio,
        "latent_align": latent_align,
        "cond_dim": cond_dim,
        "global_cond_dim": global_cond_dim,
        "input_concat_dim": eff_concat_dim,
        "diffusion": {
            "objective": objective,
            "default_sampler": default_sampler,
            "default_steps": default_steps,
            "default_cfg_scale": default_cfg_scale,
            "default_apg_scale": 1.0,
            "dist_shift": dict(dist_shift),
            "has_inpaint_concat": has_inpaint_concat,
            "cross_attn_cond_ids": cross_attn_cond_ids,
            "global_cond_ids": global_cond_ids,
            "input_concat_ids": input_concat_ids,
        },
        "tokenizer": {
            "file": "tokenizer.json",
            "max_length": max_length,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "add_eos": True,
        },
    }


def build_bundle_manifest(
    *,
    variant: str,
    runtime: str,
    sample_rate: int,
    io_channels: int,
    latent_dim: int,
    downsampling_ratio: int,
    latent_align: int,
    max_duration_sec: float,
    duration_padding_sec: float,
    diffusion: dict[str, Any],
    tokenizer: dict[str, Any],
    components: dict[str, dict[str, str]],
    cond_dim: int = 768,
    global_cond_dim: int = 768,
    input_concat_dim: int = 257,
    delegate: str | None = None,
    compiled_latent_frames: int | None = None,
) -> dict[str, Any]:
    """Constructs a version-1 `.nbsampler.json` bundle manifest dictionary."""
    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "architecture": "stable_audio_3",
        "variant": variant,
        "runtime": runtime,
        "sample_rate": int(sample_rate),
        "io_channels": int(io_channels),
        "latent_dim": int(latent_dim),
        "downsampling_ratio": int(downsampling_ratio),
        "latent_align": int(latent_align),
        "cond_dim": int(cond_dim),
        "global_cond_dim": int(global_cond_dim),
        "input_concat_dim": int(input_concat_dim),
        "max_duration_sec": float(max_duration_sec),
        "duration_padding_sec": float(duration_padding_sec),
        "diffusion": dict(diffusion),
        "tokenizer": dict(tokenizer),
        "components": {k: dict(v) for k, v in components.items()},
    }
    if delegate is not None:
        manifest["delegate"] = str(delegate)
    if compiled_latent_frames is not None:
        manifest["compiled_latent_frames"] = int(compiled_latent_frames)
    return manifest


def save_bundle_manifest(manifest: dict[str, Any], path: str) -> str:
    """Writes a `.nbsampler.json` manifest to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_bundle_manifest(path: str) -> dict[str, Any]:
    """Loads a `.nbsampler.json` manifest from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_component_path(bundle_dir: str, rel_or_abs: str) -> str:
    """Resolves a component path relative to the manifest directory."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(bundle_dir, rel_or_abs))


def validate_bundle_manifest(
    manifest: dict[str, Any],
    bundle_dir: str | None = None,
    check_files: bool = True,
) -> list[str]:
    """
    Validates a `.nbsampler.json` manifest dictionary.
    Returns a list of error strings (empty list means valid).
    """
    errors: list[str] = []

    if manifest.get("format") != BUNDLE_FORMAT:
        errors.append(f"Invalid format: expected '{BUNDLE_FORMAT}', got {manifest.get('format')!r}")
    if manifest.get("version") != BUNDLE_VERSION:
        errors.append(f"Unsupported version: expected {BUNDLE_VERSION}, got {manifest.get('version')!r}")
    runtime = manifest.get("runtime")
    if runtime not in ("executorch", "onnx", "torchscript"):
        errors.append(f"Unsupported runtime: expected executorch/onnx/torchscript, got {runtime!r}")
    if runtime == "executorch" and "compiled_latent_frames" in manifest:
        clf = manifest.get("compiled_latent_frames")
        if not isinstance(clf, int) or clf <= 0:
            errors.append(f"Invalid 'compiled_latent_frames': {clf!r}")

    for field in ("sample_rate", "io_channels", "latent_dim", "downsampling_ratio", "latent_align"):
        val = manifest.get(field)
        if not isinstance(val, (int, float)) or val <= 0:
            errors.append(f"Field '{field}' must be > 0, got {val!r}")

    diff = manifest.get("diffusion")
    if not isinstance(diff, dict):
        errors.append("Missing or invalid 'diffusion' section")
    else:
        for req in ("objective", "default_sampler", "default_steps"):
            if req not in diff:
                errors.append(f"Missing 'diffusion.{req}'")

    tok = manifest.get("tokenizer")
    if not isinstance(tok, dict):
        errors.append("Missing or invalid 'tokenizer' section")
    else:
        if "max_length" not in tok or int(tok["max_length"]) <= 0:
            errors.append("Invalid 'tokenizer.max_length'")
        if check_files and bundle_dir and tok.get("file"):
            tok_path = resolve_component_path(bundle_dir, tok["file"])
            if not os.path.isfile(tok_path):
                errors.append(f"Tokenizer file not found: {tok_path}")

    comps = manifest.get("components")
    if not isinstance(comps, dict):
        errors.append("Missing or invalid 'components' section")
    else:
        for key in REQUIRED_COMPONENTS:
            comp = comps.get(key)
            if not isinstance(comp, dict) or not comp.get("model"):
                errors.append(f"Missing component '{key}.model'")
                continue
            if check_files and bundle_dir:
                m_path = resolve_component_path(bundle_dir, comp["model"])
                if not os.path.isfile(m_path):
                    errors.append(f"Component model not found ({key}): {m_path}")
                sidecar = comp.get("sidecar")
                if sidecar:
                    s_path = resolve_component_path(bundle_dir, sidecar)
                    if not os.path.isfile(s_path):
                        errors.append(f"Component sidecar not found ({key}): {s_path}")

    return errors


def read_nbnd_v2_sidecar(bin_path: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Reads a NetBender `NBND` v2 binary weights sidecar (`.bin`).
    Returns `(metadata_dict, {tensor_name: np.ndarray(dtype=float32)})`.
    """
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"NBND":
            raise ValueError(f"Invalid NBND magic in '{bin_path}': {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"Expected NBND version >= 2 in '{bin_path}', got {version}")

        json_len = struct.unpack("<I", f.read(4))[0]
        meta: dict[str, Any] = {}
        if json_len > 0:
            meta = json.loads(f.read(json_len).decode("utf-8"))

        num_tensors = struct.unpack("<I", f.read(4))[0]
        tensors: dict[str, np.ndarray] = {}
        for _ in range(num_tensors):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            num_dims = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<q", f.read(8))[0] for _ in range(num_dims))
            num_elems = struct.unpack("<Q", f.read(8))[0]
            arr = np.frombuffer(f.read(num_elems * 4), dtype=np.float32).copy()
            if shape:
                arr = arr.reshape(shape)
            tensors[name] = arr

    return meta, tensors
