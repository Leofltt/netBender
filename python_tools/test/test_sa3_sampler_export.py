"""
Offline unit & numerical parity tests for Stable Audio 3 Sampler export pipeline (Phase 0).

Uses tiny synthetic SA3 models (real T5GemmaEncoderModel + NumberConditioner +
DiffusionTransformer + synthetic pretransform) so no HuggingFace downloads are needed.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import T5GemmaConfig, T5GemmaEncoderModel

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_audio_3.models.conditioners import MultiConditioner, NumberConditioner
from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper, DiTWrapper
from python_tools.exporters.sa3_sampler_exporter import (
    export_sa3_sampler,
    verify_sa3_onnx_parity,
)
from python_tools.sa3_bundle import (
    build_bundle_manifest,
    load_bundle_manifest,
    read_nbnd_v2_sidecar,
    resolve_component_path,
    save_bundle_manifest,
    validate_bundle_manifest,
)


class _DummyT5Conditioner(nn.Module):
    """Minimal duck-typed T5GemmaConditioner matching stable_audio_3's layout."""

    def __init__(self, output_dim: int = 32, hidden_size: int = 32, max_length: int = 16, padding_mode: str = "zero"):
        super().__init__()
        cfg = T5GemmaConfig(
            is_encoder_decoder=False,
            encoder={
                "vocab_size": 128,
                "hidden_size": hidden_size,
                "intermediate_size": hidden_size * 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": hidden_size // 4,
            },
        )
        # Store in __dict__["model"] just like real T5GemmaConditioner(enable_grad=False)
        self.__dict__["model"] = T5GemmaEncoderModel(cfg).eval().requires_grad_(False)
        self.proj_out = nn.Linear(hidden_size, output_dim) if hidden_size != output_dim else nn.Linear(hidden_size, output_dim)
        self.max_length = max_length
        self.padding_mode = padding_mode


class _DummyPretransform(nn.Module):
    """Tiny conv pretransform matching AutoencoderPretransform API."""

    def __init__(self, io_channels: int = 2, latent_dim: int = 16, ratio: int = 32):
        super().__init__()
        self.io_channels = io_channels
        self.latent_dim = latent_dim
        self.downsampling_ratio = ratio
        self.enc_conv = nn.Conv1d(io_channels, latent_dim, kernel_size=ratio, stride=ratio)
        self.dec_conv = nn.ConvTranspose1d(latent_dim, io_channels, kernel_size=ratio, stride=ratio)
        self.scale_vec = nn.Parameter(torch.ones(1, latent_dim, 1))

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.enc_conv(audio) * self.scale_vec

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_conv(z / self.scale_vec)


def _build_tiny_sa3_wrapper(norm_type: str = "dyt") -> tuple[ConditionedDiffusionModelWrapper, dict]:
    cond_dim = 32
    latent_dim = 16
    ratio = 32
    max_len = 16
    sr = 44100

    prompt_cond = _DummyT5Conditioner(output_dim=cond_dim, hidden_size=32, max_length=max_len)
    sec_cond = NumberConditioner(output_dim=cond_dim, min_val=0.0, max_val=60.0, fourier_features_type="expo")
    multi_cond = MultiConditioner({"prompt": prompt_cond, "seconds_total": sec_cond})

    dit_wrapper = DiTWrapper(
        diffusion_objective="rf_denoiser",
        io_channels=latent_dim,
        embed_dim=64,
        cond_token_dim=cond_dim,
        global_cond_dim=cond_dim,
        input_concat_dim=latent_dim + 1,
        depth=2,
        num_heads=2,
        global_cond_type="adaLN",
        timestep_features_type="expo",
        timestep_features_dim=32,
        timestep_features_logsnr=True,
        norm_type=norm_type,
        attn_kwargs={"qk_norm": norm_type if norm_type == "dyt" else "rms", "differential": True},
        zero_init_branch_outputs=False,
    )

    pretransform = _DummyPretransform(io_channels=2, latent_dim=latent_dim, ratio=ratio)

    wrapper = ConditionedDiffusionModelWrapper(
        model=dit_wrapper,
        conditioner=multi_cond,
        io_channels=latent_dim,
        sample_rate=sr,
        min_input_length=ratio,
        diffusion_objective="rf_denoiser",
        sampling_distribution_shift_options={
            "type": "logsnr",
            "anchor_length": 2000,
            "anchor_logsnr": -6.2,
            "rate": 1.0,
            "logsnr_end": 2.0,
        },
        mask_padding_attention=True,
        use_effective_length_for_schedule=True,
        pretransform=pretransform,
        cross_attn_cond_ids=["prompt", "seconds_total"],
        global_cond_ids=["seconds_total"],
        input_concat_ids=["inpaint_mask", "inpaint_masked_input"],
    ).eval()

    model_config = {
        "sample_rate": sr,
        "model": {
            "io_channels": latent_dim,
            "diffusion": {
                "diffusion_objective": "rf_denoiser",
                "cross_attention_cond_ids": ["prompt", "seconds_total"],
                "global_cond_ids": ["seconds_total"],
                "input_concat_ids": ["inpaint_mask", "inpaint_masked_input"],
                "sampling_distribution_shift_options": {
                    "type": "logsnr",
                    "anchor_length": 2000,
                    "anchor_logsnr": -6.2,
                    "rate": 1.0,
                    "logsnr_end": 2.0,
                },
                "config": {
                    "cond_token_dim": cond_dim,
                    "global_cond_dim": cond_dim,
                    "input_concat_dim": latent_dim + 1,
                },
            },
            "pretransform": {
                "config": {
                    "io_channels": 2,
                    "latent_dim": latent_dim,
                    "downsampling_ratio": ratio,
                    "encoder": {"config": {"chunk_size": 32, "strides": [16]}},
                }
            },
            "conditioning": {
                "cond_dim": cond_dim,
                "configs": [{"id": "prompt", "type": "t5gemma", "config": {"max_length": max_len}}],
            },
        },
    }
    return wrapper, model_config


class TestSA3SamplerExport(unittest.TestCase):
    def test_manifest_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy referenced files
            for fname in ("te.onnx", "te.bin", "nc.onnx", "nc.bin", "dit.onnx", "dit.bin", "ae.onnx", "ae.bin", "tokenizer.json"):
                with open(os.path.join(tmpdir, fname), "wb") as f:
                    f.write(b"ok")

            manifest = build_bundle_manifest(
                variant="small-music",
                runtime="onnx",
                sample_rate=44100,
                io_channels=2,
                latent_dim=256,
                downsampling_ratio=4096,
                latent_align=2,
                max_duration_sec=30.0,
                duration_padding_sec=6.0,
                diffusion={"objective": "rf_denoiser", "default_sampler": "pingpong", "default_steps": 8},
                tokenizer={"file": "tokenizer.json", "max_length": 128, "pad_token_id": 0, "eos_token_id": 1},
                components={
                    "text_encoder": {"model": "te.onnx", "sidecar": "te.bin"},
                    "number_cond": {"model": "nc.onnx", "sidecar": "nc.bin"},
                    "dit": {"model": "dit.onnx", "sidecar": "dit.bin"},
                    "autoencoder": {"model": "ae.onnx", "sidecar": "ae.bin"},
                },
            )
            m_path = os.path.join(tmpdir, "test.nbsampler.json")
            save_bundle_manifest(manifest, m_path)
            loaded = load_bundle_manifest(m_path)
            self.assertEqual(validate_bundle_manifest(loaded, bundle_dir=tmpdir, check_files=True), [])

            bad = dict(loaded)
            bad["format"] = "wrong"
            self.assertGreater(len(validate_bundle_manifest(bad, bundle_dir=tmpdir, check_files=False)), 0)

            os.remove(os.path.join(tmpdir, "dit.bin"))
            self.assertGreater(len(validate_bundle_manifest(loaded, bundle_dir=tmpdir, check_files=True)), 0)

    def test_save_tokenizer_json_overwrites_readonly_dest(self):
        """Read-only pre-existing tokenizer.json dest (e.g. copied from HF cache) must be replaced."""
        import python_tools.exporters.sa3_sampler_exporter as sa3_exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            src_tok = os.path.join(tmpdir, "src_tokenizer.json")
            with open(src_tok, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0", "model": {"type": "Unigram", "vocab": ["a", "b"]}}, f)
            os.chmod(src_tok, 0o444)  # HF cache files are read-only

            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            dest_tok = os.path.join(out_dir, "tokenizer.json")
            with open(dest_tok, "w", encoding="utf-8") as f:
                f.write("stale placeholder")
            os.chmod(dest_tok, 0o444)

            fake_cond = types.SimpleNamespace(max_length=16)
            info = sa3_exporter._save_tokenizer_json(fake_cond, src_tok, out_dir, verbose=False)

            self.assertTrue(os.path.isfile(dest_tok))
            with open(dest_tok, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["model"]["vocab"], ["a", "b"])
            self.assertEqual(info["file"], "tokenizer.json")
            # dest must be writable again after the copy (HF cache 0444 bits stripped)
            mode = stat.S_IMODE(os.stat(dest_tok).st_mode)
            self.assertTrue(mode & 0o200, f"dest not writable: {oct(mode)}")

    def test_export_and_parity_dyt(self):
        wrapper, model_config = _build_tiny_sa3_wrapper(norm_type="dyt")
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = export_sa3_sampler(
                model_instance=wrapper,
                model_config=model_config,
                output_dir=tmpdir,
                stem="tiny_dyt",
                format="onnx",
                bend_mode="delegate",
                promote_conditioning_heads=True,
                verify=False,
                verbose=False,
            )
            self.assertTrue(os.path.isfile(manifest_path))
            manifest = load_bundle_manifest(manifest_path)
            self.assertEqual(validate_bundle_manifest(manifest, bundle_dir=tmpdir, check_files=True), [])

            metrics = verify_sa3_onnx_parity(wrapper, manifest_path, verbose=False)
            self.assertLess(metrics["text_encoder_max_err"], 1e-4)
            self.assertLess(metrics["number_cond_max_err"], 1e-4)
            self.assertLess(metrics["dit_max_err"], 1e-4)
            self.assertLess(metrics["dit_dynamic_max_err"], 1e-4)
            self.assertLess(metrics["encode_max_err"], 1e-4)
            self.assertLess(metrics["decode_max_err"], 1e-4)
            self.assertLess(metrics["pingpong_latent_max_err"], 1e-3)
            self.assertGreater(metrics["prompt_bend_diff"], 1e-5)
            self.assertGreater(metrics["dit_bend_diff"], 1e-5)
            self.assertLess(metrics["prompt_restore_err"], 1e-6)
            self.assertLess(metrics["dit_restore_err"], 1e-6)

    def test_export_and_parity_rms_norm(self):
        wrapper, model_config = _build_tiny_sa3_wrapper(norm_type="rms_norm")
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = export_sa3_sampler(
                model_instance=wrapper,
                model_config=model_config,
                output_dir=tmpdir,
                stem="tiny_rms",
                format="onnx",
                bend_mode="delegate",
                verify=False,
                verbose=False,
            )
            metrics = verify_sa3_onnx_parity(wrapper, manifest_path, verbose=False)
            self.assertLess(metrics["dit_max_err"], 1e-4)
            self.assertLess(metrics["dit_dynamic_max_err"], 1e-4)
            self.assertLess(metrics["pingpong_latent_max_err"], 1e-3)

    def test_nbnd_v2_sidecar_integrity(self):
        wrapper, model_config = _build_tiny_sa3_wrapper(norm_type="dyt")
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = export_sa3_sampler(
                model_instance=wrapper,
                model_config=model_config,
                output_dir=tmpdir,
                stem="tiny_sidecar",
                format="onnx",
                bend_mode="delegate",
                promote_conditioning_heads=True,
                verbose=False,
            )
            manifest = load_bundle_manifest(manifest_path)
            for comp_name, expected_sub in (("text_encoder", "prompt"), ("number_cond", "prompt"), ("dit", "dit"), ("autoencoder", "ae")):
                sidecar_rel = manifest["components"][comp_name]["sidecar"]
                self.assertTrue(sidecar_rel)
                meta, tensors = read_nbnd_v2_sidecar(resolve_component_path(tmpdir, sidecar_rel))
                self.assertEqual(meta.get("netbender.bendable"), "1")
                self.assertEqual(meta.get("netbender.subsystem"), expected_sub)
                self.assertGreater(len(tensors), 0)
                for arr in tensors.values():
                    self.assertEqual(arr.dtype, np.float32)

    def test_export_pte_xnnpack_dyt_and_rmsnorm(self):
        for norm_t in ("dyt", "rms_norm"):
            with self.subTest(norm_type=norm_t):
                wrapper, model_config = _build_tiny_sa3_wrapper(norm_type=norm_t)
                with tempfile.TemporaryDirectory() as tmpdir:
                    manifest_path = export_sa3_sampler(
                        model_instance=wrapper,
                        model_config=model_config,
                        output_dir=tmpdir,
                        stem=f"tiny_pte_{norm_t}",
                        format="pte",
                        delegate="xnnpack",
                        bend_mode="delegate",
                        dummy_latent_frames=8,
                        verbose=False,
                    )
                    self.assertTrue(os.path.isfile(manifest_path))
                    manifest = load_bundle_manifest(manifest_path)
                    self.assertEqual(manifest["runtime"], "executorch")
                    self.assertEqual(manifest["delegate"], "xnnpack")
                    self.assertEqual(manifest["compiled_latent_frames"], 8)
                    self.assertEqual(validate_bundle_manifest(manifest, bundle_dir=tmpdir, check_files=True), [])

                    expected_in_counts = {"text_encoder": 2, "number_cond": 1, "dit": 6}
                    for comp_name, expected_sub in (("text_encoder", "prompt"), ("number_cond", "prompt"), ("dit", "dit"), ("autoencoder", "ae")):
                        comp = manifest["components"][comp_name]
                        self.assertTrue(os.path.isfile(resolve_component_path(tmpdir, comp["model"])))
                        meta, tensors = read_nbnd_v2_sidecar(resolve_component_path(tmpdir, comp["sidecar"]))
                        self.assertEqual(meta.get("netbender.delegate"), "xnnpack")
                        self.assertEqual(meta.get("netbender.subsystem"), expected_sub)
                        self.assertGreater(len(meta.get("promoted_inputs", [])), 0)
                        self.assertGreater(len(tensors), 0)
                        if comp_name in expected_in_counts:
                            self.assertEqual(meta.get("num_model_inputs"), expected_in_counts[comp_name])

    def test_export_pte_coreml_dyt(self):
        try:
            import coremltools  # noqa: F401
        except ImportError:
            self.skipTest("coremltools not available")

        wrapper, model_config = _build_tiny_sa3_wrapper(norm_type="dyt")
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = export_sa3_sampler(
                model_instance=wrapper,
                model_config=model_config,
                output_dir=tmpdir,
                stem="tiny_pte_coreml",
                format="pte",
                delegate="coreml",
                bend_mode="delegate",
                dummy_latent_frames=8,
                verbose=False,
            )
            self.assertTrue(os.path.isfile(manifest_path))
            manifest = load_bundle_manifest(manifest_path)
            self.assertEqual(manifest["runtime"], "executorch")
            self.assertEqual(manifest["delegate"], "coreml")
            self.assertEqual(validate_bundle_manifest(manifest, bundle_dir=tmpdir, check_files=True), [])

            ae_comp = manifest["components"]["autoencoder"]
            self.assertTrue(ae_comp["encode_model"].endswith("_encode.pte"))
            self.assertTrue(ae_comp["decode_model"].endswith("_decode.pte"))
            self.assertTrue(os.path.isfile(resolve_component_path(tmpdir, ae_comp["encode_model"])))
            self.assertTrue(os.path.isfile(resolve_component_path(tmpdir, ae_comp["decode_model"])))

            for comp_name in ("text_encoder", "number_cond", "dit", "autoencoder"):
                meta, _ = read_nbnd_v2_sidecar(
                    resolve_component_path(tmpdir, manifest["components"][comp_name]["sidecar"])
                )
                self.assertEqual(meta.get("netbender.delegate"), "coreml")
                self.assertLessEqual(len(meta.get("promoted_inputs", [])), 240)

    def test_export_pte_prompt_only_no_bending(self):
        delegates = ["xnnpack"]
        try:
            import coremltools  # noqa: F401
            delegates.append("coreml")
        except ImportError:
            pass

        for del_name in delegates:
            with self.subTest(delegate=del_name):
                wrapper, model_config = _build_tiny_sa3_wrapper(norm_type="dyt")
                with tempfile.TemporaryDirectory() as tmpdir:
                    manifest_path = export_sa3_sampler(
                        model_instance=wrapper,
                        model_config=model_config,
                        output_dir=tmpdir,
                        stem=f"tiny_prompt_only_{del_name}",
                        format="pte",
                        delegate=del_name,
                        bend_mode="none",
                        dummy_latent_frames=8,
                        verbose=False,
                    )
                    self.assertTrue(os.path.isfile(manifest_path))
                    manifest = load_bundle_manifest(manifest_path)
                    self.assertEqual(manifest["runtime"], "executorch")
                    self.assertEqual(manifest["delegate"], del_name)
                    self.assertEqual(manifest["compiled_latent_frames"], 8)
                    self.assertEqual(
                        validate_bundle_manifest(manifest, bundle_dir=tmpdir, check_files=True), []
                    )

                    expected_in_counts = {"text_encoder": 2, "number_cond": 1, "dit": 6}
                    for comp_name in ("text_encoder", "number_cond", "dit", "autoencoder"):
                        comp = manifest["components"][comp_name]
                        self.assertTrue(os.path.isfile(resolve_component_path(tmpdir, comp["model"])))
                        meta, tensors = read_nbnd_v2_sidecar(
                            resolve_component_path(tmpdir, comp["sidecar"])
                        )
                        self.assertEqual(meta.get("netbender.bendable"), "0")
                        self.assertEqual(meta.get("netbender.delegate"), del_name)
                        self.assertEqual(len(tensors), 0)
                        if comp_name in expected_in_counts:
                            self.assertEqual(meta.get("num_model_inputs"), expected_in_counts[comp_name])

    def test_cli_help(self):
        cli_path = os.path.join(project_root, "python_tools", "export_cli.py")
        res = subprocess.run(
            [sys.executable, cli_path, "sa3_sampler", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("--variant", res.stdout)
        self.assertIn("--bend-mode", res.stdout)
        self.assertIn("--delegate", res.stdout)
        self.assertIn("--latent-frames", res.stdout)


if __name__ == "__main__":
    unittest.main()
