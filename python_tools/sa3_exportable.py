"""
sa3_exportable.py - ONNX-opset-17-compatible PyTorch wrappers for Stable Audio 3.

Adapts T5GemmaConditioner, NumberConditioner, DiffusionTransformer, and the
SAME-S / AutoencoderPretransform into pure-tensor forward graphs without
flash_attn, flex_attention, torch.compile, activation checkpointing, or
unsupported opset-17 operators (e.g. aten::rms_norm).
"""

from __future__ import annotations

import types

import torch.nn.functional as F

import torch
from torch import nn


def _exportable_rms_norm_forward(self_mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Opset-17-compatible RMSNorm matching stable_audio_3.models.transformer.RMSNorm."""
    force_fp32 = bool(getattr(self_mod, "force_fp32", False))
    x_f = x.float() if force_fp32 else x
    gamma = self_mod.gamma.float() if force_fp32 else self_mod.gamma
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self_mod.eps)
    return (x_f * rms * gamma).to(x.dtype)


def _rank4_rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _rank4_apply_rope(t: torch.Tensor, freqs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    out_dtype = t.dtype
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs_f = freqs.float()[-seq_len:, :]
    t_f = t.float()
    if t_f.ndim == 4 and freqs_f.ndim == 3:
        freqs_f = freqs_f.unsqueeze(1)
    t_rot, t_pass = t_f[..., :rot_dim], t_f[..., rot_dim:]
    t_rot = (t_rot * freqs_f.cos() * scale) + (_rank4_rotate_half(t_rot) * freqs_f.sin() * scale)
    return torch.cat((t_rot.to(out_dtype), t_pass.to(out_dtype)), dim=-1)


def _rank4_attn_forward(
    self_mod,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rotary_pos_emb=None,
    rotary_pos_emb_k=None,
    causal: bool | None = None,
    flex_attention_block_mask=None,
    flex_attention_score_mod=None,
    flash_attn_sliding_window=None,
    padding_mask=None,
    varlen_metadata=None,
) -> torch.Tensor:
    """
    CoreML-safe (max rank <= 4) forward for `stable_audio_3.models.transformer.Attention`.
    Avoids stacking `[q, q_diff]` into rank 5 and `rotate_half`'s rank-6 reshape.
    """
    h, kv_h = self_mod.num_heads, self_mod.kv_heads
    kv_input = context if context is not None else x
    b, n_q, _ = x.shape
    n_k = kv_input.shape[1]
    d = self_mod.dim_heads

    if hasattr(self_mod, "to_q"):
        if self_mod.differential:
            q, q_diff = self_mod.to_q(x).chunk(2, dim=-1)
            q = q.reshape(b, n_q, h, d).permute(0, 2, 1, 3)
            q_diff = q_diff.reshape(b, n_q, h, d).permute(0, 2, 1, 3)
            k, k_diff, v = self_mod.to_kv(kv_input).chunk(3, dim=-1)
            k = k.reshape(b, n_k, kv_h, d).permute(0, 2, 1, 3)
            k_diff = k_diff.reshape(b, n_k, kv_h, d).permute(0, 2, 1, 3)
            v = v.reshape(b, n_k, kv_h, d).permute(0, 2, 1, 3)
        else:
            q = self_mod.to_q(x).reshape(b, n_q, h, d).permute(0, 2, 1, 3)
            k, v = self_mod.to_kv(kv_input).chunk(2, dim=-1)
            k = k.reshape(b, n_k, kv_h, d).permute(0, 2, 1, 3)
            v = v.reshape(b, n_k, kv_h, d).permute(0, 2, 1, 3)
            q_diff = k_diff = None
    else:
        if self_mod.differential:
            q, k, v, q_diff, k_diff = self_mod.to_qkv(x).chunk(5, dim=-1)
            q = q.reshape(b, n_q, h, d).permute(0, 2, 1, 3)
            k = k.reshape(b, n_k, h, d).permute(0, 2, 1, 3)
            v = v.reshape(b, n_k, h, d).permute(0, 2, 1, 3)
            q_diff = q_diff.reshape(b, n_q, h, d).permute(0, 2, 1, 3)
            k_diff = k_diff.reshape(b, n_k, h, d).permute(0, 2, 1, 3)
        else:
            q, k, v = self_mod.to_qkv(x).chunk(3, dim=-1)
            q = q.reshape(b, n_q, h, d).permute(0, 2, 1, 3)
            k = k.reshape(b, n_k, h, d).permute(0, 2, 1, 3)
            v = v.reshape(b, n_k, h, d).permute(0, 2, 1, 3)
            q_diff = k_diff = None

    if self_mod.qk_norm == "l2":
        q = F.normalize(q, dim=-1, eps=self_mod.qk_norm_eps)
        k = F.normalize(k, dim=-1, eps=self_mod.qk_norm_eps)
        if self_mod.differential:
            q_diff = F.normalize(q_diff, dim=-1, eps=self_mod.qk_norm_eps)
            k_diff = F.normalize(k_diff, dim=-1, eps=self_mod.qk_norm_eps)
    elif self_mod.qk_norm != "none":
        q, k = self_mod.apply_qk_layernorm(q, k)
        if self_mod.differential:
            q_diff, k_diff = self_mod.apply_qk_layernorm(q_diff, k_diff)

    if rotary_pos_emb is not None:
        freqs, _ = rotary_pos_emb
        q_freqs = freqs
        if rotary_pos_emb_k is not None:
            k_freqs, _ = rotary_pos_emb_k
        else:
            k_freqs = q_freqs
            if n_q >= n_k:
                k_freqs = (n_q / n_k) * freqs
            else:
                q_freqs = (n_k / n_q) * freqs
        q = _rank4_apply_rope(q, q_freqs)
        k = _rank4_apply_rope(k, k_freqs)
        if self_mod.differential:
            q_diff = _rank4_apply_rope(q_diff, q_freqs)
            k_diff = _rank4_apply_rope(k_diff, k_freqs)

    is_causal = bool(self_mod.causal if causal is None else causal)
    if n_q == 1 and is_causal:
        is_causal = False

    if self_mod.differential:
        out_main = self_mod.apply_attn(
            q,
            k,
            v,
            causal=is_causal,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=varlen_metadata,
        )
        out_diff = self_mod.apply_attn(
            q_diff,
            k_diff,
            v,
            causal=is_causal,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=varlen_metadata,
        )
        out = out_main - out_diff
    else:
        out = self_mod.apply_attn(
            q,
            k,
            v,
            causal=is_causal,
            flex_attention_block_mask=flex_attention_block_mask,
            flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window,
            padding_mask=padding_mask,
            varlen_metadata=varlen_metadata,
        )

    out = out.permute(0, 2, 1, 3).reshape(b, n_q, h * d)
    out = self_mod.to_out(out)

    if getattr(self_mod, "feat_scale", False):
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).to(out.dtype)
            out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1)
            out_hf = out - out_dc
            out = out + (self_mod.lambda_dc * out_dc + self_mod.lambda_hf * out_hf) * mask
        else:
            out_dc = out.mean(dim=-2, keepdim=True)
            out_hf = out - out_dc
            out = out + self_mod.lambda_dc * out_dc + self_mod.lambda_hf * out_hf

    return out


def patch_sa3_for_onnx_export(module: nn.Module) -> None:
    """
    Patches a Stable Audio 3 / Transformers module tree in-place so it traces
    cleanly to ONNX opset 17 and lowers cleanly to ExecuTorch (CoreML / XNNPACK):
      1. Replaces `RMSNorm.forward` (which invokes `F.rms_norm` / `aten::rms_norm`)
         with explicit `rsqrt(mean(x^2) + eps) * gamma`.
      2. Replaces `Attention.forward` with rank-<=4 `_rank4_attn_forward` so CoreML
         does not reject rank-5/6 tensors in differential attention + RoPE.
      3. Sets HuggingFace `_attn_implementation = "eager"` and `use_cache = False`.
      4. Strips deprecated `weight_norm` hooks where present.
    """
    for sub in module.modules():
        if sub.__class__.__name__ == "RMSNorm" and hasattr(sub, "gamma") and hasattr(sub, "eps"):
            sub.forward = types.MethodType(_exportable_rms_norm_forward, sub)

        if sub.__class__.__name__ == "Attention" and hasattr(sub, "to_out") and hasattr(sub, "apply_attn"):
            sub.forward = types.MethodType(_rank4_attn_forward, sub)

        cfg = getattr(sub, "config", None)
        if cfg is not None:
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "eager"
            if hasattr(cfg, "use_cache"):
                cfg.use_cache = False
            enc_cfg = getattr(cfg, "encoder", None)
            if enc_cfg is not None:
                if hasattr(enc_cfg, "_attn_implementation"):
                    enc_cfg._attn_implementation = "eager"
                if hasattr(enc_cfg, "use_cache"):
                    enc_cfg.use_cache = False

        try:
            torch.nn.utils.remove_weight_norm(sub)
        except (ValueError, AttributeError):
            pass


patch_sa3_for_export = patch_sa3_for_onnx_export


class ExportableTextEncoder(nn.Module):
    """
    Pure-tensor ONNX wrapper for `T5GemmaConditioner`.

    Explicitly registers `conditioner.model` (which `T5GemmaConditioner` stores in
    `__dict__["model"]` when `enable_grad=False`) as a child `nn.Module` so all
    encoder parameters are included in the exported ONNX graph.
    """

    def __init__(self, conditioner: nn.Module):
        super().__init__()
        encoder = getattr(conditioner, "model", None) or conditioner.__dict__.get("model")
        if encoder is None:
            raise ValueError("Conditioner has no `.model` encoder attribute.")

        self.encoder = encoder
        self.proj_out = getattr(conditioner, "proj_out", nn.Identity())

        raw_mode = getattr(conditioner, "padding_mode", "zero")
        self.padding_mode = str(raw_mode.value if hasattr(raw_mode, "value") else raw_mode).lower()
        if self.padding_mode == "learned":
            pad_emb = getattr(conditioner, "padding_embedding", None)
            if pad_emb is None:
                raise ValueError("padding_mode='learned' requires conditioner.padding_embedding.")
            self.padding_embedding = pad_emb
        else:
            self.register_parameter("padding_embedding", None)

        self.encoder.float()
        self.float()
        patch_sa3_for_onnx_export(self)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = out["last_hidden_state"] if isinstance(out, dict) else out[0]

        if not isinstance(self.proj_out, nn.Identity):
            proj_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_dtype)
        embeddings = self.proj_out(embeddings)

        if self.padding_mode == "zero":
            embeddings = embeddings * attention_mask.unsqueeze(-1).to(embeddings.dtype)
        elif self.padding_mode == "learned" and self.padding_embedding is not None:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            pad_vec = self.padding_embedding.view(1, 1, -1).to(embeddings.dtype).expand_as(embeddings)
            embeddings = torch.where(mask_expanded, embeddings, pad_vec)

        return embeddings


class ExportableNumberConditioner(nn.Module):
    """
    Pure-tensor ONNX wrapper for `NumberConditioner` (`seconds_total`).
    Input: `seconds_total [B]` -> Output: `time_embed [B, 1, output_dim]`.
    """

    def __init__(self, conditioner: nn.Module):
        super().__init__()
        self.min_val = float(getattr(conditioner, "min_val", 0.0))
        self.max_val = float(getattr(conditioner, "max_val", 1.0))
        self.embedder = conditioner.embedder
        self.float()
        patch_sa3_for_onnx_export(self)

    def forward(self, seconds_total: torch.Tensor) -> torch.Tensor:
        floats = seconds_total.float().clamp(self.min_val, self.max_val)
        normalized = (floats - self.min_val) / (self.max_val - self.min_val)
        embedder_params = list(self.embedder.parameters())
        if embedder_params:
            normalized = normalized.to(embedder_params[0].dtype)
        return self.embedder(normalized).unsqueeze(1)


class ExportableDiTStep(nn.Module):
    """
    Single unguided denoising step wrapper around `DiffusionTransformer._forward`.

    Always keeps timestep `t` in float32 (per `dit.py` logSNR precision requirement)
    and disables activation checkpointing for clean ONNX tracing.
    """

    def __init__(
        self,
        dit_or_wrapper: nn.Module,
        has_cross_attn: bool | None = None,
        has_global_cond: bool | None = None,
        has_input_concat: bool | None = None,
    ):
        super().__init__()
        self.dit = dit_or_wrapper.model if hasattr(dit_or_wrapper, "model") and not hasattr(dit_or_wrapper, "_forward") else dit_or_wrapper
        self.has_cross_attn = (
            bool(getattr(self.dit, "cond_token_dim", 0) > 0)
            if has_cross_attn is None
            else bool(has_cross_attn)
        )
        self.has_global_cond = (
            hasattr(self.dit, "to_global_embed")
            if has_global_cond is None
            else bool(has_global_cond)
        )
        self.has_input_concat = (
            bool(getattr(self.dit, "input_concat_dim", 0) > 0)
            if has_input_concat is None
            else bool(has_input_concat)
        )
        layers = getattr(getattr(self.dit, "transformer", None), "layers", [])
        self.has_local_add_cond = (
            len(layers) > 0 and getattr(layers[0], "to_local_embed", None) is not None
        )
        self.float()
        patch_sa3_for_onnx_export(self)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor,
        global_cond: torch.Tensor,
        input_concat_cond: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        model_dtype = next(self.dit.parameters()).dtype
        x_in = x.to(model_dtype)
        t_f32 = t.float()
        ca_in = cross_attn_cond.to(model_dtype) if self.has_cross_attn else None
        gc_in = global_cond.to(model_dtype) if self.has_global_cond else None
        ic_in = input_concat_cond.to(model_dtype) if self.has_input_concat else None
        lac_in = (
            input_concat_cond.to(model_dtype)
            if (not self.has_input_concat and self.has_local_add_cond)
            else None
        )

        out = self.dit._forward(
            x_in,
            t_f32,
            cross_attn_cond=ca_in,
            global_embed=gc_in,
            input_concat_cond=ic_in,
            local_add_cond=lac_in,
            padding_mask=padding_mask.bool(),
            use_checkpointing=False,
        )
        return out.float()


class PretransformEncodeAdapter(nn.Module):
    """Uniform pure-tensor encoder adapter for `SAMESCore` or `AutoencoderPretransform`."""

    def __init__(self, pretransform: nn.Module):
        super().__init__()
        self.pretransform = pretransform
        self.float()
        patch_sa3_for_onnx_export(self)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if hasattr(self.pretransform, "model") and hasattr(self.pretransform, "scale") and hasattr(self.pretransform.model, "encode"):
            return self.pretransform.model.encode(audio) / self.pretransform.scale
        return self.pretransform.encode(audio)


class PretransformDecodeAdapter(nn.Module):
    """Uniform pure-tensor decoder adapter for `SAMESCore` or `AutoencoderPretransform`."""

    def __init__(self, pretransform: nn.Module):
        super().__init__()
        self.pretransform = pretransform
        self.float()
        patch_sa3_for_onnx_export(self)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if hasattr(self.pretransform, "model") and hasattr(self.pretransform, "scale") and hasattr(self.pretransform.model, "decode"):
            return self.pretransform.model.decode(latents * self.pretransform.scale)
        return self.pretransform.decode(latents)


class PretransformForwardAdapter(nn.Module):
    """Uniform pure-tensor encode->decode adapter for `SAMESCore` or `AutoencoderPretransform`."""

    def __init__(self, pretransform: nn.Module):
        super().__init__()
        self.pretransform = pretransform
        self.float()
        patch_sa3_for_onnx_export(self)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if hasattr(self.pretransform, "model") and hasattr(self.pretransform, "scale") and hasattr(self.pretransform.model, "encode"):
            n = audio.shape[-1]
            z = self.pretransform.model.encode(audio) / self.pretransform.scale
            return self.pretransform.model.decode(z * self.pretransform.scale)[..., :n]
        if hasattr(self.pretransform, "encode") and hasattr(self.pretransform, "decode"):
            n = audio.shape[-1]
            return self.pretransform.decode(self.pretransform.encode(audio))[..., :n]
        return self.pretransform(audio)
