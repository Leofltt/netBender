#!/usr/bin/env python3
"""
rave_ts_to_onnx.py - Export RAVE TorchScript (.ts) models to ONNX and Bendable ONNX.

Takes any existing TorchScript RAVE model, extracts its audio forward method (or
subgraphs), converts it to standard ONNX with dynamic buffer sizes, and optionally
runs parameter promotion (onnx_bend_prep) so the model is immediately ready for
real-time circuit bending in NetBender.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

try:
    import onnx
except ImportError:
    print("Warning: 'onnx' library is not installed. Run `pip install onnx` for full verification.", file=sys.stderr)


def _register_rave_onnx_symbolics(opset_version: int = 17):
    """
    Registers custom ONNX symbolic handlers for RAVE's internal STFT and noise operators
    (view_as_complex, fft_irfft, fft_rfft, roll, hann_window) that are otherwise missing in
    PyTorch's TorchScript -> ONNX converter.
    """
    from torch.onnx import register_custom_op_symbolic

    def symbolic_view_as_complex(g, self):
        return self

    def symbolic_fft_irfft(g, self, n, dim, norm):
        # ONNX Runtime specification forbids DFT with onesided=1 & inverse=1.
        # We rebuild the exact two-sided conjugate-symmetric spectrum [..., N, 2]
        # and invoke DFT with inverse=1, onesided=0 along axis=-2, which ONNX Runtime natively executes!
        axis_freq = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
        start_1 = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        end_neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        interior = g.op("Slice", self, start_1, end_neg1, axis_freq)

        # Reverse interior along the frequency axis (-2) using negative step -1
        rev_start = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        rev_end = g.op("Constant", value_t=torch.tensor([-999999], dtype=torch.int64))
        rev_step = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        flipped = g.op("Slice", interior, rev_start, rev_end, axis_freq, rev_step)

        # Conjugate: keep real part, negate imaginary part along the complex component axis (-1)
        axis_complex = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        real_part = g.op("Slice", flipped, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)), g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)), axis_complex)
        imag_part = g.op("Slice", flipped, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)), g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64)), axis_complex)
        neg_imag = g.op("Neg", imag_part)
        conj_flipped = g.op("Concat", real_part, neg_imag, axis_i=-1)

        # Concat along frequency axis (-2) to form full two-sided Hermitian spectrum
        full_spectrum = g.op("Concat", self, conj_flipped, axis_i=-2)

        # Inverse two-sided DFT along axis -2
        inv_dft = g.op("DFT", full_spectrum, axis_i=-2, inverse_i=1, onesided_i=0)

        # Extract real part (index 0 along complex axis -1) and squeeze
        real_out = g.op("Slice", inv_dft, g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)), g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)), axis_complex)
        squeezed = g.op("Squeeze", real_out, axis_complex)

        # Scale by 1 / N to match PyTorch irfft(norm="backward") specification
        # full_spectrum has shape [..., N, 2] along axis -2
        shape_full = g.op("Shape", full_spectrum)
        axis_dim = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
        n_samples = g.op("Gather", shape_full, axis_dim, axis_i=0)
        n_float = g.op("Cast", n_samples, to_i=1)  # 1 = FLOAT
        return g.op("Div", squeezed, n_float)

    def symbolic_fft_rfft(g, self, n, dim, norm):
        # ONNX specification requires one-sided forward DFT input to have trailing dimension 1: [..., samples, 1]
        axis_c = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
        real_in = g.op("Unsqueeze", self, axis_c)
        return g.op("DFT", real_in, axis_i=-2, inverse_i=0, onesided_i=1)

    def symbolic_roll(g, self, shifts, dims):
        return self

    def symbolic_hann_window(g, window_length, *args):
        val = window_length.toIValue() if hasattr(window_length, "toIValue") else None
        if val is None and hasattr(window_length, "node") and window_length.node().hasAttribute("value"):
            val = window_length.node().t("value").item()
        L = int(val) if val is not None else 8
        return g.op("Constant", value_t=torch.hann_window(L))

    register_custom_op_symbolic("aten::view_as_complex", symbolic_view_as_complex, opset_version)
    register_custom_op_symbolic("aten::fft_irfft", symbolic_fft_irfft, opset_version)
    register_custom_op_symbolic("aten::fft_rfft", symbolic_fft_rfft, opset_version)
    register_custom_op_symbolic("aten::roll", symbolic_roll, opset_version)
    register_custom_op_symbolic("aten::hann_window", symbolic_hann_window, opset_version)


def _estimate_rave_history_samples(
    ts_model_path: str, ratio: int, latent_dim: int, default_history: int = 4096
) -> int:
    """Measure causal decode warmup delay on a fresh TorchScript copy."""
    if "speech" in ts_model_path.lower() or (latent_dim <= 8 and ratio >= 1024):
        return 28672
    if ratio < 1024:
        return default_history
    try:
        m = torch.jit.load(ts_model_path, map_location="cpu").eval()
        if not hasattr(m, "decode"):
            return default_history
        gen = torch.Generator().manual_seed(42)
        z = torch.randn(1, latent_dim, 12, generator=gen) * 2.0
        with torch.no_grad():
            y = m.decode(z)
        if not torch.isfinite(y).all():
            return default_history
        y_flat = y.reshape(y.shape[1], -1)
        num_frames = min(12, y_flat.shape[-1] // ratio)
        if num_frames <= 0:
            return default_history
        frame_rms = [
            float(torch.sqrt((y_flat[:, i * ratio : (i + 1) * ratio] ** 2).mean()))
            for i in range(num_frames)
        ]
        peak_rms = max(frame_rms)
        if peak_rms <= 1e-4:
            return default_history
        onset_idx = next((i for i, r in enumerate(frame_rms) if r >= 0.25 * peak_rms), 0)
        hist_frames = max(2, min(14, onset_idx + 2))
        return hist_frames * ratio
    except Exception:
        return default_history


def convert_rave_ts_to_onnx(
    ts_model_path: str,
    output_onnx_path: str,
    buffer_size: int = 2048,
    channels: int = 1,
    opset_version: int = 17,
    promote_for_bending: bool = True,
    methods: Optional[List[str]] = None,
    verbose: bool = True,
    trace_samples: Optional[int] = None,
    bend_mode: str = "delegate",
) -> str:
    """
    Loads a TorchScript model and exports it to ONNX.
    Supports exporting forward pass as well as companion methods (encode, decode, prior).
    If promote_for_bending is True, automatically invokes onnx_bend_prep to convert
    initializers into bendable graph inputs.
    """
    if not promote_for_bending:
        bend_mode = "none"

    if not os.path.exists(ts_model_path):
        raise FileNotFoundError(f"Model file not found: {ts_model_path}")

    # Register custom symbolics for RAVE's complex & FFT operators
    _register_rave_onnx_symbolics(opset_version=opset_version)

    if verbose:
        print(f"\n[1/3] Loading TorchScript model: {ts_model_path}")

    model = torch.jit.load(ts_model_path, map_location="cpu")
    model.eval()

    # Determine in_channels if metadata exists, else fallback to param
    in_channels = channels

    # Extract metadata properties if available on the model
    sr = 48000
    try:
        sr_found = False
        if hasattr(model, "sr"):
            val = model.sr() if callable(model.sr) else model.sr
            if val is not None and int(val) > 0:
                sr = int(val)
                sr_found = True
        if not sr_found and hasattr(model, "sample_rate"):
            val = model.sample_rate() if callable(model.sample_rate) else model.sample_rate
            if val is not None and int(val) > 0:
                sr = int(val)
                sr_found = True
        if not sr_found and hasattr(model, "named_buffers"):
            for bname, buf in model.named_buffers():
                if "sampling_rate" in bname.lower() or "sample_rate" in bname.lower():
                    sr = int(buf.item())
                    break
    except Exception:
        pass

    ratio = 2048
    try:
        if hasattr(model, "encode_params"):
            ep = model.encode_params() if callable(model.encode_params) else model.encode_params
            if hasattr(ep, "stride"):
                val = ep.stride() if callable(ep.stride) else ep.stride
                ratio = int(val)
            elif isinstance(ep, (list, tuple)) and len(ep) >= 2:
                ratio = int(ep[1])
        elif hasattr(model, "ratios"):
            ratios = model.ratios() if callable(model.ratios) else model.ratios
            if hasattr(ratios, "__iter__"):
                prod = 1
                for r in ratios:
                    prod *= int(r)
                if prod > 0:
                    ratio = prod
    except Exception:
        pass

    latent_dim = 4
    try:
        if hasattr(model, "decode_params"):
            dp = model.decode_params() if callable(model.decode_params) else model.decode_params
            if hasattr(dp, "__iter__") or isinstance(dp, (list, tuple, torch.Tensor)):
                latent_dim = int(dp[0])
        elif hasattr(model, "encode_params"):
            ep = model.encode_params() if callable(model.encode_params) else model.encode_params
            if hasattr(ep, "__iter__") or isinstance(ep, (list, tuple, torch.Tensor)):
                latent_dim = int(ep[2])
    except Exception:
        pass

    history_samples = _estimate_rave_history_samples(
        ts_model_path, ratio=ratio, latent_dim=latent_dim, default_history=4096
    )
    latent_scale = 2.5 if ("speech" in ts_model_path.lower() or (latent_dim <= 8 and ratio >= 1024)) else 1.0
    actual_trace_samples = trace_samples if trace_samples is not None else (history_samples + buffer_size)
    total_decode_frames = max(1, actual_trace_samples // max(1, ratio))

    # Dummy audio tensor [batch=1, channels=in_channels, samples=actual_trace_samples]
    dummy_input = torch.zeros(1, in_channels, actual_trace_samples, dtype=torch.float32)

    # Temporary standard onnx path
    base, ext = os.path.splitext(output_onnx_path)
    std_onnx_path = output_onnx_path if not promote_for_bending else f"{base}_raw.onnx"

    os.makedirs(os.path.dirname(os.path.abspath(std_onnx_path)), exist_ok=True)

    if verbose:
        print(f"[2/3] Exporting to standard ONNX (opset {opset_version}): {std_onnx_path}...")

    # Export forward pass with dynamic length axis
    export_kwargs = dict(
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes={
            "audio_in": {0: "batch", 2: "samples"},
            "audio_out": {0: "batch", 2: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    used_eager_fallback = False
    fallback_reason = None
    try:
        torch.onnx.export(model, (dummy_input,), std_onnx_path, **export_kwargs)
        _chk = onnx.load(std_onnx_path)
        has_control_flow = any(n.op_type in ("If", "Loop") for n in _chk.graph.node)
        if has_control_flow:
            used_eager_fallback = True
            fallback_reason = "direct TorchScript ONNX graph contains If/Loop control-flow ops"
    except Exception as direct_err:
        used_eager_fallback = True
        fallback_reason = str(direct_err)

    if used_eager_fallback:
        if verbose:
            print(
                f"  [Note] Direct TorchScript->ONNX export unsuitable ({fallback_reason}); "
                "reconstructing eager graph via TorchScriptGraphEagerModule..."
            )
        from python_tools.migrate_ts_to_pte import TorchScriptGraphEagerModule

        eager_fwd = TorchScriptGraphEagerModule(model, method_name="forward").eval()
        torch.onnx.export(eager_fwd, (dummy_input,), std_onnx_path, **export_kwargs)

    if verbose:
        print(f"  -> Exported standard ONNX successfully.")

    # Sanitize graph to bypass NaN-producing DFT noise branch in ONNX Runtime C++
    _sanitize_rave_onnx_graph(std_onnx_path, verbose=verbose)

    if used_eager_fallback:
        hist_lat_steps = history_samples // max(1, ratio)
        decode_latent_steps = max(1, total_decode_frames - hist_lat_steps)
    else:
        decode_latent_steps = max(1, (history_samples + buffer_size + ratio - 1) // max(1, ratio))

    extra_meta = {
        "netbender.architecture": "rave",
        "netbender.sample_rate": str(sr),
        "netbender.ratio": str(ratio),
        "netbender.latent_dim": str(latent_dim),
        "netbender.in_channels": str(in_channels),
        "netbender.out_channels": str(in_channels),
        "netbender.history_samples": str(history_samples),
        "netbender.output_offset": str(history_samples),
        "netbender.latent_scale": str(latent_scale),
        "netbender.decode_latent_steps": str(decode_latent_steps),
    }
    if used_eager_fallback:
        extra_meta["netbender.buffer_size"] = str(buffer_size)

    if promote_for_bending:
        # Step 3: Run parameter promotion
        if verbose:
            print(f"[3/3] Promoting weights to graph inputs for real-time bending: {output_onnx_path}...")

        from python_tools.onnx_bend_prep import promote_initializers_to_inputs

        promote_initializers_to_inputs(
            input_path=std_onnx_path,
            output_path=output_onnx_path,
            keep_initializers_as_defaults=True,
            verbose=verbose,
            extra_metadata=extra_meta,
            bend_mode=bend_mode,
        )

        # Clean up intermediate raw file if different
        if std_onnx_path != output_onnx_path and os.path.exists(std_onnx_path):
            try:
                os.remove(std_onnx_path)
            except OSError:
                pass

    # Export companion subgraphs if requested
    if methods:
        candidate_methods = ["encode", "decode", "prior"]
        available_ts_methods = [m for m in candidate_methods if hasattr(model, m)]
        export_list = []
        if "all" in methods:
            export_list = available_ts_methods
        else:
            export_list = [m for m in methods if m in available_ts_methods and m != "forward"]

        if export_list:
            sliced = slice_rave_onnx_subgraphs(
                output_onnx_path, base, export_list, latent_dim=latent_dim, verbose=verbose
            )
            if not sliced and any(m in export_list for m in ("encode", "decode")):
                from python_tools.migrate_ts_to_pte import TorchScriptGraphEagerModule

                for comp_m in ("encode", "decode"):
                    if comp_m not in export_list:
                        continue
                    comp_out = f"{base}_{comp_m}.onnx"
                    comp_raw = comp_out if not promote_for_bending else f"{base}_{comp_m}_raw.onnx"
                    if verbose:
                        print(f"  -> Exporting RAVE companion method '{comp_m}' -> {comp_out}...")
                    if comp_m == "encode":
                        comp_dummy = torch.zeros(1, in_channels, actual_trace_samples, dtype=torch.float32)
                        in_name, out_name = "audio_in", "latent_out"
                        dyn_axes = {in_name: {0: "batch", 2: "samples"}, out_name: {0: "batch", 2: "frames"}}
                    else:
                        comp_dummy = torch.zeros(1, latent_dim, total_decode_frames, dtype=torch.float32)
                        in_name, out_name = "latent_in", "audio_out"
                        dyn_axes = {in_name: {0: "batch", 2: "frames"}, out_name: {0: "batch", 2: "samples"}}
                    eager_comp = TorchScriptGraphEagerModule(model, method_name=comp_m).eval()
                    comp_kwargs = dict(
                        input_names=[in_name],
                        output_names=[out_name],
                        dynamic_axes=dyn_axes,
                        opset_version=opset_version,
                        do_constant_folding=True,
                    )
                    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                        comp_kwargs["dynamo"] = False
                    torch.onnx.export(eager_comp, (comp_dummy,), comp_raw, **comp_kwargs)
                    if comp_m == "decode":
                        _sanitize_rave_onnx_graph(comp_raw, verbose=verbose)
                    if promote_for_bending:
                        from python_tools.onnx_bend_prep import promote_initializers_to_inputs

                        promote_initializers_to_inputs(
                            input_path=comp_raw,
                            output_path=comp_out,
                            keep_initializers_as_defaults=True,
                            verbose=verbose,
                            write_sidecar=False,
                            bend_mode=bend_mode,
                        )
                        if comp_raw != comp_out and os.path.exists(comp_raw):
                            try:
                                os.remove(comp_raw)
                            except OSError:
                                pass
                    if verbose:
                        print(f"  -> Companion '{comp_m}' exported successfully.")

        # Direct companion export for 'prior' if present on the TorchScript model
        if "prior" in export_list and hasattr(model, "prior"):
            prior_onnx_path = f"{base}_prior.onnx"
            if verbose:
                print(f"  -> Exporting companion method 'prior' -> {prior_onnx_path}...")
            try:
                p_in_dim = 0
                p_in_ratio = 1
                try:
                    if hasattr(model, "prior_params"):
                        pp = model.prior_params() if callable(model.prior_params) else model.prior_params
                        if isinstance(pp, (list, tuple)) and len(pp) >= 2:
                            p_in_dim = int(pp[0])
                            p_in_ratio = int(pp[1])
                except Exception:
                    pass

                in_ch = max(1, p_in_dim)
                dummy_x = torch.zeros(1, in_ch, 2048, dtype=torch.float32)

                class PriorExportModule(torch.nn.Module):
                    def __init__(self, prior_fn, takes_audio_input: bool = False):
                        super().__init__()
                        self.prior_fn = prior_fn
                        self.takes_audio_input = takes_audio_input

                    def forward(self, x):
                        if self.takes_audio_input:
                            return self.prior_fn(x)
                        try:
                            return self.prior_fn(x)
                        except TypeError:
                            return self.prior_fn()

                wrapper = PriorExportModule(model.prior, takes_audio_input=(p_in_dim > 0))

                torch.onnx.export(
                    wrapper,
                    (dummy_x,),
                    prior_onnx_path,
                    input_names=["prior_in"],
                    output_names=["latent_out"],
                    dynamic_axes={"prior_in": {0: "batch", 2: "samples"}, "latent_out": {0: "batch", 2: "frames"}},
                    opset_version=opset_version,
                )
                if verbose:
                    print(f"  -> Companion 'prior' exported successfully.")
            except Exception as pe:
                if verbose:
                    print(f"  [Note] RAVE prior export skipped: {pe}")
                if os.path.exists(prior_onnx_path):
                    try:
                        os.remove(prior_onnx_path)
                    except OSError:
                        pass

    if verbose:
        print(f"\n[DONE] Ready for NetBender ONNX inference: {output_onnx_path}\n")

    return output_onnx_path


def slice_rave_onnx_subgraphs(full_onnx_path: str, base_path: str, methods: List[str], latent_dim: int = 4, verbose: bool = True) -> bool:
    """
    Slices a verified bendable RAVE v1 ONNX graph into companion models for 'encode' and 'decode'
    using the bottleneck latent tensor /_rave/Slice_1_output_0.
    Returns True if the RAVE v1 bottleneck tensor was found and sliced, False otherwise.
    """
    try:
        import onnx
        from onnx import helper, TensorProto
        import copy
    except ImportError:
        return False

    full_model = onnx.load(full_onnx_path)
    latent_tensor_name = "/_rave/Slice_1_output_0"

    # Find actual RAVE v1 bottleneck tensor (strictly under /_rave/ so PQMF / decoder Slice_1 nodes never collide)
    has_latent = any(latent_tensor_name in n.output for n in full_model.graph.node)
    if not has_latent:
        for n in full_model.graph.node:
            if n.name == "/_rave/Slice_1" or n.name.endswith("/_rave/Slice_1"):
                latent_tensor_name = n.output[0]
                has_latent = True
                break

    if not has_latent:
        return False

    # 1. ENCODE SUBGRAPH
    if "encode" in methods:
        enc_path = f"{base_path}_encode.onnx"
        if verbose:
            print(f"  -> Slicing companion method 'encode' -> {enc_path}...")
        enc_model = copy.deepcopy(full_model)
        del enc_model.graph.output[:]
        latent_out = helper.make_tensor_value_info("latent_out", TensorProto.FLOAT, ["batch", latent_dim, "frames"])
        enc_model.graph.output.append(latent_out)

        for n in enc_model.graph.node:
            for i, out in enumerate(n.output):
                if out == latent_tensor_name:
                    n.output[i] = "latent_out"

        needed = {"latent_out"}
        nodes_to_keep = []
        for n in reversed(enc_model.graph.node):
            if any(out in needed for out in n.output):
                needed.update(n.input)
                nodes_to_keep.append(n)
        nodes_to_keep.reverse()

        del enc_model.graph.node[:]
        enc_model.graph.node.extend(nodes_to_keep)

        enc_inits = [init for init in enc_model.graph.initializer if init.name in needed]
        del enc_model.graph.initializer[:]
        enc_model.graph.initializer.extend(enc_inits)

        enc_inputs = [inp for inp in enc_model.graph.input if inp.name in needed]
        del enc_model.graph.input[:]
        enc_model.graph.input.extend(enc_inputs)

        onnx.save(enc_model, enc_path)
        if verbose:
            print(f"  -> Companion 'encode' exported successfully ({len(enc_model.graph.node)} nodes).")

    # 2. DECODE SUBGRAPH
    if "decode" in methods:
        dec_path = f"{base_path}_decode.onnx"
        if verbose:
            print(f"  -> Slicing companion method 'decode' -> {dec_path}...")
        dec_model = copy.deepcopy(full_model)
        latent_in = helper.make_tensor_value_info("latent_in", TensorProto.FLOAT, ["batch", latent_dim, "frames"])
        dec_model.graph.input.insert(0, latent_in)

        for n in dec_model.graph.node:
            for i, inp in enumerate(n.input):
                if inp == latent_tensor_name:
                    n.input[i] = "latent_in"

        needed = {o.name for o in dec_model.graph.output}
        nodes_to_keep = []
        for n in reversed(dec_model.graph.node):
            if any(out in needed for out in n.output):
                needed.update(n.input)
                nodes_to_keep.append(n)
        nodes_to_keep.reverse()

        del dec_model.graph.node[:]
        dec_model.graph.node.extend(nodes_to_keep)

        needed.add("latent_in")
        dec_inits = [init for init in dec_model.graph.initializer if init.name in needed]
        del dec_model.graph.initializer[:]
        dec_model.graph.initializer.extend(dec_inits)

        dec_inputs = [inp for inp in dec_model.graph.input if inp.name in needed]
        del dec_model.graph.input[:]
        dec_model.graph.input.extend(dec_inputs)

        onnx.save(dec_model, dec_path)
        if verbose:
            print(f"  -> Companion 'decode' exported successfully ({len(dec_model.graph.node)} nodes).")

    return True


def _sanitize_rave_onnx_graph(onnx_path: str, verbose: bool = True):
    """
    Sanitizes RAVE's exported ONNX graph by pruning any noise synthesis branch containing
    torch.fft.irfft / DFT ops that cannot run dynamically or in ONNX Runtime / ExecuTorch.
    Supports both RAVE v1 (/_rave/decoder/Add_1) and generic RAVE v2 graphs (via DFT taint pass).
    """
    try:
        import onnx
    except ImportError:
        return

    model = onnx.load(onnx_path)

    # 1. RAVE v1 explicit pass: bypass /_rave/decoder/Add_1 to pass through tonal waveform (/Mul_1_output_0)
    bypassed_v1 = False
    for n in model.graph.node:
        if n.name == "/_rave/decoder/Add_1":
            n.op_type = "Identity"
            del n.input[:]
            n.input.append("/_rave/decoder/Mul_1_output_0")
            bypassed_v1 = True
            if verbose:
                print("  [Sanitizer] Bypassed /_rave/decoder/Add_1 noise summation with tonal waveform.")

    if bypassed_v1:
        nodes_to_keep = [n for n in model.graph.node if "/_rave/decoder/synth/2/" not in n.name]
        del model.graph.node[:]
        model.graph.node.extend(nodes_to_keep)

    # 2. Generic DFT-branch pruning (handles RAVE v2 noise modules regardless of node naming)
    has_dft = any(n.op_type == "DFT" for n in model.graph.node)
    if has_dft:
        tainted = set()
        replaced = 0
        for n in model.graph.node:
            if n.op_type == "DFT":
                tainted.update(n.output)
                continue
            if not any(inp in tainted for inp in n.input):
                continue
            if n.op_type == "Add" and len(n.input) == 2:
                clean = [inp for inp in n.input if inp not in tainted]
                if len(clean) == 1:
                    n.op_type = "Identity"
                    del n.input[:]
                    n.input.append(clean[0])
                    replaced += 1
                    continue
            tainted.update(n.output)

        needed = {o.name for o in model.graph.output}
        kept_nodes = []
        for n in reversed(model.graph.node):
            if any(o in needed for o in n.output):
                needed.update(n.input)
                kept_nodes.append(n)
        kept_nodes.reverse()
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)
        if verbose and replaced > 0:
            print(f"  [Sanitizer] Pruned {replaced} DFT/noise summation branch(es).")

    # 3. Prune unreferenced initializers
    referenced_inputs = set()
    for n in model.graph.node:
        for inp in n.input:
            referenced_inputs.add(inp)

    initializers_to_keep = [init for init in model.graph.initializer if init.name in referenced_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers_to_keep)

    onnx.save(model, onnx_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAVE TorchScript (.ts) to ONNX and Bendable ONNX for NetBender"
    )
    parser.add_argument("-i", "--input", required=True, help="Input TorchScript .ts model path")
    parser.add_argument("-o", "--output", help="Output .onnx file path")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Sample buffer size for export tracing")
    parser.add_argument("--channels", type=int, default=1, help="Input audio channels (default: 1)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to export (e.g. all, forward, encode, decode, prior). Default: all",
    )
    parser.add_argument(
        "--no-bending",
        action="store_true",
        help="Export standard static ONNX without promoting weights to inputs",
    )
    parser.add_argument(
        "--bend-mode",
        choices=["delegate", "full", "none"],
        default="delegate",
        help="Parameter promotion mode: delegate (1D biases/norms/scales), full (all weights), none (default: delegate)",
    )

    args = parser.parse_args()

    output = args.output
    if not output:
        base, _ = os.path.splitext(args.input)
        output = f"{base}_bendable.onnx"

    convert_rave_ts_to_onnx(
        ts_model_path=args.input,
        output_onnx_path=output,
        buffer_size=args.buffer_size,
        channels=args.channels,
        opset_version=args.opset,
        promote_for_bending=not args.no_bending,
        methods=args.methods,
        verbose=True,
        bend_mode=args.bend_mode,
    )


if __name__ == "__main__":
    main()
