#!/usr/bin/env python3
"""
onnx_bend_prep.py - NetBender ONNX Parameter Promotion Tool

Converts frozen ONNX initializers (weights, biases, scale/shift vectors) into
explicit graph inputs ("Exposed Ports"). This allows real-time, zero-copy
circuit bending (matrix drawing, stochastic drift, cross-talk bridging)
without recreating the ONNX Runtime session during live audio callbacks.
"""

import argparse
import json
import os
import sys
from typing import List, Optional

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
except ImportError:
    print("Error: 'onnx' Python package is required. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


DELEGATE_SAFE_KEYWORDS = (
    "norm",
    "ln",
    "gn",
    "bn",
    "bias",
    "snake",
    "alpha",
    "beta",
    "gamma",
    "scale",
    "shift",
    "gain",
    "weight_g",
    "act",
)


def is_bendable_tensor(
    name: str,
    tensor: TensorProto,
    min_elements: int = 1,
    bend_mode: str = "full",
) -> bool:
    """
    Check if an initializer qualifies as a bendable neural network parameter.
    Includes weights, biases, scale/shift vectors, normalization coefficients.
    Excludes shape constants, sequence indices, or integer scalars.
    """
    if bend_mode == "none":
        return False

    # Only float / double / float16 parameters are bendable audio weights
    if tensor.data_type not in (
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    ):
        return False

    dims = list(tensor.dims)
    total_elements = 1
    for d in dims:
        total_elements *= d

    if total_elements < min_elements:
        return False

    if bend_mode == "delegate":
        non_singleton_dims = sum(1 for d in dims if d > 1)
        if non_singleton_dims > 1:
            return False
        lower_name = name.lower()
        if any(
            ex in lower_name
            for ex in ("onnx::concat", "onnx::slice", "onnx::constantofshape", "cache", ".pad")
        ):
            return False
        if any(k in lower_name for k in DELEGATE_SAFE_KEYWORDS):
            return True
        if total_elements <= 4096:
            return True
        return False

    return True


def promote_initializers_to_inputs(
    input_path: str,
    output_path: str,
    layer_filter: Optional[List[str]] = None,
    keep_initializers_as_defaults: bool = True,
    verbose: bool = True,
    extra_metadata: Optional[dict] = None,
    write_sidecar: bool = True,
    bend_mode: str = "full",
) -> int:
    """
    Reads an ONNX model, finds specified or all eligible float initializers,
    and converts them to explicit graph inputs.
    """
    if verbose:
        print(f"Loading ONNX model from: {input_path}")

    model = onnx.load(input_path)
    graph = model.graph

    existing_input_names = {inp.name for inp in graph.input}
    initializers = list(graph.initializer)

    promoted_count = 0
    new_inputs = []
    promoted_initializers = []

    for init in initializers:
        name = init.name

        # If already an input, skip
        if name in existing_input_names:
            continue

        # Optional layer name filtering
        if layer_filter:
            matched = any(f.lower() in name.lower() for f in layer_filter)
            if not matched:
                continue

        # Verify candidate tensor type and dimensions
        if not is_bendable_tensor(name, init, bend_mode=bend_mode):
            continue

        # Create corresponding ValueInfoProto for graph.input
        dims = list(init.dims)
        value_info = helper.make_tensor_value_info(
            name=name,
            elem_type=init.data_type,
            shape=dims,
        )
        new_inputs.append(value_info)
        promoted_initializers.append(init)
        promoted_count += 1

        if verbose:
            print(f"  [+] Promoted: {name} | Shape: {dims} | Type: {init.data_type}")

    # Append new inputs to graph.input
    graph.input.extend(new_inputs)

    # In ONNX IR >= 4, having an initializer with a matching graph.input name marks it as an
    # input with a default initializer value. If keep_initializers_as_defaults is False, we strip it.
    if not keep_initializers_as_defaults:
        name_set = {inp.name for inp in new_inputs}
        remaining_inits = [init for init in graph.initializer if init.name not in name_set]
        del graph.initializer[:]
        graph.initializer.extend(remaining_inits)

    # Add metadata marker to document that this model is NetBender-promoted
    meta = model.metadata_props.add()
    meta.key = "netbender.bendable"
    meta.value = "1"

    meta_count = model.metadata_props.add()
    meta_count.key = "netbender.promoted_layers_count"
    meta_count.value = str(promoted_count)

    meta_mode = model.metadata_props.add()
    meta_mode.key = "netbender.bend_mode"
    meta_mode.value = str(bend_mode)

    if extra_metadata:
        for k, v in extra_metadata.items():
            if str(k) in ("netbender.bendable", "netbender.promoted_layers_count", "netbender.bend_mode"):
                continue
            m = model.metadata_props.add()
            m.key = str(k)
            m.value = str(v)
            if verbose:
                print(f"  [+] Metadata property: {k} = {v}")

    # Verify model validity
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"Warning: ONNX model check raised an issue: {e}", file=sys.stderr)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)

    # Export high-speed binary sidecar file (.bin) storing initial float weights
    # This enables NetBender C++ to instantly map original layer weights for bending
    if write_sidecar:
        import struct
        base_path, _ = os.path.splitext(output_path)
        sidecar_bin_path = f"{base_path}.bin"
        sidecar_meta = dict(extra_metadata) if extra_metadata else {}
        sidecar_meta["netbender.bendable"] = "1"
        sidecar_meta["netbender.promoted_layers_count"] = str(promoted_count)
        sidecar_meta["netbender.bend_mode"] = str(bend_mode)
        with open(sidecar_bin_path, "wb") as f:
            # NBND v2: Includes embedded JSON metadata string
            f.write(b"NBND\x02\x00\x00\x00")
            json_bytes = json.dumps(sidecar_meta).encode("utf-8")
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(json_bytes)
            f.write(struct.pack("<I", len(promoted_initializers)))
            for init in promoted_initializers:
                arr = numpy_helper.to_array(init).astype("float32")
                name_bytes = init.name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", len(arr.shape)))
                for dim in arr.shape:
                    f.write(struct.pack("<q", dim))
                f.write(struct.pack("<Q", arr.size))
                f.write(arr.tobytes())

    if verbose:
        print(f"\nSuccessfully promoted {promoted_count} parameters to graph inputs.")
        print(f"Saved bendable model to: {output_path}")
        if write_sidecar:
            print(f"Saved initial weights sidecar to: {sidecar_bin_path}")

    return promoted_count


def inspect_onnx_model(input_path: str):
    """Prints all inputs, outputs, and initializers of an ONNX model."""
    print(f"\n--- Inspecting ONNX Model: {input_path} ---")
    model = onnx.load(input_path)
    graph = model.graph

    print(f"Graph Name: {graph.name}")
    print("\n[Graph Inputs]:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  - {inp.name}: shape={shape}, type={inp.type.tensor_type.elem_type}")

    print("\n[Graph Outputs]:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  - {out.name}: shape={shape}, type={out.type.tensor_type.elem_type}")

    print(f"\n[Graph Initializers / Weights]: {len(graph.initializer)} total")
    for i, init in enumerate(graph.initializer[:15]):
        print(f"  - {init.name}: shape={list(init.dims)}")
    if len(graph.initializer) > 15:
        print(f"  ... and {len(graph.initializer) - 15} more.")


def main():
    parser = argparse.ArgumentParser(
        description="NetBender ONNX Parameter Promotion Tool - Prepares ONNX models for real-time audio circuit-bending."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input .onnx file")
    parser.add_argument("-o", "--output", help="Path to output .onnx file (defaults to <name>_bendable.onnx)")
    parser.add_argument("--inspect", action="store_true", help="Inspect model structure without writing changes")
    parser.add_argument(
        "--filter",
        nargs="+",
        help="Optional name substrings to selectively promote (e.g. --filter conv weight bias)",
    )
    parser.add_argument(
        "--strip-initializers",
        action="store_true",
        help="Completely remove promoted weights from initializers (pure dynamic inputs without defaults)",
    )
    parser.add_argument(
        "--bend-mode",
        choices=["delegate", "full", "none"],
        default="full",
        help="Parameter promotion mode: delegate (1D biases/norms/scales), full (all weights), none (default: full)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.inspect:
        inspect_onnx_model(args.input)
        return

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_bendable{ext}"

    promote_initializers_to_inputs(
        input_path=args.input,
        output_path=output_path,
        layer_filter=args.filter,
        keep_initializers_as_defaults=not args.strip_initializers,
        verbose=True,
        bend_mode=args.bend_mode,
    )


if __name__ == "__main__":
    main()
