#pragma once

#include "SamplerTypes.h"

#include <string>

namespace netbender::sampler {

/**
 * Loads, parses, and validates a `.nbsampler.json` bundle manifest from disk.
 * All relative component and tokenizer paths are resolved against the manifest directory.
 */
auto parse_sampler_bundle_manifest(const std::string& manifest_path,
                                   SamplerBundleMetadata& out_meta, std::string& out_error) -> bool;

/**
 * Parses and validates a `.nbsampler.json` JSON payload. When `check_files` is true,
 * verifies that referenced component models, non-empty sidecars, and tokenizer files exist.
 */
auto parse_sampler_bundle_json(const std::string& json_text, const std::string& bundle_dir,
                               SamplerBundleMetadata& out_meta, std::string& out_error,
                               bool check_files = true) -> bool;

/** Converts a `SamplerAlgorithm` enum to its lowercase manifest string name. */
[[nodiscard]] auto sampler_algorithm_to_string(SamplerAlgorithm algo) -> std::string;

/** Parses a sampler algorithm string (`pingpong`, `euler`, `dpmpp`). */
auto parse_sampler_algorithm(const std::string& name, SamplerAlgorithm& out_algo) -> bool;

}  // namespace netbender::sampler
