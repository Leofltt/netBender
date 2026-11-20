#pragma once
#include "IBackend.h"
#include <memory>
#include <string>

/**
 * BackendFactory
 * Responsible for instantiating the appropriate neural inference backend
 * based on model file extension (.pte -> ExecuTorchBackend, .onnx -> OnnxBackend, .ts ->
 * TorchBackend).
 */
class BackendFactory {
public:
    static auto create_for_file(const std::string& path) -> std::unique_ptr<IBackend>;
    static auto create_default() -> std::unique_ptr<IBackend>;
    static auto has_gpu_support() -> bool;
    static auto gpu_backend_label() -> std::string;
    static auto get_torch_version() -> std::string;
};
