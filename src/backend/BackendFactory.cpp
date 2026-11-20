#include "BackendFactory.h"
#include "TorchBackend.h"
#include "OnnxBackend.h"
#include "ExecuTorchBackend.h"
#include <algorithm>
#include <cctype>

auto BackendFactory::create_for_file(const std::string& path) -> std::unique_ptr<IBackend> {
    std::string ext;
    const size_t dot_pos = path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = path.substr(dot_pos);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }

    if (ext == ".pte") {
        return std::make_unique<ExecuTorchBackend>();
    }
    if (ext == ".onnx") {
        return std::make_unique<OnnxBackend>();
    }

    // Default to TorchScript backend (.ts or fallback)
    return std::make_unique<TorchBackend>();
}

auto BackendFactory::create_default() -> std::unique_ptr<IBackend> {
    return std::make_unique<TorchBackend>();
}

auto BackendFactory::has_gpu_support() -> bool {
#if defined(__APPLE__)
    return true;
#else
    return torch::hasCUDA() || torch::hasMPS();
#endif
}

auto BackendFactory::gpu_backend_label() -> std::string {
    if (torch::hasCUDA()) {
        return "CUDA";
    }
#if defined(__APPLE__)
    return "MPS/CoreML";
#else
    return torch::hasMPS() ? "MPS" : "CPU";
#endif
}

auto BackendFactory::get_torch_version() -> std::string {
#ifdef TORCH_VERSION
    return TORCH_VERSION;
#else
    return "unknown";
#endif
}
