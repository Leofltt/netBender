#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "CsoundSessionAdapter.h"

#include <plugin.h>

namespace netbender::csound_fe {

/**
 * Per-CSOUND* handle registry stored via Csound's global variable API and cleaned up
 * automatically on csoundReset() / csoundDestroy() via RegisterResetCallback.
 */
class CsoundRegistry {
public:
    static constexpr const char* kRegistryVarName = "__netbender_csound_registry__";

    static auto get_or_create(CSOUND* cs) -> CsoundRegistry*;
    static auto query(CSOUND* cs) -> CsoundRegistry*;

    auto insert(std::shared_ptr<CsoundSessionAdapter> adapter) -> int32_t;
    [[nodiscard]] auto get(int32_t handle) const -> std::shared_ptr<CsoundSessionAdapter>;
    auto remove(int32_t handle) -> bool;
    auto clear() -> void;

private:
    CsoundRegistry() = default;
    static int32_t on_csound_reset(CSOUND* cs, void* user_data);

    mutable std::mutex m_mutex;
    int32_t m_next_handle{1};
    std::unordered_map<int32_t, std::shared_ptr<CsoundSessionAdapter>> m_handles;
};

[[nodiscard]] auto is_csound_realtime(CSOUND* cs) -> bool;
[[nodiscard]] auto make_csound_file_finder(CSOUND* cs) -> HostFileFinder;
auto register_windows_dll_directory() -> void;

}  // namespace netbender::csound_fe
