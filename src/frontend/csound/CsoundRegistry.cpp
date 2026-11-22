#include "CsoundRegistry.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace netbender::csound_fe {

namespace {

auto starts_with_ci(const char* text, const char* prefix) -> bool {
    if (text == nullptr || prefix == nullptr) {
        return false;
    }
    while (*prefix != '\0') {
        if (*text == '\0') {
            return false;
        }
        const unsigned char a = static_cast<unsigned char>(*text);
        const unsigned char b = static_cast<unsigned char>(*prefix);
        if (std::tolower(a) != std::tolower(b)) {
            return false;
        }
        ++text;
        ++prefix;
    }
    return true;
}

}  // namespace

auto CsoundRegistry::get_or_create(CSOUND* cs) -> CsoundRegistry* {
    if (cs == nullptr) {
        return nullptr;
    }
    auto** slot = static_cast<CsoundRegistry**>(cs->QueryGlobalVariable(cs, kRegistryVarName));
    if (slot == nullptr) {
        if (cs->CreateGlobalVariable(cs, kRegistryVarName, sizeof(CsoundRegistry*)) !=
            CSOUND_SUCCESS) {
            return nullptr;
        }
        slot = static_cast<CsoundRegistry**>(cs->QueryGlobalVariable(cs, kRegistryVarName));
        if (slot == nullptr) {
            return nullptr;
        }
    }
    if (*slot == nullptr) {
        auto* reg = new CsoundRegistry();
        *slot = reg;
        cs->RegisterResetCallback(cs, reg, &CsoundRegistry::on_csound_reset);
    }
    return *slot;
}

auto CsoundRegistry::query(CSOUND* cs) -> CsoundRegistry* {
    if (cs == nullptr) {
        return nullptr;
    }
    auto** slot = static_cast<CsoundRegistry**>(cs->QueryGlobalVariable(cs, kRegistryVarName));
    return (slot != nullptr) ? *slot : nullptr;
}

int32_t CsoundRegistry::on_csound_reset(CSOUND* cs, void* user_data) {
    auto* reg = static_cast<CsoundRegistry*>(user_data);
    if (reg != nullptr) {
        reg->clear();
        delete reg;
    }
    if (cs != nullptr) {
        if (auto** slot =
                static_cast<CsoundRegistry**>(cs->QueryGlobalVariable(cs, kRegistryVarName))) {
            *slot = nullptr;
        }
        cs->DestroyGlobalVariable(cs, kRegistryVarName);
    }
    return OK;
}

auto CsoundRegistry::insert(std::shared_ptr<CsoundSessionAdapter> adapter) -> int32_t {
    if (!adapter) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    const int32_t handle = m_next_handle++;
    m_handles[handle] = std::move(adapter);
    return handle;
}

auto CsoundRegistry::get(int32_t handle) const -> std::shared_ptr<CsoundSessionAdapter> {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_handles.find(handle);
    return (it != m_handles.end()) ? it->second : nullptr;
}

auto CsoundRegistry::remove(int32_t handle) -> bool {
    std::shared_ptr<CsoundSessionAdapter> target;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_handles.find(handle);
        if (it == m_handles.end()) {
            return false;
        }
        target = std::move(it->second);
        m_handles.erase(it);
    }
    if (target) {
        target->close();
    }
    return true;
}

auto CsoundRegistry::clear() -> void {
    std::vector<std::shared_ptr<CsoundSessionAdapter>> adapters;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        adapters.reserve(m_handles.size());
        for (auto& [id, adapter] : m_handles) {
            if (adapter) {
                adapters.push_back(std::move(adapter));
            }
        }
        m_handles.clear();
    }
    for (auto& adapter : adapters) {
        adapter->close();
    }
}

auto is_csound_realtime(CSOUND* cs) -> bool {
    if (cs == nullptr || cs->GetOParms == nullptr) {
        return true;
    }
    const OPARMS* parms = cs->GetOParms(cs);
    if (parms == nullptr) {
        return true;
    }
    if (parms->sfwrite == 0) {
        return false;
    }
    if (parms->outfilename == nullptr || parms->outfilename[0] == '\0') {
        return false;
    }
    return starts_with_ci(parms->outfilename, "dac") ||
           starts_with_ci(parms->outfilename, "devaudio");
}

auto get_csound_io_buffers(CSOUND* cs, int ksmps_val) -> CsoundIoBuffers {
    const int safe_ksmps = std::max(1, ksmps_val);
#if defined(__APPLE__)
    constexpr int kDefaultSwBuf = 128;
    constexpr int kDefaultHwBuf = 512;
#else
    constexpr int kDefaultSwBuf = 256;
    constexpr int kDefaultHwBuf = 1024;
#endif

    const OPARMS* parms = (cs != nullptr && cs->GetOParms != nullptr) ? cs->GetOParms(cs) : nullptr;
    const int raw_b = (parms != nullptr) ? static_cast<int>(parms->outbufsamps) : 0;
    const int raw_B = (parms != nullptr) ? static_cast<int>(parms->oMaxLag) : 0;

    int sw = kDefaultSwBuf;
    if (raw_b < 0) {
        sw = (-raw_b) * safe_ksmps;
    } else if (raw_b > 0) {
        sw = raw_b;
    }
    sw = std::max(sw, safe_ksmps);

    int hw = kDefaultHwBuf;
    if (raw_B < 0) {
        hw = (-raw_B) * safe_ksmps;
    } else if (raw_B > 0) {
        hw = raw_B;
    }
    hw = std::max(hw, sw);

    return CsoundIoBuffers{sw, hw};
}

auto make_csound_file_finder(CSOUND* cs) -> HostFileFinder {
    return [cs](const std::string& candidate) -> std::string {
        if (cs == nullptr || cs->FindInputFile == nullptr || candidate.empty()) {
            return {};
        }
        char* found = cs->FindInputFile(cs, candidate.c_str(), "SSDIR;SFDIR;INCDIR");
        if (found == nullptr) {
            return {};
        }
        std::string result(found);
        if (cs->Free != nullptr) {
            cs->Free(cs, found);
        }
        return result;
    };
}

auto register_windows_dll_directory() -> void {
#if defined(_WIN32)
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;

    HMODULE module = nullptr;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&register_windows_dll_directory), &module) ||
        module == nullptr) {
        return;
    }

    char path_buf[MAX_PATH];
    const DWORD len = GetModuleFileNameA(module, path_buf, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        return;
    }

    std::error_code ec;
    const std::filesystem::path libs_dir =
        std::filesystem::path(path_buf).parent_path() / "netbender_libs";
    if (!std::filesystem::is_directory(libs_dir, ec)) {
        return;
    }

    const std::wstring wlibs = libs_dir.wstring();
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    AddDllDirectory(wlibs.c_str());
#endif
}

}  // namespace netbender::csound_fe
