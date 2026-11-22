#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CsoundSessionAdapter.h"
#include "CsoundRegistry.h"

#include <plugin.h>

namespace {

using netbender::csound_fe::CsoundRegistry;
using netbender::csound_fe::CsoundSessionAdapter;
using netbender::csound_fe::get_csound_io_buffers;
using netbender::csound_fe::is_csound_realtime;
using netbender::csound_fe::make_csound_file_finder;
using netbender::csound_fe::ModelSummary;
using netbender::csound_fe::OpenOptions;
using netbender::csound_fe::resolve_model_path;

auto lookup_adapter(csnd::Csound* csound, cs_float handle_arg, std::string& error_out)
    -> std::shared_ptr<CsoundSessionAdapter> {
    if (csound == nullptr) {
        error_out = "netbender: invalid Csound engine instance\n";
        return nullptr;
    }
    const int32_t handle = static_cast<int32_t>(std::llround(handle_arg));
    auto* reg = CsoundRegistry::query(csound->get_csound());
    if (reg == nullptr) {
        error_out = "netbender: invalid handle " + std::to_string(handle) + " (no models loaded)\n";
        return nullptr;
    }
    auto adapter = reg->get(handle);
    if (!adapter || !adapter->is_open()) {
        error_out = "netbender: handle " + std::to_string(handle) + " is invalid or closed\n";
        return nullptr;
    }
    return adapter;
}

auto fill_string_array(CSOUND* cs, ARRAYDAT* arr, INSDS* ctx, const std::vector<std::string>& items)
    -> int32_t {
    if (cs == nullptr || arr == nullptr) {
        return NOTOK;
    }
    if (tabinit(cs, arr, static_cast<int32_t>(items.size()), ctx) != OK) {
        return NOTOK;
    }
    const CS_TYPE* str_type = (cs->GetType != nullptr) ? cs->GetType(cs, "S") : nullptr;
    for (size_t i = 0; i < items.size(); ++i) {
        STRINGDAT* dst = csound_string_array_element(arr, i);
        if (dst == nullptr) {
            return NOTOK;
        }
        if (str_type != nullptr && str_type->copyValue != nullptr) {
            STRINGDAT src{};
            src.data = const_cast<char*>(items[i].c_str());
            src.size = items[i].size() + 1;
            src.refcount = 0;
            str_type->copyValue(cs, str_type, dst, &src, ctx);
        } else if (cs->Strdup != nullptr) {
            dst->data = cs->Strdup(cs, items[i].c_str());
            dst->size = items[i].size() + 1;
            dst->refcount = 0;
        }
    }
    return OK;
}

auto open_model_handle(csnd::Csound* csound, const std::string& raw_path, const std::string& method,
                       int buf_size, bool use_gpu, int thread_mode, int safety_sentry,
                       int dry_delay, int ksmps_val, double sr_val, int32_t& out_handle,
                       std::string& err_msg) -> bool {
    CSOUND* cs = csound->get_csound();
    const std::string resolved = resolve_model_path(raw_path, make_csound_file_finder(cs));
    if (resolved.empty()) {
        err_msg = "netbender_load: could not find model '" + raw_path +
                  "' (checked cwd, SSDIR/SFDIR/INCDIR, and ~/.netbender/models)\n";
        return false;
    }

    auto* reg = CsoundRegistry::get_or_create(cs);
    if (reg == nullptr) {
        err_msg = "netbender_load: failed to create Csound NetBender registry\n";
        return false;
    }

    const int safe_ksmps = std::max(1, ksmps_val);
    const auto io_bufs = get_csound_io_buffers(cs, safe_ksmps);

    OpenOptions opts;
    opts.model_path = resolved;
    opts.method = method;
    opts.buffer_size = buf_size;
    opts.use_gpu = use_gpu;
    opts.thread_mode = thread_mode;
    opts.safety_sentry = safety_sentry;
    opts.dry_delay = dry_delay;
    opts.host_sample_rate = (sr_val > 0.0) ? sr_val : 44100.0;
    opts.ksmps = safe_ksmps;
    opts.sw_buffer_frames = io_bufs.sw_buffer_frames;
    opts.hw_buffer_frames = io_bufs.hw_buffer_frames;
    opts.max_channels = std::max(2, static_cast<int>(csound->nchnls()));
    opts.is_realtime_output = is_csound_realtime(cs);

    auto adapter = std::make_shared<CsoundSessionAdapter>();
    std::string open_err;
    if (!adapter->open(opts, open_err)) {
        err_msg = open_err + "\n";
        return false;
    }

    const ModelSummary summary = adapter->get_summary();
    if (summary.worker_thread_enabled && summary.sw_buffer_frames > summary.model_buffer_size) {
        csound->warning("netbender_load: Csound software I/O buffer (-b " +
                        std::to_string(summary.sw_buffer_frames) +
                        " frames) exceeds model buffer_size (" +
                        std::to_string(summary.model_buffer_size) +
                        "); consider lowering -b <= ibufsize or raising ibufsize for dropout-free "
                        "async streaming\n");
    }

    out_handle = reg->insert(std::move(adapter));
    if (out_handle <= 0) {
        err_msg = "netbender_load: failed to register handle\n";
        return false;
    }
    return true;
}

template <typename ArrayElemT>
auto validate_audio_array_write(CSOUND* cs, ARRAYDAT* arr, INSDS* ctx, uint32_t ksmps_total,
                                std::string& err_out) -> bool {
    if (arr == nullptr) {
        err_out = "netbender: null output array\n";
        return false;
    }
    if (csound_array_try_prepare_write(cs, arr, ctx) != OK) {
        err_out = "netbender: cannot write shared audio array during performance pass\n";
        return false;
    }
    if (arr->data == nullptr || arr->dimensions < 1 || arr->sizes == nullptr ||
        arr->arrayMemberSize < static_cast<int32_t>(ksmps_total * sizeof(ArrayElemT))) {
        err_out = "netbender: output audio array is uninitialized or malformed\n";
        return false;
    }
    return true;
}

auto array_channel_ptr(ARRAYDAT* arr, int channel_idx) -> cs_float* {
    return reinterpret_cast<cs_float*>(reinterpret_cast<char*>(arr->data) +
                                       static_cast<size_t>(channel_idx) *
                                           static_cast<size_t>(arr->arrayMemberSize));
}

auto array_channel_const_ptr(const ARRAYDAT* arr, int channel_idx) -> const cs_float* {
    return reinterpret_cast<const cs_float*>(reinterpret_cast<const char*>(arr->data) +
                                             static_cast<size_t>(channel_idx) *
                                                 static_cast<size_t>(arr->arrayMemberSize));
}

auto resolve_layer_by_name(const CsoundSessionAdapter& adapter, const STRINGDAT& str,
                           const char* opcode_name, int& out_idx, std::string& err_out) -> bool {
    const std::string layer_name = (str.data != nullptr) ? std::string(str.data) : "";
    if (layer_name.empty()) {
        err_out = std::string(opcode_name) + ": layer name cannot be empty\n";
        return false;
    }
    const int idx = adapter.find_layer_index(layer_name);
    if (idx < 0) {
        err_out = std::string(opcode_name) + ": unknown bendable layer '" + layer_name +
                  "' (model has " + std::to_string(adapter.get_layer_count()) +
                  " bendable layers)\n";
        return false;
    }
    out_idx = idx;
    return true;
}

auto resolve_layer_by_index(const CsoundSessionAdapter& adapter, cs_float idx_arg,
                            const char* opcode_name, int& out_idx, std::string& err_out) -> bool {
    const int count = adapter.get_layer_count();
    const int idx = static_cast<int>(std::llround(idx_arg));
    if (idx < 0 || idx >= count) {
        err_out = std::string(opcode_name) + ": layer index " + std::to_string(idx) +
                  " out of range [0, " + std::to_string(count - 1) + "]\n";
        return false;
    }
    out_idx = idx;
    return true;
}

auto numeric_array_elem_ptr(ARRAYDAT* arr, size_t idx) -> cs_float* {
    return reinterpret_cast<cs_float*>(reinterpret_cast<char*>(arr->data) +
                                       idx * static_cast<size_t>(arr->arrayMemberSize));
}

auto numeric_array_const_elem_ptr(const ARRAYDAT* arr, size_t idx) -> const cs_float* {
    return reinterpret_cast<const cs_float*>(reinterpret_cast<const char*>(arr->data) +
                                             idx * static_cast<size_t>(arr->arrayMemberSize));
}

auto numeric_array_length(const ARRAYDAT* arr) -> size_t {
    if (arr == nullptr || arr->data == nullptr || arr->dimensions < 1 || arr->sizes == nullptr ||
        arr->sizes[0] < 0 || arr->arrayMemberSize < static_cast<int32_t>(sizeof(cs_float))) {
        return 0;
    }
    return static_cast<size_t>(arr->sizes[0]);
}

auto read_numeric_array_to_floats(const ARRAYDAT* arr, std::vector<float>& dst) -> bool {
    if (arr == nullptr || arr->data == nullptr || arr->dimensions < 1 || arr->sizes == nullptr ||
        arr->sizes[0] < 0 || arr->arrayMemberSize < static_cast<int32_t>(sizeof(cs_float))) {
        return false;
    }
    const size_t len = static_cast<size_t>(arr->sizes[0]);
    if (dst.size() != len) {
        dst.resize(len);
    }
    for (size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<float>(*numeric_array_const_elem_ptr(arr, i));
    }
    return true;
}

// =============================================================================
// netbender_load
// =============================================================================

struct NetBenderLoadWithMethod : csnd::Plugin<1, 7> {
    int32_t init() {
        const STRINGDAT& path_str = inargs.str_data(0);
        const STRINGDAT& meth_str = inargs.str_data(1);
        const std::string raw_path = (path_str.data != nullptr) ? std::string(path_str.data) : "";
        const std::string method = (meth_str.data != nullptr) ? std::string(meth_str.data) : "";

        int32_t handle = -1;
        std::string err;
        if (!open_model_handle(csound, raw_path, method, static_cast<int>(inargs[2]),
                               inargs[3] > cs_float{0.5}, static_cast<int>(inargs[4]),
                               static_cast<int>(inargs[5]), static_cast<int>(inargs[6]),
                               static_cast<int>(ksmps()), static_cast<double>(sr()), handle, err)) {
            return csound->init_error(err);
        }
        outargs[0] = static_cast<cs_float>(handle);
        return OK;
    }
};

struct NetBenderLoadDefault : csnd::Plugin<1, 6> {
    int32_t init() {
        const STRINGDAT& path_str = inargs.str_data(0);
        const std::string raw_path = (path_str.data != nullptr) ? std::string(path_str.data) : "";

        int32_t handle = -1;
        std::string err;
        if (!open_model_handle(csound, raw_path, "forward", static_cast<int>(inargs[1]),
                               inargs[2] > cs_float{0.5}, static_cast<int>(inargs[3]),
                               static_cast<int>(inargs[4]), static_cast<int>(inargs[5]),
                               static_cast<int>(ksmps()), static_cast<double>(sr()), handle, err)) {
            return csound->init_error(err);
        }
        outargs[0] = static_cast<cs_float>(handle);
        return OK;
    }
};

// =============================================================================
// netbender_free
// =============================================================================

struct NetBenderFree : csnd::InPlug<1> {
    int32_t init() {
        const int32_t handle = static_cast<int32_t>(std::llround(args[0]));
        auto* reg = CsoundRegistry::query(csound->get_csound());
        if (reg == nullptr || !reg->remove(handle)) {
            csound->warning("netbender_free: handle " + std::to_string(handle) +
                            " not found or already freed\n");
        }
        return OK;
    }
};

// =============================================================================
// netbender_reset
// =============================================================================

struct NetBenderResetI : csnd::InPlug<1> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        adapter->reset();
        return OK;
    }
};

struct NetBenderResetK : csnd::InPlug<2> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        cs_float prev_trig{0};
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->prev_trig = 0;
        return OK;
    }

    int32_t kperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        const cs_float cur = args[1];
        if (impl->prev_trig <= cs_float{0} && cur > cs_float{0}) {
            impl->adapter->reset();
        }
        impl->prev_trig = cur;
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

// =============================================================================
// netbender_process (mono, stereo, a[])
// =============================================================================

struct NetBenderProcessMono : csnd::Plugin<1, 4> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        return OK;
    }

    int32_t aperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        if (nsmps <= offset) {
            return OK;
        }
        const int frames = static_cast<int>(nsmps - offset);
        const cs_float* in_ptr = inargs(1) + offset;
        cs_float* out_ptr = outargs(0) + offset;
        impl->adapter->process(&in_ptr, 1, &out_ptr, 1, frames,
                               static_cast<double>(csound->_0dbfs()), static_cast<float>(inargs[2]),
                               inargs[3] > cs_float{0.5});
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderProcessStereo : csnd::Plugin<2, 5> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        return OK;
    }

    int32_t aperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        if (nsmps <= offset) {
            return OK;
        }
        const int frames = static_cast<int>(nsmps - offset);
        const cs_float* in_ptrs[2] = {inargs(1) + offset, inargs(2) + offset};
        cs_float* out_ptrs[2] = {outargs(0) + offset, outargs(1) + offset};
        impl->adapter->process(in_ptrs, 2, out_ptrs, 2, frames,
                               static_cast<double>(csound->_0dbfs()), static_cast<float>(inargs[3]),
                               inargs[4] > cs_float{0.5});
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderProcessArray : csnd::Plugin<1, 4> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        std::vector<const cs_float*> in_ptrs;
        std::vector<cs_float*> out_ptrs;
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }

        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        auto* in_arr = reinterpret_cast<const ARRAYDAT*>(inargs(1));
        const int out_ch = std::max(1, adapter->get_out_channels());
        const int in_ch = (in_arr != nullptr && in_arr->dimensions >= 1 && in_arr->sizes != nullptr)
                              ? std::max(0, in_arr->sizes[0])
                              : 0;

        if (tabinit(csound->get_csound(), out_arr, out_ch, insdshead) != OK) {
            delete impl;
            impl = nullptr;
            return csound->init_error("netbender_process: failed to allocate output audio array\n");
        }

        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->in_ptrs.assign(
            static_cast<size_t>(std::max({64, in_ch, impl->adapter->get_in_channels()})), nullptr);
        impl->out_ptrs.assign(static_cast<size_t>(std::max(64, out_ch)), nullptr);
        return OK;
    }

    int32_t aperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        auto* in_arr = reinterpret_cast<const ARRAYDAT*>(inargs(1));
        const uint32_t ksmps_total = insdshead->ksmps;

        std::string arr_err;
        if (!validate_audio_array_write<cs_float>(csound->get_csound(), out_arr, insdshead,
                                                  ksmps_total, arr_err)) {
            return csound->perf_error(arr_err, this);
        }

        const int out_ch = std::min(out_arr->sizes[0], static_cast<int>(impl->out_ptrs.size()));
        for (int c = 0; c < out_ch; ++c) {
            cs_float* ch = array_channel_ptr(out_arr, c);
            if (offset > 0) {
                std::fill(ch, ch + offset, cs_float{0});
            }
            if (nsmps < ksmps_total) {
                std::fill(ch + nsmps, ch + ksmps_total, cs_float{0});
            }
            impl->out_ptrs[static_cast<size_t>(c)] = ch + offset;
        }

        if (nsmps <= offset || out_ch <= 0) {
            return OK;
        }

        int active_in = 0;
        if (in_arr != nullptr && in_arr->data != nullptr && in_arr->dimensions >= 1 &&
            in_arr->sizes != nullptr &&
            in_arr->arrayMemberSize >= static_cast<int32_t>(ksmps_total * sizeof(cs_float))) {
            active_in = std::min(in_arr->sizes[0], static_cast<int>(impl->in_ptrs.size()));
            for (int c = 0; c < active_in; ++c) {
                impl->in_ptrs[static_cast<size_t>(c)] = array_channel_const_ptr(in_arr, c) + offset;
            }
        }

        const int frames = static_cast<int>(nsmps - offset);
        impl->adapter->process(impl->in_ptrs.data(), active_in, impl->out_ptrs.data(), out_ch,
                               frames, static_cast<double>(csound->_0dbfs()),
                               static_cast<float>(inargs[2]), inargs[3] > cs_float{0.5});
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

// =============================================================================
// netbender_gen (mono, stereo, a[])
// =============================================================================

struct NetBenderGenMono : csnd::Plugin<1, 2> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        return OK;
    }

    int32_t aperf() {
        if (impl == nullptr || !impl->adapter || nsmps <= offset) {
            return OK;
        }
        const int frames = static_cast<int>(nsmps - offset);
        cs_float* out_ptr = outargs(0) + offset;
        impl->adapter->generate(&out_ptr, 1, frames, static_cast<double>(csound->_0dbfs()),
                                inargs[1] > cs_float{0.5});
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderGenStereo : csnd::Plugin<2, 2> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        return OK;
    }

    int32_t aperf() {
        if (impl == nullptr || !impl->adapter || nsmps <= offset) {
            return OK;
        }
        const int frames = static_cast<int>(nsmps - offset);
        cs_float* out_ptrs[2] = {outargs(0) + offset, outargs(1) + offset};
        impl->adapter->generate(out_ptrs, 2, frames, static_cast<double>(csound->_0dbfs()),
                                inargs[1] > cs_float{0.5});
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderGenArray : csnd::Plugin<1, 2> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        std::vector<cs_float*> out_ptrs;
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }

        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        const int out_ch = std::max(1, adapter->get_out_channels());
        if (tabinit(csound->get_csound(), out_arr, out_ch, insdshead) != OK) {
            delete impl;
            impl = nullptr;
            return csound->init_error("netbender_gen: failed to allocate output audio array\n");
        }

        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->out_ptrs.assign(static_cast<size_t>(std::max(64, out_ch)), nullptr);
        return OK;
    }

    int32_t aperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        const uint32_t ksmps_total = insdshead->ksmps;

        std::string arr_err;
        if (!validate_audio_array_write<cs_float>(csound->get_csound(), out_arr, insdshead,
                                                  ksmps_total, arr_err)) {
            return csound->perf_error(arr_err, this);
        }

        const int out_ch = std::min(out_arr->sizes[0], static_cast<int>(impl->out_ptrs.size()));
        for (int c = 0; c < out_ch; ++c) {
            cs_float* ch = array_channel_ptr(out_arr, c);
            if (offset > 0) {
                std::fill(ch, ch + offset, cs_float{0});
            }
            if (nsmps < ksmps_total) {
                std::fill(ch + nsmps, ch + ksmps_total, cs_float{0});
            }
            impl->out_ptrs[static_cast<size_t>(c)] = ch + offset;
        }

        if (nsmps <= offset || out_ch <= 0) {
            return OK;
        }

        const int frames = static_cast<int>(nsmps - offset);
        impl->adapter->generate(impl->out_ptrs.data(), out_ch, frames,
                                static_cast<double>(csound->_0dbfs()), inargs[1] > cs_float{0.5});
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

// =============================================================================
// Introspection: netbender_info, netbender_dump, netbender_layers, netbender_methods
// =============================================================================

struct NetBenderInfo : csnd::Plugin<5, 1> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        const ModelSummary s = adapter->get_summary();
        outargs[0] = static_cast<cs_float>(s.in_channels);
        outargs[1] = static_cast<cs_float>(s.out_channels);
        outargs[2] = static_cast<cs_float>(s.model_sample_rate);
        outargs[3] = static_cast<cs_float>(s.host_latency_samples);
        outargs[4] = static_cast<cs_float>(s.layers.size());
        return OK;
    }
};

struct NetBenderDump : csnd::InPlug<1> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        csound->message(adapter->format_dump());
        return OK;
    }
};

struct NetBenderLayers : csnd::Plugin<1, 1> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        const ModelSummary s = adapter->get_summary();
        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        if (fill_string_array(csound->get_csound(), out_arr, insdshead, s.layers) != OK) {
            return csound->init_error("netbender_layers: failed to populate string array\n");
        }
        return OK;
    }
};

struct NetBenderMethods : csnd::Plugin<1, 1> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        const ModelSummary s = adapter->get_summary();
        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        if (fill_string_array(csound->get_csound(), out_arr, insdshead, s.methods) != OK) {
            return csound->init_error("netbender_methods: failed to populate string array\n");
        }
        return OK;
    }
};

// =============================================================================
// Phase 2: Live Bending, Short-Circuit, Trigger, Bridge, Latent, Sentry, Weights
// =============================================================================

struct BendImpl {
    std::shared_ptr<CsoundSessionAdapter> adapter;
    int layer_idx{-1};
    bool has_sent{false};
    float last_scale{1.0f};
    float last_offset{0.0f};
    float last_heat{0.0f};
    float last_memory{0.8f};
    int last_drift_mode{1};
    bool last_frozen{false};
};

auto run_bend_kperf(BendImpl* impl, const csnd::Param<8>& args) -> void {
    if (impl == nullptr || !impl->adapter || impl->layer_idx < 0) {
        return;
    }
    const float scale = static_cast<float>(args[2]);
    const float offset = static_cast<float>(args[3]);
    const float heat = std::max(0.0f, static_cast<float>(args[4]));
    const float raw_mem = static_cast<float>(args[5]);
    const float memory = (raw_mem < 0.0f) ? 0.8f : std::clamp(raw_mem, 0.0f, 1.0f);
    const int raw_drift = static_cast<int>(std::llround(args[6]));
    const int drift_mode = (raw_drift < 0) ? 1 : std::clamp(raw_drift, 0, 2);
    const bool frozen = (args[7] > cs_float{0.5});

    if (!impl->has_sent || std::abs(scale - impl->last_scale) > 1e-6f ||
        std::abs(offset - impl->last_offset) > 1e-6f || std::abs(heat - impl->last_heat) > 1e-6f ||
        std::abs(memory - impl->last_memory) > 1e-6f || drift_mode != impl->last_drift_mode ||
        frozen != impl->last_frozen) {
        impl->adapter->set_layer_bend(impl->layer_idx, scale, offset, heat, memory, drift_mode,
                                      frozen);
        impl->last_scale = scale;
        impl->last_offset = offset;
        impl->last_heat = heat;
        impl->last_memory = memory;
        impl->last_drift_mode = drift_mode;
        impl->last_frozen = frozen;
        impl->has_sent = true;
    }
}

struct NetBenderBendByName : csnd::InPlug<8> {
    BendImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_name(*adapter, args.str_data(1), "netbender_bend", layer_idx, err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new BendImpl();
        }
        impl->adapter = std::move(adapter);
        impl->layer_idx = layer_idx;
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        run_bend_kperf(impl, args);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderBendByIndex : csnd::InPlug<8> {
    BendImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_index(*adapter, args[1], "netbender_bend", layer_idx, err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new BendImpl();
        }
        impl->adapter = std::move(adapter);
        impl->layer_idx = layer_idx;
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        run_bend_kperf(impl, args);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderShort : csnd::InPlug<2> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        bool has_sent{false};
        bool last_active{false};
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        const bool active = (args[1] > cs_float{0.5});
        if (!impl->has_sent || active != impl->last_active) {
            impl->adapter->set_short_circuit(active);
            impl->last_active = active;
            impl->has_sent = true;
        }
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

auto apply_trigger_args(CsoundSessionAdapter& adapter, const csnd::Param<5>& args) -> void {
    const int mode = std::clamp(static_cast<int>(std::llround(args[1])), 0, 2);
    const float att = static_cast<float>(args[2]);
    const float rel = static_cast<float>(args[3]);
    const float thresh = static_cast<float>(args[4]);
    adapter.set_trigger(mode, att, rel, thresh);
}

struct NetBenderTriggerI : csnd::InPlug<5> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        apply_trigger_args(*adapter, args);
        return OK;
    }
};

struct NetBenderTriggerK : csnd::InPlug<5> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        bool has_sent{false};
        int last_mode{0};
        float last_att{-1.0f};
        float last_rel{-1.0f};
        float last_thresh{-1.0f};
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        const int mode = std::clamp(static_cast<int>(std::llround(args[1])), 0, 2);
        const float att = static_cast<float>(args[2]);
        const float rel = static_cast<float>(args[3]);
        const float thresh = static_cast<float>(args[4]);
        if (!impl->has_sent || mode != impl->last_mode || std::abs(att - impl->last_att) > 1e-6f ||
            std::abs(rel - impl->last_rel) > 1e-6f ||
            std::abs(thresh - impl->last_thresh) > 1e-6f) {
            impl->adapter->set_trigger(mode, att, rel, thresh);
            impl->last_mode = mode;
            impl->last_att = att;
            impl->last_rel = rel;
            impl->last_thresh = thresh;
            impl->has_sent = true;
        }
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct BridgeImpl {
    std::shared_ptr<CsoundSessionAdapter> adapter;
    int target_idx{-1};
    int source_idx{-1};
    bool has_sent{false};
    float last_depth{0.0f};
};

auto run_bridge_kperf(BridgeImpl* impl, const csnd::Param<4>& args) -> void {
    if (impl == nullptr || !impl->adapter || impl->target_idx < 0) {
        return;
    }
    const float depth = std::clamp(static_cast<float>(args[3]), 0.0f, 1.0f);
    if (!impl->has_sent || std::abs(depth - impl->last_depth) > 1e-6f) {
        impl->adapter->set_layer_bridge(impl->target_idx, impl->source_idx, depth);
        impl->last_depth = depth;
        impl->has_sent = true;
    }
}

struct NetBenderBridgeByName : csnd::InPlug<4> {
    BridgeImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int target_idx = -1;
        if (!resolve_layer_by_name(*adapter, args.str_data(1), "netbender_bridge", target_idx,
                                   err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int source_idx = -1;
        const STRINGDAT& src_str = args.str_data(2);
        const std::string src_name = (src_str.data != nullptr) ? std::string(src_str.data) : "";
        if (!src_name.empty()) {
            if (!resolve_layer_by_name(*adapter, src_str, "netbender_bridge", source_idx, err)) {
                delete impl;
                impl = nullptr;
                return csound->init_error(err);
            }
        }
        if (impl == nullptr) {
            impl = new BridgeImpl();
        }
        impl->adapter = std::move(adapter);
        impl->target_idx = target_idx;
        impl->source_idx = source_idx;
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        run_bridge_kperf(impl, args);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderBridgeByIndex : csnd::InPlug<4> {
    BridgeImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int target_idx = -1;
        if (!resolve_layer_by_index(*adapter, args[1], "netbender_bridge", target_idx, err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int source_idx = -1;
        const int raw_src = static_cast<int>(std::llround(args[2]));
        if (raw_src >= 0) {
            if (!resolve_layer_by_index(*adapter, args[2], "netbender_bridge", source_idx, err)) {
                delete impl;
                impl = nullptr;
                return csound->init_error(err);
            }
        }
        if (impl == nullptr) {
            impl = new BridgeImpl();
        }
        impl->adapter = std::move(adapter);
        impl->target_idx = target_idx;
        impl->source_idx = source_idx;
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        run_bridge_kperf(impl, args);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

auto apply_latent_args(CsoundSessionAdapter& adapter, const csnd::Param<7>& args) -> void {
    const bool enabled = (args[1] > cs_float{0.5});
    const float x = static_cast<float>(args[2]);
    const float y = static_cast<float>(args[3]);
    const float depth = static_cast<float>(args[4]);
    const int mode = std::clamp(static_cast<int>(std::llround(args[5])), 0, 2);
    const float slew = static_cast<float>(args[6]);
    adapter.set_latent(enabled, x, y, depth, mode, slew);
}

struct NetBenderLatentI : csnd::InPlug<7> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        apply_latent_args(*adapter, args);
        return OK;
    }
};

struct NetBenderLatentK : csnd::InPlug<7> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        bool has_sent{false};
        bool last_enabled{false};
        float last_x{0.0f};
        float last_y{0.0f};
        float last_depth{0.0f};
        int last_mode{0};
        float last_slew{0.0f};
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->has_sent = false;
        return OK;
    }

    int32_t kperf() {
        if (impl == nullptr || !impl->adapter) {
            return OK;
        }
        const bool enabled = (args[1] > cs_float{0.5});
        const float x = static_cast<float>(args[2]);
        const float y = static_cast<float>(args[3]);
        const float depth = static_cast<float>(args[4]);
        const int mode = std::clamp(static_cast<int>(std::llround(args[5])), 0, 2);
        const float slew = static_cast<float>(args[6]);

        if (!impl->has_sent || enabled != impl->last_enabled ||
            std::abs(x - impl->last_x) > 1e-6f || std::abs(y - impl->last_y) > 1e-6f ||
            std::abs(depth - impl->last_depth) > 1e-6f || mode != impl->last_mode ||
            std::abs(slew - impl->last_slew) > 1e-6f) {
            impl->adapter->set_latent(enabled, x, y, depth, mode, slew);
            impl->last_enabled = enabled;
            impl->last_x = x;
            impl->last_y = y;
            impl->last_depth = depth;
            impl->last_mode = mode;
            impl->last_slew = slew;
            impl->has_sent = true;
        }
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderSentry : csnd::Plugin<1, 3> {
    struct Impl {
        std::shared_ptr<CsoundSessionAdapter> adapter;
        bool has_set_auto{false};
        bool last_auto_reset{false};
        cs_float prev_trig{0};
    };
    Impl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new Impl();
        }
        impl->adapter = std::move(adapter);
        impl->has_set_auto = false;
        impl->last_auto_reset = false;
        impl->prev_trig = 0;
        outargs[0] = impl->adapter->get_sentry_blown() ? cs_float{1} : cs_float{0};
        return OK;
    }

    int32_t kperf() {
        if (impl == nullptr || !impl->adapter) {
            outargs[0] = cs_float{0};
            return OK;
        }
        const float raw_auto = static_cast<float>(inargs[1]);
        if (raw_auto >= 0.0f) {
            const bool auto_reset = (raw_auto > 0.5f);
            if (!impl->has_set_auto || auto_reset != impl->last_auto_reset) {
                impl->adapter->set_sentry_auto_reset(auto_reset);
                impl->last_auto_reset = auto_reset;
                impl->has_set_auto = true;
            }
        }
        const cs_float cur_trig = inargs[2];
        if (impl->prev_trig <= cs_float{0} && cur_trig > cs_float{0}) {
            impl->adapter->request_fuse_reset();
        }
        impl->prev_trig = cur_trig;
        outargs[0] = impl->adapter->get_sentry_blown() ? cs_float{1} : cs_float{0};
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

auto populate_weights_out(CSOUND* cs, ARRAYDAT* out_arr, INSDS* ctx,
                          const CsoundSessionAdapter& adapter, int layer_idx, bool original,
                          std::string& err_out) -> bool {
    const std::vector<float> weights = adapter.get_layer_weights(layer_idx, original);
    if (tabinit(cs, out_arr, static_cast<int32_t>(weights.size()), ctx) != OK ||
        out_arr->arrayMemberSize < static_cast<int32_t>(sizeof(cs_float)) ||
        (weights.size() > 0 && out_arr->data == nullptr)) {
        err_out = "netbender_getweights: failed to allocate output numeric array\n";
        return false;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        *numeric_array_elem_ptr(out_arr, i) = static_cast<cs_float>(weights[i]);
    }
    return true;
}

struct NetBenderGetWeightsByName : csnd::Plugin<1, 3> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_name(*adapter, inargs.str_data(1), "netbender_getweights", layer_idx,
                                   err)) {
            return csound->init_error(err);
        }
        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        if (!populate_weights_out(csound->get_csound(), out_arr, insdshead, *adapter, layer_idx,
                                  inargs[2] > cs_float{0.5}, err)) {
            return csound->init_error(err);
        }
        return OK;
    }
};

struct NetBenderGetWeightsByIndex : csnd::Plugin<1, 3> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, inargs[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_index(*adapter, inargs[1], "netbender_getweights", layer_idx, err)) {
            return csound->init_error(err);
        }
        auto* out_arr = reinterpret_cast<ARRAYDAT*>(outargs(0));
        if (!populate_weights_out(csound->get_csound(), out_arr, insdshead, *adapter, layer_idx,
                                  inargs[2] > cs_float{0.5}, err)) {
            return csound->init_error(err);
        }
        return OK;
    }
};

auto set_weights_immediate_helper(CsoundSessionAdapter& adapter, int layer_idx, const ARRAYDAT* arr,
                                  std::string& err_out) -> bool {
    const size_t expected = adapter.get_layer_weight_count(layer_idx);
    const size_t actual = numeric_array_length(arr);
    if (actual != expected || expected == 0) {
        err_out = "netbender_setweights: array length (" + std::to_string(actual) +
                  ") does not match layer weight count (" + std::to_string(expected) + ")\n";
        return false;
    }
    std::vector<float> weights(expected);
    if (!read_numeric_array_to_floats(arr, weights) ||
        !adapter.set_layer_weights_immediate(layer_idx, weights.data(), weights.size())) {
        err_out = "netbender_setweights: failed to apply weight array to layer\n";
        return false;
    }
    return true;
}

struct NetBenderSetWeightsIByName : csnd::InPlug<3> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_name(*adapter, args.str_data(1), "netbender_setweights", layer_idx,
                                   err)) {
            return csound->init_error(err);
        }
        if (!set_weights_immediate_helper(*adapter, layer_idx,
                                          reinterpret_cast<const ARRAYDAT*>(args(2)), err)) {
            return csound->init_error(err);
        }
        return OK;
    }
};

struct NetBenderSetWeightsIByIndex : csnd::InPlug<3> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_index(*adapter, args[1], "netbender_setweights", layer_idx, err)) {
            return csound->init_error(err);
        }
        if (!set_weights_immediate_helper(*adapter, layer_idx,
                                          reinterpret_cast<const ARRAYDAT*>(args(2)), err)) {
            return csound->init_error(err);
        }
        return OK;
    }
};

struct SetWeightsKImpl {
    std::shared_ptr<CsoundSessionAdapter> adapter;
    int layer_idx{-1};
    size_t expected_count{0};
    std::vector<float> scratch_weights;
    cs_float prev_trig{0};
    bool warned_size_mismatch{false};
};

auto init_set_weights_k_impl(SetWeightsKImpl*& impl, std::shared_ptr<CsoundSessionAdapter> adapter,
                             int layer_idx, csnd::Csound* csound) -> int32_t {
    if (!adapter->prepare_layer_weight_staging(layer_idx)) {
        delete impl;
        impl = nullptr;
        return csound->init_error(
            "netbender_setweights: failed to prepare real-time weight staging buffer\n");
    }
    const size_t expected = adapter->get_layer_weight_count(layer_idx);
    if (impl == nullptr) {
        impl = new SetWeightsKImpl();
    }
    impl->adapter = std::move(adapter);
    impl->layer_idx = layer_idx;
    impl->expected_count = expected;
    impl->scratch_weights.assign(expected, 0.0f);
    impl->prev_trig = 0;
    impl->warned_size_mismatch = false;
    return OK;
}

auto run_set_weights_kperf(SetWeightsKImpl* impl, const ARRAYDAT* arr, cs_float cur_trig,
                           csnd::Csound* csound) -> void {
    if (impl == nullptr || !impl->adapter || impl->layer_idx < 0) {
        return;
    }
    if (impl->prev_trig <= cs_float{0} && cur_trig > cs_float{0}) {
        const size_t actual = numeric_array_length(arr);
        if (actual != impl->expected_count || impl->expected_count == 0) {
            if (!impl->warned_size_mismatch) {
                csound->warning("netbender_setweights: array length (" + std::to_string(actual) +
                                ") does not match layer weight count (" +
                                std::to_string(impl->expected_count) + ")\n");
                impl->warned_size_mismatch = true;
            }
        } else if (read_numeric_array_to_floats(arr, impl->scratch_weights)) {
            impl->adapter->set_layer_weights_rt(impl->layer_idx, impl->scratch_weights.data(),
                                                impl->scratch_weights.size());
        }
    }
    impl->prev_trig = cur_trig;
}

struct NetBenderSetWeightsKByName : csnd::InPlug<4> {
    SetWeightsKImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_name(*adapter, args.str_data(1), "netbender_setweights", layer_idx,
                                   err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        return init_set_weights_k_impl(impl, std::move(adapter), layer_idx, csound);
    }

    int32_t kperf() {
        run_set_weights_kperf(impl, reinterpret_cast<const ARRAYDAT*>(args(2)), args[3], csound);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderSetWeightsKByIndex : csnd::InPlug<4> {
    SetWeightsKImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_index(*adapter, args[1], "netbender_setweights", layer_idx, err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        return init_set_weights_k_impl(impl, std::move(adapter), layer_idx, csound);
    }

    int32_t kperf() {
        run_set_weights_kperf(impl, reinterpret_cast<const ARRAYDAT*>(args(2)), args[3], csound);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderResetWeightsAllI : csnd::InPlug<1> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        adapter->reset_all_weights_immediate();
        return OK;
    }
};

struct NetBenderResetWeightsLayerI : csnd::InPlug<2> {
    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_name(*adapter, args.str_data(1), "netbender_resetweights", layer_idx,
                                   err)) {
            return csound->init_error(err);
        }
        adapter->reset_layer_weights_immediate(layer_idx);
        return OK;
    }
};

struct ResetWeightsKImpl {
    std::shared_ptr<CsoundSessionAdapter> adapter;
    int layer_idx{-1};  // < 0 resets all layers
    cs_float prev_trig{0};
};

auto run_reset_weights_kperf(ResetWeightsKImpl* impl, cs_float cur_trig) -> void {
    if (impl == nullptr || !impl->adapter) {
        return;
    }
    if (impl->prev_trig <= cs_float{0} && cur_trig > cs_float{0}) {
        if (impl->layer_idx < 0) {
            impl->adapter->request_reset_all_weights();
        } else {
            impl->adapter->request_reset_layer_weights(impl->layer_idx);
        }
    }
    impl->prev_trig = cur_trig;
}

struct NetBenderResetWeightsAllK : csnd::InPlug<2> {
    ResetWeightsKImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new ResetWeightsKImpl();
        }
        impl->adapter = std::move(adapter);
        impl->layer_idx = -1;
        impl->prev_trig = 0;
        return OK;
    }

    int32_t kperf() {
        run_reset_weights_kperf(impl, args[1]);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderResetWeightsLayerKByName : csnd::InPlug<3> {
    ResetWeightsKImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int layer_idx = -1;
        if (!resolve_layer_by_name(*adapter, args.str_data(1), "netbender_resetweights", layer_idx,
                                   err)) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        if (impl == nullptr) {
            impl = new ResetWeightsKImpl();
        }
        impl->adapter = std::move(adapter);
        impl->layer_idx = layer_idx;
        impl->prev_trig = 0;
        return OK;
    }

    int32_t kperf() {
        run_reset_weights_kperf(impl, args[2]);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

struct NetBenderResetWeightsLayerKByIndex : csnd::InPlug<3> {
    ResetWeightsKImpl* impl;

    int32_t init() {
        std::string err;
        auto adapter = lookup_adapter(csound, args[0], err);
        if (!adapter) {
            delete impl;
            impl = nullptr;
            return csound->init_error(err);
        }
        int layer_idx = -1;
        const int raw_idx = static_cast<int>(std::llround(args[1]));
        if (raw_idx >= 0) {
            if (!resolve_layer_by_index(*adapter, args[1], "netbender_resetweights", layer_idx,
                                        err)) {
                delete impl;
                impl = nullptr;
                return csound->init_error(err);
            }
        }
        if (impl == nullptr) {
            impl = new ResetWeightsKImpl();
        }
        impl->adapter = std::move(adapter);
        impl->layer_idx = layer_idx;
        impl->prev_trig = 0;
        return OK;
    }

    int32_t kperf() {
        run_reset_weights_kperf(impl, args[2]);
        return OK;
    }

    int32_t deinit() {
        delete impl;
        impl = nullptr;
        return OK;
    }
};

}  // namespace

#include <modload.h>

void csnd::on_load(csnd::Csound* csound) {
    netbender::csound_fe::register_windows_dll_directory();

    csnd::plugin<NetBenderLoadWithMethod>(csound, "netbender_load", "i", "SSjojjj",
                                          csnd::thread::i);
    csnd::plugin<NetBenderLoadDefault>(csound, "netbender_load", "i", "Sjojjj", csnd::thread::i);
    csnd::plugin<NetBenderFree>(csound, "netbender_free", "", "i", csnd::thread::i);

    csnd::plugin<NetBenderResetI>(csound, "netbender_reset", "", "i", csnd::thread::i);
    csnd::plugin<NetBenderResetK>(csound, "netbender_reset", "", "ik", csnd::thread::ik);

    csnd::plugin<NetBenderProcessMono>(csound, "netbender_process", "a", "iaPP", csnd::thread::ia);
    csnd::plugin<NetBenderProcessStereo>(csound, "netbender_process", "aa", "iaaPP",
                                         csnd::thread::ia);
    csnd::plugin<NetBenderProcessArray>(csound, "netbender_process", "a[]", "ia[]PP",
                                        csnd::thread::ia);

    csnd::plugin<NetBenderGenMono>(csound, "netbender_gen", "a", "iP", csnd::thread::ia);
    csnd::plugin<NetBenderGenStereo>(csound, "netbender_gen", "aa", "iP", csnd::thread::ia);
    csnd::plugin<NetBenderGenArray>(csound, "netbender_gen", "a[]", "iP", csnd::thread::ia);

    csnd::plugin<NetBenderInfo>(csound, "netbender_info", "iiiii", "i", csnd::thread::i);
    csnd::plugin<NetBenderDump>(csound, "netbender_dump", "", "i", csnd::thread::i);
    csnd::plugin<NetBenderLayers>(csound, "netbender_layers", "S[]", "i", csnd::thread::i);
    csnd::plugin<NetBenderMethods>(csound, "netbender_methods", "S[]", "i", csnd::thread::i);

    // Phase 2: Live circuit bending, triggers, latent terrain, sentry, and raw weight access
    csnd::plugin<NetBenderBendByName>(csound, "netbender_bend", "", "iSkkOJJO", csnd::thread::ik);
    csnd::plugin<NetBenderBendByIndex>(csound, "netbender_bend", "", "iikkOJJO", csnd::thread::ik);
    csnd::plugin<NetBenderShort>(csound, "netbender_short", "", "ik", csnd::thread::ik);

    csnd::plugin<NetBenderTriggerI>(csound, "netbender_trigger", "", "iijjj", csnd::thread::i);
    csnd::plugin<NetBenderTriggerK>(csound, "netbender_trigger", "", "ikJJJ", csnd::thread::ik);

    csnd::plugin<NetBenderBridgeByName>(csound, "netbender_bridge", "", "iSSk", csnd::thread::ik);
    csnd::plugin<NetBenderBridgeByIndex>(csound, "netbender_bridge", "", "iiik", csnd::thread::ik);

    csnd::plugin<NetBenderLatentI>(csound, "netbender_latent", "", "iiiiioo", csnd::thread::i);
    csnd::plugin<NetBenderLatentK>(csound, "netbender_latent", "", "ikkkkOO", csnd::thread::ik);

    csnd::plugin<NetBenderSentry>(csound, "netbender_sentry", "k", "iJO", csnd::thread::ik);

    csnd::plugin<NetBenderGetWeightsByName>(csound, "netbender_getweights", "i[]", "iSo",
                                            csnd::thread::i);
    csnd::plugin<NetBenderGetWeightsByIndex>(csound, "netbender_getweights", "i[]", "iio",
                                             csnd::thread::i);

    csnd::plugin<NetBenderSetWeightsIByName>(csound, "netbender_setweights", "", "iSi[]",
                                             csnd::thread::i);
    csnd::plugin<NetBenderSetWeightsIByIndex>(csound, "netbender_setweights", "", "iii[]",
                                              csnd::thread::i);
    csnd::plugin<NetBenderSetWeightsKByName>(csound, "netbender_setweights", "", "iSk[]k",
                                             csnd::thread::ik);
    csnd::plugin<NetBenderSetWeightsKByIndex>(csound, "netbender_setweights", "", "iik[]k",
                                              csnd::thread::ik);

    csnd::plugin<NetBenderResetWeightsAllI>(csound, "netbender_resetweights", "", "i",
                                            csnd::thread::i);
    csnd::plugin<NetBenderResetWeightsLayerI>(csound, "netbender_resetweights", "", "iS",
                                              csnd::thread::i);
    csnd::plugin<NetBenderResetWeightsAllK>(csound, "netbender_resetweights", "", "ik",
                                            csnd::thread::ik);
    csnd::plugin<NetBenderResetWeightsLayerKByName>(csound, "netbender_resetweights", "", "iSk",
                                                    csnd::thread::ik);
    csnd::plugin<NetBenderResetWeightsLayerKByIndex>(csound, "netbender_resetweights", "", "iik",
                                                     csnd::thread::ik);
}
