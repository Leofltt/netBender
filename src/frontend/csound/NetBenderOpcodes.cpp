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
}
