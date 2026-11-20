#include "AudioRingBuffer.h"
#include "BackendFactory.h"
#include "IBackend.h"
#include "MockBackend.h"
#include "RtGate.h"
#include "SafetySentry.h"
#include "WorkerThread.h"
#include "parsing_utils.h"
#include "rtsan_utils.h"
#include "static_buffer.h"

#if defined(TORCH_API) || defined(TORCH_VERSION) || defined(C10_MACROS_MACROS_H_) ||               \
    defined(AT_DISPATCH_ALL_TYPES) || defined(ORT_API_VERSION) ||                                  \
    defined(ONNXRUNTIME_CXX_API_H) || defined(JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED) ||             \
    defined(JUCE_CORE_H_INCLUDED)
#error "Public NetBender backend/DSP headers leaked Torch, ONNX Runtime, or JUCE headers"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " #cond << "\n";           \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

auto contains(const std::vector<std::string>& vec, const std::string& val) -> bool {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

auto test_factory_extension_dispatch() -> void {
    struct Case {
        std::string path;
        std::string expected_name;
    };

    const std::vector<Case> cases{
        {"model.ts", "TorchScript"},
        {"/UPPER/CASE/MODEL.TS", "TorchScript"},
        {"path/to/model.onnx", "ONNX Runtime"},
        {"/dir.with.dots/model.ONNX", "ONNX Runtime"},
        {"model.pte", "ExecuTorch"},
        {"/models/darbouka.PTE", "ExecuTorch"},
        {"model.unknown", "TorchScript"},
        {"model_without_extension", "TorchScript"},
        {"", "TorchScript"},
    };

    for (const auto& c : cases) {
        auto backend = BackendFactory::create_for_file(c.path);
        CHECK(backend != nullptr);
        CHECK(!backend->is_loaded());
        CHECK(backend->get_backend_name() == c.expected_name);
    }

    auto def = BackendFactory::create_default();
    CHECK(def != nullptr);
    CHECK(!def->is_loaded());
    CHECK(def->get_backend_name() == "TorchScript");
}

auto test_factory_runtime_queries() -> void {
    const std::string torch_ver = BackendFactory::get_torch_version();
    CHECK(!torch_ver.empty());
    CHECK(torch_ver != "unknown");

    const std::string gpu_label = BackendFactory::gpu_backend_label();
    CHECK(gpu_label == "CUDA" || gpu_label == "MPS/CoreML" || gpu_label == "MPS" ||
          gpu_label == "CPU");

    const bool has_gpu = BackendFactory::has_gpu_support();
#if defined(__APPLE__)
    CHECK(has_gpu);
#else
    (void)has_gpu;
#endif
}

auto test_factory_invalid_path_and_unloaded_errors() -> void {
    const std::array<std::string, 3> exts{".ts", ".onnx", ".pte"};
    for (const auto& ext : exts) {
        const std::string missing_path = "/nonexistent/netbender_phase2_missing_model" + ext;
        auto backend = BackendFactory::create_for_file(missing_path);
        CHECK(backend != nullptr);
        CHECK(!backend->is_loaded());
        CHECK(backend->get_sample_rate() > 0.0);
        CHECK(backend->reload() == -1);

        const int rc = backend->load(missing_path, 48000.0);
        CHECK(rc == -1);
        CHECK(!backend->is_loaded());
        CHECK(!backend->get_last_error().empty());
    }
}

auto test_mock_backend_lifecycle_and_introspection() -> void {
    auto owned = std::make_unique<MockBackend>();
    owned->in_channels = 2;
    owned->out_channels = 2;
    owned->in_ratio = 1;
    owned->out_ratio = 1;
    owned->higher_ratio = 512;
    owned->preferred_buffer_size = 1024;
    owned->latent_dim = 8;

    MockBackend* raw = owned.get();
    std::unique_ptr<IBackend> backend = std::move(owned);

    CHECK(!backend->is_loaded());
    CHECK(backend->get_plugin_modes().empty());
    CHECK(backend->get_available_methods().empty());
    CHECK(backend->reload() == -1);
    CHECK(backend->save_model("/tmp/mock_unloaded.bin") == -1);
    CHECK(!backend->has_method("forward"));
    CHECK(!backend->has_prior_decode());
    CHECK(!backend->has_autoencode());

    // Empty path failure
    CHECK(backend->load("", 48000.0) == -1);
    CHECK(!backend->is_loaded());
    CHECK(!backend->get_last_error().empty());

    // Simulated failure flag
    raw->fail_next_load = true;
    CHECK(backend->load("virtual/model.mock", 48000.0) == -1);
    CHECK(!backend->is_loaded());
    CHECK(!backend->get_last_error().empty());

    // Missing target method failure
    const std::string bad_method = "nonexistent_method";
    CHECK(backend->load("virtual/model.mock", 48000.0, &bad_method) == -1);
    CHECK(!backend->is_loaded());

    // Successful load with target_method
    const std::string good_method = "forward";
    CHECK(backend->load("virtual/model.mock", 48000.0, &good_method) == 0);
    CHECK(backend->is_loaded());
    CHECK(backend->get_last_error().empty());
    CHECK(backend->get_sample_rate() == 48000.0);

    // Reload & sample rate & GPU toggle
    CHECK(backend->reload() == 0);
    backend->set_sample_rate(44100.0);
    CHECK(backend->get_sample_rate() == 44100.0);
    CHECK(backend->get_backend_name() == "MockBackend");
    backend->use_gpu(true);
    CHECK(backend->get_backend_name() == "MockBackend (GPU)");

    // Method & mode introspection
    CHECK(backend->has_method("forward"));
    CHECK(backend->has_method("encode"));
    CHECK(backend->has_method("decode"));
    CHECK(backend->has_method("prior"));
    CHECK(!backend->has_method("missing"));
    CHECK(backend->has_prior_decode());
    CHECK(backend->has_autoencode());

    const auto modes = backend->get_plugin_modes();
    CHECK(contains(modes, "forward"));
    CHECK(contains(modes, "autoencode"));
    CHECK(contains(modes, "prior"));

    const auto methods = backend->get_available_methods();
    CHECK(contains(methods, "forward"));
    CHECK(contains(methods, "encode"));
    CHECK(contains(methods, "decode"));
    CHECK(contains(methods, "prior"));

    CHECK((backend->get_method_params("forward") == std::vector<int>{2, 1, 2, 1}));
    CHECK((backend->get_mode_params("prior") == std::vector<int>{0, 0, 2, 1}));
    CHECK(backend->get_higher_ratio() == 512);
    CHECK(backend->get_preferred_buffer_size() == 1024);

    const ModelInfo info = backend->get_model_info();
    CHECK(info.method_properties.count("forward") == 1);
    CHECK(info.method_properties.at("forward").channels_in == 2);
    CHECK(info.method_properties.at("forward").channels_out == 2);
    CHECK(info.attribute_properties.count("temperature") == 1);
    CHECK(info.attribute_properties.count("buf_attr") == 1);

    CHECK(backend->get_input_label("forward", 1) == "(signal) model input 1");
    CHECK(backend->get_output_label("forward", 0) == "(signal) model output 0");
}

auto test_mock_backend_inference_and_latent_hook() -> void {
    auto owned = std::make_unique<MockBackend>();
    owned->in_channels = 2;
    owned->out_channels = 2;
    owned->higher_ratio = 4;
    owned->latent_dim = 3;
    owned->gain = 2.0f;

    MockBackend* raw = owned.get();
    std::unique_ptr<IBackend> backend = std::move(owned);
    CHECK(backend->load("virtual/infer.mock", 48000.0) == 0);

    constexpr int k_vec = 8;
    std::vector<float> in_ch0{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> in_ch1{-0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f};
    std::vector<float> out_ch0(k_vec, 0.0f);
    std::vector<float> out_ch1(k_vec, 0.0f);
    std::vector<float*> in_ptrs{in_ch0.data(), in_ch1.data()};
    std::vector<float*> out_ptrs{out_ch0.data(), out_ch1.data()};

    // perform()
    backend->perform(in_ptrs, out_ptrs, "forward", 1, 2, k_vec);
    CHECK(raw->perform_count == 1);
    CHECK(raw->last_method == "forward");
    CHECK(raw->last_n_batches == 1);
    CHECK(raw->last_n_out_channels == 2);
    CHECK(raw->last_n_vec == k_vec);
    for (int i = 0; i < k_vec; ++i) {
        const size_t idx = static_cast<size_t>(i);
        CHECK(std::abs(out_ch0[idx] - in_ch0[idx] * 2.0f) < 1e-6f);
        CHECK(std::abs(out_ch1[idx] - in_ch1[idx] * 2.0f) < 1e-6f);
    }

    // perform_forward() without hook
    backend->perform_forward(in_ptrs, out_ptrs, 1, 2, k_vec, nullptr);
    CHECK(raw->forward_count == 1);
    CHECK(!raw->last_latent_hook_called);
    for (int i = 0; i < k_vec; ++i) {
        CHECK(std::abs(out_ch0[static_cast<size_t>(i)] - in_ch0[static_cast<size_t>(i)] * 2.0f) <
              1e-6f);
    }

    // perform_forward() with LatentHook modifying latent[0] = 0.5f
    int64_t seen_channels = 0;
    int64_t seen_length = 0;
    backend->perform_forward(in_ptrs, out_ptrs, 1, 2, k_vec, [&](IBackend::LatentBuffer& buf) {
        seen_channels = buf.channels;
        seen_length = buf.length;
        CHECK(buf.data != nullptr);
        buf.data[0] = 0.5f;
    });
    CHECK(raw->forward_count == 2);
    CHECK(raw->last_latent_hook_called);
    CHECK(seen_channels == 3);
    CHECK(seen_length == 2);
    for (int i = 0; i < k_vec; ++i) {
        const size_t idx = static_cast<size_t>(i);
        CHECK(std::abs(out_ch0[idx] - in_ch0[idx]) < 1e-6f);
        CHECK(std::abs(out_ch1[idx] - in_ch1[idx]) < 1e-6f);
    }

    // perform_prior_decode()
    std::fill(out_ch0.begin(), out_ch0.end(), 0.0f);
    std::fill(out_ch1.begin(), out_ch1.end(), 0.0f);
    backend->perform_prior_decode(out_ptrs, 1, 2, k_vec);
    CHECK(raw->prior_count == 1);
    for (int i = 0; i < k_vec; ++i) {
        const float expected = 2.0f * (0.2f + 0.001f * static_cast<float>(i % 64));
        CHECK(std::abs(out_ch0[static_cast<size_t>(i)] - expected) < 1e-6f);
        CHECK(std::abs(out_ch1[static_cast<size_t>(i)] - expected) < 1e-6f);
    }

    // perform_decode_gen() + LatentHook + clear_decode_gen_fifo()
    backend->perform_decode_gen(out_ptrs, 1, 2, k_vec, [&](IBackend::LatentBuffer& buf) {
        CHECK(buf.channels == 3);
        CHECK(buf.length == 2);
        buf.data[0] = 3.0f;
    });
    CHECK(raw->decode_gen_count == 1);
    CHECK(raw->last_latent_hook_called);
    CHECK(raw->decode_gen_state_size() == 6);
    for (int i = 0; i < k_vec; ++i) {
        CHECK(std::abs(out_ch0[static_cast<size_t>(i)] - 1.5f) < 1e-6f);
        CHECK(std::abs(out_ch1[static_cast<size_t>(i)] - 1.5f) < 1e-6f);
    }

    backend->clear_decode_gen_fifo();
    CHECK(raw->decode_gen_state_size() == 0);
}

auto test_mock_backend_weight_bending_and_attributes() -> void {
    std::unique_ptr<IBackend> backend = std::make_unique<MockBackend>();
    CHECK(backend->load("virtual/bend.mock", 48000.0) == 0);

    const auto layers = backend->get_available_layers();
    CHECK((layers == std::vector<std::string>{"layer0.weight", "layer1.bias"}));
    CHECK((backend->get_layer_shape("layer0.weight") == std::vector<int64_t>{2, 2}));
    CHECK((backend->get_layer_shape("layer1.bias") == std::vector<int64_t>{2}));

    const auto orig_w0 = backend->get_original_layer_weights("layer0.weight");
    const auto orig_w1 = backend->get_original_layer_weights("layer1.bias");
    CHECK(backend->get_layer_weights("layer0.weight") == orig_w0);
    CHECK(backend->get_layer_weights("layer1.bias") == orig_w1);

    const std::vector<float> bent_w0{9.0f, -9.0f, 3.0f, -3.0f};
    const std::vector<float> bent_w1{4.0f, -4.0f};
    backend->set_layer_weights("layer0.weight", bent_w0);
    backend->set_layer_weights("layer1.bias", bent_w1);
    CHECK(backend->get_layer_weights("layer0.weight") == bent_w0);
    CHECK(backend->get_original_layer_weights("layer0.weight") == orig_w0);
    CHECK(backend->get_layer_weights("layer1.bias") == bent_w1);

    backend->reset_layer_weights("layer0.weight");
    CHECK(backend->get_layer_weights("layer0.weight") == orig_w0);
    CHECK(backend->get_layer_weights("layer1.bias") == bent_w1);

    backend->reset_all_weights();
    CHECK(backend->get_layer_weights("layer0.weight") == orig_w0);
    CHECK(backend->get_layer_weights("layer1.bias") == orig_w1);

    CHECK(backend->save_model("/tmp/mock_saved.bin") == 0);
    CHECK(backend->save_model("") == -1);

    // Attributes & StaticBuffer management
    CHECK(contains(backend->get_available_attributes(), "temperature"));
    CHECK(contains(backend->get_available_attributes(), "readonly_info"));
    CHECK(contains(backend->get_settable_attributes(), "temperature"));
    CHECK(backend->has_settable_attribute("temperature"));
    CHECK(!backend->has_settable_attribute("readonly_info"));
    CHECK(backend->get_attribute_as_string("temperature") == "1.0");
    backend->set_attribute("temperature", {"0.75"}, {});
    CHECK(backend->get_attribute_as_string("temperature") == "0.75");

    CHECK((backend->get_buffer_attributes() == std::vector<std::string>{"buf_attr"}));
    CHECK(backend->is_buffer_element_of_attribute("buf_attr", 0));
    CHECK(!backend->is_buffer_element_of_attribute("temperature", 0));
    CHECK(backend->is_tensor_element_of_attribute("temperature", 0));
    CHECK(backend->get_buffer_name("buf_attr", 0) == "buf_attr");
    CHECK(backend->get_buffer_name("buf_attr", 1).empty());

    StaticBuffer<float> buf(1, 4, 48000.0);
    buf.put(42.0f, 0, 2);
    CHECK(backend->update_buffer("", buf) == -1);
    CHECK(backend->update_buffer("buf_attr", buf) == 0);
    IBackend::BufferMap buf_map{{"buf_attr", buf}};
    backend->set_attribute("buf_attr", {"buf_attr"}, buf_map);
    CHECK(backend->reset_buffer("missing_buf") == -1);
    CHECK(backend->reset_buffer("buf_attr") == 0);
    CHECK(buf.data()[2] == 0.0f);
}

auto test_parsing_utils_and_static_buffer() -> void {
    CHECK(!to_bool("0"));
    CHECK(!to_bool("false"));
    CHECK(to_bool("1"));
    CHECK(to_bool("true"));
    CHECK(to_bool("yes"));

    CHECK(to_int("42") == 42);
    CHECK(to_int("-7") == -7);
    CHECK(std::abs(to_float("3.25") - 3.25f) < 1e-6f);
    CHECK(std::abs(to_float("-0.5") - (-0.5f)) < 1e-6f);

    StaticBuffer<float> empty_buf;
    CHECK(empty_buf.dims()[0] == 0);
    CHECK(empty_buf.dims()[1] == 0);
    CHECK(empty_buf.data() == nullptr);

    StaticBuffer<float> sq_buf(2, 2, 48000.0);
    CHECK(sq_buf.dims()[0] == 2);
    CHECK(sq_buf.dims()[1] == 2);
    CHECK(sq_buf.sr() == 48000.0);
    CHECK(sq_buf.data() != nullptr);

    sq_buf.put(1.5f, 1, 1);
    CHECK(sq_buf.at(1, 1) == 1.5f);
    CHECK(sq_buf.data()[3] == 1.5f);

    StaticBuffer<float> shared_copy(sq_buf);
    CHECK(shared_copy.data() == sq_buf.data());
    CHECK(shared_copy.at(1, 1) == 1.5f);
    sq_buf.reset();
    CHECK(shared_copy.at(1, 1) == 0.0f);
}

}  // namespace

auto main() -> int {
    std::cout << "=== NetBender Backend Core Unit Tests ===\n";

    test_factory_extension_dispatch();
    std::cout << "  [1/7] BackendFactory extension dispatch: PASS\n";

    test_factory_runtime_queries();
    std::cout << "  [2/7] BackendFactory runtime queries: PASS\n";

    test_factory_invalid_path_and_unloaded_errors();
    std::cout << "  [3/7] Unloaded/invalid path handling across backends: PASS\n";

    test_mock_backend_lifecycle_and_introspection();
    std::cout << "  [4/7] MockBackend IBackend lifecycle & introspection: PASS\n";

    test_mock_backend_inference_and_latent_hook();
    std::cout << "  [5/7] MockBackend inference & LatentHook: PASS\n";

    test_mock_backend_weight_bending_and_attributes();
    std::cout << "  [6/7] MockBackend weight bending & attributes: PASS\n";

    test_parsing_utils_and_static_buffer();
    std::cout << "  [7/7] parsing_utils & StaticBuffer: PASS\n";

    std::cout << "All Backend Core tests passed.\n";
    return 0;
}
