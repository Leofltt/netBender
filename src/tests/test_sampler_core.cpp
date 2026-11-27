#include "MockSamplerBackend.h"
#include "../backend/bending/PromptBendingEngine.h"
#include "../backend/sampler/DiffusionSchedule.h"
#include "../backend/sampler/SamplerBundleLoader.h"
#include "../backend/sampler/SamplerEngine.h"
#include "../backend/sampler/SamplerPrompter.h"
#include "../backend/sampler/TextTokenizer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL [" << __FILE__ << ":" << __LINE__ << "]: " #cond << "\n";           \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

using namespace netbender::sampler;
using netbender::bending::Xoshiro128Plus;

auto test_text_tokenizer_unigram() -> void {
    const std::string tok_json = R"({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": true},
            {"id": 1, "content": "</s>", "special": true},
            {"id": 2, "content": "<unk>", "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "add_prefix_space": true
        },
        "model": {
            "type": "Unigram",
            "unk_id": 2,
            "vocab": [
                ["<pad>", 0.0],
                ["</s>", 0.0],
                ["<unk>", 0.0],
                ["▁", -5.0],
                ["▁deep", -1.2],
                ["▁bass", -1.0],
                ["▁d", -4.0],
                ["e", -3.0],
                ["ep", -3.5],
                ["<0x5A>", -8.0]
            ]
        }
    })";

    TextTokenizer tok;
    CHECK(!tok.load("/nonexistent/tokenizer.json"));
    CHECK(!tok.is_loaded());

    CHECK(tok.load_from_json_string(tok_json, 8, 0, 1, true));
    CHECK(tok.is_loaded());
    CHECK(tok.get_max_length() == 8);
    CHECK(tok.get_model_type() == TextTokenizer::ModelType::Unigram);

    // Viterbi picks whole words "▁deep", "▁bass" rather than splitting letters
    const TokenizedPrompt res = tok.encode("deep bass");
    CHECK(res.input_ids.size() == 8);
    CHECK(res.attention_mask.size() == 8);
    CHECK(res.valid_token_count == 3);
    CHECK(res.input_ids[0] == 4);  // "▁deep"
    CHECK(res.input_ids[1] == 5);  // "▁bass"
    CHECK(res.input_ids[2] == 1);  // "</s>"
    CHECK(res.attention_mask[0] == 1);
    CHECK(res.attention_mask[1] == 1);
    CHECK(res.attention_mask[2] == 1);
    CHECK(res.attention_mask[3] == 0);
    CHECK(res.input_ids[3] == 0);  // <pad>
    CHECK(res.token_pieces.size() == 3);
    CHECK(res.token_pieces[0] ==
          "\xE2\x96\x81"
          "deep");

    // Byte fallback (<0x5A> == 'Z') inside unigram
    const TokenizedPrompt byte_res = tok.encode("Z");
    CHECK(byte_res.valid_token_count == 3);
    CHECK(byte_res.input_ids[0] == 3);  // "▁"
    CHECK(byte_res.input_ids[1] == 9);  // "<0x5A>"
    CHECK(byte_res.input_ids[2] == 1);  // "</s>"

    // Truncation preserves trailing EOS inside max_length = 3
    TextTokenizer short_tok;
    CHECK(short_tok.load_from_json_string(tok_json, 3, 0, 1, true));
    const TokenizedPrompt trunc = short_tok.encode("deep bass deep bass");
    CHECK(trunc.valid_token_count == 3);
    CHECK(trunc.input_ids[0] == 4);
    CHECK(trunc.input_ids[1] == 5);
    CHECK(trunc.input_ids[2] == 1);
    CHECK(trunc.attention_mask[2] == 1);
}

auto test_text_tokenizer_bpe() -> void {
    const std::string bpe_json = R"({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": true},
            {"id": 1, "content": "<eos>", "special": true}
        ],
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "model": {
            "type": "BPE",
            "vocab": {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "a": 3,
                "b": 4,
                "c": 5,
                "ab": 6,
                "abc": 7
            },
            "merges": [
                "a b",
                "ab c"
            ]
        }
    })";

    TextTokenizer tok;
    CHECK(tok.load_from_json_string(bpe_json, 6, 0, 1, true));
    CHECK(tok.get_model_type() == TextTokenizer::ModelType::BPE);

    const TokenizedPrompt res = tok.encode("abc ab x");
    CHECK(res.valid_token_count == 4);
    CHECK(res.input_ids[0] == 7);  // "abc"
    CHECK(res.input_ids[1] == 6);  // "ab"
    CHECK(res.input_ids[2] == 2);  // "<unk>" for 'x'
    CHECK(res.input_ids[3] == 1);  // "<eos>"
    CHECK(res.input_ids[4] == 0);
    CHECK(res.attention_mask[3] == 1);
    CHECK(res.attention_mask[4] == 0);

    // T5Gemma-style BPE: Replace(" ", "▁") + TemplateProcessing without EOS
    const std::string gemma_json = R"({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": true},
            {"id": 1, "content": "<eos>", "special": true}
        ],
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious"},
        "post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}]},
        "model": {
            "type": "BPE",
            "vocab": {"<pad>": 0, "<eos>": 1, "<unk>": 2, "▁": 3, "a": 4, "b": 5, "▁b": 6},
            "merges": ["▁ b"]
        }
    })";
    TextTokenizer gemma_tok;
    CHECK(gemma_tok.load_from_json_string(gemma_json, 4, 0, 1, true));
    const TokenizedPrompt gemma_res = gemma_tok.encode("a b");
    CHECK(gemma_res.valid_token_count == 2);
    CHECK(gemma_res.input_ids[0] == 4);  // "a"
    CHECK(gemma_res.input_ids[1] == 6);  // "▁b"
    CHECK(gemma_res.input_ids[2] == 0);  // "<pad>" (no trailing <eos>)
    CHECK(gemma_res.attention_mask[1] == 1);
    CHECK(gemma_res.attention_mask[2] == 0);
}

auto test_diffusion_schedule_and_apg() -> void {
    // 1. LogSNRShift endpoints & monotonicity (matching SA3 LogSNRShift)
    DistShiftConfig logsnr_cfg;
    logsnr_cfg.type = DistShiftType::LogSNR;
    logsnr_cfg.anchor_length = 2000.0f;
    logsnr_cfg.anchor_logsnr = -6.2f;
    logsnr_cfg.rate = 1.0f;
    logsnr_cfg.logsnr_end = 2.0f;

    const auto sigmas = build_schedule(8, logsnr_cfg, 128, 1.0f, true);
    CHECK(sigmas.size() == 9);
    CHECK(std::abs(sigmas.front() - 1.0f) < 1e-6f);
    CHECK(std::abs(sigmas.back() - 0.0f) < 1e-6f);
    for (size_t i = 0; i + 1 < sigmas.size(); ++i) {
        CHECK(sigmas[i] > sigmas[i + 1]);
    }
    // Check an interior step against exact analytical formula:
    // t_lin[4] = 0.5, logsnr_start = -6.2 - log2(128 / 2000), logsnr = 2 - 0.5*(2 - logsnr_start)
    const float expected_start = -6.2f - std::log2(128.0f / 2000.0f);
    const float expected_mid = 1.0f / (1.0f + std::exp(2.0f - 0.5f * (2.0f - expected_start)));
    CHECK(std::abs(sigmas[4] - expected_mid) < 1e-5f);

    // 2. Flux and Full distribution shifts
    DistShiftConfig flux_cfg;
    flux_cfg.type = DistShiftType::Flux;
    flux_cfg.min_length = 256.0f;
    flux_cfg.max_length = 4096.0f;
    flux_cfg.alpha_min = 1.0f;
    flux_cfg.alpha_max = 4.0f;
    const auto flux_sig = build_schedule(4, flux_cfg, 4096, 1.0f, true);
    // alpha=4 at max_length => at t=0.5, 4*0.5 / (1 + 3*0.5) = 2/2.5 = 0.8
    CHECK(std::abs(flux_sig[2] - 0.8f) < 1e-5f);

    DistShiftConfig full_cfg;
    full_cfg.type = DistShiftType::Full;
    full_cfg.use_sine = true;
    const auto full_sig = build_schedule(4, full_cfg, 1024, 1.0f, true);
    CHECK(full_sig.front() == 1.0f);
    CHECK(full_sig.back() == 0.0f);

    // 3. APG projection orthogonality & masking
    const int C = 2;
    const int T = 4;
    const std::vector<float> v0{1.0f, 2.0f, -1.0f, 99.0f, -0.5f, 1.5f, 3.0f, -99.0f};
    const std::vector<float> v1{0.5f, -1.0f, 2.0f, 50.0f, 1.0f, 0.5f, -1.0f, 50.0f};
    const std::vector<uint8_t> mask{1, 1, 1, 0};  // frame 3 is padded
    std::vector<float> par(C * T, 0.0f);
    std::vector<float> orth(C * T, 0.0f);
    apg_project(v0.data(), v1.data(), C, T, mask.data(), par.data(), orth.data());

    // Padding positions must be zeroed
    CHECK(par[3] == 0.0f && par[7] == 0.0f);
    CHECK(orth[3] == 0.0f && orth[7] == 0.0f);

    // Parallel + orthogonal reconstructs valid v0, and dot(par, orth) ~ 0
    double dot = 0.0;
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < 3; ++t) {
            const size_t idx = static_cast<size_t>(c) * T + static_cast<size_t>(t);
            CHECK(std::abs((par[idx] + orth[idx]) - v0[idx]) < 1e-5f);
            dot += static_cast<double>(par[idx]) * static_cast<double>(orth[idx]);
        }
    }
    CHECK(std::abs(dot) < 1e-5);

    // 4. Deterministic Xoshiro128Plus Gaussian noise
    Xoshiro128Plus rng_a(12345ULL);
    Xoshiro128Plus rng_b(12345ULL);
    std::vector<float> n_a(1024), n_b(1024);
    fill_standard_normal(n_a.data(), n_a.size(), rng_a);
    fill_standard_normal(n_b.data(), n_b.size(), rng_b);
    for (size_t i = 0; i < n_a.size(); ++i) {
        CHECK(std::isfinite(n_a[i]));
        CHECK(n_a[i] == n_b[i]);
    }

    // 5. Euler & DPM++ exact one-step consistency on trivial identity velocity v = x - target
    std::vector<float> xe{1.0f, -1.0f};
    const std::vector<float> ve{0.5f, -0.5f};
    step_euler(xe.data(), ve.data(), 2, 1.0f, 0.5f);
    CHECK(std::abs(xe[0] - 0.75f) < 1e-6f);
    CHECK(std::abs(xe[1] + 0.75f) < 1e-6f);

    std::vector<float> xd{1.0f, -1.0f};
    std::vector<float> old_d;
    step_dpmpp(xd.data(), ve.data(), 2, 1.0f, 0.5f, 0.0f, false, old_d);
    CHECK(std::isfinite(xd[0]) && std::isfinite(xd[1]));
    step_dpmpp(xd.data(), ve.data(), 2, 0.5f, 0.25f, 1.0f, true, old_d);
    CHECK(std::isfinite(xd[0]) && std::isfinite(xd[1]));
    step_dpmpp(xd.data(), ve.data(), 2, 0.25f, 0.0f, 0.5f, true, old_d);
    CHECK(std::isfinite(xd[0]) && std::isfinite(xd[1]));
}

auto test_bundle_loader_validation() -> void {
    const std::string valid_manifest = R"({
        "format": "netbender.sampler",
        "version": 1,
        "architecture": "stable_audio_3",
        "variant": "small-music",
        "runtime": "executorch",
        "delegate": "coreml",
        "compiled_latent_frames": 128,
        "sample_rate": 44100,
        "io_channels": 2,
        "latent_dim": 256,
        "downsampling_ratio": 4096,
        "latent_align": 2,
        "cond_dim": 768,
        "global_cond_dim": 768,
        "input_concat_dim": 257,
        "max_duration_sec": 30.0,
        "duration_padding_sec": 6.0,
        "diffusion": {
            "objective": "rf_denoiser",
            "default_sampler": "pingpong",
            "default_steps": 8,
            "default_cfg_scale": 1.0,
            "default_apg_scale": 1.0,
            "dist_shift": {"type": "logsnr", "anchor_length": 2000, "anchor_logsnr": -6.2}
        },
        "tokenizer": {"file": "tokenizer.json", "max_length": 128, "pad_token_id": 0, "eos_token_id": 1},
        "components": {
            "text_encoder": {"model": "te.pte", "sidecar": "te.bin"},
            "number_cond": {"model": "nc.pte", "sidecar": "nc.bin"},
            "dit": {"model": "dit.pte", "sidecar": "dit.bin"},
            "autoencoder": {"model": "ae.pte", "encode_model": "ae_encode.pte", "decode_model": "ae_decode.pte"}
        }
    })";

    SamplerBundleMetadata meta;
    std::string err;
    CHECK(parse_sampler_bundle_json(valid_manifest, "/tmp/bundle", meta, err, false));
    CHECK(err.empty());
    CHECK(meta.runtime == "executorch");
    CHECK(meta.delegate == "coreml");
    CHECK(meta.compiled_latent_frames == 128);
    CHECK(meta.default_sampler == SamplerAlgorithm::PingPong);
    CHECK(meta.ae_decode_model == "/tmp/bundle/ae_decode.pte");

    // Reject wrong format
    const std::string bad_format = R"({"format":"wrong","version":1,"runtime":"executorch"})";
    CHECK(!parse_sampler_bundle_json(bad_format, "/tmp/bundle", meta, err, false));
    CHECK(!err.empty());
}

auto test_sampler_prompter_end_to_end() -> void {
    const std::string tok_json = R"({
        "version": "1.0",
        "added_tokens": [{"id": 0, "content": "<pad>", "special": true}, {"id": 1, "content": "</s>", "special": true}],
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true},
        "model": {
            "type": "Unigram",
            "unk_id": 2,
            "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0], ["▁warm", -1.0], ["▁synth", -1.0], ["▁kick", -1.0]]
        }
    })";

    TextTokenizer tok;
    CHECK(tok.load_from_json_string(tok_json, 16, 0, 1, true));

    netbender::sampler::test::MockSamplerBackend backend;
    backend.init_synthetic("executorch", "coreml", 16, 44100, 4096, 8, 16);

    SamplerPrompter prompter(backend, tok);

    GenerationRequest req;
    req.prompt = "warm synth";
    req.duration_sec = 0.5f;  // 0.5s -> 22050 samples (< compiled 16*4096 = 65536)
    req.steps = 8;
    req.seed = 999ULL;

    std::vector<GenerationProgress::Stage> stages;
    GeneratedSample out1;
    CHECK(prompter.generate(req, out1, nullptr,
                            [&](const GenerationProgress& p) { stages.push_back(p.stage); }));

    CHECK(!stages.empty());
    CHECK(stages.front() == GenerationProgress::Stage::Tokenizing);
    CHECK(stages.back() == GenerationProgress::Stage::Done);
    CHECK(backend.encode_prompt_calls == 1);
    CHECK(backend.encode_duration_calls == 1);
    CHECK(backend.dit_step_calls == 8);
    CHECK(backend.decode_calls == 1);
    CHECK(out1.audio.size() == 2);
    // Exact duration trimming check: 0.5s at 44100 Hz = 22050 samples
    CHECK(out1.audio[0].size() == 22050);
    CHECK(out1.audio[1].size() == 22050);
    CHECK(out1.latent_dim == 8);
    CHECK(out1.latent_frames == 16);

    float energy = 0.0f;
    for (float s : out1.audio[0]) {
        CHECK(std::isfinite(s));
        energy += std::abs(s);
    }
    CHECK(energy > 0.1f);

    // Deterministic seed repeatability
    GeneratedSample out2;
    CHECK(prompter.generate(req, out2));
    CHECK(out1.latents == out2.latents);
    CHECK(out1.audio[0] == out2.audio[0]);

    // Different seed changes output
    GenerationRequest req_diff = req;
    req_diff.seed = 1000ULL;
    GeneratedSample out3;
    CHECK(prompter.generate(req_diff, out3));
    CHECK(out1.latents != out3.latents);

    // CFG != 1.0 runs 2 DiT calls per step + encodes negative prompt
    netbender::sampler::test::MockSamplerBackend cfg_backend;
    cfg_backend.init_synthetic("executorch", "coreml", 16, 44100, 4096, 8, 16);
    SamplerPrompter cfg_prompter(cfg_backend, tok);
    GenerationRequest cfg_req = req;
    cfg_req.negative_prompt = "kick";
    cfg_req.cfg_scale = 5.0f;
    cfg_req.steps = 4;
    cfg_req.use_default_sampler = false;
    cfg_req.sampler = SamplerAlgorithm::Euler;
    GeneratedSample cfg_out;
    CHECK(cfg_prompter.generate(cfg_req, cfg_out));
    CHECK(cfg_backend.encode_prompt_calls == 2);
    CHECK(cfg_backend.dit_step_calls == 8);

    // Mid-diffusion cancellation aborts cleanly before decode
    netbender::sampler::test::MockSamplerBackend cancel_backend;
    cancel_backend.init_synthetic("executorch", "coreml", 16, 44100, 4096, 8, 16);
    SamplerPrompter cancel_prompter(cancel_backend, tok);
    std::atomic<bool> cancel_flag{false};
    GeneratedSample cancel_out;
    const bool ok_cancel =
        cancel_prompter.generate(req, cancel_out, &cancel_flag, [&](const GenerationProgress& p) {
            if (p.stage == GenerationProgress::Stage::Diffusing && p.current_step == 2) {
                cancel_flag.store(true, std::memory_order_release);
            }
        });
    CHECK(!ok_cancel);
    CHECK(cancel_backend.dit_step_calls < 8);
    CHECK(cancel_backend.decode_calls == 0);
}

auto test_sampler_engine_async_and_playback() -> void {
    const std::string tok_json = R"({
        "version": "1.0",
        "added_tokens": [{"id": 0, "content": "<pad>", "special": true}, {"id": 1, "content": "</s>", "special": true}],
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true},
        "model": {
            "type": "Unigram",
            "unk_id": 2,
            "vocab": [["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0], ["▁warm", -1.0], ["▁synth", -1.0]]
        }
    })";

    auto mock = std::make_unique<netbender::sampler::test::MockSamplerBackend>();
    mock->init_synthetic("executorch", "coreml", 16, 44100, 4096, 8, 16);

    SamplerEngine engine(std::move(mock));
    CHECK(engine.load_tokenizer_from_json(tok_json, 16, 0, 1, true));
    CHECK(engine.is_loaded());
    engine.prepare(44100.0, 512);

    GenerationRequest req;
    req.prompt = "warm synth";
    req.duration_sec = 0.2f;  // 8820 samples
    req.steps = 4;
    req.seed = 42ULL;
    engine.request_generate(req);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (engine.get_stage() != GenerationProgress::Stage::Done &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(engine.get_stage() == GenerationProgress::Stage::Done);
    CHECK(engine.get_slot_generation_id() == 1);

    auto slot = engine.get_slot_snapshot();
    CHECK(slot != nullptr);
    CHECK(slot->num_samples == 8820);
    CHECK(slot->waveform_peaks.size() == SampleSlot::k_num_waveform_bins);

    // 1. OneShot playback produces finite audio and stops at end
    engine.set_playback_mode(SamplerPlaybackMode::OneShot);
    engine.trigger_rt(60, 1.0f);
    std::vector<float> ch0(512, 0.0f), ch1(512, 0.0f);
    float* ptrs[2]{ch0.data(), ch1.data()};
    engine.process_rt(ptrs, 2, 512);
    CHECK(engine.is_playing());
    float energy = 0.0f;
    for (int i = 0; i < 512; ++i) {
        CHECK(std::isfinite(ch0[static_cast<size_t>(i)]));
        energy += std::abs(ch0[static_cast<size_t>(i)]);
    }
    CHECK(energy > 1e-3f);

    // Advance past 8820 samples (~18 blocks)
    for (int b = 0; b < 20; ++b) {
        engine.process_rt(ptrs, 2, 512);
    }
    CHECK(!engine.is_playing());

    // 2. Loop mode stays playing across multiple buffer lengths
    engine.set_playback_mode(SamplerPlaybackMode::Loop);
    engine.set_loop_region(0.1f, 0.5f);
    engine.trigger_rt(60, 1.0f);
    for (int b = 0; b < 25; ++b) {
        engine.process_rt(ptrs, 2, 512);
        for (float v : ch0) {
            CHECK(std::isfinite(v));
        }
    }
    CHECK(engine.is_playing());

    // 3. MIDI pitch (+12 semitones = 2x speed)
    engine.set_playback_mode(SamplerPlaybackMode::OneShot);
    engine.set_loop_region(0.0f, 1.0f);
    engine.trigger_rt(72, 1.0f);
    engine.process_rt(ptrs, 2, 512);
    const float pos_oct = engine.get_playhead_norm();
    engine.trigger_rt(60, 1.0f);
    engine.process_rt(ptrs, 2, 512);
    const float pos_base = engine.get_playhead_norm();
    CHECK(pos_oct > pos_base * 1.8f);

    // 4. Gate mode releases cleanly to silence
    engine.set_playback_mode(SamplerPlaybackMode::Gate);
    engine.trigger_rt(60, 1.0f);
    engine.process_rt(ptrs, 2, 512);
    CHECK(engine.is_playing());
    engine.release_rt(60);
    for (int b = 0; b < 4; ++b) {
        engine.process_rt(ptrs, 2, 512);
    }
    CHECK(!engine.is_playing());

    // 5. Selective multi-slot batch generation (slots 0 and 2) & independent loop/playback
    GenerationRequest batch_req;
    batch_req.prompt = "warm synth";
    batch_req.duration_sec = 0.2f;
    batch_req.steps = 4;
    batch_req.seed = 100ULL;
    engine.request_generate_slots(batch_req, 0b0101);  // slots 0 & 2

    const auto batch_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((engine.get_slot_generation_id(2) < 1 ||
            engine.get_stage() != GenerationProgress::Stage::Done) &&
           std::chrono::steady_clock::now() < batch_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(engine.get_stage() == GenerationProgress::Stage::Done);
    CHECK(engine.get_slot_snapshot(0) != nullptr);
    CHECK(engine.get_slot_snapshot(1) == nullptr);
    CHECK(engine.get_slot_snapshot(2) != nullptr);
    CHECK(engine.get_slot_snapshot(3) == nullptr);
    CHECK(engine.get_slot_snapshot(0)->seed == 100ULL);
    CHECK(engine.get_slot_snapshot(2)->seed == 102ULL);

    engine.set_loop_region(0, 0.05f, 0.25f);
    engine.set_loop_region(2, 0.40f, 0.80f);
    CHECK(std::abs(engine.get_loop_start(0) - 0.05f) < 1e-5f);
    CHECK(std::abs(engine.get_loop_start(2) - 0.40f) < 1e-5f);

    engine.set_playback_mode(SamplerPlaybackMode::OneShot);
    engine.trigger_slot_rt(2, 60, 1.0f);
    engine.process_rt(ptrs, 2, 512);
    CHECK(engine.is_playing());
    CHECK(engine.get_playing_slot() == 2);
    CHECK(engine.is_slot_playing(2));
    CHECK(!engine.is_slot_playing(0));

    engine.stop_slot_rt(2);
    engine.process_rt(ptrs, 2, 512);
    CHECK(!engine.is_playing());

    // 6. Dynamic slot count (8 slots) & spectrogram generation on slot 5
    engine.set_slot_count(8);
    CHECK(engine.get_slot_count() == 8);
    GenerationRequest slot5_req;
    slot5_req.prompt = "warm synth";
    slot5_req.duration_sec = 0.2f;
    slot5_req.steps = 4;
    slot5_req.seed = 200ULL;
    engine.request_generate_slots(slot5_req, 1u << 5);
    const auto s5_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((engine.get_slot_generation_id(5) < 1 ||
            engine.get_stage() != GenerationProgress::Stage::Done) &&
           std::chrono::steady_clock::now() < s5_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(engine.get_stage() == GenerationProgress::Stage::Done);
    auto slot5 = engine.get_slot_snapshot(5);
    CHECK(slot5 != nullptr);
    CHECK(slot5->spectrogram.size() ==
          static_cast<size_t>(SampleSlot::k_spec_cols * SampleSlot::k_spec_bins));
}

auto test_prompt_bending_engine() -> void {
    using netbender::bending::PromptBendingEngine;
    using netbender::bending::Xoshiro128Plus;

    PromptEmbedding a;
    a.seq_len = 4;
    a.cond_dim = 4;
    a.data = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
              0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f};

    PromptEmbedding b;
    b.seq_len = 4;
    b.cond_dim = 4;
    b.data = {0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
              0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.0f, 0.0f};

    // 1. SLERP midpoint preserves unit norm on token 0
    const auto slerp_mid = PromptBendingEngine::blend_pair(a, b, 0.5f, PromptMorphLaw::Slerp);
    CHECK(slerp_mid.seq_len == 4 && slerp_mid.cond_dim == 4);
    const float n0 =
        std::sqrt(slerp_mid.data[0] * slerp_mid.data[0] + slerp_mid.data[1] * slerp_mid.data[1]);
    CHECK(std::abs(n0 - 1.0f) < 1e-4f);

    // 2. TokenSplice & FreqSplit produce valid finite blends
    const auto spliced = PromptBendingEngine::blend_pair(a, b, 0.5f, PromptMorphLaw::TokenSplice);
    CHECK(spliced.data[0] == b.data[0]);  // token 0 (phase 0 < 0.5) from b
    CHECK(spliced.data[8] == a.data[8]);  // token 2 (phase 0.5 >= 0.5) from a

    const auto fsplit = PromptBendingEngine::blend_pair(a, b, 0.5f, PromptMorphLaw::FreqSplit);
    CHECK(fsplit.data[0] == b.data[0]);  // dim 0 < 2 from b
    CHECK(fsplit.data[7] == a.data[7]);  // dim 3 >= 2 from a

    // 3. Embedding DSP (drift + quantize + wavefold)
    PromptBendingConfig cfg;
    cfg.enabled = true;
    cfg.emb_drift = 0.25f;
    cfg.emb_quantize_levels = 8;
    cfg.emb_wavefold = 0.5f;
    Xoshiro128Plus rng(77ULL);
    PromptEmbedding dsp_emb = slerp_mid;
    PromptBendingEngine::apply_embedding_dsp(dsp_emb, cfg, rng);
    for (float v : dsp_emb.data) {
        CHECK(std::isfinite(v));
    }

    // 4. Schedule warp preserves endpoints
    std::vector<float> sigmas{1.0f, 0.75f, 0.5f, 0.25f, 0.0f};
    PromptBendingEngine::warp_schedule(sigmas, 2.0f);
    CHECK(sigmas.front() == 1.0f && sigmas.back() == 0.0f);
    CHECK(std::abs(sigmas[2] - 0.25f) < 1e-5f);

    // 5. Masked 1-4 prompt navigation (only A & D active, or only D active)
    std::array<PromptEmbedding, 4> sparse_slots{};
    sparse_slots[0] = a;
    sparse_slots[3] = b;
    const auto ad_mid = PromptBendingEngine::blend_embeddings_masked(
        sparse_slots, {true, false, false, true}, 0.5f, 0.5f, PromptMorphLaw::Linear);
    CHECK(std::abs(ad_mid.data[0] - 0.5f) < 1e-5f && std::abs(ad_mid.data[1] - 0.5f) < 1e-5f);

    const auto d_only = PromptBendingEngine::blend_embeddings_masked(
        sparse_slots, {false, false, false, true}, 0.0f, 0.0f, PromptMorphLaw::Slerp);
    CHECK(d_only.data[1] == b.data[1]);
}

}  // namespace

int main() {
    test_text_tokenizer_unigram();
    test_text_tokenizer_bpe();
    test_diffusion_schedule_and_apg();
    test_bundle_loader_validation();
    test_sampler_prompter_end_to_end();
    test_sampler_engine_async_and_playback();
    test_prompt_bending_engine();
    std::cout << "test_sampler_core: all checks passed\n";
    return 0;
}
