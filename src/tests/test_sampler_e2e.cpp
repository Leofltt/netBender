#include "../backend/sampler/ExecuTorchSamplerBackend.h"
#include "../backend/sampler/SamplerPrompter.h"
#include "../backend/sampler/TextTokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

auto write_wav_pcm16(const std::string& path, const std::vector<std::vector<float>>& channels,
                     int sample_rate) -> bool {
    if (channels.empty() || channels[0].empty() || sample_rate <= 0) {
        return false;
    }

    const std::filesystem::path out_path(path);
    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        return false;
    }

    const uint16_t num_channels = static_cast<uint16_t>(channels.size());
    const uint32_t num_frames = static_cast<uint32_t>(channels[0].size());
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    const uint32_t data_bytes = num_frames * block_align;
    const uint32_t riff_size = 36 + data_bytes;

    auto write_u32 = [&](uint32_t v) { ofs.write(reinterpret_cast<const char*>(&v), 4); };
    auto write_u16 = [&](uint16_t v) { ofs.write(reinterpret_cast<const char*>(&v), 2); };

    ofs.write("RIFF", 4);
    write_u32(riff_size);
    ofs.write("WAVE", 4);
    ofs.write("fmt ", 4);
    write_u32(16);  // PCM subchunk size
    write_u16(1);   // PCM format
    write_u16(num_channels);
    write_u32(static_cast<uint32_t>(sample_rate));
    write_u32(byte_rate);
    write_u16(block_align);
    write_u16(bits_per_sample);
    ofs.write("data", 4);
    write_u32(data_bytes);

    std::vector<int16_t> interleaved(static_cast<size_t>(num_frames) * num_channels);
    for (uint32_t i = 0; i < num_frames; ++i) {
        for (uint16_t c = 0; c < num_channels; ++c) {
            const float clamped = std::clamp(channels[c][i], -1.0f, 1.0f);
            interleaved[static_cast<size_t>(i) * num_channels + c] =
                static_cast<int16_t>(std::lround(clamped * 32767.0f));
        }
    }
    ofs.write(reinterpret_cast<const char*>(interleaved.data()),
              static_cast<std::streamsize>(interleaved.size() * sizeof(int16_t)));
    return ofs.good();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace netbender::sampler;

    std::string manifest_path = "models/sa3_small_music/sa3_small_music.nbsampler.json";
    std::string prompt = "Amen break 174 BPM";
    std::string negative_prompt;
    std::string out_wav = "out/sa3_test.wav";
    float duration_sec = 6.0f;
    int steps = 0;
    float cfg_scale = -1.0f;
    uint64_t seed = 42ULL;
    bool use_gpu = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--manifest" || arg == "-m") && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--negative" && i + 1 < argc) {
            negative_prompt = argv[++i];
        } else if ((arg == "--duration" || arg == "-d") && i + 1 < argc) {
            duration_sec = std::stof(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--cfg" && i + 1 < argc) {
            cfg_scale = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
            out_wav = argv[++i];
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: test_sampler_e2e [manifest.nbsampler.json] [options]\n"
                      << "  --manifest, -m <path>   Bundle manifest path\n"
                      << "  --prompt, -p <text>     Text prompt\n"
                      << "  --negative <text>       Negative prompt\n"
                      << "  --duration, -d <sec>    Duration in seconds (default: 6.0)\n"
                      << "  --steps <n>             Diffusion steps (default: bundle default)\n"
                      << "  --cfg <val>             CFG scale (default: bundle default)\n"
                      << "  --seed <u64>            RNG seed (default: 42)\n"
                      << "  --out, -o <wav>         Output WAV path (default: out/sa3_test.wav)\n"
                      << "  --cpu                   Prefer CPU execution\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            manifest_path = arg;
        }
    }

    std::cout << "[test_sampler_e2e] Loading bundle: " << manifest_path << "\n";
    ExecuTorchSamplerBackend backend;
    const auto t_load0 = std::chrono::steady_clock::now();
    if (backend.load_bundle(manifest_path, use_gpu) != 0) {
        std::cerr << "[FAIL] load_bundle: " << backend.get_last_error() << "\n";
        return 1;
    }
    const auto t_load1 = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();

    const SamplerBundleMetadata& meta = backend.get_metadata();
    std::cout << "[test_sampler_e2e] Loaded " << backend.get_backend_name()
              << " | variant=" << meta.variant << " | sr=" << meta.sample_rate
              << " | ratio=" << meta.downsampling_ratio
              << " | compiled_frames=" << meta.compiled_latent_frames << " (" << load_ms
              << " ms)\n";

    TextTokenizer tokenizer;
    const auto t_tok0 = std::chrono::steady_clock::now();
    if (!tokenizer.load(meta.tokenizer_path, meta.max_token_length, meta.pad_token_id,
                        meta.eos_token_id, meta.add_eos)) {
        std::cerr << "[FAIL] tokenizer.load (" << meta.tokenizer_path
                  << "): " << tokenizer.get_last_error() << "\n";
        return 1;
    }
    const auto t_tok1 = std::chrono::steady_clock::now();
    const double tok_ms = std::chrono::duration<double, std::milli>(t_tok1 - t_tok0).count();

    const TokenizedPrompt preview = tokenizer.encode(prompt);
    std::cout << "[test_sampler_e2e] Tokenizer loaded (" << tok_ms
              << " ms) | valid_tokens=" << preview.valid_token_count << " | ids=[";
    for (int i = 0; i < preview.valid_token_count; ++i) {
        std::cout << (i > 0 ? ", " : "") << preview.input_ids[static_cast<size_t>(i)];
    }
    std::cout << "]\n";

    SamplerPrompter prompter(backend, tokenizer);
    GenerationRequest req;
    req.prompt = prompt;
    req.negative_prompt = negative_prompt;
    req.duration_sec = duration_sec;
    req.steps = steps;
    req.cfg_scale = cfg_scale;
    req.seed = seed;
    req.decode_audio = true;

    GeneratedSample sample;
    const bool ok = prompter.generate(req, sample, nullptr, [](const GenerationProgress& p) {
        std::cout << "  -> stage=" << static_cast<int>(p.stage) << " (" << p.current_step << "/"
                  << p.total_steps << ") " << p.message << "\n";
    });
    if (!ok) {
        std::cerr << "[FAIL] generate: " << prompter.get_last_error() << "\n";
        return 1;
    }

    if (sample.audio.empty() || sample.audio[0].empty()) {
        std::cerr << "[FAIL] empty audio output\n";
        return 1;
    }

    float peak = 0.0f;
    double sum_sq = 0.0;
    double sum = 0.0;
    size_t total_samples = 0;
    for (const auto& ch : sample.audio) {
        for (float v : ch) {
            if (!std::isfinite(v)) {
                std::cerr << "[FAIL] non-finite audio sample detected\n";
                return 1;
            }
            peak = std::max(peak, std::abs(v));
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            sum += static_cast<double>(v);
            ++total_samples;
        }
    }
    const float rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(total_samples)));
    const float dc = static_cast<float>(sum / static_cast<double>(total_samples));

    std::cout << "[test_sampler_e2e] Generated " << sample.audio.size() << "ch x "
              << sample.audio[0].size() << " samples (" << sample.duration_sec << "s) in "
              << sample.elapsed_ms << " ms | peak=" << peak << " | rms=" << rms << " | dc=" << dc
              << "\n";

    if (peak < 1e-4f) {
        std::cerr << "[FAIL] audio output is silent (peak < 1e-4)\n";
        return 1;
    }

    if (!write_wav_pcm16(out_wav, sample.audio, sample.sample_rate)) {
        std::cerr << "[FAIL] failed to write WAV file: " << out_wav << "\n";
        return 1;
    }
    std::cout << "[PASS] Wrote WAV -> " << out_wav << "\n";
    return 0;
}
