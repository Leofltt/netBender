#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace netbender::sampler {

/** Timestep distribution-shift strategies supported by Stable Audio 3. */
enum class DistShiftType {
    Identity,
    LogSNR,
    Flux,
    Full,
};

/** ODE / SDE diffusion sampling integrators for rectified-flow & rf_denoiser SA3 models. */
enum class SamplerAlgorithm {
    PingPong,
    Euler,
    DpmPP,
};

/** Configuration parameters for timestep distribution warping. */
struct DistShiftConfig {
    DistShiftType type{DistShiftType::LogSNR};
    float anchor_length{2000.0f};
    float anchor_logsnr{-6.2f};
    float rate{1.0f};
    float logsnr_end{2.0f};
    float min_length{256.0f};
    float max_length{4096.0f};
    float alpha_min{1.0f};
    float alpha_max{1.0f};
    float base_shift{0.5f};
    float max_shift{1.15f};
    bool use_sine{false};
};

/** Parsed `.nbsampler.json` bundle metadata and resolved component paths. */
struct SamplerBundleMetadata {
    std::string format;
    int version{1};
    std::string architecture;
    std::string variant;
    std::string runtime;
    std::string delegate;
    std::string bundle_dir;
    std::string manifest_path;

    int sample_rate{44100};
    int io_channels{2};
    int latent_dim{256};
    int downsampling_ratio{4096};
    int latent_align{2};
    int cond_dim{768};
    int global_cond_dim{768};
    int input_concat_dim{257};
    int compiled_latent_frames{0};
    float max_duration_sec{30.0f};
    float duration_padding_sec{6.0f};

    std::string objective{"rf_denoiser"};
    SamplerAlgorithm default_sampler{SamplerAlgorithm::PingPong};
    int default_steps{8};
    float default_cfg_scale{1.0f};
    float default_apg_scale{1.0f};
    bool has_inpaint_concat{true};
    DistShiftConfig dist_shift;
    std::vector<std::string> cross_attn_cond_ids{"prompt", "seconds_total"};
    std::vector<std::string> global_cond_ids{"seconds_total"};
    std::vector<std::string> input_concat_ids;

    std::string tokenizer_file{"tokenizer.json"};
    std::string tokenizer_path;
    int max_token_length{128};
    int pad_token_id{0};
    int eos_token_id{1};
    bool add_eos{true};

    std::string text_encoder_model;
    std::string text_encoder_sidecar;
    std::string number_cond_model;
    std::string number_cond_sidecar;
    std::string dit_model;
    std::string dit_sidecar;
    std::string ae_model;
    std::string ae_sidecar;
    std::string ae_encode_model;
    std::string ae_decode_model;
};

/** Output of `TextTokenizer::encode` ready for `ISamplerBackend::encode_prompt`. */
struct TokenizedPrompt {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<std::string> token_pieces;
    int valid_token_count{0};
};

/** Flattened row-major prompt embedding tensor `[seq_len, cond_dim]`. */
struct PromptEmbedding {
    std::vector<float> data;
    int seq_len{0};
    int cond_dim{0};
};

/** User parameters for a single text-to-audio generation pass. */
struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    float duration_sec{6.0f};
    int steps{0};            // <= 0 uses bundle default_steps
    float cfg_scale{-1.0f};  // < 0 uses bundle default_cfg_scale
    float apg_scale{-1.0f};  // < 0 uses bundle default_apg_scale
    float cfg_norm_threshold{0.0f};
    SamplerAlgorithm sampler{SamplerAlgorithm::PingPong};
    bool use_default_sampler{true};
    uint64_t seed{42};
    bool decode_audio{true};
};

/** Fine-grained generation stage and step progress emitted during sampling. */
struct GenerationProgress {
    enum class Stage {
        Idle,
        Tokenizing,
        EncodingPrompt,
        Diffusing,
        Decoding,
        Done,
        Cancelled,
        Error,
    };

    Stage stage{Stage::Idle};
    int current_step{0};
    int total_steps{0};
    std::string message;
};

/** Final generated stereo audio waveform and latent tensor returned by `SamplerPrompter`. */
struct GeneratedSample {
    std::vector<std::vector<float>> audio;  // [io_channels][num_samples]
    std::vector<float> latents;             // [latent_dim * latent_frames], channel-major [C, T]
    int latent_dim{0};
    int latent_frames{0};
    int valid_latent_frames{0};
    int sample_rate{44100};
    float duration_sec{0.0f};
    double elapsed_ms{0.0};
};

using ProgressCallback = std::function<void(const GenerationProgress&)>;

/** Real-time playback trigger behavior for `SamplerEngine`. */
enum class SamplerPlaybackMode {
    OneShot,
    Loop,
    Gate,
};

/** Immutable generated audio + latent snapshot swapped atomically into `SamplerEngine`. */
struct SampleSlot {
    static constexpr int k_num_waveform_bins = 512;
    static constexpr int k_spec_cols = 128;
    static constexpr int k_spec_bins = 48;

    std::vector<std::vector<float>> audio;  // [channels][num_samples]
    std::vector<float> latents;             // [latent_dim * latent_frames]
    std::vector<float> waveform_peaks;      // [k_num_waveform_bins] in [0..1]
    std::vector<float> spectrogram;         // [k_spec_cols * k_spec_bins] in [0..1]
    int sample_rate{44100};
    int num_samples{0};
    int latent_dim{0};
    int latent_frames{0};
    float duration_sec{0.0f};
    std::string prompt;
    uint64_t seed{0};
};

}  // namespace netbender::sampler
