#pragma once

#include <JuceHeader.h>
#include "../ui/FlatLookAndFeel.h"
#include "ExportModal.h"
#include "PromptBendPanel.h"
#include "SamplerEngine.h"
#include "WaveformView.h"

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class SamplerMainComponent final : public juce::AudioAppComponent,
                                   private juce::MidiInputCallback,
                                   private juce::Timer,
                                   private juce::KeyListener,
                                   private juce::ChangeListener {
public:
    explicit SamplerMainComponent(const juce::String& initial_bundle_path = {});
    ~SamplerMainComponent() override;

    auto load_bundle_async(const juce::String& manifest_path) -> void;

    // AudioAppComponent
    auto prepareToPlay(int samplesPerBlockExpected, double sampleRate) -> void override;
    auto getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) -> void override;
    auto releaseResources() -> void override;

    // Component
    auto paint(juce::Graphics& g) -> void override;
    auto paintOverChildren(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    enum class ViewMode { Grid, Focus };

    // MidiInputCallback
    auto handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message)
        -> void override;

    // Timer, KeyListener & ChangeListener
    auto timerCallback() -> void override;
    auto keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent)
        -> bool override;
    auto changeListenerCallback(juce::ChangeBroadcaster* source) -> void override;

    auto refresh_midi_callbacks() -> void;
    auto open_audio_settings() -> void;
    auto choose_bundle_file() -> void;
    auto trigger_generation(uint32_t override_slot_mask = 0) -> void;
    auto set_slot_count(int count) -> void;
    auto select_active_slot(int slot_idx) -> void;
    auto set_view_mode(ViewMode mode, int focus_slot = -1) -> void;
    auto sync_slot_selection_ui(int slot_idx, bool selected) -> void;
    auto update_focus_info_label() -> void;
    auto show_waveform_context_menu(int slot_idx) -> void;
    auto open_export_modal(int default_slot_idx, bool select_all_populated,
                           bool default_apply_range) -> void;
    auto export_slots_to_files(std::vector<int> slot_indices, bool apply_range,
                               float single_start_norm, float single_end_norm,
                               SamplerExportFormat format) -> void;
    [[nodiscard]] auto collect_prompts() const -> std::array<std::string, 4>;
    auto update_load_button_style(bool loaded) -> void;
    auto sync_active_prompt_corners() -> void;

    FlatLookAndFeel m_look_and_feel;
    netbender::sampler::SamplerEngine m_engine;
    std::unique_ptr<juce::FileChooser> m_file_chooser;
    std::unique_ptr<juce::FileChooser> m_export_chooser;

    // Background bundle loader state
    std::thread m_load_thread;
    std::atomic<bool> m_is_loading_bundle{false};
    std::atomic<bool> m_bundle_load_done{false};
    std::atomic<bool> m_bundle_load_ok{false};
    juce::String m_pending_bundle_name;
    juce::String m_load_error_text;

    std::array<uint64_t, netbender::sampler::SamplerEngine::k_max_slots> m_last_seen_slot_ids{};
    std::array<SamplerWaveformView::DisplayMode, netbender::sampler::SamplerEngine::k_max_slots>
        m_slot_display_modes{};
    double m_progress_value{0.0};
    ViewMode m_view_mode{ViewMode::Grid};

    // Header
    juce::Label m_title_label;
    juce::Label m_bundle_label;
    juce::ComboBox m_slot_count_combo;
    juce::TextButton m_export_button{"EXPORT"};
    juce::TextButton m_load_button{"LOAD BUNDLE"};
    juce::TextButton m_audio_midi_button{"AUDIO / MIDI"};

    // Prompt bar (2x2 Prompts A/B/C/D)
    std::array<juce::Label, 4> m_prompt_labels;
    std::array<juce::TextEditor, 4> m_prompt_editors;
    std::array<juce::TextButton, netbender::sampler::SamplerEngine::k_max_slots>
        m_slot_select_buttons;
    juce::TextButton m_generate_button{"GENERATE"};
    juce::ProgressBar m_progress_bar{m_progress_value};
    juce::Label m_status_label;

    // Focus bar (visible in Focus mode)
    juce::TextButton m_back_to_grid_button{"< GRID"};
    std::array<juce::TextButton, netbender::sampler::SamplerEngine::k_max_slots>
        m_focus_slot_buttons;
    juce::Label m_focus_info_label;

    // Waveforms (Grid + Focus) & Prompt Bend Panel
    std::array<SamplerWaveformView, netbender::sampler::SamplerEngine::k_max_slots> m_grid_views;
    SamplerWaveformView m_focus_view;
    PromptBendPanel m_bend_panel;
    std::array<PromptBendPanel::State, netbender::sampler::SamplerEngine::k_max_slots>
        m_slot_bend_states{};
    int m_loaded_bend_slot{0};
    int m_user_selected_slot{0};
    int m_auto_regen_target_slot{0};
    bool m_bundle_loaded{false};
    bool m_auto_regen_pending{false};
    juce::uint32 m_auto_regen_deadline_ms{0};

    // Transport & playback controls
    juce::Label m_mode_label{"", "MODE"};
    juce::ComboBox m_mode_combo;
    juce::Label m_speed_label{"", "SPEED"};
    juce::Slider m_speed_slider;
    juce::Label m_gain_label{"", "GAIN"};
    juce::Slider m_gain_slider;
    juce::Label m_seed_label{"", "SEED"};
    juce::TextEditor m_seed_editor;
};
