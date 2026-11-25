#pragma once

#include <JuceHeader.h>
#include "../ui/FlatLookAndFeel.h"
#include "SamplerEngine.h"

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

    auto loadBundleAsync(const juce::String& manifest_path) -> void;

    // AudioAppComponent
    auto prepareToPlay(int samplesPerBlockExpected, double sampleRate) -> void override;
    auto getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) -> void override;
    auto releaseResources() -> void override;

    // Component
    auto paint(juce::Graphics& g) -> void override;
    auto resized() -> void override;

private:
    enum class ViewMode { Grid, Focus };
    enum class DisplayMode { Waveform, Spectrum };
    enum class ExportFormat { Wav24 = 1, Wav16 = 2, WavFloat32 = 3, Aiff24 = 4 };

    class WaveformView final : public juce::Component {
    public:
        std::function<void(float, float)> onLoopChanged;
        std::function<void(int)> onFocusRequested;
        std::function<void(int)> onTogglePlay;
        std::function<void(int)> onToggleRegen;
        std::function<void(int, DisplayMode)> onDisplayModeChanged;
        std::function<void(int)> onRightClick;

        auto setSlotIndex(int idx) -> void;
        auto setCompactMode(bool compact) -> void;
        auto setSelectedForGen(bool selected) -> void;
        auto setGenerating(bool generating) -> void;
        auto setFocused(bool focused) -> void;
        auto setDisplayMode(DisplayMode mode) -> void;
        [[nodiscard]] auto getDisplayMode() const noexcept -> DisplayMode { return m_displayMode; }

        auto setSlot(std::shared_ptr<const netbender::sampler::SampleSlot> slot) -> void;
        auto setPlayhead(float norm, bool playing) -> void;
        auto setLoopRegion(float start_norm, float end_norm) -> void;

        auto paint(juce::Graphics& g) -> void override;
        auto mouseDown(const juce::MouseEvent& e) -> void override;
        auto mouseDrag(const juce::MouseEvent& e) -> void override;
        auto mouseUp(const juce::MouseEvent& e) -> void override;

    private:
        [[nodiscard]] auto regenBadgeBounds() const -> juce::Rectangle<float>;
        [[nodiscard]] auto playBadgeBounds() const -> juce::Rectangle<float>;
        [[nodiscard]] auto specBadgeBounds() const -> juce::Rectangle<float>;
        auto rebuildSpecImage() -> void;

        enum class DragTarget { None, StartHandle, EndHandle };
        int m_slotIndex{0};
        bool m_compactMode{false};
        bool m_selectedForGen{true};
        bool m_generating{false};
        bool m_focused{false};
        DisplayMode m_displayMode{DisplayMode::Waveform};
        std::shared_ptr<const netbender::sampler::SampleSlot> m_slot;
        juce::Image m_specImage;
        float m_playhead{0.0f};
        bool m_playing{false};
        float m_loop_start{0.0f};
        float m_loop_end{1.0f};
        DragTarget m_drag{DragTarget::None};
    };

    // MidiInputCallback
    auto handleIncomingMidiMessage(juce::MidiInput* source, const juce::MidiMessage& message)
        -> void override;

    // Timer, KeyListener & ChangeListener
    auto timerCallback() -> void override;
    auto keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent)
        -> bool override;
    auto changeListenerCallback(juce::ChangeBroadcaster* source) -> void override;

    auto refreshMidiCallbacks() -> void;
    auto openAudioSettings() -> void;
    auto chooseBundleFile() -> void;
    auto triggerGeneration() -> void;
    auto setSlotCount(int count) -> void;
    auto setViewMode(ViewMode mode, int focus_slot = -1) -> void;
    auto syncSlotSelectionUi(int slot_idx, bool selected) -> void;
    auto updateFocusInfoLabel() -> void;
    auto showWaveformContextMenu(int slot_idx) -> void;
    auto openExportModal(int default_slot_idx, bool select_all_populated, bool default_apply_range)
        -> void;
    auto exportSlotToFile(int slot_idx, bool apply_range, float start_norm, float end_norm,
                          ExportFormat format) -> void;
    auto exportSlotsToFiles(std::vector<int> slot_indices, bool apply_range,
                            float single_start_norm, float single_end_norm, ExportFormat format)
        -> void;

    class SamplerLookAndFeel final : public FlatLookAndFeel {
    public:
        auto drawProgressBar(juce::Graphics& g, juce::ProgressBar& bar, int width, int height,
                             double progress, const juce::String& textToShow) -> void override;
    };

    SamplerLookAndFeel m_lookAndFeel;
    netbender::sampler::SamplerEngine m_engine;
    std::unique_ptr<juce::FileChooser> m_fileChooser;
    std::unique_ptr<juce::FileChooser> m_exportChooser;

    // Background bundle loader state
    std::thread m_loadThread;
    std::atomic<bool> m_isLoadingBundle{false};
    std::atomic<bool> m_bundleLoadDone{false};
    std::atomic<bool> m_bundleLoadOk{false};
    juce::String m_pendingBundleName;
    juce::String m_loadErrorText;

    std::array<uint64_t, netbender::sampler::SamplerEngine::k_max_slots> m_lastSeenSlotIds{};
    std::array<DisplayMode, netbender::sampler::SamplerEngine::k_max_slots> m_slotDisplayModes{};
    double m_progressValue{0.0};
    ViewMode m_viewMode{ViewMode::Grid};

    // Header
    juce::Label m_titleLabel;
    juce::Label m_bundleLabel;
    juce::ComboBox m_slotCountCombo;
    juce::TextButton m_exportButton{"EXPORT"};
    juce::TextButton m_loadButton{"LOAD BUNDLE"};
    juce::TextButton m_audioMidiButton{"AUDIO / MIDI"};

    // Prompt bar
    juce::TextEditor m_promptEditor;
    std::array<juce::TextButton, netbender::sampler::SamplerEngine::k_max_slots>
        m_slotSelectButtons;
    juce::TextButton m_generateButton{"GENERATE"};
    juce::ProgressBar m_progressBar{m_progressValue};
    juce::Label m_statusLabel;

    // Focus bar (visible in Focus mode)
    juce::TextButton m_backToGridButton{"< GRID"};
    std::array<juce::TextButton, netbender::sampler::SamplerEngine::k_max_slots> m_focusSlotButtons;
    juce::Label m_focusInfoLabel;

    // Waveforms (Grid + Focus)
    std::array<WaveformView, netbender::sampler::SamplerEngine::k_max_slots> m_gridViews;
    WaveformView m_focusView;

    // Transport & generation controls
    juce::TextButton m_triggerButton{"TRIGGER"};
    juce::TextButton m_stopButton{"STOP"};
    juce::Label m_modeLabel{"", "MODE"};
    juce::ComboBox m_modeCombo;
    juce::Label m_durationLabel{"", "DUR (s)"};
    juce::Slider m_durationSlider;
    juce::Label m_stepsLabel{"", "STEPS"};
    juce::Slider m_stepsSlider;
    juce::Label m_speedLabel{"", "SPEED"};
    juce::Slider m_speedSlider;
    juce::Label m_gainLabel{"", "GAIN"};
    juce::Slider m_gainSlider;
    juce::Label m_seedLabel{"", "SEED"};
    juce::TextEditor m_seedEditor;
    juce::TextButton m_diceButton{"DICE"};
};
