#include "SamplerMainComponent.h"
#include "../ui/UIConstants.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace {

auto format_stage(netbender::sampler::GenerationProgress::Stage stage, int cur, int tot, double ms,
                  int batch_idx, int batch_total) -> juce::String {
    using Stage = netbender::sampler::GenerationProgress::Stage;
    const juce::String prefix =
        (batch_total > 1 && batch_idx > 0)
            ? ("Slot " + juce::String(batch_idx) + "/" + juce::String(batch_total) + " — ")
            : juce::String();
    switch (stage) {
        case Stage::Idle:
            return "Ready";
        case Stage::Tokenizing:
            return prefix + "Tokenizing prompt...";
        case Stage::EncodingPrompt:
            return prefix + "Encoding T5Gemma prompt...";
        case Stage::Diffusing:
            return prefix + "Diffusing step " + juce::String(cur) + " / " + juce::String(tot);
        case Stage::Decoding:
            return prefix + "Decoding SAME-S audio...";
        case Stage::Done:
            return "Generated " + juce::String(std::max(1, batch_total)) + " slot(s) in " +
                   juce::String(ms, 0) + " ms";
        case Stage::Cancelled:
            return "Cancelled";
        case Stage::Error:
            return "Generation error";
    }
    return "Ready";
}

}  // namespace

SamplerMainComponent::SamplerMainComponent(const juce::String& initial_bundle_path) {
    juce::LookAndFeel::setDefaultLookAndFeel(&m_look_and_feel);
    setLookAndFeel(&m_look_and_feel);
    setWantsKeyboardFocus(true);
    addKeyListener(this);

    // Header
    m_title_label.setText("NETBENDER SAMPLER", juce::dontSendNotification);
    m_title_label.setFont(juce::FontOptions(UIConstants::Typography::font_title, juce::Font::bold));
    m_title_label.setColour(juce::Label::textColourId, UIConstants::Colors::text_white);
    addAndMakeVisible(m_title_label);

    m_bundle_label.setText("No bundle loaded", juce::dontSendNotification);
    m_bundle_label.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
    addAndMakeVisible(m_bundle_label);

    m_slot_count_combo.addItem("4 Slots", 4);
    m_slot_count_combo.addItem("8 Slots", 8);
    m_slot_count_combo.addItem("12 Slots", 12);
    m_slot_count_combo.setSelectedId(4, juce::dontSendNotification);
    m_slot_count_combo.onChange = [this]() { set_slot_count(m_slot_count_combo.getSelectedId()); };
    addAndMakeVisible(m_slot_count_combo);

    m_export_button.onClick = [this]() {
        const int slot_idx = m_engine.get_active_slot();
        const float s = m_engine.get_loop_start(slot_idx);
        const float e = m_engine.get_loop_end(slot_idx);
        const bool has_custom = (s > 0.005f || e < 0.995f);
        open_export_modal(slot_idx, true, has_custom);
    };
    addAndMakeVisible(m_export_button);

    m_load_button.onClick = [this]() { choose_bundle_file(); };
    update_load_button_style(false);
    addAndMakeVisible(m_load_button);

    m_audio_midi_button.onClick = [this]() { open_audio_settings(); };
    addAndMakeVisible(m_audio_midi_button);

    // 2x2 Prompts (A, B, C, D)
    constexpr std::array<const char*, 4> k_corner_names{"A", "B", "C", "D"};
    constexpr std::array<const char*, 4> k_corner_hints{
        "Prompt A (top-left)...",
        "Prompt B (top-right, optional)...",
        "Prompt C (bottom-left, optional)...",
        "Prompt D (bottom-right, optional)...",
    };
    for (int i = 0; i < 4; ++i) {
        auto& lbl = m_prompt_labels[static_cast<size_t>(i)];
        lbl.setText(k_corner_names[static_cast<size_t>(i)], juce::dontSendNotification);
        lbl.setFont(juce::FontOptions(UIConstants::Typography::font_small, juce::Font::bold));
        lbl.setColour(juce::Label::textColourId, UIConstants::Colors::lilac);
        lbl.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl);

        auto& ed = m_prompt_editors[static_cast<size_t>(i)];
        ed.setTextToShowWhenEmpty(k_corner_hints[static_cast<size_t>(i)],
                                  UIConstants::Colors::text_dimmed);
        ed.onReturnKey = [this]() { trigger_generation(); };
        ed.onTextChange = [this]() { sync_active_prompt_corners(); };
        addAndMakeVisible(ed);
    }
    m_prompt_editors[0].setText("Amen break 174 BPM", false);

    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        auto& btn = m_slot_select_buttons[static_cast<size_t>(i)];
        btn.setButtonText(juce::String(i + 1));
        btn.setClickingTogglesState(true);
        btn.setToggleState(true, juce::dontSendNotification);
        btn.setColour(juce::TextButton::buttonOnColourId, UIConstants::Colors::violet);
        btn.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::surface_card);
        btn.setTooltip("Include Slot " + juce::String(i + 1) + " in next Generate");
        btn.onClick = [this, i]() {
            sync_slot_selection_ui(i,
                                   m_slot_select_buttons[static_cast<size_t>(i)].getToggleState());
        };
        addAndMakeVisible(btn);
    }

    m_generate_button.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::violet);
    m_generate_button.onClick = [this]() {
        using Stage = netbender::sampler::GenerationProgress::Stage;
        const auto st = m_engine.get_stage();
        if (st == Stage::Tokenizing || st == Stage::EncodingPrompt || st == Stage::Diffusing ||
            st == Stage::Decoding) {
            m_engine.cancel_generation();
        } else {
            static thread_local std::mt19937_64 rng{std::random_device{}()};
            m_seed_editor.setText(juce::String(rng() % 1000000ULL), false);
            trigger_generation();
        }
    };
    addAndMakeVisible(m_generate_button);

    addAndMakeVisible(m_progress_bar);
    m_status_label.setText("Idle", juce::dontSendNotification);
    m_status_label.setColour(juce::Label::textColourId, UIConstants::Colors::text_light);
    addAndMakeVisible(m_status_label);

    // Focus bar
    m_back_to_grid_button.onClick = [this]() { set_view_mode(ViewMode::Grid); };
    addChildComponent(m_back_to_grid_button);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        auto& fb = m_focus_slot_buttons[static_cast<size_t>(i)];
        fb.setButtonText(juce::String(i + 1));
        fb.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::surface_card);
        fb.onClick = [this, i]() { set_view_mode(ViewMode::Focus, i); };
        addChildComponent(fb);
    }
    m_focus_info_label.setColour(juce::Label::textColourId, UIConstants::Colors::text_light);
    m_focus_info_label.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    addChildComponent(m_focus_info_label);

    // Grid Waveforms & Focus Waveform
    m_slot_display_modes.fill(SamplerWaveformView::DisplayMode::Waveform);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        auto& gv = m_grid_views[static_cast<size_t>(i)];
        gv.set_slot_index(i);
        gv.set_compact_mode(true);
        gv.set_selected_for_gen(true);
        gv.set_focused(i == 0);
        gv.on_focus_requested = [this](int slot_idx) { set_view_mode(ViewMode::Focus, slot_idx); };
        gv.on_toggle_play = [this](int slot_idx) {
            select_active_slot(slot_idx);
            if (m_engine.is_slot_playing(slot_idx)) {
                m_engine.stop_slot_rt(slot_idx);
            } else {
                m_engine.trigger_slot_rt(slot_idx, 60, 1.0f);
            }
        };
        gv.on_toggle_regen = [this](int slot_idx) {
            const bool next =
                !m_slot_select_buttons[static_cast<size_t>(slot_idx)].getToggleState();
            sync_slot_selection_ui(slot_idx, next);
        };
        gv.on_display_mode_changed = [this](int slot_idx, SamplerWaveformView::DisplayMode mode) {
            m_slot_display_modes[static_cast<size_t>(slot_idx)] = mode;
            if (slot_idx == m_engine.get_active_slot()) {
                m_focus_view.set_display_mode(mode);
            }
        };
        gv.on_right_click = [this](int slot_idx) { show_waveform_context_menu(slot_idx); };
        addAndMakeVisible(gv);
    }

    m_focus_view.set_compact_mode(false);
    m_focus_view.set_slot_index(0);
    m_focus_view.on_loop_changed = [this](float s, float e) {
        const int active = m_engine.get_active_slot();
        m_engine.set_loop_region(active, s, e);
        m_grid_views[static_cast<size_t>(active)].set_loop_region(s, e);
        update_focus_info_label();
    };
    m_focus_view.on_display_mode_changed = [this](int slot_idx,
                                                  SamplerWaveformView::DisplayMode mode) {
        m_slot_display_modes[static_cast<size_t>(slot_idx)] = mode;
        m_grid_views[static_cast<size_t>(slot_idx)].set_display_mode(mode);
    };
    m_focus_view.on_right_click = [this](int slot_idx) { show_waveform_context_menu(slot_idx); };
    addChildComponent(m_focus_view);

    m_bend_panel.on_state_edited = [this]() {
        m_slot_bend_states[static_cast<size_t>(m_user_selected_slot)] =
            m_bend_panel.get_state(collect_prompts());
    };
    m_bend_panel.on_bend_changed = [this]() {
        if (!m_engine.is_loaded()) {
            return;
        }
        m_auto_regen_target_slot = m_user_selected_slot;
        m_auto_regen_pending = true;
        m_auto_regen_deadline_ms = juce::Time::getMillisecondCounter() + 150;
    };
    addAndMakeVisible(m_bend_panel);

    juce::File dev_prompts =
        juce::File::getCurrentWorkingDirectory().getChildFile("dev_prompts.txt");
    if (!dev_prompts.existsAsFile()) {
        juce::File dir =
            juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
        for (int depth = 0; depth < 12 && dir.exists(); ++depth) {
            const juce::File candidate = dir.getChildFile("dev_prompts.txt");
            if (candidate.existsAsFile()) {
                dev_prompts = candidate;
                break;
            }
            const juce::File parent = dir.getParentDirectory();
            if (parent == dir) {
                break;
            }
            dir = parent;
        }
    }
    if (dev_prompts.existsAsFile()) {
        juce::StringArray raw_lines;
        dev_prompts.readLines(raw_lines);
        juce::StringArray prompts;
        for (const auto& line : raw_lines) {
            const auto trimmed = line.trim();
            if (trimmed.isNotEmpty() && !trimmed.startsWithChar('#')) {
                prompts.add(trimmed);
            }
        }
        for (int i = 0; i < 4 && i < prompts.size(); ++i) {
            m_prompt_editors[static_cast<size_t>(i)].setText(prompts[i], false);
        }
    }

    for (auto& st : m_slot_bend_states) {
        st.config.enabled = true;
    }
    sync_active_prompt_corners();

    set_slot_count(4);

    // Playback & seed controls
    addAndMakeVisible(m_mode_label);
    m_mode_combo.addItem("One-Shot", 1);
    m_mode_combo.addItem("Loop", 2);
    m_mode_combo.addItem("Gate", 3);
    m_mode_combo.setSelectedId(2, juce::dontSendNotification);
    m_engine.set_playback_mode(netbender::sampler::SamplerPlaybackMode::Loop);
    m_mode_combo.onChange = [this]() {
        const int id = m_mode_combo.getSelectedId();
        using Mode = netbender::sampler::SamplerPlaybackMode;
        if (id == 1) {
            m_engine.set_playback_mode(Mode::OneShot);
        } else if (id == 2) {
            m_engine.set_playback_mode(Mode::Loop);
        } else {
            m_engine.set_playback_mode(Mode::Gate);
        }
    };
    addAndMakeVisible(m_mode_combo);

    auto setup_slider = [this](juce::Label& lbl, juce::Slider& sl, double min_v, double max_v,
                               double step, double init_v) {
        lbl.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
        lbl.setFont(juce::FontOptions(UIConstants::Typography::font_small));
        addAndMakeVisible(lbl);
        sl.setSliderStyle(juce::Slider::LinearHorizontal);
        sl.setTextBoxStyle(juce::Slider::TextBoxRight, false, 42, 18);
        sl.setRange(min_v, max_v, step);
        sl.setValue(init_v, juce::dontSendNotification);
        addAndMakeVisible(sl);
    };

    setup_slider(m_speed_label, m_speed_slider, -2.0, 2.0, 0.01, 1.0);
    setup_slider(m_gain_label, m_gain_slider, 0.0, 1.5, 0.01, 1.0);

    m_speed_slider.onValueChange = [this]() {
        m_engine.set_speed(static_cast<float>(m_speed_slider.getValue()));
    };
    m_gain_slider.onValueChange = [this]() {
        m_engine.set_gain(static_cast<float>(m_gain_slider.getValue()));
    };

    m_seed_label.setColour(juce::Label::textColourId, UIConstants::Colors::text_muted);
    m_seed_label.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    addAndMakeVisible(m_seed_label);
    m_seed_editor.setText("42", false);
    m_seed_editor.setInputRestrictions(12, "0123456789");
    m_seed_editor.onReturnKey = [this]() { trigger_generation(); };
    addAndMakeVisible(m_seed_editor);

    setAudioChannels(0, 2);
    deviceManager.addChangeListener(this);
    refresh_midi_callbacks();
    startTimerHz(30);

    setSize(1040, 600);

    if (initial_bundle_path.isNotEmpty()) {
        load_bundle_async(initial_bundle_path);
    } else {
        const juce::File default_bundle = juce::File::getCurrentWorkingDirectory().getChildFile(
            "models/sa3_small_music/sa3_small_music.nbsampler.json");
        if (default_bundle.existsAsFile()) {
            load_bundle_async(default_bundle.getFullPathName());
        }
    }
}

SamplerMainComponent::~SamplerMainComponent() {
    deviceManager.removeChangeListener(this);
    removeKeyListener(this);
    stopTimer();
    if (m_load_thread.joinable()) {
        m_load_thread.join();
    }
    for (const auto& info : juce::MidiInput::getAvailableDevices()) {
        deviceManager.removeMidiInputDeviceCallback(info.identifier, this);
    }
    shutdownAudio();
    juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
    setLookAndFeel(nullptr);
}

auto SamplerMainComponent::prepareToPlay(int samplesPerBlockExpected, double sampleRate) -> void {
    m_engine.prepare(sampleRate, samplesPerBlockExpected);
}

auto SamplerMainComponent::getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)
    -> void {
    auto* buf = bufferToFill.buffer;
    if (buf == nullptr) {
        return;
    }
    const int total_ch = buf->getNumChannels();
    if (total_ch <= 0 || bufferToFill.numSamples <= 0) {
        return;
    }
    float* ptrs[2]{
        buf->getWritePointer(0, bufferToFill.startSample),
        (total_ch > 1) ? buf->getWritePointer(1, bufferToFill.startSample) : nullptr,
    };
    m_engine.process_rt(ptrs, std::min(total_ch, 2), bufferToFill.numSamples);
    for (int c = 2; c < total_ch; ++c) {
        buf->clear(c, bufferToFill.startSample, bufferToFill.numSamples);
    }
}

auto SamplerMainComponent::releaseResources() -> void {}

auto SamplerMainComponent::handleIncomingMidiMessage(juce::MidiInput*,
                                                     const juce::MidiMessage& message) -> void {
    if (message.isNoteOn()) {
        m_engine.trigger_rt(message.getNoteNumber(), message.getFloatVelocity());
    } else if (message.isNoteOff()) {
        m_engine.release_rt(message.getNoteNumber());
    } else if (message.isAllNotesOff() || message.isAllSoundOff()) {
        m_engine.stop_rt();
    }
}

auto SamplerMainComponent::refresh_midi_callbacks() -> void {
    for (const auto& info : juce::MidiInput::getAvailableDevices()) {
        deviceManager.removeMidiInputDeviceCallback(info.identifier, this);
        if (deviceManager.isMidiInputDeviceEnabled(info.identifier)) {
            deviceManager.addMidiInputDeviceCallback(info.identifier, this);
        }
    }
}

auto SamplerMainComponent::changeListenerCallback(juce::ChangeBroadcaster* source) -> void {
    if (source == &deviceManager) {
        refresh_midi_callbacks();
    }
}

auto SamplerMainComponent::open_audio_settings() -> void {
    auto selector = std::make_unique<juce::AudioDeviceSelectorComponent>(deviceManager, 0, 0, 2, 2,
                                                                         true, false, true, false);
    selector->setSize(520, 420);
    juce::DialogWindow::LaunchOptions opts;
    opts.content.setOwned(selector.release());
    opts.dialogTitle = "Audio / MIDI Settings";
    opts.dialogBackgroundColour = UIConstants::Colors::surface_panel;
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true;
    opts.resizable = false;
    opts.launchAsync();
}

auto SamplerMainComponent::choose_bundle_file() -> void {
    m_file_chooser = std::make_unique<juce::FileChooser>(
        "Select Stable Audio 3 Sampler Bundle (.nbsampler.json)",
        juce::File::getCurrentWorkingDirectory().getChildFile("models"), "*.nbsampler.json");
    constexpr auto flags =
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    m_file_chooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
        const auto file = fc.getResult();
        if (file.existsAsFile()) {
            load_bundle_async(file.getFullPathName());
        }
    });
}

auto SamplerMainComponent::load_bundle_async(const juce::String& manifest_path) -> void {
    if (m_is_loading_bundle.load()) {
        return;
    }
    if (m_load_thread.joinable()) {
        m_load_thread.join();
    }
    m_is_loading_bundle.store(true);
    m_bundle_load_done.store(false);
    m_pending_bundle_name = juce::File(manifest_path).getFileNameWithoutExtension();
    m_bundle_label.setText("Loading " + m_pending_bundle_name + "...", juce::dontSendNotification);
    m_status_label.setText("Loading CoreML / ExecuTorch bundle...", juce::dontSendNotification);

    const std::string path_str = manifest_path.toStdString();
    m_load_thread = std::thread([this, path_str]() {
        const bool ok = m_engine.load_bundle(path_str, true);
        if (!ok) {
            m_load_error_text = m_engine.get_last_error();
        }
        m_bundle_load_ok.store(ok, std::memory_order_release);
        m_bundle_load_done.store(true, std::memory_order_release);
        m_is_loading_bundle.store(false, std::memory_order_release);
    });
}

auto SamplerMainComponent::collect_prompts() const -> std::array<std::string, 4> {
    return {
        m_prompt_editors[0].getText().trim().toStdString(),
        m_prompt_editors[1].getText().trim().toStdString(),
        m_prompt_editors[2].getText().trim().toStdString(),
        m_prompt_editors[3].getText().trim().toStdString(),
    };
}

auto SamplerMainComponent::update_load_button_style(bool loaded) -> void {
    m_bundle_loaded = loaded;
    if (loaded) {
        m_load_button.setColour(juce::TextButton::buttonColourId, UIConstants::Colors::violet);
        m_load_button.setColour(juce::TextButton::textColourOffId, UIConstants::Colors::text_white);
    } else {
        m_load_button.setColour(juce::TextButton::buttonColourId,
                                UIConstants::Colors::surface_card);
        m_load_button.setColour(juce::TextButton::textColourOffId, UIConstants::Colors::text_white);
    }
    m_load_button.repaint();
    repaint(m_load_button.getBounds().expanded(2));
}

auto SamplerMainComponent::sync_active_prompt_corners() -> void {
    const auto prompts = collect_prompts();
    m_bend_panel.set_active_corners({
        !prompts[0].empty(),
        !prompts[1].empty(),
        !prompts[2].empty(),
        !prompts[3].empty(),
    });
}

auto SamplerMainComponent::trigger_generation(uint32_t override_slot_mask) -> void {
    if (!m_engine.is_loaded()) {
        m_status_label.setText("Load a .nbsampler.json bundle first", juce::dontSendNotification);
        return;
    }
    const int slot_count = m_engine.get_slot_count();
    uint32_t slot_mask = override_slot_mask;
    if (slot_mask == 0) {
        for (int i = 0; i < slot_count; ++i) {
            if (m_slot_select_buttons[static_cast<size_t>(i)].getToggleState()) {
                slot_mask |= (1u << i);
            }
        }
    }
    if (slot_mask == 0) {
        m_status_label.setText(
            "Select at least one slot (1-" + juce::String(slot_count) + ") to generate",
            juce::dontSendNotification);
        return;
    }

    const auto prompts = collect_prompts();
    std::string primary_prompt;
    for (const auto& p : prompts) {
        if (!p.empty()) {
            primary_prompt = p;
            break;
        }
    }

    const auto active_state = m_bend_panel.get_state(prompts);
    if (m_user_selected_slot >= 0 &&
        m_user_selected_slot < netbender::sampler::SamplerEngine::k_max_slots) {
        m_slot_bend_states[static_cast<size_t>(m_user_selected_slot)] = active_state;
    }

    const uint64_t base_seed =
        static_cast<uint64_t>(std::max<int64_t>(0, m_seed_editor.getText().getLargeIntValue()));
    const int cols = (slot_count <= 4) ? 2 : 4;
    const int rows = (slot_count + cols - 1) / cols;

    std::array<netbender::sampler::GenerationRequest,
               netbender::sampler::SamplerEngine::k_max_slots>
        slot_reqs{};
    for (int i = 0; i < slot_count; ++i) {
        if ((slot_mask & (1u << i)) == 0) {
            continue;
        }
        PromptBendPanel::State st = active_state;
        if (override_slot_mask == 0) {
            const auto snap = m_engine.get_slot_snapshot(i);
            const bool is_empty = (!snap || snap->num_samples <= 0);
            st = is_empty ? active_state : m_slot_bend_states[static_cast<size_t>(i)];
            if (is_empty) {
                const float gx =
                    (cols > 1) ? static_cast<float>(i % cols) / static_cast<float>(cols - 1) : 0.0f;
                const float gy =
                    (rows > 1) ? static_cast<float>(i / cols) / static_cast<float>(rows - 1) : 0.0f;
                st.config.morph_x = gx;
                st.config.morph_y = gy;
            }
            st.config.prompts = prompts;
        }
        m_slot_bend_states[static_cast<size_t>(i)] = st;
        if (i == m_user_selected_slot) {
            m_bend_panel.load_state(st);
        }

        auto& req = slot_reqs[static_cast<size_t>(i)];
        req.prompt = primary_prompt;
        req.duration_sec = st.duration_sec;
        req.steps = st.steps;
        req.seed = base_seed + static_cast<uint64_t>(i);
        req.decode_audio = true;
        req.prompt_bend = st.config;
        if (req.prompt_bend.enabled) {
            req.cfg_scale = st.cfg_scale;
        }
    }
    m_engine.request_generate_slots(slot_reqs, slot_mask);
}

auto SamplerMainComponent::select_active_slot(int slot_idx) -> void {
    const int clamped = std::clamp(slot_idx, 0, m_engine.get_slot_count() - 1);
    if (m_loaded_bend_slot >= 0 &&
        m_loaded_bend_slot < netbender::sampler::SamplerEngine::k_max_slots) {
        m_slot_bend_states[static_cast<size_t>(m_loaded_bend_slot)] =
            m_bend_panel.get_state(collect_prompts());
    }
    m_user_selected_slot = clamped;
    m_loaded_bend_slot = clamped;
    m_engine.set_active_slot(clamped);
    m_bend_panel.load_state(m_slot_bend_states[static_cast<size_t>(clamped)]);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        m_grid_views[static_cast<size_t>(i)].set_focused(i == clamped);
    }
}

auto SamplerMainComponent::set_slot_count(int count) -> void {
    m_engine.set_slot_count(count);
    const int snapped = m_engine.get_slot_count();
    if (m_loaded_bend_slot >= snapped) {
        select_active_slot(snapped - 1);
    }
    const bool is_focus = (m_view_mode == ViewMode::Focus);

    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        const bool in_range = (i < snapped);
        m_slot_select_buttons[static_cast<size_t>(i)].setVisible(in_range);
        m_grid_views[static_cast<size_t>(i)].setVisible(!is_focus && in_range);
        m_focus_slot_buttons[static_cast<size_t>(i)].setVisible(is_focus && in_range);
    }

    if (is_focus) {
        set_view_mode(ViewMode::Focus, m_engine.get_active_slot());
    } else {
        resized();
        repaint();
    }
}

auto SamplerMainComponent::sync_slot_selection_ui(int slot_idx, bool selected) -> void {
    if (slot_idx < 0 || slot_idx >= netbender::sampler::SamplerEngine::k_max_slots) {
        return;
    }
    m_slot_select_buttons[static_cast<size_t>(slot_idx)].setToggleState(selected,
                                                                        juce::dontSendNotification);
    m_grid_views[static_cast<size_t>(slot_idx)].set_selected_for_gen(selected);
}

auto SamplerMainComponent::set_view_mode(ViewMode mode, int focus_slot) -> void {
    const int slot_count = m_engine.get_slot_count();
    if (focus_slot >= 0 && focus_slot < slot_count) {
        select_active_slot(focus_slot);
    }
    m_view_mode = mode;
    const int active = std::clamp(m_engine.get_active_slot(), 0, slot_count - 1);

    const bool is_focus = (m_view_mode == ViewMode::Focus);
    m_back_to_grid_button.setVisible(is_focus);
    m_focus_info_label.setVisible(is_focus);
    m_focus_view.setVisible(is_focus);
    for (int i = 0; i < netbender::sampler::SamplerEngine::k_max_slots; ++i) {
        const bool in_range = (i < slot_count);
        m_grid_views[static_cast<size_t>(i)].setVisible(!is_focus && in_range);
        m_grid_views[static_cast<size_t>(i)].set_focused(i == active);
        auto& fb = m_focus_slot_buttons[static_cast<size_t>(i)];
        fb.setVisible(is_focus && in_range);
        fb.setColour(juce::TextButton::buttonColourId, (i == active)
                                                           ? UIConstants::Colors::violet
                                                           : UIConstants::Colors::surface_card);
    }

    if (is_focus) {
        m_focus_view.set_slot_index(active);
        m_focus_view.set_display_mode(m_slot_display_modes[static_cast<size_t>(active)]);
        m_focus_view.set_slot(m_engine.get_slot_snapshot(active));
        m_focus_view.set_loop_region(m_engine.get_loop_start(active),
                                     m_engine.get_loop_end(active));
        update_focus_info_label();
    }

    resized();
    repaint();
}

auto SamplerMainComponent::update_focus_info_label() -> void {
    const int active = m_engine.get_active_slot();
    const auto snap = m_engine.get_slot_snapshot(active);
    juce::String info = "SLOT " + juce::String(active + 1);
    if (snap && snap->num_samples > 0) {
        const float dur = snap->duration_sec;
        const float s_sec = m_engine.get_loop_start(active) * dur;
        const float e_sec = m_engine.get_loop_end(active) * dur;
        info += "  •  Seed #" + juce::String(snap->seed) + "  •  " + juce::String(dur, 2) +
                "s  •  Range: " + juce::String(s_sec, 2) + "s - " + juce::String(e_sec, 2) + "s";
    } else {
        info += "  •  (empty)";
    }
    m_focus_info_label.setText(info, juce::dontSendNotification);
}

auto SamplerMainComponent::show_waveform_context_menu(int slot_idx) -> void {
    const auto snap = m_engine.get_slot_snapshot(slot_idx);
    const bool has_audio = (snap && snap->num_samples > 0);
    const float s = m_engine.get_loop_start(slot_idx);
    const float e = m_engine.get_loop_end(slot_idx);

    juce::PopupMenu menu;
    if (has_audio) {
        menu.addItem(1, "Export Selection as WAV...");
        menu.addItem(2, "Export Full Sample as WAV...");
        menu.addItem(3, "Export Options...");
        menu.addSeparator();
        menu.addItem(4, m_engine.is_slot_playing(slot_idx) ? "Stop" : "Play");
    } else {
        menu.addItem(10, "Slot " + juce::String(slot_idx + 1) + " is empty", false, false);
    }

    if (m_view_mode == ViewMode::Grid) {
        menu.addItem(5, "Focus Slot " + juce::String(slot_idx + 1));
    } else {
        menu.addItem(6, "Back to Grid");
    }

    juce::Component::SafePointer<SamplerMainComponent> safe_this(this);
    menu.showMenuAsync(juce::PopupMenu::Options{}, [safe_this, slot_idx, s, e](int result) {
        if (safe_this == nullptr || result == 0) {
            return;
        }
        if (result == 1) {
            safe_this->export_slots_to_files({slot_idx}, true, s, e, SamplerExportFormat::Wav24);
        } else if (result == 2) {
            safe_this->export_slots_to_files({slot_idx}, false, 0.0f, 1.0f,
                                             SamplerExportFormat::Wav24);
        } else if (result == 3) {
            safe_this->open_export_modal(slot_idx, false, true);
        } else if (result == 4) {
            if (safe_this->m_engine.is_slot_playing(slot_idx)) {
                safe_this->m_engine.stop_slot_rt(slot_idx);
            } else {
                safe_this->m_engine.trigger_slot_rt(slot_idx, 60, 1.0f);
            }
        } else if (result == 5) {
            safe_this->set_view_mode(ViewMode::Focus, slot_idx);
        } else if (result == 6) {
            safe_this->set_view_mode(ViewMode::Grid);
        }
    });
}

auto SamplerMainComponent::open_export_modal(int default_slot_idx, bool select_all_populated,
                                             bool default_apply_range) -> void {
    juce::Component::SafePointer<SamplerMainComponent> safe_this(this);
    auto content = std::make_unique<ExportModalContent>(
        m_engine, default_slot_idx, select_all_populated, default_apply_range,
        [safe_this](std::vector<int> slots, bool apply_range, float s_norm, float e_norm,
                    SamplerExportFormat fmt) {
            if (safe_this != nullptr) {
                safe_this->export_slots_to_files(std::move(slots), apply_range, s_norm, e_norm,
                                                 fmt);
            }
        });

    juce::DialogWindow::LaunchOptions opts;
    opts.content.setOwned(content.release());
    opts.dialogTitle = "Export Sample(s)";
    opts.dialogBackgroundColour = UIConstants::Colors::surface_panel;
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = true;
    opts.resizable = false;
    opts.launchAsync();
}

auto SamplerMainComponent::export_slots_to_files(std::vector<int> slot_indices, bool apply_range,
                                                 float single_start_norm, float single_end_norm,
                                                 SamplerExportFormat format) -> void {
    if (slot_indices.empty()) {
        return;
    }

    auto initial_dir = juce::File::getSpecialLocation(juce::File::userDesktopDirectory);
    if (!initial_dir.exists()) {
        initial_dir = juce::File::getCurrentWorkingDirectory();
    }

    const bool is_aiff = (format == SamplerExportFormat::Aiff24);
    const juce::String ext = is_aiff ? ".aiff" : ".wav";
    const juce::String pattern = is_aiff ? "*.aiff;*.aif" : "*.wav";

    juce::Component::SafePointer<SamplerMainComponent> safe_this(this);

    if (slot_indices.size() == 1) {
        const int slot_idx = slot_indices[0];
        const auto snap = m_engine.get_slot_snapshot(slot_idx);
        if (!snap || snap->audio.empty() || snap->num_samples <= 0) {
            m_status_label.setText("Slot " + juce::String(slot_idx + 1) + " has no audio to export",
                                   juce::dontSendNotification);
            return;
        }
        const juce::String default_name =
            sanitize_export_filename(snap->prompt) + "_slot" + juce::String(slot_idx + 1) + ext;

        m_export_chooser = std::make_unique<juce::FileChooser>(
            "Export Audio Sample", initial_dir.getChildFile(default_name), pattern);
        constexpr auto flags = juce::FileBrowserComponent::saveMode |
                               juce::FileBrowserComponent::canSelectFiles |
                               juce::FileBrowserComponent::warnAboutOverwriting;
        m_export_chooser->launchAsync(
            flags, [safe_this, snap, apply_range, single_start_norm, single_end_norm,
                    format](const juce::FileChooser& fc) {
                if (safe_this == nullptr) {
                    return;
                }
                const juce::File dest = fc.getResult();
                if (dest == juce::File{}) {
                    return;
                }
                juce::String err;
                const bool ok = write_slot_audio_to_file(
                    *snap, dest, apply_range, single_start_norm, single_end_norm, format, err);
                safe_this->m_status_label.setText(
                    ok ? ("Exported " + dest.getFileName()) : ("Export failed: " + err),
                    juce::dontSendNotification);
            });
        return;
    }

    m_export_chooser = std::make_unique<juce::FileChooser>(
        "Select Folder to Export " + juce::String(slot_indices.size()) + " Samples", initial_dir);
    constexpr auto dir_flags =
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories;
    m_export_chooser->launchAsync(dir_flags, [safe_this, slot_indices = std::move(slot_indices),
                                              apply_range, format,
                                              ext](const juce::FileChooser& fc) {
        if (safe_this == nullptr) {
            return;
        }
        const juce::File dir = fc.getResult();
        if (!dir.isDirectory()) {
            return;
        }
        int exported = 0;
        juce::String last_err;
        for (int slot_idx : slot_indices) {
            const auto snap = safe_this->m_engine.get_slot_snapshot(slot_idx);
            if (!snap || snap->num_samples <= 0) {
                continue;
            }
            const float s_norm = safe_this->m_engine.get_loop_start(slot_idx);
            const float e_norm = safe_this->m_engine.get_loop_end(slot_idx);
            const juce::String fname =
                sanitize_export_filename(snap->prompt) + "_slot" + juce::String(slot_idx + 1) + ext;
            if (write_slot_audio_to_file(*snap, dir.getChildFile(fname), apply_range, s_norm,
                                         e_norm, format, last_err)) {
                ++exported;
            }
        }
        if (exported > 0) {
            safe_this->m_status_label.setText(
                "Exported " + juce::String(exported) + " slot(s) to " + dir.getFileName(),
                juce::dontSendNotification);
        } else {
            safe_this->m_status_label.setText("Export failed: " + last_err,
                                              juce::dontSendNotification);
        }
    });
}

auto SamplerMainComponent::timerCallback() -> void {
    if (m_auto_regen_pending && juce::Time::getMillisecondCounter() >= m_auto_regen_deadline_ms) {
        m_auto_regen_pending = false;
        const int target = std::clamp(m_auto_regen_target_slot, 0, m_engine.get_slot_count() - 1);
        trigger_generation(1u << target);
    }

    if (m_bundle_load_done.exchange(false)) {
        const bool ok = m_bundle_load_ok.load();
        update_load_button_style(ok);
        if (ok) {
            const auto& meta = m_engine.get_metadata();
            m_bundle_label.setText(m_pending_bundle_name + " [" + juce::String(meta.delegate) + "]",
                                   juce::dontSendNotification);
            m_status_label.setText("Bundle ready", juce::dontSendNotification);
            if (meta.max_duration_sec > 1.0f) {
                const double max_d = std::min(11.5, static_cast<double>(meta.max_duration_sec));
                m_bend_panel.set_duration_range(1.0, max_d);
            }
        } else {
            m_bundle_label.setText("Load failed", juce::dontSendNotification);
            m_status_label.setText(m_load_error_text, juce::dontSendNotification);
        }
    }

    const int active_slot = m_engine.get_active_slot();
    const int playing_slot = m_engine.get_playing_slot();
    const bool any_playing = m_engine.is_playing();
    const float playhead = m_engine.get_playhead_norm();
    const int gen_slot = m_engine.get_batch_current_slot();

    const int slot_count = m_engine.get_slot_count();
    for (int i = 0; i < slot_count; ++i) {
        const uint64_t sid = m_engine.get_slot_generation_id(i);
        if (sid != m_last_seen_slot_ids[static_cast<size_t>(i)]) {
            m_last_seen_slot_ids[static_cast<size_t>(i)] = sid;
            auto snap = m_engine.get_slot_snapshot(i);
            const float ls = m_engine.get_loop_start(i);
            const float le = m_engine.get_loop_end(i);
            m_grid_views[static_cast<size_t>(i)].set_slot(snap);
            m_grid_views[static_cast<size_t>(i)].set_loop_region(ls, le);
            if (i == active_slot) {
                m_focus_view.set_slot(snap);
                m_focus_view.set_loop_region(ls, le);
                update_focus_info_label();
            }
        }
        const bool slot_playing = any_playing && (playing_slot == i);
        m_grid_views[static_cast<size_t>(i)].set_playhead(slot_playing ? playhead : 0.0f,
                                                          slot_playing);
        m_grid_views[static_cast<size_t>(i)].set_generating(gen_slot == i);
    }

    const bool active_playing = any_playing && (playing_slot == active_slot);
    m_focus_view.set_playhead(active_playing ? playhead : 0.0f, active_playing);
    m_focus_view.set_generating(gen_slot == active_slot);

    using Stage = netbender::sampler::GenerationProgress::Stage;
    const Stage stage = m_engine.get_stage();
    const int cur = m_engine.get_current_step();
    const int tot = std::max(1, m_engine.get_total_steps());
    const int batch_idx = m_engine.get_batch_index();
    const int batch_tot = m_engine.get_batch_total();
    const bool busy = (stage == Stage::Tokenizing || stage == Stage::EncodingPrompt ||
                       stage == Stage::Diffusing || stage == Stage::Decoding);

    m_generate_button.setButtonText(busy ? "CANCEL" : "GENERATE");
    if (busy) {
        const double slot_frac =
            (stage == Stage::Decoding) ? 0.95 : static_cast<double>(cur) / static_cast<double>(tot);
        const int b_tot = std::max(1, batch_tot);
        const int b_done = std::max(0, batch_idx - 1);
        m_progress_value = (static_cast<double>(b_done) + slot_frac) / static_cast<double>(b_tot);
        m_status_label.setText(
            format_stage(stage, cur, tot, m_engine.get_last_gen_ms(), batch_idx, batch_tot),
            juce::dontSendNotification);
    } else if (!m_is_loading_bundle.load()) {
        m_progress_value = (stage == Stage::Done) ? 1.0 : 0.0;
        if (stage != Stage::Idle) {
            m_status_label.setText(stage == Stage::Error
                                       ? juce::String(m_engine.get_last_error())
                                       : format_stage(stage, cur, tot, m_engine.get_last_gen_ms(),
                                                      batch_idx, batch_tot),
                                   juce::dontSendNotification);
        }
    }
}

auto SamplerMainComponent::keyPressed(const juce::KeyPress& key, juce::Component* origin) -> bool {
    if (origin == &m_seed_editor) {
        return false;
    }
    for (const auto& ed : m_prompt_editors) {
        if (origin == &ed) {
            return false;
        }
    }
    if (key == juce::KeyPress::escapeKey && m_view_mode == ViewMode::Focus) {
        set_view_mode(ViewMode::Grid);
        return true;
    }
    if (key == juce::KeyPress::spaceKey) {
        const int sel = std::clamp(m_user_selected_slot, 0, m_engine.get_slot_count() - 1);
        if (m_engine.is_slot_playing(sel)) {
            m_engine.stop_slot_rt(sel);
        } else {
            m_engine.trigger_slot_rt(sel, 60, 1.0f);
        }
        return true;
    }
    if (!key.getModifiers().isCommandDown() && !key.getModifiers().isCtrlDown()) {
        const auto ch = key.getTextCharacter();
        if (ch == 'g') {
            const int sel = std::clamp(m_user_selected_slot, 0, m_engine.get_slot_count() - 1);
            const bool next = !m_slot_select_buttons[static_cast<size_t>(sel)].getToggleState();
            sync_slot_selection_ui(sel, next);
            return true;
        }
        if (ch == 'h' || ch == 'j' || ch == 'k' || ch == 'l') {
            const int slot_count = m_engine.get_slot_count();
            const int cols = (slot_count == 4) ? 2 : 4;
            const int rows = (slot_count + cols - 1) / cols;
            const int cur = std::clamp(m_user_selected_slot, 0, slot_count - 1);
            int row = cur / cols;
            int col = cur % cols;
            if (ch == 'h') {
                col = std::max(0, col - 1);
            } else if (ch == 'l') {
                col = std::min(cols - 1, col + 1);
            } else if (ch == 'k') {
                row = std::max(0, row - 1);
            } else {
                row = std::min(rows - 1, row + 1);
            }
            const int next = std::clamp(row * cols + col, 0, slot_count - 1);
            if (m_view_mode == ViewMode::Focus) {
                set_view_mode(ViewMode::Focus, next);
            } else {
                select_active_slot(next);
            }
            return true;
        }
    }
    return false;
}

auto SamplerMainComponent::paint(juce::Graphics& g) -> void {
    const juce::ColourGradient bg(UIConstants::Colors::background_gradient_top, 0.0f, 0.0f,
                                  UIConstants::Colors::background_gradient_bottom, 0.0f,
                                  static_cast<float>(getHeight()), false);
    g.setGradientFill(bg);
    g.fillAll();
}

auto SamplerMainComponent::paintOverChildren(juce::Graphics& g) -> void {
    if (!m_bundle_loaded && m_load_button.isVisible()) {
        g.setColour(UIConstants::Colors::violet);
        g.drawRoundedRectangle(m_load_button.getBounds().toFloat().reduced(0.75f), 4.0f, 1.75f);
    }
}

auto SamplerMainComponent::resized() -> void {
    auto area = getLocalBounds().reduced(14);

    // Header (36 px)
    auto header = area.removeFromTop(36);
    m_title_label.setBounds(header.removeFromLeft(220));
    m_audio_midi_button.setBounds(header.removeFromRight(115));
    header.removeFromRight(8);
    m_load_button.setBounds(header.removeFromRight(115));
    header.removeFromRight(8);
    m_export_button.setBounds(header.removeFromRight(90));
    header.removeFromRight(8);
    m_slot_count_combo.setBounds(header.removeFromRight(95));
    header.removeFromRight(10);
    m_bundle_label.setBounds(header);

    area.removeFromTop(10);

    // Prompt area (56 px: 2x2 A/B/C/D prompts on left, slot toggle grid + GENERATE/SEED on right)
    const int slot_count = m_engine.get_slot_count();
    auto prompt_area = area.removeFromTop(56);
    auto gen_col = prompt_area.removeFromRight(128);
    m_generate_button.setBounds(gen_col.removeFromTop(28));
    gen_col.removeFromTop(4);
    auto seed_row = gen_col;
    m_seed_label.setBounds(seed_row.removeFromLeft(34));
    m_seed_editor.setBounds(seed_row);
    prompt_area.removeFromRight(8);

    const int sel_cols = (slot_count == 4) ? 2 : 4;
    const int sel_rows = (slot_count + sel_cols - 1) / sel_cols;
    constexpr int k_sel_gap = 2;
    constexpr int k_sel_btn_w = 26;
    const int sel_grid_w = sel_cols * k_sel_btn_w + (sel_cols - 1) * k_sel_gap;
    auto sel_grid = prompt_area.removeFromRight(sel_grid_w);
    const int sel_btn_h =
        std::max(16, (sel_grid.getHeight() - (sel_rows - 1) * k_sel_gap) / sel_rows);
    const int sel_grid_h = sel_rows * sel_btn_h + (sel_rows - 1) * k_sel_gap;
    const int sel_y0 = sel_grid.getY() + (sel_grid.getHeight() - sel_grid_h) / 2;
    for (int i = 0; i < slot_count; ++i) {
        const int col = i % sel_cols;
        const int row = i / sel_cols;
        const int x = sel_grid.getX() + col * (k_sel_btn_w + k_sel_gap);
        const int y = sel_y0 + row * (sel_btn_h + k_sel_gap);
        m_slot_select_buttons[static_cast<size_t>(i)].setBounds(x, y, k_sel_btn_w, sel_btn_h);
    }
    prompt_area.removeFromRight(8);

    auto p_row_top = prompt_area.removeFromTop(26);
    prompt_area.removeFromTop(4);
    auto p_row_bot = prompt_area.removeFromTop(26);
    const int half_w = std::max(40, (p_row_top.getWidth() - 6) / 2);

    auto place_prompt = [](juce::Rectangle<int> cell, juce::Label& lbl, juce::TextEditor& ed) {
        lbl.setBounds(cell.removeFromLeft(18));
        ed.setBounds(cell);
    };
    place_prompt(p_row_top.removeFromLeft(half_w), m_prompt_labels[0], m_prompt_editors[0]);
    p_row_top.removeFromLeft(6);
    place_prompt(p_row_top, m_prompt_labels[1], m_prompt_editors[1]);
    place_prompt(p_row_bot.removeFromLeft(half_w), m_prompt_labels[2], m_prompt_editors[2]);
    p_row_bot.removeFromLeft(6);
    place_prompt(p_row_bot, m_prompt_labels[3], m_prompt_editors[3]);

    area.removeFromTop(8);

    // Progress, status & playback row (24 px)
    auto status_row = area.removeFromTop(24);
    m_gain_slider.setBounds(status_row.removeFromRight(115));
    m_gain_label.setBounds(status_row.removeFromRight(34));
    status_row.removeFromRight(8);
    m_speed_slider.setBounds(status_row.removeFromRight(125));
    m_speed_label.setBounds(status_row.removeFromRight(42));
    status_row.removeFromRight(8);
    m_mode_combo.setBounds(status_row.removeFromRight(84));
    m_mode_label.setBounds(status_row.removeFromRight(38));
    status_row.removeFromRight(10);

    m_progress_bar.setBounds(status_row.removeFromLeft(160));
    status_row.removeFromLeft(8);
    m_status_label.setBounds(status_row);

    area.removeFromTop(10);

    m_bend_panel.setBounds(area.removeFromRight(276));
    area.removeFromRight(10);

    // Center: Dynamic Grid (2x2, 4x2, 4x3) OR Focus Bar + Focus Waveform
    if (m_view_mode == ViewMode::Grid) {
        constexpr int k_gap = 8;
        const int cols = (slot_count <= 4) ? 2 : 4;
        const int rows = (slot_count + cols - 1) / cols;
        const int cell_w = std::max(10, (area.getWidth() - k_gap * (cols - 1)) / cols);
        const int cell_h = std::max(10, (area.getHeight() - k_gap * (rows - 1)) / rows);
        for (int i = 0; i < slot_count; ++i) {
            const int r = i / cols;
            const int c = i % cols;
            m_grid_views[static_cast<size_t>(i)].setBounds(area.getX() + c * (cell_w + k_gap),
                                                           area.getY() + r * (cell_h + k_gap),
                                                           cell_w, cell_h);
        }
    } else {
        auto focus_bar = area.removeFromTop(28);
        m_back_to_grid_button.setBounds(focus_bar.removeFromLeft(80));
        focus_bar.removeFromLeft(8);
        const int fb_w = (slot_count >= 12) ? 24 : 28;
        for (int i = 0; i < slot_count; ++i) {
            m_focus_slot_buttons[static_cast<size_t>(i)].setBounds(focus_bar.removeFromLeft(fb_w));
            focus_bar.removeFromLeft(3);
        }
        focus_bar.removeFromLeft(8);
        m_focus_info_label.setBounds(focus_bar);
        area.removeFromTop(8);
        m_focus_view.setBounds(area);
    }
}
