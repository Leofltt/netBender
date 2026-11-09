#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "ui/UIConstants.h"

//==============================================================================
NetBenderAudioProcessorEditor::NetBenderAudioProcessorEditor (NetBenderAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setLookAndFeel (&flatLf);

    // =========================================================================
    // Header Toolbar Listeners & Synchronization
    // =========================================================================
    load_button.addListener (this);
    gpu_button.addListener (this);
    method_combo.addListener (this);
    dry_wet_slider.addListener (this);
    save_model_button.addListener (this);
    fuse_button.addListener (this);
    short_circuit_button.addListener (this);
    trigger_mode_combo.addListener (this);

    buffer_status_label.setText (juce::String (audioProcessor.getBufferSize()), juce::dontSendNotification);
    gpu_button.setButtonText (audioProcessor.isUsingGpu() ? "GPU" : "CPU");
    gpu_button.setColour (juce::TextButton::buttonColourId,
                         audioProcessor.isUsingGpu() ? UIConstants::Colors::dark_cyan : UIConstants::Colors::button_secondary_bg);
    gpu_button.setColour (juce::TextButton::textColourOffId,
                         audioProcessor.isUsingGpu() ? juce::Colours::lightcyan : UIConstants::Colors::text_light);
    dry_wet_slider.setValue (audioProcessor.getDryWet());
    trigger_mode_combo.setSelectedId ((int)audioProcessor.getTriggerMode() + 1, juce::dontSendNotification);
    short_circuit_button.setVisible (audioProcessor.getTriggerMode() == NetBenderAudioProcessor::TriggerMode::Momentary);

    addAndMakeVisible (headerBar);

    // =========================================================================
    // Bending Canvas & Layer Management
    // =========================================================================
    addAndMakeVisible (bendingGroup);

    layerBar.view_weights_button.addListener (this);
    layerBar.view_latent_button.addListener (this);
    layerBar.display_curve_button.addListener (this);
    layerBar.display_matrix_button.addListener (this);
    layerBar.category_combo.addListener (this);
    layerBar.layer_combo.addListener (this);
    layerBar.reset_layer_button.addListener (this);
    layerBar.reset_all_button.addListener (this);
    bendingGroup.addAndMakeVisible (layerBar);

    bendingGroup.addAndMakeVisible (weightCanvas);
    weightCanvas.onWeightsModified = [this] (const std::vector<float>& modifiedWeights)
    {
        if (currentBendingLayer.isNotEmpty() && !modifiedWeights.empty())
        {
            float scale = (float)scale_slider.getValue();
            float offset = (float)offset_slider.getValue();

            baseDrawnWeights.resize (modifiedWeights.size());
            if (std::abs (scale) > 0.00001f)
            {
                for (size_t i = 0; i < modifiedWeights.size(); ++i)
                    baseDrawnWeights[i] = (modifiedWeights[i] - offset) / scale;
            }
            else
            {
                baseDrawnWeights = modifiedWeights;
            }

            audioProcessor.setLayerDrawnWeights (currentBendingLayer.toStdString(), baseDrawnWeights);

            auto mode = audioProcessor.getTriggerMode();
            if (mode == NetBenderAudioProcessor::TriggerMode::Continuous)
            {
                currentWeights = modifiedWeights;
                weightCanvas.setTargetBentWeights ({});
            }
            else
            {
                weightCanvas.setTargetBentWeights (modifiedWeights);
            }
        }
    };

    bendingGroup.addChildComponent (latentPad);
    latentPad.onCoordsChanged = [this] (float x, float y)
    {
        audioProcessor.setLatentCoords (x, y);
        if (auto* px = audioProcessor.getLatentXParam())
            px->setValueNotifyingHost (px->convertTo0to1 (x));
        if (auto* py = audioProcessor.getLatentYParam())
            py->setValueNotifyingHost (py->convertTo0to1 (y));
    };

    // =========================================================================
    // Side Panel & Control Tabs Listeners
    // =========================================================================
    sidePanel.tab_mutate_button.addListener (this);
    sidePanel.tab_harmonics_button.addListener (this);
    sidePanel.tab_bridge_button.addListener (this);

    mutate_enable_toggle.addListener (this);
    scale_slider.addListener (this);
    offset_slider.addListener (this);
    drift_enable_toggle.addListener (this);
    heat_slider.addListener (this);
    memory_slider.addListener (this);
    drift_mode_combo.addListener (this);
    freeze_button.addListener (this);

    harmonic_enable_toggle.addListener (this);
    harmonic_freq_slider.addListener (this);
    harmonic_partials_slider.addListener (this);
    harmonic_morph_slider.addListener (this);
    harmonic_depth_slider.addListener (this);
    harmonic_mode_combo.addListener (this);
    harmonic_apply_button.addListener (this);

    bridge_enable_toggle.addListener (this);
    bridge_combo.addListener (this);
    bridge_depth_slider.addListener (this);

    latent_enable_button.addListener (this);
    latent_depth_slider.setValue (audioProcessor.getLatentDepth());
    latent_depth_slider.addListener (this);
    latent_mode_combo.setSelectedId ((int)audioProcessor.getLatentMode() + 1, juce::dontSendNotification);
    latent_mode_combo.addListener (this);
    latent_slew_slider.setValue (audioProcessor.getLatentSlew());
    latent_slew_slider.addListener (this);

    bendingGroup.addAndMakeVisible (sidePanel);

    // =========================================================================
    // Status Bar & Initialization
    // =========================================================================
    infoBendingLabel.setText ("Select a layer to bend weights.", juce::dontSendNotification);
    infoBendingLabel.setJustificationType (juce::Justification::centredLeft);
    infoBendingLabel.setColour (juce::Label::textColourId, juce::Colours::silver);
    bendingGroup.addAndMakeVisible (infoBendingLabel);

    set_view_mode (ViewMode::Weights);
    set_side_tab (SideTab::Mutate);

    setSize (UIConstants::Layout::window_default_width, UIConstants::Layout::window_default_height);
    setResizable (true, true);
    setResizeLimits (UIConstants::Layout::window_min_width, UIConstants::Layout::window_min_height,
                     UIConstants::Layout::window_max_width, UIConstants::Layout::window_max_height);

    update_model_info();
    startTimerHz (UIConstants::Layout::editor_timer_hz);
}

NetBenderAudioProcessorEditor::~NetBenderAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

//==============================================================================
auto NetBenderAudioProcessorEditor::paint(juce::Graphics& g) -> void {
    juce::ColourGradient gradient(UIConstants::Colors::background_gradient_top, 0, 0,
                                   UIConstants::Colors::background_gradient_bottom, 0, (float)getHeight(), false);
    g.setGradientFill(gradient);
    g.fillAll();

    g.setColour(UIConstants::Colors::divider_line);
    g.drawHorizontalLine(UIConstants::Layout::header_divider_y, 0.0f, (float)getWidth());
}

auto NetBenderAudioProcessorEditor::resized() -> void {
    using namespace UIConstants::Layout;

    auto area = getLocalBounds().reduced(gap_large);

    headerBar.setBounds(area.removeFromTop(header_height));
    area.removeFromTop(gap_standard);

    bendingGroup.setBounds(area);
    auto groupArea = bendingGroup.getLocalBounds().reduced(gap_xlarge);
    groupArea.removeFromTop(gap_large);

    layerBar.setBounds(groupArea.removeFromTop(layer_bar_height));
    groupArea.removeFromTop(gap_medium);

    auto bottomRow = groupArea.removeFromBottom(status_bar_height);
    infoBendingLabel.setBounds(bottomRow);
    groupArea.removeFromBottom(gap_compact);

    auto sideArea = groupArea.removeFromRight(side_panel_width);
    groupArea.removeFromRight(gap_large);

    weightCanvas.setBounds(groupArea);
    latentPad.setBounds(groupArea);

    sidePanel.setBounds(sideArea);
}

//==============================================================================
auto NetBenderAudioProcessorEditor::comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) -> void {
    if (comboBoxThatHasChanged == &method_combo)
    {
        audioProcessor.setCurrentMethod (method_combo.getText());
        update_model_info();
    }
    else if (comboBoxThatHasChanged == &layer_combo)
    {
        select_layer (layer_combo.getText());
    }
    else if (comboBoxThatHasChanged == &category_combo)
    {
        int catId = category_combo.getSelectedId();
        layer_combo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        int id = 1;
        for (const auto& l : layers)
        {
            auto cat = NetBenderAudioProcessor::classifyLayer (l);
            bool include = false;
            if (catId == 1) include = true; // All
            else if (catId == 2 && cat == NetBenderAudioProcessor::LayerCategory::Norm) include = true;
            else if (catId == 3 && cat == NetBenderAudioProcessor::LayerCategory::Conv) include = true;
            else if (catId == 4 && cat == NetBenderAudioProcessor::LayerCategory::Attention) include = true;
            else if (catId == 5 && cat == NetBenderAudioProcessor::LayerCategory::Linear) include = true;
            else if (catId == 6 && cat == NetBenderAudioProcessor::LayerCategory::Activation) include = true;
            else if (catId == 7 && cat == NetBenderAudioProcessor::LayerCategory::Recurrent) include = true;
            else if (catId == 8 && cat == NetBenderAudioProcessor::LayerCategory::Bias) include = true;

            if (include)
                layer_combo.addItem (l, id++);
        }
        if (layer_combo.getNumItems() > 0)
        {
            layer_combo.setSelectedId (1);
            select_layer (layer_combo.getText());
        }
    }
    else if (comboBoxThatHasChanged == &trigger_mode_combo)
    {
        int modeIdx = trigger_mode_combo.getSelectedId() - 1;
        auto mode = (NetBenderAudioProcessor::TriggerMode)modeIdx;
        audioProcessor.setTriggerMode (mode);
        short_circuit_button.setVisible (mode == NetBenderAudioProcessor::TriggerMode::Momentary);
        resized();

        if (currentBendingLayer.isNotEmpty())
            apply_knob_bending();
    }
    else if (comboBoxThatHasChanged == &drift_mode_combo)
    {
        int modeIdx = drift_mode_combo.getSelectedId() - 1;
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerDriftMode (currentBendingLayer.toStdString(), (NetBenderAudioProcessor::DriftMode)modeIdx);
    }
    else if (comboBoxThatHasChanged == &bridge_combo)
    {
        if (currentBendingLayer.isNotEmpty())
        {
            juce::String src = (bridge_combo.getSelectedId() > 1) ? bridge_combo.getText() : "";
            audioProcessor.setLayerBridgeSource (currentBendingLayer.toStdString(), src.toStdString());
            apply_knob_bending();
        }
    }
    else if (comboBoxThatHasChanged == &harmonic_mode_combo)
    {
        updateHarmonicGhostPreview();
    }
    else if (comboBoxThatHasChanged == &latent_mode_combo)
    {
        int modeIdx = latent_mode_combo.getSelectedId() - 1;
        audioProcessor.setLatentMode ((NetBenderAudioProcessor::LatentMode)modeIdx);
        updateLatentControlsVisibility();
        resized();
    }
}

auto NetBenderAudioProcessorEditor::sliderValueChanged(juce::Slider* slider) -> void {
    if (slider == &harmonic_freq_slider || slider == &harmonic_partials_slider
        || slider == &harmonic_morph_slider || slider == &harmonic_depth_slider)
    {
        updateHarmonicGhostPreview();
    }
    else if (slider == &dry_wet_slider)
    {
        float val = (float)dry_wet_slider.getValue();
        audioProcessor.setDryWet (val);
        lastKnownDryWet = val;
    }
    else if (slider == &heat_slider)
    {
        float val = (float)heat_slider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getHeatParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownHeat = val;
    }
    else if (slider == &memory_slider)
    {
        float val = (float)memory_slider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getMemoryParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownMemory = val;
    }
    else if (slider == &scale_slider)
    {
        float val = (float)scale_slider.getValue();
        if (auto* param = audioProcessor.getScaleParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownScale = val;
        apply_knob_bending();
    }
    else if (slider == &offset_slider)
    {
        float val = (float)offset_slider.getValue();
        if (auto* param = audioProcessor.getOffsetParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownOffset = val;
        apply_knob_bending();
    }
    else if (slider == &bridge_depth_slider)
    {
        float val = (float)bridge_depth_slider.getValue();
        if (currentBendingLayer.isNotEmpty())
        {
            audioProcessor.setLayerBridgeDepth (currentBendingLayer.toStdString(), val);
            apply_knob_bending();
        }
    }
    else if (slider == &latent_depth_slider)
    {
        float val = (float)latent_depth_slider.getValue();
        audioProcessor.setLatentDepth (val);
        if (auto* param = audioProcessor.getLatentDepthParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
    }
    else if (slider == &latent_slew_slider)
    {
        float val = (float)latent_slew_slider.getValue();
        audioProcessor.setLatentSlew (val);
    }
    else
    {
        apply_knob_bending();
    }
}

auto NetBenderAudioProcessorEditor::buttonClicked(juce::Button* button) -> void {
    if (button == &fuse_button)
    {
        audioProcessor.resetFuse();
        fuse_button.setButtonText ("FUSE: OK");
        fuse_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::forest_green);
        fuse_button.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        if (currentBendingLayer.isNotEmpty())
            select_layer (currentBendingLayer);
    }
    else if (button == &gpu_button)
    {
        bool nextGpu = !audioProcessor.isUsingGpu();
        audioProcessor.setUseGpu (nextGpu);
        update_model_info();
    }
    else if (button == &short_circuit_button)
    {
        // Toggle or pulse momentary short-circuit gate
        bool active = !audioProcessor.isShortCircuitActive();
        audioProcessor.triggerShortCircuit (active);
        short_circuit_button.setColour (juce::TextButton::buttonColourId,
                                      active ? UIConstants::Colors::short_circuit_active : UIConstants::Colors::crimson);
    }
    else if (button == &freeze_button)
    {
        bool frozen = freeze_button.getToggleState();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerFrozen (currentBendingLayer.toStdString(), frozen);

        if (frozen)
        {
            freeze_button.setColour (juce::ToggleButton::textColourId, UIConstants::Colors::drift_orange);
            if (currentBendingLayer.isNotEmpty())
            {
                currentWeights = audioProcessor.getBendingManager().get_layer_live_weights (currentBendingLayer.toStdString());
                if (currentWeights.empty())
                    currentWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
                float scale = (float)scale_slider.getValue();
                float offset = (float)offset_slider.getValue();
                baseDrawnWeights.resize (currentWeights.size());
                if (std::abs (scale) > 0.00001f)
                {
                    for (size_t i = 0; i < currentWeights.size(); ++i)
                        baseDrawnWeights[i] = (currentWeights[i] - offset) / scale;
                }
                else
                {
                    baseDrawnWeights = currentWeights;
                }
                audioProcessor.setLayerBaseWeights (currentBendingLayer.toStdString(), baseDrawnWeights);
                weightCanvas.updateCurrentWeights (currentWeights);
            }
        }
        else
        {
            freeze_button.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
        }
    }
    else if (button == &load_button)
    {
        fileChooser = std::make_unique<juce::FileChooser> (
            "Select a Neural Audio Model (.pte, .onnx, .ts)...",
            juce::File::getSpecialLocation (juce::File::userHomeDirectory),
            "*.pte;*.onnx;*.ts"
        );
        
        auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
        fileChooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
        {
            auto file = fc.getResult();
            if (file.existsAsFile())
            {
                currentBendingLayer = "";
                if (audioProcessor.loadModel (file))
                {
                    update_model_info();
                }
                else
                {
                    status_label.setText ("Error: Failed to load " + file.getFileName(), juce::dontSendNotification);
                    status_label.setColour (juce::Label::textColourId, juce::Colours::orangered);
                }
            }
        });
    }
    else if (button == &reset_layer_button)
    {
        reset_layer_weights();
    }
    else if (button == &reset_all_button)
    {
        reset_all_layer_weights();
    }
    else if (button == &save_model_button)
    {
        save_model_to_file();
    }
    else if (button == &harmonic_apply_button)
    {
        applyHarmonicWeights();
    }
    else if (button == &view_weights_button)
    {
        set_view_mode (ViewMode::Weights);
    }
    else if (button == &view_latent_button)
    {
        set_view_mode (ViewMode::Latent);
    }
    else if (button == &latent_enable_button)
    {
        bool en = latent_enable_button.getToggleState();
        audioProcessor.setLatentEnabled (en);
        latentPad.setEnabledState (en, audioProcessor.hasAutoencode());
    }
    else if (button == &tab_mutate_button)
    {
        set_side_tab (SideTab::Mutate);
    }
    else if (button == &tab_harmonics_button)
    {
        set_side_tab (SideTab::Harmonics);
    }
    else if (button == &tab_bridge_button)
    {
        set_side_tab (SideTab::Bridge);
    }
    else if (button == &mutate_enable_toggle)
    {
        apply_knob_bending();
    }
    else if (button == &drift_enable_toggle)
    {
        if (!drift_enable_toggle.getToggleState() && currentBendingLayer.isNotEmpty())
        {
            auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
            state.driftOffsets.assign (state.driftOffsets.size(), 0.0f);
            audioProcessor.setLayerState (currentBendingLayer.toStdString(), state);
        }
    }
    else if (button == &bridge_enable_toggle)
    {
        bool en = bridge_enable_toggle.getToggleState();
        weightCanvas.setShowBridgeVisuals (en);
        apply_knob_bending();
    }
    else if (button == &harmonic_enable_toggle)
    {
        bool en = harmonic_enable_toggle.getToggleState();
        weightCanvas.setShowHarmonics (en);
        updateHarmonicGhostPreview();
    }
    else if (button == &display_curve_button)
    {
        weightCanvas.setRenderMode (WeightBendingComponent::RenderMode::Curve);
        display_curve_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::deep_violet);
        display_curve_button.setColour (juce::TextButton::textColourOffId, UIConstants::Colors::text_white);
        display_matrix_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::button_secondary_bg);
        display_matrix_button.setColour (juce::TextButton::textColourOffId, UIConstants::Colors::text_dimmed);
    }
    else if (button == &display_matrix_button)
    {
        weightCanvas.setRenderMode (WeightBendingComponent::RenderMode::Matrix);
        display_matrix_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::deep_violet);
        display_matrix_button.setColour (juce::TextButton::textColourOffId, UIConstants::Colors::text_white);
        display_curve_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::button_secondary_bg);
        display_curve_button.setColour (juce::TextButton::textColourOffId, UIConstants::Colors::text_dimmed);
    }
}

//==============================================================================
auto NetBenderAudioProcessorEditor::timerCallback() -> void {
    if (audioProcessor.isModelLoaded() && status_label.getText() == "No model loaded.")
    {
        update_model_info();
    }

    // If DAW host is automating parameters, reflect changes into UI knobs
    if (auto* p = audioProcessor.getScaleParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownScale) > 0.005f && !scale_slider.isMouseButtonDown())
        {
            lastKnownScale = pVal;
            scale_slider.setValue (pVal, juce::dontSendNotification);
            apply_knob_bending();
        }
    }
    if (auto* p = audioProcessor.getOffsetParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownOffset) > 0.001f && !offset_slider.isMouseButtonDown())
        {
            lastKnownOffset = pVal;
            offset_slider.setValue (pVal, juce::dontSendNotification);
            apply_knob_bending();
        }
    }
    if (auto* p = audioProcessor.getHeatParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownHeat) > 0.001f && !heat_slider.isMouseButtonDown())
        {
            lastKnownHeat = pVal;
            heat_slider.setValue (pVal, juce::dontSendNotification);
            if (currentBendingLayer.isNotEmpty())
                audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), pVal);
        }
    }
    if (auto* p = audioProcessor.getMemoryParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownMemory) > 0.001f && !memory_slider.isMouseButtonDown())
        {
            lastKnownMemory = pVal;
            memory_slider.setValue (pVal, juce::dontSendNotification);
            if (currentBendingLayer.isNotEmpty())
                audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), pVal);
        }
    }
    if (auto* p = audioProcessor.getDryWetParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownDryWet) > 0.005f && !dry_wet_slider.isMouseButtonDown())
        {
            lastKnownDryWet = pVal;
            dry_wet_slider.setValue (pVal, juce::dontSendNotification);
            audioProcessor.setDryWet (pVal);
        }
    }

    if (auto* p = audioProcessor.getLatentDepthParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - (float)latent_depth_slider.getValue()) > 0.005f && !latent_depth_slider.isMouseButtonDown())
        {
            latent_depth_slider.setValue (pVal, juce::dontSendNotification);
            audioProcessor.setLatentDepth (pVal);
        }
    }

    if (!latentPad.isMouseButtonDownAnywhere())
    {
        latentPad.setCoords (audioProcessor.getLatentX(), audioProcessor.getLatentY());
    }
    latentPad.setEnabledState (audioProcessor.isLatentEnabled(), audioProcessor.hasAutoencode());
    if (latent_enable_button.getToggleState() != audioProcessor.isLatentEnabled())
    {
        latent_enable_button.setToggleState (audioProcessor.isLatentEnabled(), juce::dontSendNotification);
    }

    if (audioProcessor.isFuseBlown())
    {
        fuse_button.setButtonText ("FUSE: BLOWN!");
        fuse_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::fuse_blown);
        fuse_button.setColour (juce::TextButton::textColourOffId, juce::Colours::yellow);
    }
    else
    {
        float env = audioProcessor.getMomentaryEnvelope();
        bool bendingActive = false;
        if (audioProcessor.getTriggerMode() == NetBenderAudioProcessor::TriggerMode::Continuous)
        {
            bendingActive = audioProcessor.hasActiveBending();
        }
        else
        {
            bendingActive = (env > 0.01f);
        }

        if (bendingActive)
        {
            fuse_button.setButtonText ("FUSE: ACTIVE");
            fuse_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::fuse_glowing_green);
            fuse_button.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        }
        else
        {
            fuse_button.setButtonText ("FUSE: OK");
            fuse_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::forest_green);
            fuse_button.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        }
    }

    float env = audioProcessor.getMomentaryEnvelope();
    if (env > 0.05f)
    {
        short_circuit_button.setColour (juce::TextButton::buttonColourId,
            UIConstants::Colors::crimson.interpolatedWith (UIConstants::Colors::short_circuit_active, env));
    }
    else
    {
        short_circuit_button.setColour (juce::TextButton::buttonColourId, UIConstants::Colors::crimson);
    }

    // Update canvas curves:
    // 1) In Momentary or Transient modes, display the target bent curve (neon pink dashed line)
    // 2) Display live weights currently active inside the model (purple curve with gradient fill)
    if (audioProcessor.isModelLoaded() && currentBendingLayer.isNotEmpty() && !weightCanvas.isCurrentlyDrawing())
    {
        auto triggerMode = audioProcessor.getTriggerMode();

        if (triggerMode == NetBenderAudioProcessor::TriggerMode::Continuous)
        {
            weightCanvas.setTargetBentWeights ({});
            
            if (heat_slider.getValue() > 0.0001 && !freeze_button.getToggleState())
            {
                auto liveWeights = audioProcessor.getBendingManager().get_layer_live_weights (currentBendingLayer.toStdString());
                if (!liveWeights.empty())
                    weightCanvas.updateCurrentWeights (liveWeights);
            }
        }
        else
        {
            auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
            // In Momentary / Transient modes:
            // Calculate target bent curve only when parameters or source change:
            if (!state.drawnWeights.empty())
            {
                bool bridgeEnabled = bridge_enable_toggle.getToggleState();
                bool mutateEnabled = mutate_enable_toggle.getToggleState();
                float effScale = mutateEnabled ? state.scale : 1.0f;
                float effOffset = mutateEnabled ? state.offset : 0.0f;

                bool paramsChanged = (bridgeEnabled != lastBridgeEnabled) ||
                                     (std::abs(state.bridgeDepth - lastBridgeDepth) > 0.001f) ||
                                     (state.bridgeSourceLayer != lastBridgeSource) ||
                                     (std::abs(effScale - lastTargetScale) > 0.001f) ||
                                     (std::abs(effOffset - lastTargetOffset) > 0.001f) ||
                                     targetCurveDirty;

                if (paramsChanged)
                {
                    lastBridgeEnabled = bridgeEnabled;
                    lastBridgeDepth = state.bridgeDepth;
                    lastBridgeSource = state.bridgeSourceLayer;
                    lastTargetScale = effScale;
                    lastTargetOffset = effOffset;
                    targetCurveDirty = false;

                    std::vector<float> bridgedBase = state.drawnWeights;
                    std::vector<float> srcTiled;
                    if (bridgeEnabled && state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != currentBendingLayer.toStdString())
                    {
                        auto srcW = audioProcessor.getBackend().get_original_layer_weights (state.bridgeSourceLayer);
                        if (!srcW.empty())
                        {
                            float alpha = state.bridgeDepth;
                            size_t srcSize = srcW.size();
                            srcTiled.resize (bridgedBase.size());
                            for (size_t i = 0; i < bridgedBase.size(); ++i)
                            {
                                srcTiled[i] = srcW[i % srcSize];
                                bridgedBase[i] = (1.0f - alpha) * state.drawnWeights[i] + alpha * srcTiled[i];
                            }
                        }
                    }
                    weightCanvas.setShowBridgeVisuals (bridgeEnabled);
                    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, false);

                    std::vector<float> targetBent(bridgedBase.size());
                    for (size_t i = 0; i < bridgedBase.size(); ++i)
                        targetBent[i] = bridgedBase[i] * effScale + effOffset;
                    weightCanvas.setTargetBentWeights (targetBent);
                }
            }
            else
            {
                weightCanvas.setTargetBentWeights ({});
            }

            bool isBendingEngaged = (audioProcessor.getMomentaryEnvelope() > 0.001f);
            if (isBendingEngaged || state.heat > 0.0001f)
            {
                auto liveWeights = audioProcessor.getBendingManager().get_layer_live_weights (currentBendingLayer.toStdString());
                if (!liveWeights.empty())
                {
                    weightCanvas.updateCurrentWeights (liveWeights);
                }
            }
        }
    }
}

auto NetBenderAudioProcessorEditor::update_model_info() -> void {
    bool usingGpu = audioProcessor.isUsingGpu();
    gpu_button.setButtonText (usingGpu ? "GPU" : "CPU");
    gpu_button.setColour (juce::TextButton::buttonColourId,
                         usingGpu ? UIConstants::Colors::dark_cyan : UIConstants::Colors::button_secondary_bg);
    gpu_button.setColour (juce::TextButton::textColourOffId,
                         usingGpu ? juce::Colours::lightcyan : UIConstants::Colors::text_light);

    if (audioProcessor.isModelLoaded())
    {
        juce::File file (audioProcessor.getModelPath());
        double modelSr = audioProcessor.getModelSampleRate();
        juce::String srStr = (modelSr > 0) ? (" @ " + juce::String ((int)modelSr) + "Hz") : "";
        if (audioProcessor.isResamplingActive())
            srStr += " [Resampled -> " + juce::String ((int)audioProcessor.getSampleRate()) + "Hz]";

        juce::String backendName (audioProcessor.getBackend().get_backend_name());
        status_label.setText ("Loaded: " + file.getFileName() + " [" + backendName + "]" + srStr, juce::dontSendNotification);
        status_label.setColour (juce::Label::textColourId, juce::Colours::lightgreen);

        // Update Buffer readout badge
        buffer_status_label.setText (juce::String (audioProcessor.getBufferSize()), juce::dontSendNotification);

        // Update Mode combo box
        method_combo.clear (juce::dontSendNotification);
        auto modes = audioProcessor.getAvailableModes();
        int id = 1;
        for (const auto& m : modes)
        {
            method_combo.addItem (m, id++);
        }
        method_combo.setText (audioProcessor.getCurrentMode(), juce::dontSendNotification);

        // Update Layer combo box respecting selected Category
        layer_combo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        int catId = category_combo.getSelectedId();
        id = 1;
        for (const auto& l : layers)
        {
            auto cat = NetBenderAudioProcessor::classifyLayer (l);
            bool include = false;
            if (catId <= 1) include = true;
            else if (catId == 2 && cat == NetBenderAudioProcessor::LayerCategory::Norm) include = true;
            else if (catId == 3 && cat == NetBenderAudioProcessor::LayerCategory::Conv) include = true;
            else if (catId == 4 && cat == NetBenderAudioProcessor::LayerCategory::Attention) include = true;
            else if (catId == 5 && cat == NetBenderAudioProcessor::LayerCategory::Linear) include = true;
            else if (catId == 6 && cat == NetBenderAudioProcessor::LayerCategory::Activation) include = true;
            else if (catId == 7 && cat == NetBenderAudioProcessor::LayerCategory::Recurrent) include = true;
            else if (catId == 8 && cat == NetBenderAudioProcessor::LayerCategory::Bias) include = true;

            if (include)
                layer_combo.addItem (l, id++);
        }

        // Populate Bridge Combo with [No Bridge] + all model layers
        bridge_combo.clear (juce::dontSendNotification);
        bridge_combo.addItem ("(Off)", 1);
        int bridgeId = 2;
        for (const auto& l : layers)
        {
            bridge_combo.addItem (l, bridgeId++);
        }
        bridge_combo.setSelectedId (1, juce::dontSendNotification);
        
        bool layerFound = false;
        for (int i = 0; i < layer_combo.getNumItems(); ++i)
        {
            if (layer_combo.getItemText(i) == currentBendingLayer)
            {
                layerFound = true;
                break;
            }
        }

        if (layerFound && currentBendingLayer.isNotEmpty())
        {
            layer_combo.setText (currentBendingLayer, juce::dontSendNotification);
            select_layer (currentBendingLayer);
        }
        else if (layer_combo.getNumItems() > 0)
        {
            layer_combo.setSelectedId (1, juce::dontSendNotification);
            select_layer (layer_combo.getItemText(0));
        }
        else
        {
            originalWeights.clear();
            currentWeights.clear();
            currentBendingLayer = "";
            weightCanvas.setWeights ({}, {});
            infoBendingLabel.setText ("Model has no trainable parameters.", juce::dontSendNotification);
        }
    }
    else
    {
        status_label.setText ("No model loaded.", juce::dontSendNotification);
        status_label.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
        buffer_status_label.setText ("-", juce::dontSendNotification);
        method_combo.clear (juce::dontSendNotification);
        layer_combo.clear (juce::dontSendNotification);
        infoBendingLabel.setText ("Load a model to view weights.", juce::dontSendNotification);
        originalWeights.clear();
        currentWeights.clear();
        currentBendingLayer = "";
        weightCanvas.setWeights ({}, {});
        update_knob_context_labels (NetBenderAudioProcessor::LayerCategory::Other);
    }
}

auto NetBenderAudioProcessorEditor::select_layer(const juce::String& layerName) -> void {
    if (layerName.isEmpty()) return;
    
    currentBendingLayer = layerName;
    audioProcessor.setActiveLayerName (currentBendingLayer);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (layerName.toStdString());
    currentWeights = audioProcessor.getBackend().get_layer_weights (layerName.toStdString());

    auto state = audioProcessor.getLayerState (layerName.toStdString());
    if (state.originalWeights.empty())
    {
        state.originalWeights = originalWeights;
        audioProcessor.setLayerOriginalWeights (layerName.toStdString(), originalWeights);
        if (state.drawnWeights.empty())
        {
            baseDrawnWeights = originalWeights;
            state.drawnWeights = originalWeights;
        }
        else
        {
            baseDrawnWeights = state.drawnWeights;
        }
    }
    else if (state.drawnWeights.empty())
    {
        baseDrawnWeights = originalWeights;
        state.drawnWeights = originalWeights;
    }
    else
    {
        baseDrawnWeights = state.drawnWeights;
    }

    scale_slider.removeListener (this);
    offset_slider.removeListener (this);
    heat_slider.removeListener (this);
    memory_slider.removeListener (this);
    drift_mode_combo.removeListener (this);
    
    scale_slider.setValue (state.scale, juce::dontSendNotification);
    offset_slider.setValue (state.offset, juce::dontSendNotification);
    heat_slider.setValue (state.heat, juce::dontSendNotification);
    memory_slider.setValue (state.memory, juce::dontSendNotification);
    drift_mode_combo.setSelectedId ((int)state.driftMode + 1, juce::dontSendNotification);

    freeze_button.setToggleState (state.frozen, juce::dontSendNotification);
    freeze_button.setColour (juce::ToggleButton::textColourId,
                            state.frozen ? UIConstants::Colors::drift_orange : UIConstants::Colors::text_light);

    // Sync DAW host automatable parameters to active layer values
    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (state.scale));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (state.offset));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (state.heat));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (state.memory));

    // Sync bridge controls with this layer's state
    bridge_combo.removeListener (this);
    bridge_depth_slider.removeListener (this);

    if (state.bridgeSourceLayer.empty())
    {
        bridge_combo.setSelectedId (1, juce::dontSendNotification);
    }
    else
    {
        int foundId = 1;
        for (int i = 0; i < bridge_combo.getNumItems(); ++i)
        {
            if (bridge_combo.getItemText(i).toStdString() == state.bridgeSourceLayer)
            {
                foundId = bridge_combo.getItemId(i);
                break;
            }
        }
        bridge_combo.setSelectedId (foundId, juce::dontSendNotification);
    }
    bridge_depth_slider.setValue (state.bridgeDepth, juce::dontSendNotification);

    bridge_combo.addListener (this);
    bridge_depth_slider.addListener (this);

    auto shape = audioProcessor.getBackend().get_layer_shape (layerName.toStdString());
    weightCanvas.setLayerShape (shape);
    weightCanvas.setWeights (originalWeights, currentWeights);

    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    bool bridgeEnabled = bridge_enable_toggle.getToggleState();
    if (bridgeEnabled && state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != layerName.toStdString())
    {
        auto srcW = audioProcessor.getBackend().get_original_layer_weights (state.bridgeSourceLayer);
        if (!srcW.empty())
        {
            float alpha = state.bridgeDepth;
            size_t srcSize = srcW.size();
            srcTiled.resize (bridgedBase.size());
            for (size_t i = 0; i < bridgedBase.size(); ++i)
            {
                srcTiled[i] = srcW[i % srcSize];
                bridgedBase[i] = (1.0f - alpha) * baseDrawnWeights[i] + alpha * srcTiled[i];
            }
        }
    }
    bool isContinuous = (audioProcessor.getTriggerMode() == NetBenderAudioProcessor::TriggerMode::Continuous);
    weightCanvas.setShowBridgeVisuals (bridgeEnabled);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, isContinuous);

    if (!isContinuous)
    {
        std::vector<float> targetBent(bridgedBase.size());
        bool mutateEnabled = mutate_enable_toggle.getToggleState();
        float effScale = mutateEnabled ? state.scale : 1.0f;
        float effOffset = mutateEnabled ? state.offset : 0.0f;
        for (size_t i = 0; i < bridgedBase.size(); ++i)
            targetBent[i] = bridgedBase[i] * effScale + effOffset;
        weightCanvas.setTargetBentWeights (targetBent);
    }
    else
    {
        weightCanvas.setTargetBentWeights ({});
    }
    
    auto category = NetBenderAudioProcessor::classifyLayer (currentBendingLayer.toStdString());
    juce::String catName = "Other";
    if (category == NetBenderAudioProcessor::LayerCategory::Norm) catName = "Norm / Dynamics";
    else if (category == NetBenderAudioProcessor::LayerCategory::Conv) catName = "Conv Kernel";
    else if (category == NetBenderAudioProcessor::LayerCategory::Attention) catName = "Attention";
    else if (category == NetBenderAudioProcessor::LayerCategory::Linear) catName = "Linear / Dense";
    else if (category == NetBenderAudioProcessor::LayerCategory::Activation) catName = "Activation (Snake)";
    else if (category == NetBenderAudioProcessor::LayerCategory::Recurrent) catName = "Recurrent (LSTM)";
    else if (category == NetBenderAudioProcessor::LayerCategory::Bias) catName = "Bias";

    infoBendingLabel.setText ("Trace: [" + catName + "]  |  Layer: " + currentBendingLayer + "  |  " + juce::String (originalWeights.size()) + " params" + (audioProcessor.hasAutoencode() ? "  |  [Latent Hook Ready]" : ""), juce::dontSendNotification);

    // Update contextual knob labels (Gamma / Beta for Norm, Scale / Offset for other traces)
    update_knob_context_labels (category);

    scale_slider.addListener (this);
    offset_slider.addListener (this);
    heat_slider.addListener (this);
    memory_slider.addListener (this);
    drift_mode_combo.addListener (this);

    lastKnownScale = state.scale;
    lastKnownOffset = state.offset;
    lastKnownHeat = state.heat;
    lastKnownMemory = state.memory;
    targetCurveDirty = true;

    updateHarmonicGhostPreview();
}

auto NetBenderAudioProcessorEditor::apply_knob_bending() -> void {
    if (currentBendingLayer.isEmpty()) return;
    if (baseDrawnWeights.empty())
        baseDrawnWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    
    bool mutateEnabled = mutate_enable_toggle.getToggleState();
    float scale = mutateEnabled ? (float)scale_slider.getValue() : 1.0f;
    float offset = mutateEnabled ? (float)offset_slider.getValue() : 0.0f;

    // Store per-layer scale and offset
    audioProcessor.setLayerScale (currentBendingLayer.toStdString(), scale);
    audioProcessor.setLayerOffset (currentBendingLayer.toStdString(), offset);

    // Fetch bridge state
    auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    bool bridgeEnabled = bridge_enable_toggle.getToggleState();

    if (bridgeEnabled && state.bridgeDepth > 0.001f && !state.bridgeSourceLayer.empty() && state.bridgeSourceLayer != currentBendingLayer.toStdString())
    {
        auto srcW = audioProcessor.getBackend().get_original_layer_weights (state.bridgeSourceLayer);
        if (!srcW.empty())
        {
            float alpha = state.bridgeDepth;
            size_t srcSize = srcW.size();
            srcTiled.resize (bridgedBase.size());
            for (size_t i = 0; i < bridgedBase.size(); ++i)
            {
                srcTiled[i] = srcW[i % srcSize];
                bridgedBase[i] = (1.0f - alpha) * baseDrawnWeights[i] + alpha * srcTiled[i];
            }
        }
    }

    auto mode = audioProcessor.getTriggerMode();
    bool isContinuous = (mode == NetBenderAudioProcessor::TriggerMode::Continuous);
    weightCanvas.setShowBridgeVisuals (bridgeEnabled);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, isContinuous);

    std::vector<float> bentTarget(bridgedBase.size());
    for (size_t i = 0; i < bridgedBase.size(); ++i)
    {
        bentTarget[i] = bridgedBase[i] * scale + offset;
    }

    if (isContinuous)
    {
        currentWeights = bentTarget;
        weightCanvas.updateCurrentWeights (currentWeights);
        weightCanvas.setTargetBentWeights ({});
    }
    else
    {
        weightCanvas.setTargetBentWeights (bentTarget);
    }
    lastTargetScale = scale;
    lastTargetOffset = offset;
    lastBridgeDepth = state.bridgeDepth;
    lastBridgeSource = state.bridgeSourceLayer;
    lastBridgeEnabled = bridgeEnabled;
}

auto NetBenderAudioProcessorEditor::reset_layer_weights() -> void {
    if (currentBendingLayer.isEmpty()) return;
    
    audioProcessor.clearLayerBending (currentBendingLayer.toStdString());
    
    scale_slider.removeListener (this);
    offset_slider.removeListener (this);
    heat_slider.removeListener (this);
    memory_slider.removeListener (this);
    bridge_combo.removeListener (this);
    bridge_depth_slider.removeListener (this);

    scale_slider.setValue (1.0, juce::dontSendNotification);
    offset_slider.setValue (0.0, juce::dontSendNotification);
    heat_slider.setValue (0.0, juce::dontSendNotification);
    memory_slider.setValue (0.8, juce::dontSendNotification);
    bridge_combo.setSelectedId (1, juce::dontSendNotification);
    bridge_depth_slider.setValue (0.0, juce::dontSendNotification);
    freeze_button.setToggleState (false, juce::dontSendNotification);
    freeze_button.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (1.0f));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.8f));

    lastKnownScale = 1.0f;
    lastKnownOffset = 0.0f;
    lastKnownHeat = 0.0f;
    lastKnownMemory = 0.8f;

    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;

    scale_slider.addListener (this);
    offset_slider.addListener (this);
    heat_slider.addListener (this);
    memory_slider.addListener (this);
    bridge_combo.addListener (this);
    bridge_depth_slider.addListener (this);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    currentWeights = originalWeights;
    baseDrawnWeights = originalWeights;

    weightCanvas.setWeights (originalWeights, currentWeights);
    weightCanvas.setTargetBentWeights ({});
    weightCanvas.setBridgeWeights ({}, {}, 0.0f, audioProcessor.getTriggerMode() == NetBenderAudioProcessor::TriggerMode::Continuous);
    updateHarmonicGhostPreview();
}

auto NetBenderAudioProcessorEditor::reset_all_layer_weights() -> void {
    audioProcessor.clearAllLayerBending();
    
    scale_slider.removeListener (this);
    offset_slider.removeListener (this);
    heat_slider.removeListener (this);
    memory_slider.removeListener (this);
    bridge_combo.removeListener (this);
    bridge_depth_slider.removeListener (this);

    scale_slider.setValue (1.0, juce::dontSendNotification);
    offset_slider.setValue (0.0, juce::dontSendNotification);
    heat_slider.setValue (0.0, juce::dontSendNotification);
    memory_slider.setValue (0.8, juce::dontSendNotification);
    bridge_combo.setSelectedId (1, juce::dontSendNotification);
    bridge_depth_slider.setValue (0.0, juce::dontSendNotification);
    freeze_button.setToggleState (false, juce::dontSendNotification);
    freeze_button.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (1.0f));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (0.0f));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (0.8f));

    lastKnownScale = 1.0f;
    lastKnownOffset = 0.0f;
    lastKnownHeat = 0.0f;
    lastKnownMemory = 0.8f;

    lastKnobScale = 1.0;
    lastKnobOffset = 0.0;

    scale_slider.addListener (this);
    offset_slider.addListener (this);
    heat_slider.addListener (this);
    memory_slider.addListener (this);
    bridge_combo.addListener (this);
    bridge_depth_slider.addListener (this);

    if (currentBendingLayer.isNotEmpty())
    {
        originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
        currentWeights = originalWeights;
        baseDrawnWeights = originalWeights;
        weightCanvas.setWeights (originalWeights, currentWeights);
        weightCanvas.setTargetBentWeights ({});
        weightCanvas.setBridgeWeights ({}, {}, 0.0f, audioProcessor.getTriggerMode() == NetBenderAudioProcessor::TriggerMode::Continuous);
        updateHarmonicGhostPreview();
    }
}

auto NetBenderAudioProcessorEditor::save_model_to_file() -> void {
    if (!audioProcessor.isModelLoaded())
    {
        status_label.setText ("Cannot save: No model loaded.", juce::dontSendNotification);
        status_label.setColour (juce::Label::textColourId, juce::Colours::orangered);
        return;
    }

    bool isPte = audioProcessor.getModelPath().endsWithIgnoreCase (".pte");
    bool isOnnx = audioProcessor.getModelPath().endsWithIgnoreCase (".onnx");
    juce::String defaultName = isPte ? "bended_model.pte" : (isOnnx ? "bended_model.onnx" : "bended_model.ts");
    juce::String filePattern = isPte ? "*.pte" : (isOnnx ? "*.onnx" : "*.ts");

    fileChooser = std::make_unique<juce::FileChooser> (
        "Save Bended Model (" + filePattern + ")...",
        juce::File::getSpecialLocation (juce::File::userHomeDirectory).getChildFile (defaultName),
        filePattern
    );

    auto flags = juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting;
    fileChooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
    {
        auto file = fc.getResult();
        if (file.getFullPathName().isNotEmpty())
        {
            int err = audioProcessor.getBackend().save_model (file.getFullPathName().toStdString());
            if (err == 0)
            {
                status_label.setText ("Saved bended model: " + file.getFileName(), juce::dontSendNotification);
                status_label.setColour (juce::Label::textColourId, juce::Colours::lightgreen);
            }
            else
            {
                status_label.setText ("Error: Failed to save model.", juce::dontSendNotification);
                status_label.setColour (juce::Label::textColourId, juce::Colours::orangered);
            }
        }
    });
}

auto NetBenderAudioProcessorEditor::update_knob_context_labels(NetBenderAudioProcessor::LayerCategory category) -> void {
    if (category == NetBenderAudioProcessor::LayerCategory::Norm)
    {
        // Normalization / Dynamics context: Gamma (Gain) & Beta (Bias)
        scale_label.setText (juce::CharPointer_UTF8 ("Gamma (\xce\xb3)"), juce::dontSendNotification);
        scale_label.setColour (juce::Label::textColourId, UIConstants::Colors::lilac);

        offset_label.setText (juce::CharPointer_UTF8 ("Beta (\xce\xb2)"), juce::dontSendNotification);
        offset_label.setColour (juce::Label::textColourId, UIConstants::Colors::sky_blue);
    }
    else if (category == NetBenderAudioProcessor::LayerCategory::Activation)
    {
        // Snake / periodic activation context: Alpha (Freq scaling) & Offset
        scale_label.setText (juce::CharPointer_UTF8 ("Alpha (\xce\xb1)"), juce::dontSendNotification);
        scale_label.setColour (juce::Label::textColourId, UIConstants::Colors::amber_accent);

        offset_label.setText ("Offset", juce::dontSendNotification);
        offset_label.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    }
    else
    {
        // Standard convolution / dense / bias context: Scale & Offset
        scale_label.setText ("Scale", juce::dontSendNotification);
        scale_label.setColour (juce::Label::textColourId, juce::Colours::lightgrey);

        offset_label.setText ("Offset", juce::dontSendNotification);
        offset_label.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    }
}

auto NetBenderAudioProcessorEditor::updateHarmonicGhostPreview() -> void {
    if (!harmonic_enable_toggle.getToggleState() || currentBendingLayer.isEmpty() || baseDrawnWeights.empty())
    {
        weightCanvas.setHarmonicGhostWeights ({}, false);
        return;
    }

    size_t N = baseDrawnWeights.size();
    std::vector<float> ghost (N);

    float f = (float)harmonic_freq_slider.getValue();
    int partials = (int)harmonic_partials_slider.getValue();
    float morph = (float)harmonic_morph_slider.getValue();
    float gain = (float)harmonic_depth_slider.getValue();
    int mode = harmonic_mode_combo.getSelectedId(); // 1 = Add, 2 = Ring, 3 = Replace

    // TwoPi constant
    const float twoPi = 6.28318530717958647692f;

    for (size_t i = 0; i < N; ++i)
    {
        float x = (N > 1) ? (float)i / (float)(N - 1) : 0.0f;
        float synthVal = 0.0f;

        for (int k = 1; k <= partials; ++k)
        {
            float harmonicAmp = 1.0f / std::sqrt ((float)k); // Soft 1/sqrt(k) harmonic decay
            synthVal += harmonicAmp * std::sin (twoPi * (float)k * f * x + morph);
        }
        synthVal *= gain;

        if (mode == 1) // Add
        {
            ghost[i] = baseDrawnWeights[i] + synthVal;
        }
        else if (mode == 2) // Ring (multiply)
        {
            ghost[i] = baseDrawnWeights[i] * (1.0f + synthVal);
        }
        else // Replace
        {
            ghost[i] = synthVal;
        }
    }

    weightCanvas.setHarmonicGhostWeights (ghost, true);
}

auto NetBenderAudioProcessorEditor::applyHarmonicWeights() -> void {
    if (currentBendingLayer.isEmpty() || baseDrawnWeights.empty())
        return;

    size_t N = baseDrawnWeights.size();
    std::vector<float> stamped (N);

    float f = (float)harmonic_freq_slider.getValue();
    int partials = (int)harmonic_partials_slider.getValue();
    float morph = (float)harmonic_morph_slider.getValue();
    float gain = (float)harmonic_depth_slider.getValue();
    int mode = harmonic_mode_combo.getSelectedId();

    const float twoPi = 6.28318530717958647692f;

    for (size_t i = 0; i < N; ++i)
    {
        float x = (N > 1) ? (float)i / (float)(N - 1) : 0.0f;
        float synthVal = 0.0f;

        for (int k = 1; k <= partials; ++k)
        {
            float harmonicAmp = 1.0f / std::sqrt ((float)k);
            synthVal += harmonicAmp * std::sin (twoPi * (float)k * f * x + morph);
        }
        synthVal *= gain;

        if (mode == 1)
            stamped[i] = baseDrawnWeights[i] + synthVal;
        else if (mode == 2)
            stamped[i] = baseDrawnWeights[i] * (1.0f + synthVal);
        else
            stamped[i] = synthVal;
    }

    // Persist as new baseDrawnWeights
    baseDrawnWeights = stamped;
    audioProcessor.setLayerDrawnWeights (currentBendingLayer.toStdString(), baseDrawnWeights);

    // Hide ghost preview once stamped
    weightCanvas.setHarmonicGhostWeights ({}, false);

    // Apply via existing knob bending pipeline
    apply_knob_bending();

    infoBendingLabel.setText ("Harmonic wave stamped to " + currentBendingLayer, juce::dontSendNotification);
}

auto NetBenderAudioProcessorEditor::set_side_tab(SideTab newTab) -> void {
    currentSideTab = newTab;
    sidePanel.set_side_tab(newTab);
    repaint();
}

auto NetBenderAudioProcessorEditor::set_view_mode(ViewMode newMode) -> void {
    currentViewMode = newMode;

    bool isWeights = (currentViewMode == ViewMode::Weights);
    bool isLatent = (currentViewMode == ViewMode::Latent);

    const auto& activeBg = UIConstants::Colors::tab_active_bg;
    const auto& inactiveBg = UIConstants::Colors::tab_inactive_bg;
    const auto& activeText = UIConstants::Colors::text_white;
    const auto& inactiveText = UIConstants::Colors::text_dimmed;

    view_weights_button.setColour(juce::TextButton::buttonColourId, isWeights ? activeBg : inactiveBg);
    view_weights_button.setColour(juce::TextButton::textColourOffId, isWeights ? activeText : inactiveText);

    view_latent_button.setColour(juce::TextButton::buttonColourId, isLatent ? activeBg : inactiveBg);
    view_latent_button.setColour(juce::TextButton::textColourOffId, isLatent ? activeText : inactiveText);

    weightCanvas.setVisible(isWeights);
    latentPad.setVisible(isLatent);

    category_label.setVisible(isWeights);
    category_combo.setVisible(isWeights);
    layer_label.setVisible(isWeights);
    layer_combo.setVisible(isWeights);
    reset_layer_button.setVisible(isWeights);
    reset_all_button.setVisible(isWeights);
    display_curve_button.setVisible(isWeights);
    display_matrix_button.setVisible(isWeights);

    sidePanel.set_view_mode(newMode);

    if (isLatent) {
        if (audioProcessor.hasAutoencode())
            infoBendingLabel.setText("Latent Topographic Vector Pad: Drag cursor to modulate autoencoder bottleneck.", juce::dontSendNotification);
        else
            infoBendingLabel.setText("Latent View: Standby. Loaded model does not expose paired encode + decode methods.", juce::dontSendNotification);
    } else {
        infoBendingLabel.setText(currentBendingLayer.isNotEmpty() ? ("Layer: " + currentBendingLayer) : "Select a layer to bend weights.", juce::dontSendNotification);
    }

    repaint();
}

auto NetBenderAudioProcessorEditor::updateLatentControlsVisibility() -> void {
    side_panel.latent_panel.resized();
}

