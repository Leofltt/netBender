#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
NNBendingAudioProcessorEditor::NNBendingAudioProcessorEditor (NNBendingAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    setLookAndFeel (&flatLf);

    // =========================================================================
    // Header Toolbar Listeners & Synchronization
    // =========================================================================
    loadButton.addListener (this);
    gpuButton.addListener (this);
    methodCombo.addListener (this);
    dryWetSlider.addListener (this);
    saveModelButton.addListener (this);
    fuseButton.addListener (this);
    shortCircuitButton.addListener (this);
    triggerModeCombo.addListener (this);

    bufferStatusLabel.setText (juce::String (audioProcessor.getBufferSize()), juce::dontSendNotification);
    gpuButton.setButtonText (audioProcessor.isUsingGpu() ? "GPU" : "CPU");
    gpuButton.setColour (juce::TextButton::buttonColourId,
                         audioProcessor.isUsingGpu() ? juce::Colour::fromString ("#ff0e7490") : juce::Colour::fromString ("#ff1c1926"));
    gpuButton.setColour (juce::TextButton::textColourOffId,
                         audioProcessor.isUsingGpu() ? juce::Colours::lightcyan : juce::Colours::lightgrey);
    dryWetSlider.setValue (audioProcessor.getDryWet());
    triggerModeCombo.setSelectedId ((int)audioProcessor.getTriggerMode() + 1, juce::dontSendNotification);
    shortCircuitButton.setVisible (audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Momentary);

    addAndMakeVisible (headerBar);

    // =========================================================================
    // Bending Canvas & Layer Management
    // =========================================================================
    addAndMakeVisible (bendingGroup);

    layerBar.viewWeightsButton.addListener (this);
    layerBar.viewLatentButton.addListener (this);
    layerBar.displayCurveButton.addListener (this);
    layerBar.displayMatrixButton.addListener (this);
    layerBar.categoryCombo.addListener (this);
    layerBar.layerCombo.addListener (this);
    layerBar.resetLayerButton.addListener (this);
    layerBar.resetAllButton.addListener (this);
    bendingGroup.addAndMakeVisible (layerBar);

    bendingGroup.addAndMakeVisible (weightCanvas);
    weightCanvas.onWeightsModified = [this] (const std::vector<float>& modifiedWeights)
    {
        if (currentBendingLayer.isNotEmpty() && !modifiedWeights.empty())
        {
            float scale = (float)scaleSlider.getValue();
            float offset = (float)offsetSlider.getValue();

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
            if (mode == NNBendingAudioProcessor::TriggerMode::Continuous)
            {
                currentWeights = modifiedWeights;
                audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), currentWeights);
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
    sidePanel.tabMutateButton.addListener (this);
    sidePanel.tabHarmonicsButton.addListener (this);
    sidePanel.tabBridgeButton.addListener (this);

    mutateEnableToggle.addListener (this);
    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    driftEnableToggle.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    driftModeCombo.addListener (this);
    freezeButton.addListener (this);

    harmonicEnableToggle.addListener (this);
    harmonicFreqSlider.addListener (this);
    harmonicPartialsSlider.addListener (this);
    harmonicMorphSlider.addListener (this);
    harmonicDepthSlider.addListener (this);
    harmonicModeCombo.addListener (this);
    harmonicApplyButton.addListener (this);

    bridgeEnableToggle.addListener (this);
    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    latentEnableButton.addListener (this);
    latentDepthSlider.setValue (audioProcessor.getLatentDepth());
    latentDepthSlider.addListener (this);
    latentModeCombo.setSelectedId ((int)audioProcessor.getLatentMode() + 1, juce::dontSendNotification);
    latentModeCombo.addListener (this);
    latentSlewSlider.setValue (audioProcessor.getLatentSlew());
    latentSlewSlider.addListener (this);

    bendingGroup.addAndMakeVisible (sidePanel);

    // =========================================================================
    // Status Bar & Initialization
    // =========================================================================
    infoBendingLabel.setText ("Select a layer to bend weights.", juce::dontSendNotification);
    infoBendingLabel.setJustificationType (juce::Justification::centredLeft);
    infoBendingLabel.setColour (juce::Label::textColourId, juce::Colours::silver);
    bendingGroup.addAndMakeVisible (infoBendingLabel);

    setViewMode (ViewMode::Weights);
    setSideTab (SideTab::Mutate);

    setSize (1100, 640);
    setResizable (true, true);
    setResizeLimits (1060, 600, 1600, 1000);

    updateModelInfo();
    startTimerHz (60);
}

NNBendingAudioProcessorEditor::~NNBendingAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

//==============================================================================
void NNBendingAudioProcessorEditor::paint (juce::Graphics& g)
{
    juce::Colour color1 = juce::Colour::fromString ("#ff0f0e13");
    juce::Colour color2 = juce::Colour::fromString ("#ff1a1724");
    juce::ColourGradient gradient (color1, 0, 0, color2, 0, (float)getHeight(), false);
    g.setGradientFill (gradient);
    g.fillAll();

    g.setColour (juce::Colour::fromString ("#ff382f4c"));
    g.drawHorizontalLine (52, 0.0f, (float)getWidth());
}

void NNBendingAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    headerBar.setBounds (area.removeFromTop (34));
    area.removeFromTop (10);

    bendingGroup.setBounds (area);
    auto groupArea = bendingGroup.getLocalBounds().reduced (14);
    groupArea.removeFromTop (12);

    layerBar.setBounds (groupArea.removeFromTop (32));
    groupArea.removeFromTop (8);

    auto bottomRow = groupArea.removeFromBottom (20);
    infoBendingLabel.setBounds (bottomRow);
    groupArea.removeFromBottom (6);

    auto controlsWidth = 240;
    auto sideArea = groupArea.removeFromRight (controlsWidth);
    groupArea.removeFromRight (12);

    weightCanvas.setBounds (groupArea);
    latentPad.setBounds (groupArea);

    sidePanel.setBounds (sideArea);
}

//==============================================================================
void NNBendingAudioProcessorEditor::comboBoxChanged (juce::ComboBox* comboBoxThatHasChanged)
{
    if (comboBoxThatHasChanged == &methodCombo)
    {
        audioProcessor.setCurrentMethod (methodCombo.getText());
        updateModelInfo();
    }
    else if (comboBoxThatHasChanged == &layerCombo)
    {
        selectLayer (layerCombo.getText());
    }
    else if (comboBoxThatHasChanged == &categoryCombo)
    {
        int catId = categoryCombo.getSelectedId();
        layerCombo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        int id = 1;
        for (const auto& l : layers)
        {
            auto cat = NNBendingAudioProcessor::classifyLayer (l);
            bool include = false;
            if (catId == 1) include = true; // All
            else if (catId == 2 && cat == NNBendingAudioProcessor::LayerCategory::Norm) include = true;
            else if (catId == 3 && cat == NNBendingAudioProcessor::LayerCategory::Conv) include = true;
            else if (catId == 4 && cat == NNBendingAudioProcessor::LayerCategory::Attention) include = true;
            else if (catId == 5 && cat == NNBendingAudioProcessor::LayerCategory::Linear) include = true;
            else if (catId == 6 && cat == NNBendingAudioProcessor::LayerCategory::Activation) include = true;
            else if (catId == 7 && cat == NNBendingAudioProcessor::LayerCategory::Recurrent) include = true;
            else if (catId == 8 && cat == NNBendingAudioProcessor::LayerCategory::Bias) include = true;

            if (include)
                layerCombo.addItem (l, id++);
        }
        if (layerCombo.getNumItems() > 0)
        {
            layerCombo.setSelectedId (1);
            selectLayer (layerCombo.getText());
        }
    }
    else if (comboBoxThatHasChanged == &triggerModeCombo)
    {
        int modeIdx = triggerModeCombo.getSelectedId() - 1;
        auto mode = (NNBendingAudioProcessor::TriggerMode)modeIdx;
        audioProcessor.setTriggerMode (mode);
        shortCircuitButton.setVisible (mode == NNBendingAudioProcessor::TriggerMode::Momentary);
        resized();

        if (currentBendingLayer.isNotEmpty())
            applyKnobBending();
    }
    else if (comboBoxThatHasChanged == &driftModeCombo)
    {
        int modeIdx = driftModeCombo.getSelectedId() - 1;
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerDriftMode (currentBendingLayer.toStdString(), (NNBendingAudioProcessor::DriftMode)modeIdx);
    }
    else if (comboBoxThatHasChanged == &bridgeCombo)
    {
        if (currentBendingLayer.isNotEmpty())
        {
            juce::String src = (bridgeCombo.getSelectedId() > 1) ? bridgeCombo.getText() : "";
            audioProcessor.setLayerBridgeSource (currentBendingLayer.toStdString(), src.toStdString());
            applyKnobBending();
        }
    }
    else if (comboBoxThatHasChanged == &harmonicModeCombo)
    {
        updateHarmonicGhostPreview();
    }
    else if (comboBoxThatHasChanged == &latentModeCombo)
    {
        int modeIdx = latentModeCombo.getSelectedId() - 1;
        audioProcessor.setLatentMode ((NNBendingAudioProcessor::LatentMode)modeIdx);
        updateLatentControlsVisibility();
        resized();
    }
}

void NNBendingAudioProcessorEditor::sliderValueChanged (juce::Slider* slider)
{
    if (slider == &harmonicFreqSlider || slider == &harmonicPartialsSlider
        || slider == &harmonicMorphSlider || slider == &harmonicDepthSlider)
    {
        updateHarmonicGhostPreview();
    }
    else if (slider == &dryWetSlider)
    {
        float val = (float)dryWetSlider.getValue();
        audioProcessor.setDryWet (val);
        lastKnownDryWet = val;
    }
    else if (slider == &heatSlider)
    {
        float val = (float)heatSlider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getHeatParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownHeat = val;
    }
    else if (slider == &memorySlider)
    {
        float val = (float)memorySlider.getValue();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), val);
        if (auto* param = audioProcessor.getMemoryParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownMemory = val;
    }
    else if (slider == &scaleSlider)
    {
        float val = (float)scaleSlider.getValue();
        if (auto* param = audioProcessor.getScaleParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownScale = val;
        applyKnobBending();
    }
    else if (slider == &offsetSlider)
    {
        float val = (float)offsetSlider.getValue();
        if (auto* param = audioProcessor.getOffsetParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
        lastKnownOffset = val;
        applyKnobBending();
    }
    else if (slider == &bridgeDepthSlider)
    {
        float val = (float)bridgeDepthSlider.getValue();
        if (currentBendingLayer.isNotEmpty())
        {
            audioProcessor.setLayerBridgeDepth (currentBendingLayer.toStdString(), val);
            applyKnobBending();
        }
    }
    else if (slider == &latentDepthSlider)
    {
        float val = (float)latentDepthSlider.getValue();
        audioProcessor.setLatentDepth (val);
        if (auto* param = audioProcessor.getLatentDepthParam())
            param->setValueNotifyingHost (param->convertTo0to1 (val));
    }
    else if (slider == &latentSlewSlider)
    {
        float val = (float)latentSlewSlider.getValue();
        audioProcessor.setLatentSlew (val);
    }
    else
    {
        applyKnobBending();
    }
}

void NNBendingAudioProcessorEditor::buttonClicked (juce::Button* button)
{
    if (button == &fuseButton)
    {
        audioProcessor.resetFuse();
        fuseButton.setButtonText ("FUSE: OK");
        fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1b3a2a"));
        fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        if (currentBendingLayer.isNotEmpty())
            selectLayer (currentBendingLayer);
    }
    else if (button == &gpuButton)
    {
        bool nextGpu = !audioProcessor.isUsingGpu();
        audioProcessor.setUseGpu (nextGpu);
        updateModelInfo();
    }
    else if (button == &shortCircuitButton)
    {
        // Toggle or pulse momentary short-circuit gate
        bool active = !audioProcessor.isShortCircuitActive();
        audioProcessor.triggerShortCircuit (active);
        shortCircuitButton.setColour (juce::TextButton::buttonColourId,
                                      active ? juce::Colour::fromString ("#ffff3300") : juce::Colour::fromString ("#ff8b2500"));
    }
    else if (button == &freezeButton)
    {
        bool frozen = freezeButton.getToggleState();
        if (currentBendingLayer.isNotEmpty())
            audioProcessor.setLayerFrozen (currentBendingLayer.toStdString(), frozen);

        if (frozen)
        {
            freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffff9933"));
            if (currentBendingLayer.isNotEmpty())
            {
                currentWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
                float scale = (float)scaleSlider.getValue();
                float offset = (float)offsetSlider.getValue();
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
            freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
        }
    }
    else if (button == &loadButton)
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
                    updateModelInfo();
                }
                else
                {
                    statusLabel.setText ("Error: Failed to load " + file.getFileName(), juce::dontSendNotification);
                    statusLabel.setColour (juce::Label::textColourId, juce::Colours::orangered);
                }
            }
        });
    }
    else if (button == &resetLayerButton)
    {
        resetLayerWeights();
    }
    else if (button == &resetAllButton)
    {
        resetAllLayerWeights();
    }
    else if (button == &saveModelButton)
    {
        saveModelToFile();
    }
    else if (button == &harmonicApplyButton)
    {
        applyHarmonicWeights();
    }
    else if (button == &viewWeightsButton)
    {
        setViewMode (ViewMode::Weights);
    }
    else if (button == &viewLatentButton)
    {
        setViewMode (ViewMode::Latent);
    }
    else if (button == &latentEnableButton)
    {
        bool en = latentEnableButton.getToggleState();
        audioProcessor.setLatentEnabled (en);
        latentPad.setEnabledState (en, audioProcessor.hasAutoencode());
    }
    else if (button == &tabMutateButton)
    {
        setSideTab (SideTab::Mutate);
    }
    else if (button == &tabHarmonicsButton)
    {
        setSideTab (SideTab::Harmonics);
    }
    else if (button == &tabBridgeButton)
    {
        setSideTab (SideTab::Bridge);
    }
    else if (button == &mutateEnableToggle)
    {
        applyKnobBending();
    }
    else if (button == &driftEnableToggle)
    {
        if (!driftEnableToggle.getToggleState() && currentBendingLayer.isNotEmpty())
        {
            auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
            state.driftOffsets.assign (state.driftOffsets.size(), 0.0f);
            audioProcessor.setLayerState (currentBendingLayer.toStdString(), state);
        }
    }
    else if (button == &bridgeEnableToggle)
    {
        bool en = bridgeEnableToggle.getToggleState();
        weightCanvas.setShowBridgeVisuals (en);
        applyKnobBending();
    }
    else if (button == &harmonicEnableToggle)
    {
        bool en = harmonicEnableToggle.getToggleState();
        weightCanvas.setShowHarmonics (en);
        updateHarmonicGhostPreview();
    }
    else if (button == &displayCurveButton)
    {
        weightCanvas.setRenderMode (WeightBendingComponent::RenderMode::Curve);
        displayCurveButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        displayCurveButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        displayMatrixButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        displayMatrixButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
    }
    else if (button == &displayMatrixButton)
    {
        weightCanvas.setRenderMode (WeightBendingComponent::RenderMode::Matrix);
        displayMatrixButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        displayMatrixButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        displayCurveButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        displayCurveButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
    }
}

//==============================================================================
void NNBendingAudioProcessorEditor::timerCallback()
{
    if (audioProcessor.isModelLoaded() && statusLabel.getText() == "No model loaded.")
    {
        updateModelInfo();
    }

    // If DAW host is automating parameters, reflect changes into UI knobs
    if (auto* p = audioProcessor.getScaleParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownScale) > 0.005f && !scaleSlider.isMouseButtonDown())
        {
            lastKnownScale = pVal;
            scaleSlider.setValue (pVal, juce::dontSendNotification);
            applyKnobBending();
        }
    }
    if (auto* p = audioProcessor.getOffsetParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownOffset) > 0.001f && !offsetSlider.isMouseButtonDown())
        {
            lastKnownOffset = pVal;
            offsetSlider.setValue (pVal, juce::dontSendNotification);
            applyKnobBending();
        }
    }
    if (auto* p = audioProcessor.getHeatParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownHeat) > 0.001f && !heatSlider.isMouseButtonDown())
        {
            lastKnownHeat = pVal;
            heatSlider.setValue (pVal, juce::dontSendNotification);
            if (currentBendingLayer.isNotEmpty())
                audioProcessor.setLayerHeat (currentBendingLayer.toStdString(), pVal);
        }
    }
    if (auto* p = audioProcessor.getMemoryParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownMemory) > 0.001f && !memorySlider.isMouseButtonDown())
        {
            lastKnownMemory = pVal;
            memorySlider.setValue (pVal, juce::dontSendNotification);
            if (currentBendingLayer.isNotEmpty())
                audioProcessor.setLayerMemory (currentBendingLayer.toStdString(), pVal);
        }
    }
    if (auto* p = audioProcessor.getDryWetParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - lastKnownDryWet) > 0.005f && !dryWetSlider.isMouseButtonDown())
        {
            lastKnownDryWet = pVal;
            dryWetSlider.setValue (pVal, juce::dontSendNotification);
            audioProcessor.setDryWet (pVal);
        }
    }

    if (auto* p = audioProcessor.getLatentDepthParam())
    {
        float pVal = p->get();
        if (std::abs (pVal - (float)latentDepthSlider.getValue()) > 0.005f && !latentDepthSlider.isMouseButtonDown())
        {
            latentDepthSlider.setValue (pVal, juce::dontSendNotification);
            audioProcessor.setLatentDepth (pVal);
        }
    }

    if (!latentPad.isMouseButtonDownAnywhere())
    {
        latentPad.setCoords (audioProcessor.getLatentX(), audioProcessor.getLatentY());
    }
    latentPad.setEnabledState (audioProcessor.isLatentEnabled(), audioProcessor.hasAutoencode());
    if (latentEnableButton.getToggleState() != audioProcessor.isLatentEnabled())
    {
        latentEnableButton.setToggleState (audioProcessor.isLatentEnabled(), juce::dontSendNotification);
    }

    if (audioProcessor.isFuseBlown())
    {
        fuseButton.setButtonText ("FUSE: BLOWN!");
        fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff990000"));
        fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::yellow);
    }
    else
    {
        float env = audioProcessor.getMomentaryEnvelope();
        bool bendingActive = false;
        if (audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous)
        {
            bendingActive = audioProcessor.hasActiveBending();
        }
        else
        {
            bendingActive = (env > 0.01f);
        }

        if (bendingActive)
        {
            fuseButton.setButtonText ("FUSE: ACTIVE");
            fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff006633")); // Vibrant glowing green
            fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        }
        else
        {
            fuseButton.setButtonText ("FUSE: OK");
            fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1b3a2a")); // Idle dark green
            fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        }
    }

    float env = audioProcessor.getMomentaryEnvelope();
    if (env > 0.05f)
    {
        shortCircuitButton.setColour (juce::TextButton::buttonColourId,
            juce::Colour::fromString ("#ff8b2500").interpolatedWith (juce::Colour::fromString ("#ffff3300"), env));
    }
    else
    {
        shortCircuitButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff8b2500"));
    }

    // Update canvas curves:
    // 1) In Momentary or Transient modes, display the target bent curve (neon pink dashed line)
    // 2) Display live weights currently active inside the model (purple curve with gradient fill)
    if (audioProcessor.isModelLoaded() && currentBendingLayer.isNotEmpty() && !weightCanvas.isCurrentlyDrawing())
    {
        auto triggerMode = audioProcessor.getTriggerMode();
        auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());

        if (triggerMode == NNBendingAudioProcessor::TriggerMode::Continuous)
        {
            weightCanvas.setTargetBentWeights ({});
            
            if (state.heat > 0.0001f && !state.frozen)
            {
                auto liveWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
                if (!liveWeights.empty())
                    weightCanvas.updateCurrentWeights (liveWeights);
            }
        }
        else
        {
            // In Momentary / Transient modes:
            // Calculate target bent curve only when parameters or source change:
            if (!state.drawnWeights.empty())
            {
                bool bridgeEnabled = bridgeEnableToggle.getToggleState();
                bool mutateEnabled = mutateEnableToggle.getToggleState();
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
                auto liveWeights = audioProcessor.getBackend().get_layer_weights (currentBendingLayer.toStdString());
                if (!liveWeights.empty())
                {
                    weightCanvas.updateCurrentWeights (liveWeights);
                }
            }
        }
    }
}

void NNBendingAudioProcessorEditor::updateModelInfo()
{
    bool usingGpu = audioProcessor.isUsingGpu();
    gpuButton.setButtonText (usingGpu ? "GPU" : "CPU");
    gpuButton.setColour (juce::TextButton::buttonColourId,
                         usingGpu ? juce::Colour::fromString ("#ff0e7490") : juce::Colour::fromString ("#ff1c1926"));
    gpuButton.setColour (juce::TextButton::textColourOffId,
                         usingGpu ? juce::Colours::lightcyan : juce::Colours::lightgrey);

    if (audioProcessor.isModelLoaded())
    {
        juce::File file (audioProcessor.getModelPath());
        double modelSr = audioProcessor.getModelSampleRate();
        juce::String srStr = (modelSr > 0) ? (" @ " + juce::String ((int)modelSr) + "Hz") : "";
        if (audioProcessor.isResamplingActive())
            srStr += " [Resampled -> " + juce::String ((int)audioProcessor.getSampleRate()) + "Hz]";

        juce::String backendName (audioProcessor.getBackend().get_backend_name());
        statusLabel.setText ("Loaded: " + file.getFileName() + " [" + backendName + "]" + srStr, juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::lightgreen);

        // Update Buffer readout badge
        bufferStatusLabel.setText (juce::String (audioProcessor.getBufferSize()), juce::dontSendNotification);

        // Update Mode combo box
        methodCombo.clear (juce::dontSendNotification);
        auto modes = audioProcessor.getAvailableModes();
        int id = 1;
        for (const auto& m : modes)
        {
            methodCombo.addItem (m, id++);
        }
        methodCombo.setText (audioProcessor.getCurrentMode(), juce::dontSendNotification);

        // Update Layer combo box respecting selected Category
        layerCombo.clear (juce::dontSendNotification);
        auto layers = audioProcessor.getBackend().get_available_layers();
        int catId = categoryCombo.getSelectedId();
        id = 1;
        for (const auto& l : layers)
        {
            auto cat = NNBendingAudioProcessor::classifyLayer (l);
            bool include = false;
            if (catId <= 1) include = true;
            else if (catId == 2 && cat == NNBendingAudioProcessor::LayerCategory::Norm) include = true;
            else if (catId == 3 && cat == NNBendingAudioProcessor::LayerCategory::Conv) include = true;
            else if (catId == 4 && cat == NNBendingAudioProcessor::LayerCategory::Attention) include = true;
            else if (catId == 5 && cat == NNBendingAudioProcessor::LayerCategory::Linear) include = true;
            else if (catId == 6 && cat == NNBendingAudioProcessor::LayerCategory::Activation) include = true;
            else if (catId == 7 && cat == NNBendingAudioProcessor::LayerCategory::Recurrent) include = true;
            else if (catId == 8 && cat == NNBendingAudioProcessor::LayerCategory::Bias) include = true;

            if (include)
                layerCombo.addItem (l, id++);
        }

        // Populate Bridge Combo with [No Bridge] + all model layers
        bridgeCombo.clear (juce::dontSendNotification);
        bridgeCombo.addItem ("(Off)", 1);
        int bridgeId = 2;
        for (const auto& l : layers)
        {
            bridgeCombo.addItem (l, bridgeId++);
        }
        bridgeCombo.setSelectedId (1, juce::dontSendNotification);
        
        bool layerFound = false;
        for (int i = 0; i < layerCombo.getNumItems(); ++i)
        {
            if (layerCombo.getItemText(i) == currentBendingLayer)
            {
                layerFound = true;
                break;
            }
        }

        if (layerFound && currentBendingLayer.isNotEmpty())
        {
            layerCombo.setText (currentBendingLayer, juce::dontSendNotification);
            selectLayer (currentBendingLayer);
        }
        else if (layerCombo.getNumItems() > 0)
        {
            layerCombo.setSelectedId (1, juce::dontSendNotification);
            selectLayer (layerCombo.getItemText(0));
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
        statusLabel.setText ("No model loaded.", juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
        bufferStatusLabel.setText ("-", juce::dontSendNotification);
        methodCombo.clear (juce::dontSendNotification);
        layerCombo.clear (juce::dontSendNotification);
        infoBendingLabel.setText ("Load a model to view weights.", juce::dontSendNotification);
        originalWeights.clear();
        currentWeights.clear();
        currentBendingLayer = "";
        weightCanvas.setWeights ({}, {});
        updateKnobContextLabels (NNBendingAudioProcessor::LayerCategory::Other);
    }
}

void NNBendingAudioProcessorEditor::selectLayer (const juce::String& layerName)
{
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
    }

    if (state.drawnWeights.empty())
    {
        baseDrawnWeights = originalWeights;
        state.drawnWeights = originalWeights;
        audioProcessor.setLayerDrawnWeights (layerName.toStdString(), baseDrawnWeights);
    }
    else
    {
        baseDrawnWeights = state.drawnWeights;
    }

    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);
    memorySlider.removeListener (this);
    driftModeCombo.removeListener (this);
    
    scaleSlider.setValue (state.scale, juce::dontSendNotification);
    offsetSlider.setValue (state.offset, juce::dontSendNotification);
    heatSlider.setValue (state.heat, juce::dontSendNotification);
    memorySlider.setValue (state.memory, juce::dontSendNotification);
    driftModeCombo.setSelectedId ((int)state.driftMode + 1, juce::dontSendNotification);

    freezeButton.setToggleState (state.frozen, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId,
                            state.frozen ? juce::Colour::fromString ("#ffff9933") : juce::Colours::lightgrey);

    // Sync DAW host automatable parameters to active layer values
    if (auto* p = audioProcessor.getScaleParam())  p->setValueNotifyingHost (p->convertTo0to1 (state.scale));
    if (auto* p = audioProcessor.getOffsetParam()) p->setValueNotifyingHost (p->convertTo0to1 (state.offset));
    if (auto* p = audioProcessor.getHeatParam())   p->setValueNotifyingHost (p->convertTo0to1 (state.heat));
    if (auto* p = audioProcessor.getMemoryParam()) p->setValueNotifyingHost (p->convertTo0to1 (state.memory));

    // Sync bridge controls with this layer's state
    bridgeCombo.removeListener (this);
    bridgeDepthSlider.removeListener (this);

    if (state.bridgeSourceLayer.empty())
    {
        bridgeCombo.setSelectedId (1, juce::dontSendNotification);
    }
    else
    {
        int foundId = 1;
        for (int i = 0; i < bridgeCombo.getNumItems(); ++i)
        {
            if (bridgeCombo.getItemText(i).toStdString() == state.bridgeSourceLayer)
            {
                foundId = bridgeCombo.getItemId(i);
                break;
            }
        }
        bridgeCombo.setSelectedId (foundId, juce::dontSendNotification);
    }
    bridgeDepthSlider.setValue (state.bridgeDepth, juce::dontSendNotification);

    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    auto shape = audioProcessor.getBackend().get_layer_shape (layerName.toStdString());
    weightCanvas.setLayerShape (shape);
    weightCanvas.setWeights (originalWeights, currentWeights);

    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    bool bridgeEnabled = bridgeEnableToggle.getToggleState();
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
    bool isContinuous = (audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous);
    weightCanvas.setShowBridgeVisuals (bridgeEnabled);
    weightCanvas.setBridgeWeights (srcTiled, bridgedBase, bridgeEnabled ? state.bridgeDepth : 0.0f, isContinuous);

    if (!isContinuous)
    {
        std::vector<float> targetBent(bridgedBase.size());
        bool mutateEnabled = mutateEnableToggle.getToggleState();
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
    
    auto category = NNBendingAudioProcessor::classifyLayer (currentBendingLayer.toStdString());
    juce::String catName = "Other";
    if (category == NNBendingAudioProcessor::LayerCategory::Norm) catName = "Norm / Dynamics";
    else if (category == NNBendingAudioProcessor::LayerCategory::Conv) catName = "Conv Kernel";
    else if (category == NNBendingAudioProcessor::LayerCategory::Attention) catName = "Attention";
    else if (category == NNBendingAudioProcessor::LayerCategory::Linear) catName = "Linear / Dense";
    else if (category == NNBendingAudioProcessor::LayerCategory::Activation) catName = "Activation (Snake)";
    else if (category == NNBendingAudioProcessor::LayerCategory::Recurrent) catName = "Recurrent (LSTM)";
    else if (category == NNBendingAudioProcessor::LayerCategory::Bias) catName = "Bias";

    infoBendingLabel.setText ("Trace: [" + catName + "]  |  Layer: " + currentBendingLayer + "  |  " + juce::String (originalWeights.size()) + " params" + (audioProcessor.hasAutoencode() ? "  |  [Latent Hook Ready]" : ""), juce::dontSendNotification);

    // Update contextual knob labels (Gamma / Beta for Norm, Scale / Offset for other traces)
    updateKnobContextLabels (category);

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    driftModeCombo.addListener (this);

    lastKnownScale = state.scale;
    lastKnownOffset = state.offset;
    lastKnownHeat = state.heat;
    lastKnownMemory = state.memory;
    targetCurveDirty = true;

    updateHarmonicGhostPreview();
}

void NNBendingAudioProcessorEditor::applyKnobBending()
{
    if (currentBendingLayer.isEmpty()) return;
    if (baseDrawnWeights.empty())
        baseDrawnWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    
    bool mutateEnabled = mutateEnableToggle.getToggleState();
    float scale = mutateEnabled ? (float)scaleSlider.getValue() : 1.0f;
    float offset = mutateEnabled ? (float)offsetSlider.getValue() : 0.0f;

    // Store per-layer scale and offset
    audioProcessor.setLayerScale (currentBendingLayer.toStdString(), scale);
    audioProcessor.setLayerOffset (currentBendingLayer.toStdString(), offset);

    // Fetch bridge state
    auto state = audioProcessor.getLayerState (currentBendingLayer.toStdString());
    std::vector<float> bridgedBase = baseDrawnWeights;
    std::vector<float> srcTiled;
    bool bridgeEnabled = bridgeEnableToggle.getToggleState();

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
    bool isContinuous = (mode == NNBendingAudioProcessor::TriggerMode::Continuous);
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
        audioProcessor.getBackend().set_layer_weights (currentBendingLayer.toStdString(), currentWeights);
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

void NNBendingAudioProcessorEditor::resetLayerWeights()
{
    if (currentBendingLayer.isEmpty()) return;
    
    audioProcessor.getBackend().reset_layer_weights (currentBendingLayer.toStdString());
    audioProcessor.clearLayerBending (currentBendingLayer.toStdString());
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);
    memorySlider.removeListener (this);
    bridgeCombo.removeListener (this);
    bridgeDepthSlider.removeListener (this);

    scaleSlider.setValue (1.0, juce::dontSendNotification);
    offsetSlider.setValue (0.0, juce::dontSendNotification);
    heatSlider.setValue (0.0, juce::dontSendNotification);
    memorySlider.setValue (0.8, juce::dontSendNotification);
    bridgeCombo.setSelectedId (1, juce::dontSendNotification);
    bridgeDepthSlider.setValue (0.0, juce::dontSendNotification);
    freezeButton.setToggleState (false, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

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

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
    currentWeights = originalWeights;
    baseDrawnWeights = originalWeights;

    weightCanvas.setWeights (originalWeights, currentWeights);
    weightCanvas.setTargetBentWeights ({});
    weightCanvas.setBridgeWeights ({}, {}, 0.0f, audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous);
    updateHarmonicGhostPreview();
}

void NNBendingAudioProcessorEditor::resetAllLayerWeights()
{
    audioProcessor.getBackend().reset_all_layer_weights();
    audioProcessor.clearAllLayerBending();
    
    scaleSlider.removeListener (this);
    offsetSlider.removeListener (this);
    heatSlider.removeListener (this);
    memorySlider.removeListener (this);
    bridgeCombo.removeListener (this);
    bridgeDepthSlider.removeListener (this);

    scaleSlider.setValue (1.0, juce::dontSendNotification);
    offsetSlider.setValue (0.0, juce::dontSendNotification);
    heatSlider.setValue (0.0, juce::dontSendNotification);
    memorySlider.setValue (0.8, juce::dontSendNotification);
    bridgeCombo.setSelectedId (1, juce::dontSendNotification);
    bridgeDepthSlider.setValue (0.0, juce::dontSendNotification);
    freezeButton.setToggleState (false, juce::dontSendNotification);
    freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);

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

    scaleSlider.addListener (this);
    offsetSlider.addListener (this);
    heatSlider.addListener (this);
    memorySlider.addListener (this);
    bridgeCombo.addListener (this);
    bridgeDepthSlider.addListener (this);

    if (currentBendingLayer.isNotEmpty())
    {
        originalWeights = audioProcessor.getBackend().get_original_layer_weights (currentBendingLayer.toStdString());
        currentWeights = originalWeights;
        baseDrawnWeights = originalWeights;
        weightCanvas.setWeights (originalWeights, currentWeights);
        weightCanvas.setTargetBentWeights ({});
        weightCanvas.setBridgeWeights ({}, {}, 0.0f, audioProcessor.getTriggerMode() == NNBendingAudioProcessor::TriggerMode::Continuous);
        updateHarmonicGhostPreview();
    }
}

void NNBendingAudioProcessorEditor::saveModelToFile()
{
    if (!audioProcessor.isModelLoaded())
    {
        statusLabel.setText ("Cannot save: No model loaded.", juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::orangered);
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
                statusLabel.setText ("Saved bended model: " + file.getFileName(), juce::dontSendNotification);
                statusLabel.setColour (juce::Label::textColourId, juce::Colours::lightgreen);
            }
            else
            {
                statusLabel.setText ("Error: Failed to save model.", juce::dontSendNotification);
                statusLabel.setColour (juce::Label::textColourId, juce::Colours::orangered);
            }
        }
    });
}

void NNBendingAudioProcessorEditor::updateKnobContextLabels (NNBendingAudioProcessor::LayerCategory category)
{
    if (category == NNBendingAudioProcessor::LayerCategory::Norm)
    {
        // Normalization / Dynamics context: Gamma (Gain) & Beta (Bias)
        scaleLabel.setText (juce::CharPointer_UTF8 ("Gamma (\xce\xb3)"), juce::dontSendNotification);
        scaleLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ffc084fc")); // Soft violet

        offsetLabel.setText (juce::CharPointer_UTF8 ("Beta (\xce\xb2)"), juce::dontSendNotification);
        offsetLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ff38bdf8")); // Sky blue
    }
    else if (category == NNBendingAudioProcessor::LayerCategory::Activation)
    {
        // Snake / periodic activation context: Alpha (Freq scaling) & Offset
        scaleLabel.setText (juce::CharPointer_UTF8 ("Alpha (\xce\xb1)"), juce::dontSendNotification);
        scaleLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#fff59e0b")); // Amber

        offsetLabel.setText ("Offset", juce::dontSendNotification);
        offsetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    }
    else
    {
        // Standard convolution / dense / bias context: Scale & Offset
        scaleLabel.setText ("Scale", juce::dontSendNotification);
        scaleLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);

        offsetLabel.setText ("Offset", juce::dontSendNotification);
        offsetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    }
}

void NNBendingAudioProcessorEditor::updateHarmonicGhostPreview()
{
    if (!harmonicEnableToggle.getToggleState() || currentBendingLayer.isEmpty() || baseDrawnWeights.empty())
    {
        weightCanvas.setHarmonicGhostWeights ({}, false);
        return;
    }

    size_t N = baseDrawnWeights.size();
    std::vector<float> ghost (N);

    float f = (float)harmonicFreqSlider.getValue();
    int partials = (int)harmonicPartialsSlider.getValue();
    float morph = (float)harmonicMorphSlider.getValue();
    float gain = (float)harmonicDepthSlider.getValue();
    int mode = harmonicModeCombo.getSelectedId(); // 1 = Add, 2 = Ring, 3 = Replace

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

void NNBendingAudioProcessorEditor::applyHarmonicWeights()
{
    if (currentBendingLayer.isEmpty() || baseDrawnWeights.empty())
        return;

    size_t N = baseDrawnWeights.size();
    std::vector<float> stamped (N);

    float f = (float)harmonicFreqSlider.getValue();
    int partials = (int)harmonicPartialsSlider.getValue();
    float morph = (float)harmonicMorphSlider.getValue();
    float gain = (float)harmonicDepthSlider.getValue();
    int mode = harmonicModeCombo.getSelectedId();

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
    applyKnobBending();

    infoBendingLabel.setText ("Harmonic wave stamped to " + currentBendingLayer, juce::dontSendNotification);
}

void NNBendingAudioProcessorEditor::setSideTab (SideTab newTab)
{
    currentSideTab = newTab;
    sidePanel.setSideTab (newTab);
    repaint();
}

void NNBendingAudioProcessorEditor::setViewMode (ViewMode newMode)
{
    currentViewMode = newMode;

    bool isWeights = (currentViewMode == ViewMode::Weights);
    bool isLatent = (currentViewMode == ViewMode::Latent);

    juce::Colour activeBg = juce::Colour::fromString ("#ff4a1d72"); // Vivid violet
    juce::Colour inactiveBg = juce::Colour::fromString ("#ff1c1926"); // Dark charcoal
    juce::Colour activeText = juce::Colours::white;
    juce::Colour inactiveText = juce::Colours::darkgrey;

    viewWeightsButton.setColour (juce::TextButton::buttonColourId, isWeights ? activeBg : inactiveBg);
    viewWeightsButton.setColour (juce::TextButton::textColourOffId, isWeights ? activeText : inactiveText);

    viewLatentButton.setColour (juce::TextButton::buttonColourId, isLatent ? activeBg : inactiveBg);
    viewLatentButton.setColour (juce::TextButton::textColourOffId, isLatent ? activeText : inactiveText);

    weightCanvas.setVisible (isWeights);
    latentPad.setVisible (isLatent);

    categoryLabel.setVisible (isWeights);
    categoryCombo.setVisible (isWeights);
    layerLabel.setVisible (isWeights);
    layerCombo.setVisible (isWeights);
    resetLayerButton.setVisible (isWeights);
    resetAllButton.setVisible (isWeights);
    displayCurveButton.setVisible (isWeights);
    displayMatrixButton.setVisible (isWeights);

    sidePanel.setViewMode (newMode);

    if (isLatent)
    {
        if (audioProcessor.hasAutoencode())
            infoBendingLabel.setText ("Latent Topographic Vector Pad: Drag cursor to modulate autoencoder bottleneck.", juce::dontSendNotification);
        else
            infoBendingLabel.setText ("Latent View: Standby. Loaded model does not expose paired encode + decode methods.", juce::dontSendNotification);
    }
    else
    {
        infoBendingLabel.setText (currentBendingLayer.isNotEmpty() ? ("Layer: " + currentBendingLayer) : "Select a layer to bend weights.", juce::dontSendNotification);
    }

    repaint();
}

void NNBendingAudioProcessorEditor::updateLatentControlsVisibility()
{
    sidePanel.latentPanel.resized();
}

