#pragma once
#include "PluginProcessor.h"
#include <JuceHeader.h>
#include "ui/FlatLookAndFeel.h"
#include "ui/HeaderComponent.h"
#include "ui/LayerBarComponent.h"
#include "ui/SidePanelComponent.h"
#include "ui/WeightBendingComponent.h"
#include "ui/LatentTerrainComponent.h"
#include <vector>

class NNBendingAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                       public juce::ComboBox::Listener,
                                       public juce::Slider::Listener,
                                       public juce::Button::Listener,
                                       private juce::Timer
{
public:
    NNBendingAudioProcessorEditor (NNBendingAudioProcessor&);
    ~NNBendingAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    void comboBoxChanged (juce::ComboBox* comboBoxThatHasChanged) override;
    void sliderValueChanged (juce::Slider* slider) override;
    void buttonClicked (juce::Button* button) override;

    using ViewMode = SidePanelComponent::ViewMode;
    using SideTab = SidePanelComponent::SideTab;

    FlatLookAndFeel flatLf;

    HeaderComponent headerBar;
    LayerBarComponent layerBar;
    SidePanelComponent sidePanel;
    WeightBendingComponent weightCanvas;
    LatentTerrainComponent latentPad;

    // Direct references to UI widgets for full compatibility
    juce::Label& titleLabel { headerBar.titleLabel };
    juce::TextButton& loadButton { headerBar.loadButton };
    juce::Label& statusLabel { headerBar.statusLabel };
    juce::Label& methodLabel { headerBar.methodLabel };
    juce::ComboBox& methodCombo { headerBar.methodCombo };
    juce::Label& bufferLabel { headerBar.bufferLabel };
    juce::Label& bufferStatusLabel { headerBar.bufferStatusLabel };
    juce::TextButton& gpuButton { headerBar.gpuButton };
    juce::Label& dryWetLabel { headerBar.dryWetLabel };
    juce::Slider& dryWetSlider { headerBar.dryWetSlider };
    juce::TextButton& saveModelButton { headerBar.saveModelButton };
    juce::TextButton& fuseButton { headerBar.fuseButton };
    juce::TextButton& shortCircuitButton { headerBar.shortCircuitButton };
    juce::Label& triggerModeLabel { headerBar.triggerModeLabel };
    juce::ComboBox& triggerModeCombo { headerBar.triggerModeCombo };

    juce::TextButton& viewWeightsButton { layerBar.viewWeightsButton };
    juce::TextButton& viewLatentButton { layerBar.viewLatentButton };
    juce::TextButton& displayCurveButton { layerBar.displayCurveButton };
    juce::TextButton& displayMatrixButton { layerBar.displayMatrixButton };
    juce::Label& categoryLabel { layerBar.categoryLabel };
    juce::ComboBox& categoryCombo { layerBar.categoryCombo };
    juce::Label& layerLabel { layerBar.layerLabel };
    juce::ComboBox& layerCombo { layerBar.layerCombo };
    juce::TextButton& resetLayerButton { layerBar.resetLayerButton };
    juce::TextButton& resetAllButton { layerBar.resetAllButton };

    juce::TextButton& tabMutateButton { sidePanel.tabMutateButton };
    juce::TextButton& tabHarmonicsButton { sidePanel.tabHarmonicsButton };
    juce::TextButton& tabBridgeButton { sidePanel.tabBridgeButton };
    FlatTabLookAndFeel& flatTabLf { flatLf };

    juce::ToggleButton& mutateEnableToggle { sidePanel.mutatePanel.mutateEnableToggle };
    juce::Label& scaleLabel { sidePanel.mutatePanel.scaleLabel };
    juce::Slider& scaleSlider { sidePanel.mutatePanel.scaleSlider };
    juce::Label& offsetLabel { sidePanel.mutatePanel.offsetLabel };
    juce::Slider& offsetSlider { sidePanel.mutatePanel.offsetSlider };

    juce::ToggleButton& driftEnableToggle { sidePanel.mutatePanel.driftEnableToggle };
    juce::Label& heatLabel { sidePanel.mutatePanel.heatLabel };
    juce::Slider& heatSlider { sidePanel.mutatePanel.heatSlider };
    juce::Label& memoryLabel { sidePanel.mutatePanel.memoryLabel };
    juce::Slider& memorySlider { sidePanel.mutatePanel.memorySlider };
    juce::Label& driftModeLabel { sidePanel.mutatePanel.driftModeLabel };
    juce::ComboBox& driftModeCombo { sidePanel.mutatePanel.driftModeCombo };
    juce::ToggleButton& freezeButton { sidePanel.mutatePanel.freezeButton };

    juce::ToggleButton& harmonicEnableToggle { sidePanel.harmonicsPanel.harmonicEnableToggle };
    juce::Label& harmonicFreqLabel { sidePanel.harmonicsPanel.harmonicFreqLabel };
    juce::Slider& harmonicFreqSlider { sidePanel.harmonicsPanel.harmonicFreqSlider };
    juce::Label& harmonicPartialsLabel { sidePanel.harmonicsPanel.harmonicPartialsLabel };
    juce::Slider& harmonicPartialsSlider { sidePanel.harmonicsPanel.harmonicPartialsSlider };
    juce::Label& harmonicMorphLabel { sidePanel.harmonicsPanel.harmonicMorphLabel };
    juce::Slider& harmonicMorphSlider { sidePanel.harmonicsPanel.harmonicMorphSlider };
    juce::Label& harmonicDepthLabel { sidePanel.harmonicsPanel.harmonicDepthLabel };
    juce::Slider& harmonicDepthSlider { sidePanel.harmonicsPanel.harmonicDepthSlider };
    juce::Label& harmonicModeLabel { sidePanel.harmonicsPanel.harmonicModeLabel };
    juce::ComboBox& harmonicModeCombo { sidePanel.harmonicsPanel.harmonicModeCombo };
    juce::TextButton& harmonicApplyButton { sidePanel.harmonicsPanel.harmonicApplyButton };

    juce::ToggleButton& bridgeEnableToggle { sidePanel.bridgePanel.bridgeEnableToggle };
    juce::Label& bridgeLabel { sidePanel.bridgePanel.bridgeLabel };
    juce::ComboBox& bridgeCombo { sidePanel.bridgePanel.bridgeCombo };
    juce::Label& bridgeDepthLabel { sidePanel.bridgePanel.bridgeDepthLabel };
    juce::Slider& bridgeDepthSlider { sidePanel.bridgePanel.bridgeDepthSlider };

    juce::ToggleButton& latentEnableButton { sidePanel.latentPanel.latentEnableButton };
    juce::Label& latentDepthLabel { sidePanel.latentPanel.latentDepthLabel };
    juce::Slider& latentDepthSlider { sidePanel.latentPanel.latentDepthSlider };
    juce::Label& latentModeLabel { sidePanel.latentPanel.latentModeLabel };
    juce::ComboBox& latentModeCombo { sidePanel.latentPanel.latentModeCombo };
    juce::Label& latentSlewLabel { sidePanel.latentPanel.latentSlewLabel };
    juce::Slider& latentSlewSlider { sidePanel.latentPanel.latentSlewSlider };

private:
    void timerCallback() override;
    void updateModelInfo();
    void selectLayer (const juce::String& layerName);
    void applyKnobBending();
    void resetLayerWeights();
    void resetAllLayerWeights();
    void saveModelToFile();
    void updateKnobContextLabels (NNBendingAudioProcessor::LayerCategory category);
    void setViewMode (ViewMode newMode);
    void setSideTab (SideTab newTab);

    NNBendingAudioProcessor& audioProcessor;

    ViewMode currentViewMode { ViewMode::Weights };
    SideTab currentSideTab { SideTab::Mutate };

    juce::GroupComponent bendingGroup { "Bending", "Interactive Weight Bending" };

    void updateLatentControlsVisibility();
    void updateHarmonicGhostPreview();
    void applyHarmonicWeights();

    juce::Label infoBendingLabel;

    std::unique_ptr<juce::FileChooser> fileChooser;

    juce::String currentBendingLayer;
    std::vector<float> originalWeights;
    std::vector<float> currentWeights;
    std::vector<float> baseDrawnWeights;
    double lastKnobScale { 1.0 };
    double lastKnobOffset { 0.0 };
    float lastKnownScale { 1.0f };
    float lastKnownOffset { 0.0f };
    float lastKnownHeat { 0.0f };
    float lastKnownMemory { 0.8f };
    float lastKnownDryWet { 1.0f };
    std::string lastBridgeSource;
    float lastBridgeDepth { -1.0f };
    bool lastBridgeEnabled { false };
    float lastTargetScale { -999.0f };
    float lastTargetOffset { -999.0f };
    bool targetCurveDirty { true };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NNBendingAudioProcessorEditor)
};
