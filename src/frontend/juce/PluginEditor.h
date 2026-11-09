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

class NetBenderAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                       public juce::ComboBox::Listener,
                                       public juce::Slider::Listener,
                                       public juce::Button::Listener,
                                       private juce::Timer
{
public:
    NetBenderAudioProcessorEditor(NetBenderAudioProcessor&);
    ~NetBenderAudioProcessorEditor() override;

    auto paint(juce::Graphics&) -> void override;
    auto resized() -> void override;

    auto comboBoxChanged(juce::ComboBox* comboBoxThatHasChanged) -> void override;
    auto sliderValueChanged(juce::Slider* slider) -> void override;
    auto buttonClicked(juce::Button* button) -> void override;

    using ViewMode = SidePanelComponent::ViewMode;
    using SideTab = SidePanelComponent::SideTab;

    FlatLookAndFeel flatLf;

    HeaderComponent header_bar;
    LayerBarComponent layer_bar;
    SidePanelComponent side_panel;
    WeightBendingComponent weight_canvas;
    LatentTerrainComponent latent_pad;

    // Compatibility references
    HeaderComponent& headerBar{header_bar};
    LayerBarComponent& layerBar{layer_bar};
    SidePanelComponent& sidePanel{side_panel};
    WeightBendingComponent& weightCanvas{weight_canvas};
    LatentTerrainComponent& latentPad{latent_pad};

    // Canonical snake_case references
    juce::Label& title_label{header_bar.title_label};
    juce::TextButton& load_button{header_bar.load_button};
    juce::Label& status_label{header_bar.status_label};
    juce::Label& method_label{header_bar.method_label};
    juce::ComboBox& method_combo{header_bar.method_combo};
    juce::Label& buffer_label{header_bar.buffer_label};
    juce::Label& buffer_status_label{header_bar.buffer_status_label};
    juce::TextButton& gpu_button{header_bar.gpu_button};
    juce::Label& dry_wet_label{header_bar.dry_wet_label};
    juce::Slider& dry_wet_slider{header_bar.dry_wet_slider};
    juce::TextButton& save_model_button{header_bar.save_model_button};
    juce::TextButton& fuse_button{header_bar.fuse_button};
    juce::TextButton& short_circuit_button{header_bar.short_circuit_button};
    juce::Label& trigger_mode_label{header_bar.trigger_mode_label};
    juce::ComboBox& trigger_mode_combo{header_bar.trigger_mode_combo};

    juce::TextButton& view_weights_button{layer_bar.view_weights_button};
    juce::TextButton& view_latent_button{layer_bar.view_latent_button};
    juce::TextButton& display_curve_button{layer_bar.display_curve_button};
    juce::TextButton& display_matrix_button{layer_bar.display_matrix_button};
    juce::Label& category_label{layer_bar.category_label};
    juce::ComboBox& category_combo{layer_bar.category_combo};
    juce::Label& layer_label{layer_bar.layer_label};
    juce::ComboBox& layer_combo{layer_bar.layer_combo};
    juce::TextButton& reset_layer_button{layer_bar.reset_layer_button};
    juce::TextButton& reset_all_button{layer_bar.reset_all_button};

    juce::TextButton& tab_mutate_button{side_panel.tab_mutate_button};
    juce::TextButton& tab_harmonics_button{side_panel.tab_harmonics_button};
    juce::TextButton& tab_bridge_button{side_panel.tab_bridge_button};
    FlatTabLookAndFeel& flat_tab_lf{flatLf};

    juce::ToggleButton& mutate_enable_toggle{side_panel.mutate_panel.mutate_enable_toggle};
    juce::Label& scale_label{side_panel.mutate_panel.scale_label};
    juce::Slider& scale_slider{side_panel.mutate_panel.scale_slider};
    juce::Label& offset_label{side_panel.mutate_panel.offset_label};
    juce::Slider& offset_slider{side_panel.mutate_panel.offset_slider};

    juce::ToggleButton& drift_enable_toggle{side_panel.mutate_panel.drift_enable_toggle};
    juce::Label& heat_label{side_panel.mutate_panel.heat_label};
    juce::Slider& heat_slider{side_panel.mutate_panel.heat_slider};
    juce::Label& memory_label{side_panel.mutate_panel.memory_label};
    juce::Slider& memory_slider{side_panel.mutate_panel.memory_slider};
    juce::Label& drift_mode_label{side_panel.mutate_panel.drift_mode_label};
    juce::ComboBox& drift_mode_combo{side_panel.mutate_panel.drift_mode_combo};
    juce::ToggleButton& freeze_button{side_panel.mutate_panel.freeze_button};

    juce::ToggleButton& harmonic_enable_toggle{side_panel.harmonics_panel.harmonic_enable_toggle};
    juce::Label& harmonic_freq_label{side_panel.harmonics_panel.harmonic_freq_label};
    juce::Slider& harmonic_freq_slider{side_panel.harmonics_panel.harmonic_freq_slider};
    juce::Label& harmonic_partials_label{side_panel.harmonics_panel.harmonic_partials_label};
    juce::Slider& harmonic_partials_slider{side_panel.harmonics_panel.harmonic_partials_slider};
    juce::Label& harmonic_morph_label{side_panel.harmonics_panel.harmonic_morph_label};
    juce::Slider& harmonic_morph_slider{side_panel.harmonics_panel.harmonic_morph_slider};
    juce::Label& harmonic_depth_label{side_panel.harmonics_panel.harmonic_depth_label};
    juce::Slider& harmonic_depth_slider{side_panel.harmonics_panel.harmonic_depth_slider};
    juce::Label& harmonic_mode_label{side_panel.harmonics_panel.harmonic_mode_label};
    juce::ComboBox& harmonic_mode_combo{side_panel.harmonics_panel.harmonic_mode_combo};
    juce::TextButton& harmonic_apply_button{side_panel.harmonics_panel.harmonic_apply_button};

    juce::ToggleButton& bridge_enable_toggle{side_panel.bridge_panel.bridge_enable_toggle};
    juce::Label& bridge_label{side_panel.bridge_panel.bridge_label};
    juce::ComboBox& bridge_combo{side_panel.bridge_panel.bridge_combo};
    juce::Label& bridge_depth_label{side_panel.bridge_panel.bridge_depth_label};
    juce::Slider& bridge_depth_slider{side_panel.bridge_panel.bridge_depth_slider};

    juce::ToggleButton& latent_enable_button{side_panel.latent_panel.latent_enable_button};
    juce::Label& latent_depth_label{side_panel.latent_panel.latent_depth_label};
    juce::Slider& latent_depth_slider{side_panel.latent_panel.latent_depth_slider};
    juce::Label& latent_mode_label{side_panel.latent_panel.latent_mode_label};
    juce::ComboBox& latent_mode_combo{side_panel.latent_panel.latent_mode_combo};
    juce::Label& latent_slew_label{side_panel.latent_panel.latent_slew_label};
    juce::Slider& latent_slew_slider{side_panel.latent_panel.latent_slew_slider};

private:
    auto timerCallback() -> void override;
    auto update_model_info() -> void;
    auto select_layer(const juce::String& layer_name) -> void;
    auto apply_knob_bending() -> void;
    auto reset_layer_weights() -> void;
    auto reset_all_layer_weights() -> void;
    auto save_model_to_file() -> void;
    auto update_knob_context_labels(LayerCategory category) -> void;
    auto set_view_mode(ViewMode new_mode) -> void;
    auto set_side_tab(SideTab new_tab) -> void;

    NetBenderAudioProcessor& audioProcessor;

    ViewMode currentViewMode{ViewMode::Weights};
    SideTab currentSideTab{SideTab::Mutate};

    juce::GroupComponent bendingGroup{"Bending", "Interactive Weight Bending"};

    auto updateLatentControlsVisibility() -> void;
    auto updateHarmonicGhostPreview() -> void;
    auto applyHarmonicWeights() -> void;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NetBenderAudioProcessorEditor)
};
