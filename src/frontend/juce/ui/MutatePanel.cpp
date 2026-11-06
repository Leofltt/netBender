#include "MutatePanel.h"
#include "UIConstants.h"

MutatePanel::MutatePanel() {
    using namespace UIConstants::Colors;
    using namespace UIConstants::Layout;

    mutate_enable_toggle.setToggleState(true, juce::dontSendNotification);
    mutate_enable_toggle.setColour(juce::ToggleButton::textColourId, lilac);
    mutate_enable_toggle.setColour(juce::ToggleButton::tickColourId, lilac);
    addAndMakeVisible(mutate_enable_toggle);

    scale_label.setText("Scale", juce::dontSendNotification);
    scale_label.setJustificationType(juce::Justification::centredLeft);
    scale_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(scale_label);

    scale_slider.setRange(0.0, 5.0, 0.01);
    scale_slider.setValue(1.0);
    scale_slider.setSliderStyle(juce::Slider::LinearBar);
    scale_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    scale_slider.setColour(juce::Slider::trackColourId, violet);
    addAndMakeVisible(scale_slider);

    offset_label.setText("Offset", juce::dontSendNotification);
    offset_label.setJustificationType(juce::Justification::centredLeft);
    offset_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(offset_label);

    offset_slider.setRange(-2.0, 2.0, 0.001);
    offset_slider.setValue(0.0);
    offset_slider.setSliderStyle(juce::Slider::LinearBar);
    offset_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    offset_slider.setColour(juce::Slider::trackColourId, cyan);
    addAndMakeVisible(offset_slider);

    drift_enable_toggle.setToggleState(false, juce::dontSendNotification);
    drift_enable_toggle.setColour(juce::ToggleButton::textColourId, drift_orange);
    drift_enable_toggle.setColour(juce::ToggleButton::tickColourId, drift_orange);
    addAndMakeVisible(drift_enable_toggle);

    heat_label.setText("Heat (Jitter)", juce::dontSendNotification);
    heat_label.setJustificationType(juce::Justification::centredLeft);
    heat_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(heat_label);

    heat_slider.setRange(0.0, 1.0, 0.001);
    heat_slider.setValue(0.0);
    heat_slider.setSliderStyle(juce::Slider::LinearBar);
    heat_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    heat_slider.setColour(juce::Slider::trackColourId, flame_orange);
    addAndMakeVisible(heat_slider);

    memory_label.setText("Memory (Drag)", juce::dontSendNotification);
    memory_label.setJustificationType(juce::Justification::centredLeft);
    memory_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(memory_label);

    memory_slider.setRange(0.0, 1.0, 0.001);
    memory_slider.setValue(0.8);
    memory_slider.setSliderStyle(juce::Slider::LinearBar);
    memory_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    memory_slider.setColour(juce::Slider::trackColourId, deep_sky_blue);
    addAndMakeVisible(memory_slider);

    drift_mode_label.setText("Drift Algorithm:", juce::dontSendNotification);
    drift_mode_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(drift_mode_label);

    drift_mode_combo.addItem("Thermal OU", 1);
    drift_mode_combo.addItem("Random Walk", 2);
    drift_mode_combo.addItem("Glitch Noise", 3);
    drift_mode_combo.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(drift_mode_combo);

    freeze_button.setButtonText("Freeze Drift");
    freeze_button.setColour(juce::ToggleButton::textColourId, text_light);
    freeze_button.setColour(juce::ToggleButton::tickColourId, drift_orange);
    addAndMakeVisible(freeze_button);
}

auto MutatePanel::resized() -> void {
    using namespace UIConstants::Layout;

    auto area = getLocalBounds();

    mutate_enable_toggle.setBounds(area.removeFromTop(toggle_height));
    area.removeFromTop(gap_small);

    scale_label.setBounds(area.removeFromTop(label_height));
    scale_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    offset_label.setBounds(area.removeFromTop(label_height));
    offset_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_xlarge);

    drift_enable_toggle.setBounds(area.removeFromTop(toggle_height));
    area.removeFromTop(gap_small);

    heat_label.setBounds(area.removeFromTop(label_height));
    heat_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    memory_label.setBounds(area.removeFromTop(label_height));
    memory_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    drift_mode_label.setBounds(area.removeFromTop(label_height));
    drift_mode_combo.setBounds(area.removeFromTop(toggle_height));
    area.removeFromTop(gap_standard);

    freeze_button.setBounds(area.removeFromTop(toggle_height));
}
