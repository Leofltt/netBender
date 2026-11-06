#include "HarmonicsPanel.h"
#include "UIConstants.h"

HarmonicsPanel::HarmonicsPanel() {
    using namespace UIConstants::Colors;
    using namespace UIConstants::Layout;

    harmonic_enable_toggle.setToggleState(false, juce::dontSendNotification);
    harmonic_enable_toggle.setColour(juce::ToggleButton::textColourId, amber_light);
    harmonic_enable_toggle.setColour(juce::ToggleButton::tickColourId, amber_light);
    addAndMakeVisible(harmonic_enable_toggle);

    harmonic_freq_label.setText("Frequency", juce::dontSendNotification);
    harmonic_freq_label.setJustificationType(juce::Justification::centredLeft);
    harmonic_freq_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(harmonic_freq_label);

    harmonic_freq_slider.setRange(0.5, 32.0, 0.1);
    harmonic_freq_slider.setValue(2.0);
    harmonic_freq_slider.setSliderStyle(juce::Slider::LinearBar);
    harmonic_freq_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    harmonic_freq_slider.setColour(juce::Slider::trackColourId, amber);
    addAndMakeVisible(harmonic_freq_slider);

    harmonic_partials_label.setText("Partials", juce::dontSendNotification);
    harmonic_partials_label.setJustificationType(juce::Justification::centredLeft);
    harmonic_partials_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(harmonic_partials_label);

    harmonic_partials_slider.setRange(1.0, 8.0, 1.0);
    harmonic_partials_slider.setValue(1.0);
    harmonic_partials_slider.setSliderStyle(juce::Slider::LinearBar);
    harmonic_partials_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    harmonic_partials_slider.setColour(juce::Slider::trackColourId, amber);
    addAndMakeVisible(harmonic_partials_slider);

    harmonic_morph_label.setText("Morph Phase", juce::dontSendNotification);
    harmonic_morph_label.setJustificationType(juce::Justification::centredLeft);
    harmonic_morph_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(harmonic_morph_label);

    harmonic_morph_slider.setRange(0.0, 6.2831853, 0.01);
    harmonic_morph_slider.setValue(0.0);
    harmonic_morph_slider.setSliderStyle(juce::Slider::LinearBar);
    harmonic_morph_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    harmonic_morph_slider.setColour(juce::Slider::trackColourId, amber);
    addAndMakeVisible(harmonic_morph_slider);

    harmonic_depth_label.setText("Wave Gain", juce::dontSendNotification);
    harmonic_depth_label.setJustificationType(juce::Justification::centredLeft);
    harmonic_depth_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(harmonic_depth_label);

    harmonic_depth_slider.setRange(0.0, 2.0, 0.01);
    harmonic_depth_slider.setValue(0.5);
    harmonic_depth_slider.setSliderStyle(juce::Slider::LinearBar);
    harmonic_depth_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    harmonic_depth_slider.setColour(juce::Slider::trackColourId, amber);
    addAndMakeVisible(harmonic_depth_slider);

    harmonic_mode_label.setText("Combine Mode:", juce::dontSendNotification);
    harmonic_mode_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(harmonic_mode_label);

    harmonic_mode_combo.addItem("Add (+)", 1);
    harmonic_mode_combo.addItem("Ring (*)", 2);
    harmonic_mode_combo.addItem("Replace", 3);
    harmonic_mode_combo.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(harmonic_mode_combo);

    harmonic_apply_button.setButtonText("Stamp Wave");
    harmonic_apply_button.setColour(juce::TextButton::buttonColourId, amber_dark);
    harmonic_apply_button.setColour(juce::TextButton::textColourOffId, amber_bright);
    addAndMakeVisible(harmonic_apply_button);
}

auto HarmonicsPanel::resized() -> void {
    using namespace UIConstants::Layout;

    auto area = getLocalBounds();

    harmonic_enable_toggle.setBounds(area.removeFromTop(toggle_height));
    area.removeFromTop(gap_small);

    harmonic_freq_label.setBounds(area.removeFromTop(label_height));
    harmonic_freq_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    harmonic_partials_label.setBounds(area.removeFromTop(label_height));
    harmonic_partials_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    harmonic_morph_label.setBounds(area.removeFromTop(label_height));
    harmonic_morph_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    harmonic_depth_label.setBounds(area.removeFromTop(label_height));
    harmonic_depth_slider.setBounds(area.removeFromTop(row_height_compact));
    area.removeFromTop(gap_medium);

    harmonic_mode_label.setBounds(area.removeFromTop(label_height));
    harmonic_mode_combo.setBounds(area.removeFromTop(toggle_height));
    area.removeFromTop(gap_xlarge);

    harmonic_apply_button.setBounds(area.removeFromTop(action_button_height));
}
