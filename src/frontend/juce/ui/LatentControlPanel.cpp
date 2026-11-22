#include "LatentControlPanel.h"
#include "UIConstants.h"

LatentControlPanel::LatentControlPanel() {
    using namespace UIConstants::Colors;
    using namespace UIConstants::Layout;

    latent_enable_button.setButtonText("Enable Latent Hook");
    latent_enable_button.setToggleState(true, juce::dontSendNotification);
    latent_enable_button.setColour(juce::ToggleButton::textColourId, magenta);
    latent_enable_button.setColour(juce::ToggleButton::tickColourId, magenta);
    addAndMakeVisible(latent_enable_button);

    latent_depth_label.setText("Latent Depth", juce::dontSendNotification);
    latent_depth_label.setJustificationType(juce::Justification::centredLeft);
    latent_depth_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(latent_depth_label);

    latent_depth_slider.setRange(0.0, 2.0, 0.01);
    latent_depth_slider.setValue(1.0);
    latent_depth_slider.setSliderStyle(juce::Slider::LinearBar);
    latent_depth_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false,
                                        slider_box_width_standard, slider_box_height_standard);
    latent_depth_slider.setColour(juce::Slider::trackColourId, magenta);
    addAndMakeVisible(latent_depth_slider);

    latent_mode_label.setText("Orbit Mode:", juce::dontSendNotification);
    latent_mode_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(latent_mode_label);

    latent_mode_combo.addItem("Fourier Orbit", 1);
    latent_mode_combo.addItem("Latent Slew", 2);
    latent_mode_combo.addItem("Quantize", 3);
    latent_mode_combo.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(latent_mode_combo);

    latent_slew_label.setText("Slew Speed", juce::dontSendNotification);
    latent_slew_label.setJustificationType(juce::Justification::centredLeft);
    latent_slew_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(latent_slew_label);

    latent_slew_slider.setRange(0.001, 1.0, 0.001);
    latent_slew_slider.setValue(0.25);
    latent_slew_slider.setSliderStyle(juce::Slider::LinearBar);
    latent_slew_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard,
                                       slider_box_height_standard);
    latent_slew_slider.setColour(juce::Slider::trackColourId, sky_blue);
    addAndMakeVisible(latent_slew_slider);

    decode_gen_smoothness_label.setText("Gen Smoothness", juce::dontSendNotification);
    decode_gen_smoothness_label.setJustificationType(juce::Justification::centredLeft);
    decode_gen_smoothness_label.setColour(juce::Label::textColourId, text_light);
    addChildComponent(decode_gen_smoothness_label);

    decode_gen_smoothness_slider.setRange(0.0, 1.0, 0.01);
    decode_gen_smoothness_slider.setValue(0.30);
    decode_gen_smoothness_slider.setSliderStyle(juce::Slider::LinearBar);
    decode_gen_smoothness_slider.setTextBoxStyle(
        juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    decode_gen_smoothness_slider.setColour(juce::Slider::trackColourId, sky_blue);
    addChildComponent(decode_gen_smoothness_slider);
}

auto LatentControlPanel::set_decode_gen_active(bool active) -> void {
    m_decodeGenActive = active;
    decode_gen_smoothness_label.setVisible(active);
    decode_gen_smoothness_slider.setVisible(active);
    resized();
}

auto LatentControlPanel::resized() -> void {
    using namespace UIConstants::Layout;

    auto area = getLocalBounds();

    latent_enable_button.setBounds(area.removeFromTop(toggle_height));
    area.removeFromTop(gap_standard);

    latent_mode_label.setBounds(area.removeFromTop(label_height));
    latent_mode_combo.setBounds(area.removeFromTop(row_height_standard));
    area.removeFromTop(gap_large);

    int mode_id = latent_mode_combo.getSelectedId();

    if (mode_id == 2) {
        latent_slew_label.setBounds(area.removeFromTop(label_height));
        latent_slew_slider.setBounds(area.removeFromTop(toggle_height));
        latent_depth_label.setBounds({});
        latent_depth_slider.setBounds({});
    } else {
        latent_depth_label.setBounds(area.removeFromTop(label_height));
        latent_depth_slider.setBounds(area.removeFromTop(toggle_height));
        latent_slew_label.setBounds({});
        latent_slew_slider.setBounds({});
    }

    if (m_decodeGenActive) {
        area.removeFromTop(gap_large);
        decode_gen_smoothness_label.setBounds(area.removeFromTop(label_height));
        decode_gen_smoothness_slider.setBounds(area.removeFromTop(toggle_height));
    } else {
        decode_gen_smoothness_label.setBounds({});
        decode_gen_smoothness_slider.setBounds({});
    }
}
