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
    latent_depth_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
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
    latent_slew_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_standard, slider_box_height_standard);
    latent_slew_slider.setColour(juce::Slider::trackColourId, sky_blue);
    addAndMakeVisible(latent_slew_slider);
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
}
