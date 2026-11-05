#include "HeaderComponent.h"
#include "UIConstants.h"

HeaderComponent::HeaderComponent() {
    using namespace UIConstants::Colors;
    using namespace UIConstants::Layout;
    using namespace UIConstants::Typography;

    title_label.setText("NetBender", juce::dontSendNotification);
    title_label.setFont(juce::FontOptions(font_title, juce::Font::bold));
    title_label.setJustificationType(juce::Justification::centredLeft);
    title_label.setColour(juce::Label::textColourId, juce::Colours::lightcyan);
    addAndMakeVisible(title_label);

    load_button.setButtonText("Load Model");
    load_button.setColour(juce::TextButton::buttonColourId, dark_violet);
    load_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(load_button);

    status_label.setText("No model loaded.", juce::dontSendNotification);
    status_label.setJustificationType(juce::Justification::centredLeft);
    status_label.setColour(juce::Label::textColourId, juce::Colours::darkgrey);
    addAndMakeVisible(status_label);

    method_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(method_label);
    addAndMakeVisible(method_combo);

    buffer_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(buffer_label);
    buffer_status_label.setText("-", juce::dontSendNotification);
    buffer_status_label.setJustificationType(juce::Justification::centred);
    buffer_status_label.setColour(juce::Label::backgroundColourId, surface_card);
    buffer_status_label.setColour(juce::Label::outlineColourId, border_medium);
    buffer_status_label.setColour(juce::Label::textColourId, juce::Colours::cyan);
    addAndMakeVisible(buffer_status_label);

    gpu_button.setButtonText("CPU");
    gpu_button.setColour(juce::TextButton::buttonColourId, button_secondary_bg);
    gpu_button.setColour(juce::TextButton::textColourOffId, text_light);
    addAndMakeVisible(gpu_button);

    dry_wet_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(dry_wet_label);

    dry_wet_slider.setRange(0.0, 1.0, 0.01);
    dry_wet_slider.setValue(1.0);
    dry_wet_slider.setSliderStyle(juce::Slider::LinearBar);
    dry_wet_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_compact, slider_box_height_compact);
    dry_wet_slider.setColour(juce::Slider::trackColourId, violet);
    addAndMakeVisible(dry_wet_slider);

    save_model_button.setColour(juce::TextButton::buttonColourId, navy);
    save_model_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(save_model_button);

    fuse_button.setButtonText("FUSE: OK");
    fuse_button.setColour(juce::TextButton::buttonColourId, forest_green);
    fuse_button.setColour(juce::TextButton::textColourOffId, juce::Colours::lightgreen);
    addAndMakeVisible(fuse_button);

    short_circuit_button.setButtonText("SHORT [!]");
    short_circuit_button.setColour(juce::TextButton::buttonColourId, crimson);
    short_circuit_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(short_circuit_button);

    trigger_mode_label.setColour(juce::Label::textColourId, text_light);
    addAndMakeVisible(trigger_mode_label);
    trigger_mode_combo.addItem("Continuous", 1);
    trigger_mode_combo.addItem("Momentary", 2);
    trigger_mode_combo.addItem("Transient", 3);
    trigger_mode_combo.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(trigger_mode_combo);
    short_circuit_button.setVisible(false);
}

auto HeaderComponent::resized() -> void {
    using namespace UIConstants::Layout;
    using namespace UIConstants::Header;

    auto headerRow = getLocalBounds();

    save_model_button.setBounds(headerRow.removeFromRight(save_model_width));
    headerRow.removeFromRight(gap_standard);

    fuse_button.setBounds(headerRow.removeFromRight(fuse_width));
    headerRow.removeFromRight(gap_medium);

    if (short_circuit_button.isVisible()) {
        short_circuit_button.setBounds(headerRow.removeFromRight(short_circuit_width));
        headerRow.removeFromRight(gap_medium);
    }

    trigger_mode_combo.setBounds(headerRow.removeFromRight(trigger_combo_width));
    trigger_mode_label.setBounds(headerRow.removeFromRight(trigger_label_width));
    headerRow.removeFromRight(gap_medium);

    dry_wet_slider.setBounds(headerRow.removeFromRight(dry_wet_slider_width));
    dry_wet_label.setBounds(headerRow.removeFromRight(dry_wet_label_width));
    headerRow.removeFromRight(gap_standard);

    buffer_status_label.setBounds(headerRow.removeFromRight(buffer_status_width));
    buffer_label.setBounds(headerRow.removeFromRight(buffer_label_width));
    headerRow.removeFromRight(gap_medium);

    gpu_button.setBounds(headerRow.removeFromRight(gpu_button_width));
    headerRow.removeFromRight(gap_medium);

    load_button.setBounds(headerRow.removeFromLeft(load_button_width));
    headerRow.removeFromLeft(gap_medium);

    method_label.setBounds(headerRow.removeFromLeft(method_label_width));
    method_combo.setBounds(headerRow.removeFromLeft(method_combo_width));
    headerRow.removeFromLeft(gap_medium);

    status_label.setBounds(headerRow);
}
