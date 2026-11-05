#include "BridgePanel.h"
#include "UIConstants.h"

BridgePanel::BridgePanel() {
    using namespace UIConstants::Layout;

    addAndMakeVisible(bridge_enable_toggle);
    addAndMakeVisible(bridge_label);
    bridge_combo.setTextWhenNothingSelected("Select Target / Source...");
    addAndMakeVisible(bridge_combo);
    addAndMakeVisible(bridge_depth_label);
    bridge_depth_slider.setSliderStyle(juce::Slider::LinearHorizontal);
    bridge_depth_slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, slider_box_width_wide, slider_box_height_wide);
    addAndMakeVisible(bridge_depth_slider);
}

auto BridgePanel::resized() -> void {
    using namespace UIConstants::Layout;

    auto b = getLocalBounds().reduced(gap_standard);
    bridge_enable_toggle.setBounds(b.removeFromTop(row_height_standard));
    b.removeFromTop(gap_standard);
    bridge_label.setBounds(b.removeFromTop(label_height_tall));
    bridge_combo.setBounds(b.removeFromTop(row_height_standard));
    b.removeFromTop(gap_large);
    bridge_depth_label.setBounds(b.removeFromTop(label_height_tall));
    bridge_depth_slider.setBounds(b.removeFromTop(row_height_standard));
}
