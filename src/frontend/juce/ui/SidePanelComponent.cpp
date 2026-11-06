#include "SidePanelComponent.h"
#include "UIConstants.h"

SidePanelComponent::SidePanelComponent() {
    using namespace UIConstants::Colors;

    tab_mutate_button.setColour(juce::TextButton::buttonColourId, tab_active_bg);
    tab_mutate_button.setColour(juce::TextButton::textColourOffId, text_white);
    addAndMakeVisible(tab_mutate_button);

    tab_harmonics_button.setColour(juce::TextButton::buttonColourId, tab_inactive_bg);
    tab_harmonics_button.setColour(juce::TextButton::textColourOffId, text_dimmed);
    addAndMakeVisible(tab_harmonics_button);

    tab_bridge_button.setColour(juce::TextButton::buttonColourId, tab_inactive_bg);
    tab_bridge_button.setColour(juce::TextButton::textColourOffId, text_dimmed);
    addAndMakeVisible(tab_bridge_button);

    addAndMakeVisible(mutate_panel);
    addChildComponent(harmonics_panel);
    addChildComponent(bridge_panel);
    addChildComponent(latent_panel);

    set_side_tab(SideTab::Mutate);
}

auto SidePanelComponent::set_side_tab(SideTab new_tab) -> void {
    using namespace UIConstants::Colors;

    m_current_side_tab = new_tab;

    bool is_weights = (m_current_view_mode == ViewMode::Weights);
    bool is_mutate = is_weights && (m_current_side_tab == SideTab::Mutate);
    bool is_harmonics = is_weights && (m_current_side_tab == SideTab::Harmonics);
    bool is_bridge = is_weights && (m_current_side_tab == SideTab::Bridge);

    const auto& active_bg = tab_active_bg;
    const auto& inactive_bg = tab_inactive_bg;
    const auto& active_text = text_white;
    const auto& inactive_text = text_dimmed;

    tab_mutate_button.setColour(juce::TextButton::buttonColourId, is_mutate ? active_bg : inactive_bg);
    tab_mutate_button.setColour(juce::TextButton::textColourOffId, is_mutate ? active_text : inactive_text);

    tab_harmonics_button.setColour(juce::TextButton::buttonColourId, is_harmonics ? active_bg : inactive_bg);
    tab_harmonics_button.setColour(juce::TextButton::textColourOffId, is_harmonics ? active_text : inactive_text);

    tab_bridge_button.setColour(juce::TextButton::buttonColourId, is_bridge ? active_bg : inactive_bg);
    tab_bridge_button.setColour(juce::TextButton::textColourOffId, is_bridge ? active_text : inactive_text);

    mutate_panel.setVisible(is_mutate);
    harmonics_panel.setVisible(is_harmonics);
    bridge_panel.setVisible(is_bridge);
}

auto SidePanelComponent::set_view_mode(ViewMode new_mode) -> void {
    m_current_view_mode = new_mode;
    bool is_weights = (m_current_view_mode == ViewMode::Weights);
    bool is_latent = (m_current_view_mode == ViewMode::Latent);

    tab_mutate_button.setVisible(is_weights);
    tab_harmonics_button.setVisible(is_weights);
    tab_bridge_button.setVisible(is_weights);

    latent_panel.setVisible(is_latent);

    if (is_weights) {
        set_side_tab(m_current_side_tab);
    } else {
        mutate_panel.setVisible(false);
        harmonics_panel.setVisible(false);
        bridge_panel.setVisible(false);
    }

    resized();
}

auto SidePanelComponent::resized() -> void {
    using namespace UIConstants::Layout;

    auto area = getLocalBounds();

    if (m_current_view_mode == ViewMode::Weights) {
        auto tab_header = area.removeFromTop(tab_height);
        int tab_w = (area.getWidth() - gap_tiny * 2) / 3;
        tab_mutate_button.setBounds(tab_header.removeFromLeft(tab_w));
        tab_header.removeFromLeft(gap_tiny);
        tab_harmonics_button.setBounds(tab_header.removeFromLeft(tab_w));
        tab_header.removeFromLeft(gap_tiny);
        tab_bridge_button.setBounds(tab_header);

        area.removeFromTop(gap_standard);

        mutate_panel.setBounds(area);
        harmonics_panel.setBounds(area);
        bridge_panel.setBounds(area);
    } else {
        latent_panel.setBounds(area);
    }
}
