#pragma once
#include <JuceHeader.h>
#include "MutatePanel.h"
#include "HarmonicsPanel.h"
#include "BridgePanel.h"
#include "LatentControlPanel.h"

class SidePanelComponent : public juce::Component {
public:
    enum class SideTab { Mutate, Harmonics, Bridge };
    enum class ViewMode { Weights, Latent };

    SidePanelComponent();
    ~SidePanelComponent() override = default;

    auto set_side_tab(SideTab new_tab) -> void;
    auto set_view_mode(ViewMode new_mode) -> void;

    auto get_current_side_tab() const noexcept -> SideTab { return m_current_side_tab; }
    auto get_current_view_mode() const noexcept -> ViewMode { return m_current_view_mode; }

    // Compatibility aliases
    auto setSideTab(SideTab new_tab) -> void { set_side_tab(new_tab); }
    auto setViewMode(ViewMode new_mode) -> void { set_view_mode(new_mode); }
    auto getCurrentSideTab() const noexcept -> SideTab { return get_current_side_tab(); }
    auto getCurrentViewMode() const noexcept -> ViewMode { return get_current_view_mode(); }

    auto resized() -> void override;

    juce::TextButton tab_mutate_button{"Mutate"};
    juce::TextButton tab_harmonics_button{"Harmonics"};
    juce::TextButton tab_bridge_button{"Bridge"};

    MutatePanel mutate_panel;
    HarmonicsPanel harmonics_panel;
    BridgePanel bridge_panel;
    LatentControlPanel latent_panel;

    // Compatibility references
    juce::TextButton& tabMutateButton{tab_mutate_button};
    juce::TextButton& tabHarmonicsButton{tab_harmonics_button};
    juce::TextButton& tabBridgeButton{tab_bridge_button};
    MutatePanel& mutatePanel{mutate_panel};
    HarmonicsPanel& harmonicsPanel{harmonics_panel};
    BridgePanel& bridgePanel{bridge_panel};
    LatentControlPanel& latentPanel{latent_panel};

private:
    SideTab m_current_side_tab{SideTab::Mutate};
    ViewMode m_current_view_mode{ViewMode::Weights};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SidePanelComponent)
};
