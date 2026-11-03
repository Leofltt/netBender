#pragma once
#include <JuceHeader.h>
#include "MutatePanel.h"
#include "HarmonicsPanel.h"
#include "BridgePanel.h"
#include "LatentControlPanel.h"

class SidePanelComponent : public juce::Component
{
public:
    enum class SideTab { Mutate, Harmonics, Bridge };
    enum class ViewMode { Weights, Latent };

    SidePanelComponent()
    {
        tabMutateButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        tabMutateButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (tabMutateButton);

        tabHarmonicsButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        tabHarmonicsButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
        addAndMakeVisible (tabHarmonicsButton);

        tabBridgeButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        tabBridgeButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
        addAndMakeVisible (tabBridgeButton);

        addAndMakeVisible (mutatePanel);
        addChildComponent (harmonicsPanel);
        addChildComponent (bridgePanel);
        addChildComponent (latentPanel);

        setSideTab (SideTab::Mutate);
    }

    ~SidePanelComponent() override = default;

    void setSideTab (SideTab newTab)
    {
        currentSideTab = newTab;

        bool isWeights = (currentViewMode == ViewMode::Weights);
        bool isMutate = isWeights && (currentSideTab == SideTab::Mutate);
        bool isHarmonics = isWeights && (currentSideTab == SideTab::Harmonics);
        bool isBridge = isWeights && (currentSideTab == SideTab::Bridge);

        juce::Colour activeBg = juce::Colour::fromString ("#ff4a1d72"); // Vivid violet
        juce::Colour inactiveBg = juce::Colour::fromString ("#ff1c1926"); // Dark charcoal
        juce::Colour activeText = juce::Colours::white;
        juce::Colour inactiveText = juce::Colours::darkgrey;

        tabMutateButton.setColour (juce::TextButton::buttonColourId, isMutate ? activeBg : inactiveBg);
        tabMutateButton.setColour (juce::TextButton::textColourOffId, isMutate ? activeText : inactiveText);

        tabHarmonicsButton.setColour (juce::TextButton::buttonColourId, isHarmonics ? activeBg : inactiveBg);
        tabHarmonicsButton.setColour (juce::TextButton::textColourOffId, isHarmonics ? activeText : inactiveText);

        tabBridgeButton.setColour (juce::TextButton::buttonColourId, isBridge ? activeBg : inactiveBg);
        tabBridgeButton.setColour (juce::TextButton::textColourOffId, isBridge ? activeText : inactiveText);

        mutatePanel.setVisible (isMutate);
        harmonicsPanel.setVisible (isHarmonics);
        bridgePanel.setVisible (isBridge);
    }

    void setViewMode (ViewMode newMode)
    {
        currentViewMode = newMode;
        bool isWeights = (currentViewMode == ViewMode::Weights);
        bool isLatent = (currentViewMode == ViewMode::Latent);

        tabMutateButton.setVisible (isWeights);
        tabHarmonicsButton.setVisible (isWeights);
        tabBridgeButton.setVisible (isWeights);

        latentPanel.setVisible (isLatent);

        if (isWeights)
            setSideTab (currentSideTab);
        else
        {
            mutatePanel.setVisible (false);
            harmonicsPanel.setVisible (false);
            bridgePanel.setVisible (false);
        }

        resized();
    }

    SideTab getCurrentSideTab() const noexcept { return currentSideTab; }
    ViewMode getCurrentViewMode() const noexcept { return currentViewMode; }

    void resized() override
    {
        auto area = getLocalBounds();

        if (currentViewMode == ViewMode::Weights)
        {
            auto tabHeader = area.removeFromTop (26);
            int tabW = (area.getWidth() - 6) / 3;
            tabMutateButton.setBounds (tabHeader.removeFromLeft (tabW));
            tabHeader.removeFromLeft (3);
            tabHarmonicsButton.setBounds (tabHeader.removeFromLeft (tabW));
            tabHeader.removeFromLeft (3);
            tabBridgeButton.setBounds (tabHeader);

            area.removeFromTop (10);

            mutatePanel.setBounds (area);
            harmonicsPanel.setBounds (area);
            bridgePanel.setBounds (area);
        }
        else
        {
            latentPanel.setBounds (area);
        }
    }

    juce::TextButton tabMutateButton { "Mutate" };
    juce::TextButton tabHarmonicsButton { "Harmonics" };
    juce::TextButton tabBridgeButton { "Bridge" };

    MutatePanel mutatePanel;
    HarmonicsPanel harmonicsPanel;
    BridgePanel bridgePanel;
    LatentControlPanel latentPanel;

private:
    SideTab currentSideTab { SideTab::Mutate };
    ViewMode currentViewMode { ViewMode::Weights };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SidePanelComponent)
};
