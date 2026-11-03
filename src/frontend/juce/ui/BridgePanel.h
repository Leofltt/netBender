#pragma once
#include <JuceHeader.h>

class BridgePanel : public juce::Component
{
public:
    BridgePanel()
    {
        bridgeEnableToggle.setToggleState (false, juce::dontSendNotification);
        bridgeEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffc26a38"));
        bridgeEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffc26a38"));
        addAndMakeVisible (bridgeEnableToggle);

        bridgeLabel.setText ("Bridge Wire Source:", juce::dontSendNotification);
        bridgeLabel.setJustificationType (juce::Justification::centredLeft);
        bridgeLabel.setColour (juce::Label::textColourId, juce::Colour::fromString ("#ffc26a38")); // Warm copper
        addAndMakeVisible (bridgeLabel);

        addAndMakeVisible (bridgeCombo);

        bridgeDepthLabel.setText ("Cross-Talk Depth", juce::dontSendNotification);
        bridgeDepthLabel.setJustificationType (juce::Justification::centredLeft);
        bridgeDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (bridgeDepthLabel);

        bridgeDepthSlider.setRange (0.0, 1.0, 0.01);
        bridgeDepthSlider.setValue (0.0);
        bridgeDepthSlider.setSliderStyle (juce::Slider::LinearBar);
        bridgeDepthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        bridgeDepthSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd99b26")); // Warm golden ochre
        addAndMakeVisible (bridgeDepthSlider);
    }

    ~BridgePanel() override = default;

    void resized() override
    {
        auto area = getLocalBounds();

        bridgeEnableToggle.setBounds (area.removeFromTop (24));
        area.removeFromTop (8);

        bridgeLabel.setBounds (area.removeFromTop (16));
        bridgeCombo.setBounds (area.removeFromTop (24));
        area.removeFromTop (12);

        bridgeDepthLabel.setBounds (area.removeFromTop (16));
        bridgeDepthSlider.setBounds (area.removeFromTop (22));
    }

    juce::ToggleButton bridgeEnableToggle { "Enable Cross-Talk" };
    juce::Label bridgeLabel { {}, "Bridge Wire Source:" };
    juce::ComboBox bridgeCombo;
    juce::Label bridgeDepthLabel { {}, "Cross-Talk Depth" };
    juce::Slider bridgeDepthSlider;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BridgePanel)
};
