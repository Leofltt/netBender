#pragma once
#include <JuceHeader.h>

class HarmonicsPanel : public juce::Component
{
public:
    HarmonicsPanel()
    {
        harmonicEnableToggle.setToggleState (false, juce::dontSendNotification);
        harmonicEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#fffbbf24"));
        harmonicEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#fffbbf24"));
        addAndMakeVisible (harmonicEnableToggle);

        harmonicFreqLabel.setText ("Frequency", juce::dontSendNotification);
        harmonicFreqLabel.setJustificationType (juce::Justification::centredLeft);
        harmonicFreqLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (harmonicFreqLabel);

        harmonicFreqSlider.setRange (0.5, 32.0, 0.1);
        harmonicFreqSlider.setValue (2.0);
        harmonicFreqSlider.setSliderStyle (juce::Slider::LinearBar);
        harmonicFreqSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        harmonicFreqSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706")); // Warm amber
        addAndMakeVisible (harmonicFreqSlider);

        harmonicPartialsLabel.setText ("Partials", juce::dontSendNotification);
        harmonicPartialsLabel.setJustificationType (juce::Justification::centredLeft);
        harmonicPartialsLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (harmonicPartialsLabel);

        harmonicPartialsSlider.setRange (1.0, 8.0, 1.0);
        harmonicPartialsSlider.setValue (1.0);
        harmonicPartialsSlider.setSliderStyle (juce::Slider::LinearBar);
        harmonicPartialsSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        harmonicPartialsSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706"));
        addAndMakeVisible (harmonicPartialsSlider);

        harmonicMorphLabel.setText ("Morph Phase", juce::dontSendNotification);
        harmonicMorphLabel.setJustificationType (juce::Justification::centredLeft);
        harmonicMorphLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (harmonicMorphLabel);

        harmonicMorphSlider.setRange (0.0, 6.2831853, 0.01);
        harmonicMorphSlider.setValue (0.0);
        harmonicMorphSlider.setSliderStyle (juce::Slider::LinearBar);
        harmonicMorphSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        harmonicMorphSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706"));
        addAndMakeVisible (harmonicMorphSlider);

        harmonicDepthLabel.setText ("Wave Gain", juce::dontSendNotification);
        harmonicDepthLabel.setJustificationType (juce::Justification::centredLeft);
        harmonicDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (harmonicDepthLabel);

        harmonicDepthSlider.setRange (0.0, 2.0, 0.01);
        harmonicDepthSlider.setValue (0.5);
        harmonicDepthSlider.setSliderStyle (juce::Slider::LinearBar);
        harmonicDepthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        harmonicDepthSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd97706"));
        addAndMakeVisible (harmonicDepthSlider);

        harmonicModeLabel.setText ("Combine Mode:", juce::dontSendNotification);
        harmonicModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (harmonicModeLabel);

        harmonicModeCombo.addItem ("Add (+)", 1);
        harmonicModeCombo.addItem ("Ring (*)", 2);
        harmonicModeCombo.addItem ("Replace", 3);
        harmonicModeCombo.setSelectedId (1, juce::dontSendNotification);
        addAndMakeVisible (harmonicModeCombo);

        harmonicApplyButton.setButtonText ("Stamp Wave");
        harmonicApplyButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a3712"));
        harmonicApplyButton.setColour (juce::TextButton::textColourOffId, juce::Colour::fromString ("#fffde047"));
        addAndMakeVisible (harmonicApplyButton);
    }

    ~HarmonicsPanel() override = default;

    void resized() override
    {
        auto area = getLocalBounds();

        harmonicEnableToggle.setBounds (area.removeFromTop (24));
        area.removeFromTop (4);

        harmonicFreqLabel.setBounds (area.removeFromTop (16));
        harmonicFreqSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        harmonicPartialsLabel.setBounds (area.removeFromTop (16));
        harmonicPartialsSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        harmonicMorphLabel.setBounds (area.removeFromTop (16));
        harmonicMorphSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        harmonicDepthLabel.setBounds (area.removeFromTop (16));
        harmonicDepthSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        harmonicModeLabel.setBounds (area.removeFromTop (16));
        harmonicModeCombo.setBounds (area.removeFromTop (24));
        area.removeFromTop (14);

        harmonicApplyButton.setBounds (area.removeFromTop (28));
    }

    juce::ToggleButton harmonicEnableToggle { "Enable Harmonics" };
    juce::Label harmonicFreqLabel { {}, "Frequency" };
    juce::Slider harmonicFreqSlider;
    juce::Label harmonicPartialsLabel { {}, "Partials" };
    juce::Slider harmonicPartialsSlider;
    juce::Label harmonicMorphLabel { {}, "Morph Phase" };
    juce::Slider harmonicMorphSlider;
    juce::Label harmonicDepthLabel { {}, "Wave Gain" };
    juce::Slider harmonicDepthSlider;
    juce::Label harmonicModeLabel { {}, "Combine Mode:" };
    juce::ComboBox harmonicModeCombo;
    juce::TextButton harmonicApplyButton { "Stamp Wave" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HarmonicsPanel)
};
