#pragma once
#include <JuceHeader.h>

class MutatePanel : public juce::Component
{
public:
    MutatePanel()
    {
        mutateEnableToggle.setToggleState (true, juce::dontSendNotification);
        mutateEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffc084fc"));
        mutateEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffc084fc"));
        addAndMakeVisible (mutateEnableToggle);

        scaleLabel.setText ("Scale", juce::dontSendNotification);
        scaleLabel.setJustificationType (juce::Justification::centredLeft);
        scaleLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (scaleLabel);

        scaleSlider.setRange (0.0, 5.0, 0.01);
        scaleSlider.setValue (1.0);
        scaleSlider.setSliderStyle (juce::Slider::LinearBar);
        scaleSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        scaleSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff7c3aed")); // Vibrant violet
        addAndMakeVisible (scaleSlider);

        offsetLabel.setText ("Offset", juce::dontSendNotification);
        offsetLabel.setJustificationType (juce::Justification::centredLeft);
        offsetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (offsetLabel);

        offsetSlider.setRange (-2.0, 2.0, 0.001);
        offsetSlider.setValue (0.0);
        offsetSlider.setSliderStyle (juce::Slider::LinearBar);
        offsetSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        offsetSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff06b6d4")); // Bright cyan
        addAndMakeVisible (offsetSlider);

        driftEnableToggle.setToggleState (false, juce::dontSendNotification);
        driftEnableToggle.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffff9933"));
        driftEnableToggle.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffff9933"));
        addAndMakeVisible (driftEnableToggle);

        heatLabel.setText ("Heat (Jitter)", juce::dontSendNotification);
        heatLabel.setJustificationType (juce::Justification::centredLeft);
        heatLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (heatLabel);

        heatSlider.setRange (0.0, 1.0, 0.001);
        heatSlider.setValue (0.0);
        heatSlider.setSliderStyle (juce::Slider::LinearBar);
        heatSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        heatSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffea580c")); // Flame orange
        addAndMakeVisible (heatSlider);

        memoryLabel.setText ("Memory (Drag)", juce::dontSendNotification);
        memoryLabel.setJustificationType (juce::Justification::centredLeft);
        memoryLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (memoryLabel);

        memorySlider.setRange (0.0, 1.0, 0.001);
        memorySlider.setValue (0.8);
        memorySlider.setSliderStyle (juce::Slider::LinearBar);
        memorySlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        memorySlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff0284c7")); // Deep sky blue
        addAndMakeVisible (memorySlider);

        driftModeLabel.setText ("Drift Algorithm:", juce::dontSendNotification);
        driftModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (driftModeLabel);

        driftModeCombo.addItem ("Thermal OU", 1);
        driftModeCombo.addItem ("Random Walk", 2);
        driftModeCombo.addItem ("Glitch Noise", 3);
        driftModeCombo.setSelectedId (1, juce::dontSendNotification);
        addAndMakeVisible (driftModeCombo);

        freezeButton.setButtonText ("Freeze Drift");
        freezeButton.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
        freezeButton.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffff9933"));
        addAndMakeVisible (freezeButton);
    }

    ~MutatePanel() override = default;

    void resized() override
    {
        auto area = getLocalBounds();

        mutateEnableToggle.setBounds (area.removeFromTop (24));
        area.removeFromTop (4);

        scaleLabel.setBounds (area.removeFromTop (16));
        scaleSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        offsetLabel.setBounds (area.removeFromTop (16));
        offsetSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (14);

        driftEnableToggle.setBounds (area.removeFromTop (24));
        area.removeFromTop (4);

        heatLabel.setBounds (area.removeFromTop (16));
        heatSlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        memoryLabel.setBounds (area.removeFromTop (16));
        memorySlider.setBounds (area.removeFromTop (22));
        area.removeFromTop (8);

        driftModeLabel.setBounds (area.removeFromTop (16));
        driftModeCombo.setBounds (area.removeFromTop (24));
        area.removeFromTop (10);

        freezeButton.setBounds (area.removeFromTop (24));
    }

    juce::ToggleButton mutateEnableToggle { "Enable Transform" };
    juce::Label scaleLabel { {}, "Scale" };
    juce::Slider scaleSlider;
    juce::Label offsetLabel { {}, "Offset" };
    juce::Slider offsetSlider;

    juce::ToggleButton driftEnableToggle { "Enable Thermal Drift" };
    juce::Label heatLabel { {}, "Heat (Jitter)" };
    juce::Slider heatSlider;
    juce::Label memoryLabel { {}, "Memory (Drag)" };
    juce::Slider memorySlider;
    juce::Label driftModeLabel { {}, "Drift Algorithm:" };
    juce::ComboBox driftModeCombo;
    juce::ToggleButton freezeButton { "Freeze Drift" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MutatePanel)
};
