#pragma once
#include <JuceHeader.h>

class HeaderComponent : public juce::Component
{
public:
    HeaderComponent()
    {
        titleLabel.setText ("NetBender", juce::dontSendNotification);
        titleLabel.setFont (juce::FontOptions (18.0f, juce::Font::bold));
        titleLabel.setJustificationType (juce::Justification::centredLeft);
        titleLabel.setColour (juce::Label::textColourId, juce::Colours::lightcyan);
        addAndMakeVisible (titleLabel);

        loadButton.setButtonText ("Load Model");
        loadButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff581c87"));
        loadButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (loadButton);

        statusLabel.setText ("No model loaded.", juce::dontSendNotification);
        statusLabel.setJustificationType (juce::Justification::centredLeft);
        statusLabel.setColour (juce::Label::textColourId, juce::Colours::darkgrey);
        addAndMakeVisible (statusLabel);

        methodLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (methodLabel);
        addAndMakeVisible (methodCombo);

        bufferLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (bufferLabel);
        bufferStatusLabel.setText ("-", juce::dontSendNotification);
        bufferStatusLabel.setJustificationType (juce::Justification::centred);
        bufferStatusLabel.setColour (juce::Label::backgroundColourId, juce::Colour::fromString ("#ff1c1926"));
        bufferStatusLabel.setColour (juce::Label::outlineColourId, juce::Colour::fromString ("#ff3d3554"));
        bufferStatusLabel.setColour (juce::Label::textColourId, juce::Colours::cyan);
        addAndMakeVisible (bufferStatusLabel);

        gpuButton.setButtonText ("CPU");
        gpuButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        gpuButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgrey);
        addAndMakeVisible (gpuButton);

        dryWetLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (dryWetLabel);

        dryWetSlider.setRange (0.0, 1.0, 0.01);
        dryWetSlider.setValue (1.0);
        dryWetSlider.setSliderStyle (juce::Slider::LinearBar);
        dryWetSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 45, 18);
        dryWetSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff7c3aed"));
        addAndMakeVisible (dryWetSlider);

        saveModelButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1e3a5f"));
        saveModelButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (saveModelButton);

        fuseButton.setButtonText ("FUSE: OK");
        fuseButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1b3a2a"));
        fuseButton.setColour (juce::TextButton::textColourOffId, juce::Colours::lightgreen);
        addAndMakeVisible (fuseButton);

        shortCircuitButton.setButtonText ("SHORT [!]");
        shortCircuitButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff8b2500"));
        shortCircuitButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (shortCircuitButton);

        triggerModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (triggerModeLabel);
        triggerModeCombo.addItem ("Continuous", 1);
        triggerModeCombo.addItem ("Momentary", 2);
        triggerModeCombo.addItem ("Transient", 3);
        triggerModeCombo.setSelectedId (1, juce::dontSendNotification);
        addAndMakeVisible (triggerModeCombo);
        shortCircuitButton.setVisible (false);
    }

    ~HeaderComponent() override = default;

    void resized() override
    {
        auto headerRow = getLocalBounds();

        saveModelButton.setBounds (headerRow.removeFromRight (125));
        headerRow.removeFromRight (10);

        fuseButton.setBounds (headerRow.removeFromRight (88));
        headerRow.removeFromRight (8);

        if (shortCircuitButton.isVisible())
        {
            shortCircuitButton.setBounds (headerRow.removeFromRight (82));
            headerRow.removeFromRight (8);
        }

        triggerModeCombo.setBounds (headerRow.removeFromRight (90));
        triggerModeLabel.setBounds (headerRow.removeFromRight (48));
        headerRow.removeFromRight (8);

        dryWetSlider.setBounds (headerRow.removeFromRight (85));
        dryWetLabel.setBounds (headerRow.removeFromRight (50));
        headerRow.removeFromRight (10);

        bufferStatusLabel.setBounds (headerRow.removeFromRight (44));
        bufferLabel.setBounds (headerRow.removeFromRight (40));
        headerRow.removeFromRight (8);

        gpuButton.setBounds (headerRow.removeFromRight (48));
        headerRow.removeFromRight (8);

        loadButton.setBounds (headerRow.removeFromLeft (90));
        headerRow.removeFromLeft (8);

        methodLabel.setBounds (headerRow.removeFromLeft (40));
        methodCombo.setBounds (headerRow.removeFromLeft (90));
        headerRow.removeFromLeft (8);

        statusLabel.setBounds (headerRow);
    }

    juce::Label titleLabel;
    juce::TextButton loadButton { "Load Model" };
    juce::Label statusLabel;

    juce::Label methodLabel { {}, "Mode:" };
    juce::ComboBox methodCombo;

    juce::Label bufferLabel { {}, "Buffer:" };
    juce::Label bufferStatusLabel;
    juce::TextButton gpuButton { "CPU" };

    juce::Label dryWetLabel { {}, "Dry/Wet:" };
    juce::Slider dryWetSlider;

    juce::TextButton saveModelButton { "Save Model" };
    juce::TextButton fuseButton { "FUSE: OK" };
    juce::TextButton shortCircuitButton { "SHORT [!]" };
    juce::Label triggerModeLabel { {}, "Trigger:" };
    juce::ComboBox triggerModeCombo;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HeaderComponent)
};
