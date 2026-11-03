#pragma once
#include <JuceHeader.h>

class LatentControlPanel : public juce::Component
{
public:
    LatentControlPanel()
    {
        latentEnableButton.setButtonText ("Enable Latent Hook");
        latentEnableButton.setToggleState (true, juce::dontSendNotification);
        latentEnableButton.setColour (juce::ToggleButton::textColourId, juce::Colour::fromString ("#ffd946ef"));
        latentEnableButton.setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffd946ef"));
        addAndMakeVisible (latentEnableButton);

        latentDepthLabel.setText ("Latent Depth", juce::dontSendNotification);
        latentDepthLabel.setJustificationType (juce::Justification::centredLeft);
        latentDepthLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (latentDepthLabel);

        latentDepthSlider.setRange (0.0, 2.0, 0.01);
        latentDepthSlider.setValue (1.0);
        latentDepthSlider.setSliderStyle (juce::Slider::LinearBar);
        latentDepthSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        latentDepthSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ffd946ef"));
        addAndMakeVisible (latentDepthSlider);

        latentModeLabel.setText ("Orbit Mode:", juce::dontSendNotification);
        latentModeLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (latentModeLabel);

        latentModeCombo.addItem ("Fourier Orbit", 1);
        latentModeCombo.addItem ("Latent Slew", 2);
        latentModeCombo.addItem ("Quantize", 3);
        latentModeCombo.setSelectedId (1, juce::dontSendNotification);
        addAndMakeVisible (latentModeCombo);

        latentSlewLabel.setText ("Slew Speed", juce::dontSendNotification);
        latentSlewLabel.setJustificationType (juce::Justification::centredLeft);
        latentSlewLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (latentSlewLabel);

        latentSlewSlider.setRange (0.001, 1.0, 0.001);
        latentSlewSlider.setValue (0.25);
        latentSlewSlider.setSliderStyle (juce::Slider::LinearBar);
        latentSlewSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 18);
        latentSlewSlider.setColour (juce::Slider::trackColourId, juce::Colour::fromString ("#ff38bdf8"));
        addAndMakeVisible (latentSlewSlider);
    }

    ~LatentControlPanel() override = default;

    void resized() override
    {
        auto area = getLocalBounds();

        latentEnableButton.setBounds (area.removeFromTop (24));
        area.removeFromTop (10);

        latentModeLabel.setBounds (area.removeFromTop (16));
        latentModeCombo.setBounds (area.removeFromTop (26));
        area.removeFromTop (12);

        int modeId = latentModeCombo.getSelectedId(); // 1 = Fourier Orbit, 2 = Latent Slew, 3 = Quantize

        if (modeId == 2) // Latent Slew mode
        {
            latentSlewLabel.setBounds (area.removeFromTop (16));
            latentSlewSlider.setBounds (area.removeFromTop (24));
            latentDepthLabel.setBounds ({});
            latentDepthSlider.setBounds ({});
        }
        else // Fourier Orbit or Quantize mode
        {
            latentDepthLabel.setBounds (area.removeFromTop (16));
            latentDepthSlider.setBounds (area.removeFromTop (24));
            latentSlewLabel.setBounds ({});
            latentSlewSlider.setBounds ({});
        }
    }

    juce::ToggleButton latentEnableButton { "Enable Latent Hook" };
    juce::Label latentDepthLabel { {}, "Latent Depth" };
    juce::Slider latentDepthSlider;
    juce::Label latentModeLabel { {}, "Orbit Mode:" };
    juce::ComboBox latentModeCombo;
    juce::Label latentSlewLabel { {}, "Slew Speed" };
    juce::Slider latentSlewSlider;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LatentControlPanel)
};
