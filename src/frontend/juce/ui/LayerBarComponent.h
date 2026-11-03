#pragma once
#include <JuceHeader.h>

class LayerBarComponent : public juce::Component
{
public:
    LayerBarComponent()
    {
        viewWeightsButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        viewWeightsButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (viewWeightsButton);

        viewLatentButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        viewLatentButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
        addAndMakeVisible (viewLatentButton);

        displayCurveButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff4a1d72"));
        displayCurveButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (displayCurveButton);

        displayMatrixButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        displayMatrixButton.setColour (juce::TextButton::textColourOffId, juce::Colours::darkgrey);
        addAndMakeVisible (displayMatrixButton);

        categoryLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (categoryLabel);

        categoryCombo.addItem ("All Traces", 1);
        categoryCombo.addItem ("Norm / Dynamics", 2);
        categoryCombo.addItem ("Conv Kernels", 3);
        categoryCombo.addItem ("Attention", 4);
        categoryCombo.addItem ("Linear / Dense", 5);
        categoryCombo.addItem ("Activation (Snake)", 6);
        categoryCombo.addItem ("Recurrent (LSTM)", 7);
        categoryCombo.addItem ("Biases", 8);
        categoryCombo.setSelectedId (1, juce::dontSendNotification);
        addAndMakeVisible (categoryCombo);

        layerLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (layerLabel);
        addAndMakeVisible (layerCombo);

        resetLayerButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff801e2a"));
        resetLayerButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (resetLayerButton);

        resetAllButton.setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff5a1018"));
        resetAllButton.setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        addAndMakeVisible (resetAllButton);
    }

    ~LayerBarComponent() override = default;

    void resized() override
    {
        auto layerRow = getLocalBounds();

        viewWeightsButton.setBounds (layerRow.removeFromLeft (68));
        layerRow.removeFromLeft (4);
        viewLatentButton.setBounds (layerRow.removeFromLeft (64));
        layerRow.removeFromLeft (8);

        displayCurveButton.setBounds (layerRow.removeFromLeft (44));
        layerRow.removeFromLeft (3);
        displayMatrixButton.setBounds (layerRow.removeFromLeft (44));
        layerRow.removeFromLeft (10);

        categoryLabel.setBounds (layerRow.removeFromLeft (40));
        categoryCombo.setBounds (layerRow.removeFromLeft (110));
        layerRow.removeFromLeft (8);

        layerLabel.setBounds (layerRow.removeFromLeft (40));

        resetAllButton.setBounds (layerRow.removeFromRight (125));
        layerRow.removeFromRight (6);
        resetLayerButton.setBounds (layerRow.removeFromRight (105));
        layerRow.removeFromRight (8);

        layerCombo.setBounds (layerRow);
    }

    juce::TextButton viewWeightsButton { "weights" };
    juce::TextButton viewLatentButton { "latent" };

    juce::TextButton displayCurveButton { "1D" };
    juce::TextButton displayMatrixButton { "2D" };

    juce::Label categoryLabel { {}, "Trace:" };
    juce::ComboBox categoryCombo;

    juce::Label layerLabel { {}, "Layer:" };
    juce::ComboBox layerCombo;

    juce::TextButton resetLayerButton { "Reset Layer" };
    juce::TextButton resetAllButton { "Reset All Layers" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LayerBarComponent)
};
