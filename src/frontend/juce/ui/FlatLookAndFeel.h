#pragma once
#include <JuceHeader.h>

struct FlatLookAndFeel : public juce::LookAndFeel_V4 {
    FlatLookAndFeel();
    ~FlatLookAndFeel() override = default;

    // =========================================================================
    // Buttons
    // =========================================================================
    auto drawButtonBackground(juce::Graphics& g, juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) -> void override;

    auto drawButtonText(juce::Graphics& g, juce::TextButton& button,
                        bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) -> void override;

    // =========================================================================
    // ComboBox
    // =========================================================================
    auto drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                      int buttonX, int buttonY, int buttonW, int buttonH,
                      juce::ComboBox& box) -> void override;

    auto getComboBoxFont(juce::ComboBox& box) -> juce::Font override;

    auto positionComboBoxText(juce::ComboBox& box, juce::Label& label) -> void override;

    // =========================================================================
    // PopupMenu
    // =========================================================================
    auto drawPopupMenuBackground(juce::Graphics& g, int width, int height) -> void override;

    auto drawPopupMenuItem(juce::Graphics& g, const juce::Rectangle<int>& area,
                           bool isSeparator, bool isActive,
                           bool isHighlighted, bool isTicked,
                           bool hasSubMenu, const juce::String& text,
                           const juce::String& shortcutKeyText,
                           const juce::Drawable* icon, const juce::Colour* textColourToUse) -> void override;

    auto drawPopupMenuUpDownArrow(juce::Graphics& g, int width, int height, bool isScrollUpArrow) -> void override;

    // =========================================================================
    // ToggleButton / TickBox
    // =========================================================================
    auto drawTickBox(juce::Graphics& g, juce::Component& component,
                     float x, float y, float w, float h,
                     bool ticked, bool isEnabled,
                     bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) -> void override;

    // =========================================================================
    // ScrollBar
    // =========================================================================
    auto drawScrollbar(juce::Graphics& g, juce::ScrollBar& scrollbar, int x, int y, int width, int height,
                       bool isScrollbarVertical, int thumbStartPosition, int thumbSize, bool isMouseOver, bool isMouseDown) -> void override;

    // =========================================================================
    // GroupComponent
    // =========================================================================
    auto drawGroupComponentOutline(juce::Graphics& g, int width, int height,
                                   const juce::String& text, const juce::Justification& position,
                                   juce::GroupComponent& group) -> void override;
};

using FlatTabLookAndFeel = FlatLookAndFeel;
