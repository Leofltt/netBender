#pragma once
#include <JuceHeader.h>

struct FlatLookAndFeel : public juce::LookAndFeel_V4
{
    FlatLookAndFeel()
    {
        // Default dark flat palette for widgets
        setColour (juce::ComboBox::backgroundColourId, juce::Colour::fromString ("#ff1c1926"));
        setColour (juce::ComboBox::outlineColourId, juce::Colour::fromString ("#ff3d3554"));
        setColour (juce::ComboBox::textColourId, juce::Colour::fromString ("#ffe2e8f0"));
        setColour (juce::ComboBox::arrowColourId, juce::Colour::fromString ("#ffa78bfa"));
        setColour (juce::ComboBox::focusedOutlineColourId, juce::Colour::fromString ("#ff7c3aed"));

        setColour (juce::PopupMenu::backgroundColourId, juce::Colour::fromString ("#ff181622"));
        setColour (juce::PopupMenu::textColourId, juce::Colour::fromString ("#ffe2e8f0"));
        setColour (juce::PopupMenu::highlightedBackgroundColourId, juce::Colour::fromString ("#ff4a1d72"));
        setColour (juce::PopupMenu::highlightedTextColourId, juce::Colours::white);

        setColour (juce::TextButton::buttonColourId, juce::Colour::fromString ("#ff1c1926"));
        setColour (juce::TextButton::textColourOffId, juce::Colours::white);
        setColour (juce::TextButton::textColourOnId, juce::Colours::white);

        setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
        setColour (juce::ToggleButton::tickColourId, juce::Colour::fromString ("#ffc084fc"));
        setColour (juce::ToggleButton::tickDisabledColourId, juce::Colour::fromString ("#ff3d3554"));

        setColour (juce::ScrollBar::ColourIds::thumbColourId, juce::Colour::fromString ("#ff3d3554"));

        setColour (juce::GroupComponent::outlineColourId, juce::Colour::fromString ("#ff382f4c"));
        setColour (juce::GroupComponent::textColourId, juce::Colour::fromString ("#ffc084fc"));
    }

    // =========================================================================
    // Buttons (Flat rectangular style with subtle hover/down highlight)
    // =========================================================================
    void drawButtonBackground (juce::Graphics& g, juce::Button& button,
                                const juce::Colour& backgroundColour,
                                bool shouldDrawButtonAsHighlighted,
                                bool shouldDrawButtonAsDown) override
    {
        auto bounds = button.getLocalBounds().toFloat();
        auto bg = backgroundColour;
        if (! button.isEnabled())
            bg = bg.withMultipliedAlpha (0.45f);

        g.setColour (bg);
        g.fillRect (bounds);

        if (shouldDrawButtonAsDown || button.getToggleState())
        {
            g.setColour (juce::Colours::white.withAlpha (0.14f));
            g.fillRect (bounds);
        }
        else if (shouldDrawButtonAsHighlighted)
        {
            g.setColour (juce::Colours::white.withAlpha (0.07f));
            g.fillRect (bounds);
        }

        auto outlineCol = shouldDrawButtonAsHighlighted ? bg.brighter (0.35f).withAlpha (0.75f)
                                                        : bg.brighter (0.18f).withAlpha (0.45f);
        g.setColour (outlineCol);
        g.drawRect (bounds, 1.0f);
    }

    void drawButtonText (juce::Graphics& g, juce::TextButton& button,
                         bool /*shouldDrawButtonAsHighlighted*/, bool /*shouldDrawButtonAsDown*/) override
    {
        auto font = getTextButtonFont (button, button.getHeight());
        g.setFont (font);
        g.setColour (button.findColour (button.getToggleState() ? juce::TextButton::textColourOnId
                                                                : juce::TextButton::textColourOffId)
                           .withMultipliedAlpha (button.isEnabled() ? 1.0f : 0.5f));

        auto yIndent = juce::jmin (4, button.proportionOfHeight (0.3f));
        g.drawFittedText (button.getButtonText(),
                          4, yIndent, button.getWidth() - 8, button.getHeight() - yIndent * 2,
                          juce::Justification::centred, 2);
    }

    // =========================================================================
    // ComboBox (Flat rectangular style with sharp 1px border and flat chevron)
    // =========================================================================
    void drawComboBox (juce::Graphics& g, int width, int height, bool isButtonDown,
                       int /*buttonX*/, int /*buttonY*/, int /*buttonW*/, int /*buttonH*/,
                       juce::ComboBox& box) override
    {
        auto bounds = juce::Rectangle<float> (0.0f, 0.0f, (float) width, (float) height);

        auto bg = box.findColour (juce::ComboBox::backgroundColourId);
        if (! box.isEnabled())
            bg = bg.withMultipliedAlpha (0.45f);

        g.setColour (bg);
        g.fillRect (bounds);

        if (isButtonDown)
        {
            g.setColour (juce::Colours::white.withAlpha (0.10f));
            g.fillRect (bounds);
        }
        else if (box.isMouseOver())
        {
            g.setColour (juce::Colours::white.withAlpha (0.05f));
            g.fillRect (bounds);
        }

        auto outlineCol = box.hasKeyboardFocus (false)
                            ? box.findColour (juce::ComboBox::focusedOutlineColourId)
                            : (box.isMouseOver() ? box.findColour (juce::ComboBox::outlineColourId).brighter (0.3f)
                                                 : box.findColour (juce::ComboBox::outlineColourId));
        g.setColour (outlineCol);
        g.drawRect (bounds, 1.0f);

        // Flat chevron down arrow
        float arrowZoneWidth = 22.0f;
        juce::Rectangle<float> arrowZone ((float) width - arrowZoneWidth, 0.0f, arrowZoneWidth, (float) height);

        juce::Path arrow;
        float arrowW = 8.0f;
        float arrowH = 4.0f;
        float cx = arrowZone.getCentreX();
        float cy = arrowZone.getCentreY();
        arrow.startNewSubPath (cx - arrowW * 0.5f, cy - arrowH * 0.5f);
        arrow.lineTo (cx, cy + arrowH * 0.5f);
        arrow.lineTo (cx + arrowW * 0.5f, cy - arrowH * 0.5f);

        auto arrowCol = box.findColour (juce::ComboBox::arrowColourId)
                           .withMultipliedAlpha (box.isEnabled() ? (box.isMouseOver() ? 1.0f : 0.8f) : 0.25f);
        g.setColour (arrowCol);
        g.strokePath (arrow, juce::PathStrokeType (1.5f, juce::PathStrokeType::mitered, juce::PathStrokeType::square));
    }

    juce::Font getComboBoxFont (juce::ComboBox& box) override
    {
        return juce::Font (juce::FontOptions (juce::jlimit (11.0f, 13.0f, (float) box.getHeight() * 0.52f)));
    }

    void positionComboBoxText (juce::ComboBox& box, juce::Label& label) override
    {
        label.setBounds (5, 1, box.getWidth() - 24, box.getHeight() - 2);
        label.setFont (getComboBoxFont (box));
        label.setJustificationType (juce::Justification::centredLeft);
        label.setColour (juce::Label::textColourId,
                         box.findColour (juce::ComboBox::textColourId)
                            .withMultipliedAlpha (box.isEnabled() ? 1.0f : 0.5f));
    }

    // =========================================================================
    // PopupMenu (Flat dark background with crisp outline and flat item hover)
    // =========================================================================
    void drawPopupMenuBackground (juce::Graphics& g, int width, int height) override
    {
        g.fillAll (findColour (juce::PopupMenu::backgroundColourId));
        g.setColour (findColour (juce::ComboBox::outlineColourId));
        g.drawRect (0, 0, width, height, 1);
    }

    void drawPopupMenuItem (juce::Graphics& g, const juce::Rectangle<int>& area,
                            const bool isSeparator, const bool isActive,
                            const bool isHighlighted, const bool isTicked,
                            const bool hasSubMenu, const juce::String& text,
                            const juce::String& shortcutKeyText,
                            const juce::Drawable* icon, const juce::Colour* const textColourToUse) override
    {
        if (isSeparator)
        {
            auto r = area.reduced (4, 0);
            r.removeFromTop (roundToInt (((float) r.getHeight() * 0.5f) - 0.5f));
            g.setColour (findColour (juce::ComboBox::outlineColourId).withAlpha (0.5f));
            g.fillRect (r.removeFromTop (1));
            return;
        }

        auto r = area;
        if (isHighlighted && isActive)
        {
            g.setColour (findColour (juce::PopupMenu::highlightedBackgroundColourId));
            g.fillRect (r);
            g.setColour (findColour (juce::PopupMenu::highlightedTextColourId));
        }
        else
        {
            auto baseTextCol = (textColourToUse != nullptr) ? *textColourToUse
                                                            : findColour (juce::PopupMenu::textColourId);
            g.setColour (baseTextCol.withMultipliedAlpha (isActive ? 1.0f : 0.45f));
        }

        auto font = getPopupMenuFont();
        auto maxFontHeight = (float) r.getHeight() / 1.3f;
        if (font.getHeight() > maxFontHeight)
            font.setHeight (maxFontHeight);
        g.setFont (font);

        auto textBounds = r.reduced (8, 0);
        if (isTicked)
        {
            auto tickRect = textBounds.removeFromLeft (12).toFloat();
            g.fillRect (tickRect.withSizeKeepingCentre (5.0f, 5.0f));
        }

        g.drawFittedText (text, textBounds, juce::Justification::centredLeft, 1);

        if (hasSubMenu)
        {
            auto arrowH = 0.5f * font.getAscent();
            auto x = static_cast<float> (r.removeFromRight ((int) arrowH + 6).getX());
            auto halfH = static_cast<float> (r.getCentreY());

            juce::Path path;
            path.startNewSubPath (x, halfH - arrowH * 0.5f);
            path.lineTo (x + arrowH * 0.6f, halfH);
            path.lineTo (x, halfH + arrowH * 0.5f);
            g.strokePath (path, juce::PathStrokeType (1.5f));
        }
    }

    void drawPopupMenuUpDownArrow (juce::Graphics& g, int width, int height, bool isScrollUpArrow) override
    {
        g.setColour (findColour (juce::PopupMenu::backgroundColourId));
        g.fillRect (0, 0, width, height);

        float cx = (float) width * 0.5f;
        float cy = (float) height * 0.5f;
        float arrowW = 8.0f;
        float arrowH = 4.0f;

        juce::Path arrow;
        if (isScrollUpArrow)
        {
            arrow.startNewSubPath (cx - arrowW * 0.5f, cy + arrowH * 0.5f);
            arrow.lineTo (cx, cy - arrowH * 0.5f);
            arrow.lineTo (cx + arrowW * 0.5f, cy + arrowH * 0.5f);
        }
        else
        {
            arrow.startNewSubPath (cx - arrowW * 0.5f, cy - arrowH * 0.5f);
            arrow.lineTo (cx, cy + arrowH * 0.5f);
            arrow.lineTo (cx + arrowW * 0.5f, cy - arrowH * 0.5f);
        }

        g.setColour (findColour (juce::ComboBox::arrowColourId));
        g.strokePath (arrow, juce::PathStrokeType (1.5f, juce::PathStrokeType::mitered, juce::PathStrokeType::square));
    }

    // =========================================================================
    // ToggleButton / TickBox (Flat crisp box with square tick indicator)
    // =========================================================================
    void drawTickBox (juce::Graphics& g, juce::Component& component,
                      float x, float y, float w, float h,
                      const bool ticked,
                      const bool isEnabled,
                      const bool shouldDrawButtonAsHighlighted,
                      const bool /*shouldDrawButtonAsDown*/) override
    {
        juce::Rectangle<float> tickBounds (x, y, w, h);

        g.setColour (juce::Colour::fromString ("#ff1c1926"));
        g.fillRect (tickBounds);

        auto tickCol = component.findColour (juce::ToggleButton::tickColourId);
        if (! isEnabled)
            tickCol = tickCol.withMultipliedAlpha (0.4f);
        else if (shouldDrawButtonAsHighlighted)
            tickCol = tickCol.brighter (0.25f);

        auto outlineCol = ticked ? tickCol : juce::Colour::fromString ("#ff3d3554");
        if (shouldDrawButtonAsHighlighted && ! ticked)
            outlineCol = outlineCol.brighter (0.3f);

        g.setColour (outlineCol);
        g.drawRect (tickBounds, 1.0f);

        if (ticked)
        {
            g.setColour (tickCol);
            auto inner = tickBounds.reduced (3.0f);
            g.fillRect (inner);
        }
    }

    // =========================================================================
    // ScrollBar (Crisp flat rectangular thumb)
    // =========================================================================
    void drawScrollbar (juce::Graphics& g, juce::ScrollBar& scrollbar, int x, int y, int width, int height,
                        bool isScrollbarVertical, int thumbStartPosition, int thumbSize, bool isMouseOver, bool /*isMouseDown*/) override
    {
        juce::Rectangle<int> thumbBounds;
        if (isScrollbarVertical)
            thumbBounds = { x + 1, thumbStartPosition, width - 2, thumbSize };
        else
            thumbBounds = { thumbStartPosition, y + 1, thumbSize, height - 2 };

        auto c = scrollbar.findColour (juce::ScrollBar::ColourIds::thumbColourId);
        g.setColour (isMouseOver ? c.brighter (0.3f) : c);
        g.fillRect (thumbBounds);
    }

    // =========================================================================
    // GroupComponent (Flat rectangular border)
    // =========================================================================
    void drawGroupComponentOutline (juce::Graphics& g, int width, int height,
                                    const juce::String& text, const juce::Justification& position,
                                    juce::GroupComponent& group) override
    {
        const float textH = 14.0f;
        const float indent = 3.0f;
        const float textEdgeGap = 4.0f;

        juce::Font f (juce::FontOptions (textH, juce::Font::bold));
        auto x = indent;
        auto y = f.getAscent() - 2.0f;
        auto w = juce::jmax (0.0f, (float) width - x * 2.0f);
        auto h = juce::jmax (0.0f, (float) height - y - indent);

        auto textW = text.isEmpty() ? 0.0f
                                    : juce::jlimit (0.0f, juce::jmax (0.0f, w - textEdgeGap * 2.0f),
                                                    juce::GlyphArrangement::getStringWidth (f, text) + textEdgeGap * 2.0f);
        auto textX = 10.0f;
        if (position.testFlags (juce::Justification::horizontallyCentred))
            textX = (w - textW) * 0.5f;
        else if (position.testFlags (juce::Justification::right))
            textX = w - textW - textEdgeGap;

        juce::Path p;
        p.startNewSubPath (x + textX + textW, y);
        p.lineTo (x + w, y);
        p.lineTo (x + w, y + h);
        p.lineTo (x, y + h);
        p.lineTo (x, y);
        p.lineTo (x + textX, y);

        g.setColour (group.findColour (juce::GroupComponent::outlineColourId));
        g.strokePath (p, juce::PathStrokeType (1.0f));

        if (text.isNotEmpty())
        {
            g.setColour (group.findColour (juce::GroupComponent::textColourId));
            g.setFont (f);
            g.drawText (text,
                        juce::roundToInt (x + textX + textEdgeGap), 0,
                        juce::roundToInt (textW - textEdgeGap * 2.0f), juce::roundToInt (textH),
                        juce::Justification::centred, true);
        }
    }
};

using FlatTabLookAndFeel = FlatLookAndFeel;
