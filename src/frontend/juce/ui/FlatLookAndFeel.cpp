#include "FlatLookAndFeel.h"
#include "UIConstants.h"

FlatLookAndFeel::FlatLookAndFeel() {
    using namespace UIConstants::Colors;

    // Default dark flat palette for widgets
    setColour(juce::ComboBox::backgroundColourId, surface_card);
    setColour(juce::ComboBox::outlineColourId, border_medium);
    setColour(juce::ComboBox::textColourId, text_secondary);
    setColour(juce::ComboBox::arrowColourId, lavender);
    setColour(juce::ComboBox::focusedOutlineColourId, violet);

    setColour(juce::PopupMenu::backgroundColourId, popup_background);
    setColour(juce::PopupMenu::textColourId, text_secondary);
    setColour(juce::PopupMenu::highlightedBackgroundColourId, deep_violet);
    setColour(juce::PopupMenu::highlightedTextColourId, text_white);

    setColour(juce::TextButton::buttonColourId, button_secondary_bg);
    setColour(juce::TextButton::textColourOffId, text_white);
    setColour(juce::TextButton::textColourOnId, text_white);

    setColour(juce::ToggleButton::textColourId, text_light);
    setColour(juce::ToggleButton::tickColourId, lilac);
    setColour(juce::ToggleButton::tickDisabledColourId, border_medium);

    setColour(juce::ScrollBar::ColourIds::thumbColourId, border_medium);

    setColour(juce::GroupComponent::outlineColourId, divider_line);
    setColour(juce::GroupComponent::textColourId, lilac);
}

auto FlatLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                           const juce::Colour& backgroundColour,
                                           bool shouldDrawButtonAsHighlighted,
                                           bool shouldDrawButtonAsDown) -> void {
    auto bounds = button.getLocalBounds().toFloat();
    auto bg = backgroundColour;
    if (!button.isEnabled()) {
        bg = bg.withMultipliedAlpha(0.45f);
    }

    g.setColour(bg);
    g.fillRect(bounds);

    if (shouldDrawButtonAsDown || button.getToggleState()) {
        g.setColour(juce::Colours::white.withAlpha(0.14f));
        g.fillRect(bounds);
    } else if (shouldDrawButtonAsHighlighted) {
        g.setColour(juce::Colours::white.withAlpha(0.07f));
        g.fillRect(bounds);
    }

    auto outlineCol = shouldDrawButtonAsHighlighted ? bg.brighter(0.35f).withAlpha(0.75f)
                                                    : bg.brighter(0.18f).withAlpha(0.45f);
    g.setColour(outlineCol);
    g.drawRect(bounds, 1.0f);
}

auto FlatLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button,
                                     bool /*shouldDrawButtonAsHighlighted*/, bool /*shouldDrawButtonAsDown*/) -> void {
    auto font = getTextButtonFont(button, button.getHeight());
    g.setFont(font);
    g.setColour(button.findColour(button.getToggleState() ? juce::TextButton::textColourOnId
                                                          : juce::TextButton::textColourOffId)
                    .withMultipliedAlpha(button.isEnabled() ? 1.0f : 0.5f));

    auto yIndent = juce::jmin(4, button.proportionOfHeight(0.3f));
    g.drawFittedText(button.getButtonText(),
                     4, yIndent, button.getWidth() - 8, button.getHeight() - yIndent * 2,
                     juce::Justification::centred, 2);
}

auto FlatLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                                  int /*buttonX*/, int /*buttonY*/, int /*buttonW*/, int /*buttonH*/,
                                  juce::ComboBox& box) -> void {
    auto bounds = juce::Rectangle<float>(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));

    auto bg = box.findColour(juce::ComboBox::backgroundColourId);
    if (!box.isEnabled()) {
        bg = bg.withMultipliedAlpha(0.45f);
    }

    g.setColour(bg);
    g.fillRect(bounds);

    if (isButtonDown) {
        g.setColour(juce::Colours::white.withAlpha(0.10f));
        g.fillRect(bounds);
    } else if (box.isMouseOver()) {
        g.setColour(juce::Colours::white.withAlpha(0.05f));
        g.fillRect(bounds);
    }

    auto outlineCol = box.hasKeyboardFocus(false)
                        ? box.findColour(juce::ComboBox::focusedOutlineColourId)
                        : (box.isMouseOver() ? box.findColour(juce::ComboBox::outlineColourId).brighter(0.3f)
                                             : box.findColour(juce::ComboBox::outlineColourId));
    g.setColour(outlineCol);
    g.drawRect(bounds, 1.0f);

    float arrowZoneWidth = 22.0f;
    juce::Rectangle<float> arrowZone(static_cast<float>(width) - arrowZoneWidth, 0.0f, arrowZoneWidth, static_cast<float>(height));

    juce::Path arrow;
    float arrowW = 8.0f;
    float arrowH = 4.0f;
    float cx = arrowZone.getCentreX();
    float cy = arrowZone.getCentreY();
    arrow.startNewSubPath(cx - arrowW * 0.5f, cy - arrowH * 0.5f);
    arrow.lineTo(cx, cy + arrowH * 0.5f);
    arrow.lineTo(cx + arrowW * 0.5f, cy - arrowH * 0.5f);

    auto arrowCol = box.findColour(juce::ComboBox::arrowColourId)
                       .withMultipliedAlpha(box.isEnabled() ? (box.isMouseOver() ? 1.0f : 0.8f) : 0.25f);
    g.setColour(arrowCol);
    g.strokePath(arrow, juce::PathStrokeType(1.5f, juce::PathStrokeType::mitered, juce::PathStrokeType::square));
}

auto FlatLookAndFeel::getComboBoxFont(juce::ComboBox& box) -> juce::Font {
    return juce::Font(juce::FontOptions(juce::jlimit(11.0f, 13.0f, static_cast<float>(box.getHeight()) * 0.52f)));
}

auto FlatLookAndFeel::positionComboBoxText(juce::ComboBox& box, juce::Label& label) -> void {
    label.setBounds(5, 1, box.getWidth() - 24, box.getHeight() - 2);
    label.setFont(getComboBoxFont(box));
    label.setJustificationType(juce::Justification::centredLeft);
    label.setColour(juce::Label::textColourId,
                     box.findColour(juce::ComboBox::textColourId)
                        .withMultipliedAlpha(box.isEnabled() ? 1.0f : 0.5f));
}

auto FlatLookAndFeel::drawPopupMenuBackground(juce::Graphics& g, int width, int height) -> void {
    g.fillAll(findColour(juce::PopupMenu::backgroundColourId));
    g.setColour(findColour(juce::ComboBox::outlineColourId));
    g.drawRect(0, 0, width, height, 1);
}

auto FlatLookAndFeel::drawPopupMenuItem(juce::Graphics& g, const juce::Rectangle<int>& area,
                                       const bool isSeparator, const bool isActive,
                                       const bool isHighlighted, const bool isTicked,
                                       const bool hasSubMenu, const juce::String& text,
                                       const juce::String& /*shortcutKeyText*/,
                                       const juce::Drawable* /*icon*/, const juce::Colour* const textColourToUse) -> void {
    if (isSeparator) {
        auto r = area.reduced(4, 0);
        r.removeFromTop(roundToInt((static_cast<float>(r.getHeight()) * 0.5f) - 0.5f));
        g.setColour(findColour(juce::ComboBox::outlineColourId).withAlpha(0.5f));
        g.fillRect(r.removeFromTop(1));
        return;
    }

    auto r = area;
    if (isHighlighted && isActive) {
        g.setColour(findColour(juce::PopupMenu::highlightedBackgroundColourId));
        g.fillRect(r);
        g.setColour(findColour(juce::PopupMenu::highlightedTextColourId));
    } else {
        auto baseTextCol = (textColourToUse != nullptr) ? *textColourToUse
                                                        : findColour(juce::PopupMenu::textColourId);
        g.setColour(baseTextCol.withMultipliedAlpha(isActive ? 1.0f : 0.45f));
    }

    auto font = getPopupMenuFont();
    auto maxFontHeight = static_cast<float>(r.getHeight()) / 1.3f;
    if (font.getHeight() > maxFontHeight) {
        font.setHeight(maxFontHeight);
    }
    g.setFont(font);

    auto textBounds = r.reduced(8, 0);
    if (isTicked) {
        auto tickRect = textBounds.removeFromLeft(12).toFloat();
        g.fillRect(tickRect.withSizeKeepingCentre(5.0f, 5.0f));
    }

    g.drawFittedText(text, textBounds, juce::Justification::centredLeft, 1);

    if (hasSubMenu) {
        auto arrowH = 0.5f * font.getAscent();
        auto x = static_cast<float>(r.removeFromRight(static_cast<int>(arrowH) + 6).getX());
        auto halfH = static_cast<float>(r.getCentreY());

        juce::Path path;
        path.startNewSubPath(x, halfH - arrowH * 0.5f);
        path.lineTo(x + arrowH * 0.6f, halfH);
        path.lineTo(x, halfH + arrowH * 0.5f);
        g.strokePath(path, juce::PathStrokeType(1.5f));
    }
}

auto FlatLookAndFeel::drawPopupMenuUpDownArrow(juce::Graphics& g, int width, int height, bool isScrollUpArrow) -> void {
    g.setColour(findColour(juce::PopupMenu::backgroundColourId));
    g.fillRect(0, 0, width, height);

    float cx = static_cast<float>(width) * 0.5f;
    float cy = static_cast<float>(height) * 0.5f;
    float arrowW = 8.0f;
    float arrowH = 4.0f;

    juce::Path arrow;
    if (isScrollUpArrow) {
        arrow.startNewSubPath(cx - arrowW * 0.5f, cy + arrowH * 0.5f);
        arrow.lineTo(cx, cy - arrowH * 0.5f);
        arrow.lineTo(cx + arrowW * 0.5f, cy + arrowH * 0.5f);
    } else {
        arrow.startNewSubPath(cx - arrowW * 0.5f, cy - arrowH * 0.5f);
        arrow.lineTo(cx, cy + arrowH * 0.5f);
        arrow.lineTo(cx + arrowW * 0.5f, cy - arrowH * 0.5f);
    }

    g.setColour(findColour(juce::ComboBox::arrowColourId));
    g.strokePath(arrow, juce::PathStrokeType(1.5f, juce::PathStrokeType::mitered, juce::PathStrokeType::square));
}

auto FlatLookAndFeel::drawTickBox(juce::Graphics& g, juce::Component& component,
                                 float x, float y, float w, float h,
                                 const bool ticked,
                                 const bool isEnabled,
                                 const bool shouldDrawButtonAsHighlighted,
                                 const bool /*shouldDrawButtonAsDown*/) -> void {
    juce::Rectangle<float> tickBounds(x, y, w, h);

    g.setColour(UIConstants::Colors::surface_card);
    g.fillRect(tickBounds);

    auto tickCol = component.findColour(juce::ToggleButton::tickColourId);
    if (!isEnabled) {
        tickCol = tickCol.withMultipliedAlpha(0.4f);
    } else if (shouldDrawButtonAsHighlighted) {
        tickCol = tickCol.brighter(0.25f);
    }

    auto outlineCol = ticked ? tickCol : UIConstants::Colors::border_medium;
    if (shouldDrawButtonAsHighlighted && !ticked) {
        outlineCol = outlineCol.brighter(0.3f);
    }

    g.setColour(outlineCol);
    g.drawRect(tickBounds, UIConstants::Layout::border_width_thin);

    if (ticked) {
        g.setColour(tickCol);
        auto inner = tickBounds.reduced(UIConstants::Layout::tick_box_inner_inset);
        g.fillRect(inner);
    }
}

auto FlatLookAndFeel::drawScrollbar(juce::Graphics& g, juce::ScrollBar& scrollbar, int x, int y, int width, int height,
                                    bool isScrollbarVertical, int thumbStartPosition, int thumbSize, bool isMouseOver, bool /*isMouseDown*/) -> void {
    juce::Rectangle<int> thumbBounds;
    if (isScrollbarVertical) {
        thumbBounds = {x + 1, thumbStartPosition, width - 2, thumbSize};
    } else {
        thumbBounds = {thumbStartPosition, y + 1, thumbSize, height - 2};
    }

    auto c = scrollbar.findColour(juce::ScrollBar::ColourIds::thumbColourId);
    g.setColour(isMouseOver ? c.brighter(0.3f) : c);
    g.fillRect(thumbBounds);
}

auto FlatLookAndFeel::drawGroupComponentOutline(juce::Graphics& g, int width, int height,
                                                const juce::String& text, const juce::Justification& position,
                                                juce::GroupComponent& group) -> void {
    constexpr float textH = 14.0f;
    constexpr float indent = 3.0f;
    constexpr float textEdgeGap = 4.0f;

    juce::Font f(juce::FontOptions(textH, juce::Font::bold));
    auto x = indent;
    auto y = f.getAscent() - 2.0f;
    auto w = juce::jmax(0.0f, static_cast<float>(width) - x * 2.0f);
    auto h = juce::jmax(0.0f, static_cast<float>(height) - y - indent);

    auto textW = text.isEmpty() ? 0.0f
                                : juce::jlimit(0.0f, juce::jmax(0.0f, w - textEdgeGap * 2.0f),
                                               juce::GlyphArrangement::getStringWidth(f, text) + textEdgeGap * 2.0f);
    auto textX = 10.0f;
    if (position.testFlags(juce::Justification::horizontallyCentred)) {
        textX = (w - textW) * 0.5f;
    } else if (position.testFlags(juce::Justification::right)) {
        textX = w - textW - textEdgeGap;
    }

    juce::Path p;
    p.startNewSubPath(x + textX + textW, y);
    p.lineTo(x + w, y);
    p.lineTo(x + w, y + h);
    p.lineTo(x, y + h);
    p.lineTo(x, y);
    p.lineTo(x + textX, y);

    g.setColour(group.findColour(juce::GroupComponent::outlineColourId));
    g.strokePath(p, juce::PathStrokeType(1.0f));

    if (text.isNotEmpty()) {
        g.setColour(group.findColour(juce::GroupComponent::textColourId));
        g.setFont(f);
        g.drawText(text,
                   juce::roundToInt(x + textX + textEdgeGap), 0,
                   juce::roundToInt(textW - textEdgeGap * 2.0f), juce::roundToInt(textH),
                   juce::Justification::centred, true);
    }
}
