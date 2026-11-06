#include "LatentTerrainComponent.h"
#include "UIConstants.h"
#include <cmath>
#include <algorithm>

LatentTerrainComponent::LatentTerrainComponent() {
    setOpaque(true);
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
}

auto LatentTerrainComponent::set_coords(float x, float y) -> void {
    m_x = juce::jlimit(-1.0f, 1.0f, x);
    m_y = juce::jlimit(-1.0f, 1.0f, y);

    auto padArea = getLocalBounds().toFloat().reduced(6.0f);
    float pw = padArea.getWidth();
    float ph = padArea.getHeight();
    float cx = padArea.getX() + pw * 0.5f;
    float cy = padArea.getY() + ph * 0.5f;
    float cursorX = cx + m_x * (pw * 0.48f);
    float cursorY = cy - m_y * (ph * 0.48f);
    record_trail_point(cursorX, cursorY);

    repaint();
}

auto LatentTerrainComponent::set_enabled_state(bool is_enabled, bool has_bottleneck) -> void {
    m_isEnabled = is_enabled;
    m_hasBottleneck = has_bottleneck;
    repaint();
}

auto LatentTerrainComponent::paint(juce::Graphics& g) -> void {
    auto bounds = getLocalBounds().toFloat();

    g.setColour(UIConstants::Latent::background);
    g.fillRect(bounds);

    auto padArea = bounds.reduced(6.0f);
    float pw = padArea.getWidth();
    float ph = padArea.getHeight();

    g.setColour(UIConstants::Latent::inner_pad);
    g.fillRect(padArea);

    g.setColour(UIConstants::Latent::border);
    g.drawRect(padArea, UIConstants::Layout::border_width_thin);

    if (!m_hasBottleneck) {
        g.setColour(UIConstants::Colors::text_dimmed);
        g.setFont(juce::FontOptions(UIConstants::Typography::font_heading));
        g.drawText("Current model has no autoencode / latent bottleneck.\n(Requires paired encode + decode methods, e.g. RAVE)",
                   padArea.toNearestInt(), juce::Justification::centred, true);
        return;
    }

    float cx = padArea.getX() + pw * 0.5f;
    float cy = padArea.getY() + ph * 0.5f;

    // Render multi-pole potential / topographic latent heat map via precomputed texture blit
    {
        juce::Graphics::ScopedSaveState sss(g);
        g.reduceClipRegion(padArea.toNearestInt());

        update_background_terrain_image(static_cast<int>(pw), static_cast<int>(ph));
        if (m_backgroundTerrainImage.isValid()) {
            g.setImageResamplingQuality(juce::Graphics::ResamplingQuality::lowResamplingQuality);
            g.drawImage(m_backgroundTerrainImage, padArea, juce::RectanglePlacement::stretchToFit);
        }
    }

    {
        juce::Graphics::ScopedSaveState sss(g);
        g.reduceClipRegion(padArea.toNearestInt());

        constexpr int gridSteps = 16;
        for (int i = 0; i <= gridSteps; ++i) {
            float frac = static_cast<float>(i) / static_cast<float>(gridSteps);
            float xLine = padArea.getX() + frac * pw;
            float yLine = padArea.getY() + frac * ph;

            g.setColour(UIConstants::Latent::grid.withAlpha(0.25f));
            g.drawVerticalLine(static_cast<int>(xLine), padArea.getY(), padArea.getBottom());
            g.drawHorizontalLine(static_cast<int>(yLine), padArea.getX(), padArea.getRight());
        }
    }

    g.setColour(UIConstants::Latent::crosshair.withAlpha(0.6f));
    g.drawHorizontalLine(static_cast<int>(cy), padArea.getX(), padArea.getRight());
    g.drawVerticalLine(static_cast<int>(cx), padArea.getY(), padArea.getBottom());

    float maxRadius = std::min(pw, ph) * 0.46f;
    for (int step = 1; step <= 4; ++step) {
        float r = maxRadius * (step / 4.0f);
        g.setColour(UIConstants::Latent::rings.withAlpha(step == 4 ? 0.55f : 0.30f));
        g.drawEllipse(cx - r, cy - r, r * 2.0f, r * 2.0f, step == 4 ? 1.2f : 0.8f);

        g.setColour(juce::Colours::white.withAlpha(0.28f));
        g.setFont(juce::FontOptions(9.0f));
        g.drawText(juce::String(step * 0.25f, 2),
                   static_cast<int>(cx + 4), static_cast<int>(cy - r - 6), 30, 12, juce::Justification::centredLeft);
    }

    {
        g.setColour(UIConstants::Latent::rays.withAlpha(0.35f));
        float diagDist = maxRadius * 0.95f;
        float cos45 = 0.7071f * diagDist;
        g.drawLine(cx - cos45, cy - cos45, cx + cos45, cy + cos45, 0.8f);
        g.drawLine(cx - cos45, cy + cos45, cx + cos45, cy - cos45, 0.8f);
    }

    float cursorX = cx + m_x * (pw * 0.48f);
    float cursorY = cy - m_y * (ph * 0.48f);

    {
        float distFromOrigin = std::sqrt(m_x * m_x + m_y * m_y);
        juce::Colour rayColour = m_isEnabled ? UIConstants::Latent::active_ray : UIConstants::Latent::inactive;
        g.setColour(rayColour.withAlpha(0.40f));
        g.drawLine(cx, cy, cursorX, cursorY, 1.0f);
        if (distFromOrigin > 0.15f) {
            float midX = (cx + cursorX) * 0.5f;
            float midY = (cy + cursorY) * 0.5f;
            g.setColour(UIConstants::Colors::cyan.withAlpha(0.85f));
            g.setFont(juce::FontOptions(9.0f));
            g.drawText(juce::String::formatted("r=%.2f", distFromOrigin),
                       static_cast<int>(midX + 4), static_cast<int>(midY - 6), 40, 12, juce::Justification::centredLeft);
        }
    }

    if (m_trail.size() >= 2) {
        juce::Path trailPath;
        trailPath.startNewSubPath(m_trail[0].x, m_trail[0].y);
        for (size_t i = 1; i < m_trail.size(); ++i) {
            trailPath.lineTo(m_trail[i].x, m_trail[i].y);
        }

        juce::Colour trailColour = m_isEnabled ? UIConstants::Latent::active_trail : UIConstants::Latent::inactive;
        g.setColour(trailColour.withAlpha(0.35f));
        g.strokePath(trailPath, juce::PathStrokeType(1.2f));
        for (size_t i = 0; i < m_trail.size(); ++i) {
            float alpha = static_cast<float>(i + 1) / static_cast<float>(m_trail.size()) * 0.5f;
            g.setColour(trailColour.withAlpha(alpha));
            g.fillEllipse(m_trail[i].x - 2.0f, m_trail[i].y - 2.0f, 4.0f, 4.0f);
        }
    }

    juce::Colour glowColour = m_isEnabled ? UIConstants::Latent::active_trail : UIConstants::Latent::inactive;

    juce::ColourGradient glowGrad(
        glowColour.withAlpha(0.40f), cursorX, cursorY,
        glowColour.withAlpha(0.0f), cursorX, cursorY + 32.0f, true
    );
    g.setGradientFill(glowGrad);
    g.fillEllipse(cursorX - 32.0f, cursorY - 32.0f, 64.0f, 64.0f);

    g.setColour(glowColour.withAlpha(0.95f));
    g.drawEllipse(cursorX - 10.0f, cursorY - 10.0f, 20.0f, 20.0f, 1.2f);
    g.drawEllipse(cursorX - 3.0f, cursorY - 3.0f, 6.0f, 6.0f, 1.8f);
    g.setColour(UIConstants::Colors::text_white.withAlpha(0.85f));
    g.drawHorizontalLine(static_cast<int>(cursorY), cursorX - 16.0f, cursorX - 5.0f);
    g.drawHorizontalLine(static_cast<int>(cursorY), cursorX + 5.0f, cursorX + 16.0f);
    g.drawVerticalLine(static_cast<int>(cursorX), cursorY - 16.0f, cursorY - 5.0f);
    g.drawVerticalLine(static_cast<int>(cursorX), cursorY + 5.0f, cursorY + 16.0f);

    g.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    g.setColour(UIConstants::Colors::text_white.withAlpha(0.85f));
    float mag = std::sqrt(m_x * m_x + m_y * m_y);
    float angleRad = std::atan2(m_y, m_x);
    float angleDeg = angleRad * (180.0f / 3.14159265f);
    if (angleDeg < 0.0f) {
        angleDeg += 360.0f;
    }

    juce::String coordsText = juce::String::formatted("X: %+.3f   Y: %+.3f   |   Mag: %.3f   θ: %.1f°", m_x, m_y, mag, angleDeg);
    g.drawText(coordsText, padArea.reduced(10.0f, 6.0f).toNearestInt(), juce::Justification::bottomLeft, true);

    juce::String modeText = m_isEnabled ? "LATENT HOOK: ACTIVE" : "STANDBY (Click to Modulate)";
    g.setColour(m_isEnabled ? UIConstants::Latent::active_trail : UIConstants::Colors::text_dimmed);
    g.drawText(modeText, padArea.reduced(10.0f, 6.0f).toNearestInt(), juce::Justification::topRight, true);
}

auto LatentTerrainComponent::mouseDown(const juce::MouseEvent& e) -> void {
    update_from_mouse(e);
}

auto LatentTerrainComponent::mouseDrag(const juce::MouseEvent& e) -> void {
    update_from_mouse(e);
}

auto LatentTerrainComponent::mouseDoubleClick(const juce::MouseEvent& /*e*/) -> void {
    set_coords(0.0f, 0.0f);
    if (onCoordsChanged) {
        onCoordsChanged(0.0f, 0.0f);
    }
}

auto LatentTerrainComponent::update_background_terrain_image(int w, int h) -> void {
    if (w <= 0 || h <= 0) {
        return;
    }
    constexpr int steps = 36;
    if (!m_backgroundTerrainImage.isValid() || m_backgroundTerrainImage.getWidth() != steps || m_backgroundTerrainImage.getHeight() != steps) {
        m_backgroundTerrainImage = juce::Image(juce::Image::ARGB, steps, steps, false);
        juce::Image::BitmapData bmp(m_backgroundTerrainImage, juce::Image::BitmapData::writeOnly);

        for (int sy = 0; sy < steps; ++sy) {
            float pyNorm = 1.0f - 2.0f * ((sy + 0.5f) / static_cast<float>(steps));
            auto* line = reinterpret_cast<uint32_t*>(bmp.getLinePointer(sy));

            for (int sx = 0; sx < steps; ++sx) {
                float pxNorm = 2.0f * ((sx + 0.5f) / static_cast<float>(steps)) - 1.0f;
                float dOrigin = std::sqrt(pxNorm * pxNorm + pyNorm * pyNorm);

                float field1 = std::sin(pxNorm * 3.14159f * 1.5f) * std::cos(pyNorm * 3.14159f * 1.5f);
                float field2 = 0.5f * std::cos(dOrigin * 6.28318f);
                float field = 0.5f * (field1 + field2);
                float val = juce::jlimit(0.0f, 1.0f, 0.45f + field * 0.35f);

                juce::Colour cellCol;
                if (val < 0.45f) {
                    float t = val / 0.45f;
                    cellCol = UIConstants::Latent::terrain_low.interpolatedWith(UIConstants::Latent::terrain_mid_low, t);
                } else if (val < 0.70f) {
                    float t = (val - 0.45f) / 0.25f;
                    cellCol = UIConstants::Latent::terrain_mid_low.interpolatedWith(UIConstants::Latent::terrain_mid_high, t);
                } else {
                    float t = (val - 0.70f) / 0.30f;
                    cellCol = UIConstants::Latent::terrain_mid_high.interpolatedWith(UIConstants::Latent::terrain_high, t);
                }
                line[sx] = cellCol.withAlpha(0.70f).getARGB();
            }
        }
    }
}

auto LatentTerrainComponent::record_trail_point(float px, float py) -> void {
    m_trail.push_back({ px, py });
    if (m_trail.size() > 24) {
        m_trail.erase(m_trail.begin());
    }
}

auto LatentTerrainComponent::update_from_mouse(const juce::MouseEvent& e) -> void {
    auto padArea = getLocalBounds().toFloat().reduced(6.0f);
    float pw = padArea.getWidth();
    float ph = padArea.getHeight();

    float cx = padArea.getX() + pw * 0.5f;
    float cy = padArea.getY() + ph * 0.5f;

    float normX = (e.position.x - cx) / (pw * 0.48f);
    float normY = -(e.position.y - cy) / (ph * 0.48f);

    m_x = juce::jlimit(-1.0f, 1.0f, normX);
    m_y = juce::jlimit(-1.0f, 1.0f, normY);

    float cursorX = cx + m_x * (pw * 0.48f);
    float cursorY = cy - m_y * (ph * 0.48f);
    record_trail_point(cursorX, cursorY);

    repaint();

    if (onCoordsChanged) {
        onCoordsChanged(m_x, m_y);
    }
}
