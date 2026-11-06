#include "WeightBendingComponent.h"
#include "UIConstants.h"
#include <cmath>
#include <algorithm>

WeightBendingComponent::WeightBendingComponent() {
    setOpaque(true);
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
}

auto WeightBendingComponent::is_currently_drawing() const -> bool {
    return m_dragMode == DragMode::Drawing;
}

auto WeightBendingComponent::set_render_mode(RenderMode mode) -> void {
    if (m_renderMode != mode) {
        m_renderMode = mode;
        repaint();
    }
}

auto WeightBendingComponent::set_layer_shape(const std::vector<int64_t>& shape) -> void {
    m_layerShape = shape;
    calculate_matrix_dimensions();
    m_matrixImageDirty = true;
    repaint();
}

auto WeightBendingComponent::set_weights(const std::vector<float>& original, const std::vector<float>& current) -> void {
    m_originalWeights = original;
    m_currentWeights = current;
    calculate_matrix_dimensions();
    recalculate_stats();
    m_matrixImageDirty = true;
    reset_view();
    repaint();
}

auto WeightBendingComponent::update_current_weights(const std::vector<float>& current) -> void {
    m_currentWeights = current;
    calculate_matrix_dimensions();
    recalculate_stats();
    m_matrixImageDirty = true;
    repaint();
}

auto WeightBendingComponent::set_target_bent_weights(const std::vector<float>& target) -> void {
    m_targetBentWeights = target;
    repaint();
}

auto WeightBendingComponent::set_bridge_weights(const std::vector<float>& source_weights, const std::vector<float>& mix_weights, float depth, bool is_continuous) -> void {
    m_bridgeSourceWeights = source_weights;
    m_bridgeMixWeights = mix_weights;
    m_bridgeDepth = depth;
    m_isContinuousMode = is_continuous;
    repaint();
}

auto WeightBendingComponent::set_harmonic_ghost_weights(const std::vector<float>& ghost_weights, bool show_ghost) -> void {
    m_harmonicGhostWeights = ghost_weights;
    m_showHarmonicGhost = show_ghost;
    repaint();
}

auto WeightBendingComponent::set_show_bridge_visuals(bool show) -> void {
    if (m_showBridgeVisuals != show) {
        m_showBridgeVisuals = show;
        repaint();
    }
}

auto WeightBendingComponent::set_show_harmonics(bool show) -> void {
    if (m_showHarmonics != show) {
        m_showHarmonics = show;
        repaint();
    }
}

auto WeightBendingComponent::reset_view() -> void {
    m_viewStartX = 0.0f;
    m_viewSpanX = 1.0f;
    m_zoomY = 1.0f;
    m_panY = 0.0f;
    repaint();
}

auto WeightBendingComponent::get_plot_area() const -> juce::Rectangle<float> {
    auto bounds = getLocalBounds().toFloat();
    constexpr float barSize = UIConstants::Canvas::scroll_bar_size;
    return juce::Rectangle<float>(bounds.getX(), bounds.getY(),
                                  std::max(10.0f, bounds.getWidth() - barSize),
                                  std::max(10.0f, bounds.getHeight() - barSize));
}

auto WeightBendingComponent::get_h_scroll_bar_area() const -> juce::Rectangle<float> {
    auto bounds = getLocalBounds().toFloat();
    constexpr float barSize = UIConstants::Canvas::scroll_bar_size;
    return juce::Rectangle<float>(bounds.getX(), bounds.getBottom() - barSize,
                                  std::max(10.0f, bounds.getWidth() - barSize), barSize);
}

auto WeightBendingComponent::get_v_scroll_bar_area() const -> juce::Rectangle<float> {
    auto bounds = getLocalBounds().toFloat();
    constexpr float barSize = UIConstants::Canvas::scroll_bar_size;
    return juce::Rectangle<float>(bounds.getRight() - barSize, bounds.getY(),
                                  barSize, std::max(10.0f, bounds.getHeight() - barSize));
}

auto WeightBendingComponent::get_corner_button_area() const -> juce::Rectangle<float> {
    auto bounds = getLocalBounds().toFloat();
    constexpr float barSize = UIConstants::Canvas::scroll_bar_size;
    return juce::Rectangle<float>(bounds.getRight() - barSize, bounds.getBottom() - barSize, barSize, barSize);
}

auto WeightBendingComponent::paint(juce::Graphics& g) -> void {
    auto bounds = getLocalBounds().toFloat();

    g.setColour(UIConstants::Canvas::background);
    g.fillRect(bounds);

    auto plotArea = get_plot_area();

    if (m_currentWeights.empty()) {
        g.setColour(UIConstants::Colors::text_dimmed);
        g.setFont(juce::FontOptions(UIConstants::Typography::font_heading));
        g.drawText("No layer selected or layer has 0 parameters.",
                   plotArea.toNearestInt(), juce::Justification::centred, true);
        draw_edge_scroll_bars(g);
        return;
    }

    g.setColour(UIConstants::Canvas::inner_canvas);
    g.fillRect(plotArea);

    if (m_renderMode == RenderMode::Matrix) {
        paint_matrix_view(g, plotArea);
    } else {
        paint_curve_view(g, plotArea);
    }

    g.setColour(UIConstants::Canvas::border);
    g.drawRect(plotArea, UIConstants::Layout::border_width_thin);

    if (m_renderMode == RenderMode::Curve) {
        draw_edge_scroll_bars(g);
    }
}

auto WeightBendingComponent::paint_curve_view(juce::Graphics& g, juce::Rectangle<float> plotArea) -> void {
    float pw = plotArea.getWidth();
    float ph = plotArea.getHeight();

    juce::Graphics::ScopedSaveState sss(g);
    g.reduceClipRegion(plotArea.toNearestInt());

    float zeroY = value_to_y(0.0f, ph);
    g.setColour(UIConstants::Canvas::grid_minor);

    for (float v : { -2.0f, -1.0f, -0.5f, 0.5f, 1.0f, 2.0f }) {
        float y = value_to_y(v, ph);
        if (y >= 0 && y <= ph) {
            g.drawHorizontalLine(static_cast<int>(y), 0.0f, pw);
        }
    }

    g.setColour(UIConstants::Canvas::grid_major);
    g.drawHorizontalLine(static_cast<int>(zeroY), 0.0f, pw);

    int totalWeights = static_cast<int>(m_currentWeights.size());
    float idxStart = m_viewStartX * (totalWeights - 1);
    float idxEnd = (m_viewStartX + m_viewSpanX) * (totalWeights - 1);

    g.setColour(UIConstants::Canvas::center_axis);
    for (int step = 1; step <= 4; ++step) {
        float fraction = step / 5.0f;
        float x = fraction * pw;
        g.drawVerticalLine(static_cast<int>(x), 0.0f, ph);
    }

    // 1. Gradient fill under current live weights
    juce::Path bentPath;
    build_path_for_weights(bentPath, m_currentWeights, pw, ph);

    if (!bentPath.isEmpty()) {
        juce::Path fillPath = bentPath;
        fillPath.lineTo(pw, zeroY);
        fillPath.lineTo(0.0f, zeroY);
        fillPath.closeSubPath();

        juce::ColourGradient grad(
            UIConstants::Colors::lilac.withAlpha(0.18f), 0, 0,
            UIConstants::Colors::dark_violet.withAlpha(0.03f), 0, ph, false
        );
        g.setGradientFill(grad);
        g.fillPath(fillPath);
    }

    // 2. Original baseline curve
    if (!m_originalWeights.empty()) {
        juce::Path origPath;
        build_path_for_weights(origPath, m_originalWeights, pw, ph);
        g.setColour(UIConstants::Canvas::wave_baseline.withAlpha(0.55f));
        g.strokePath(origPath, juce::PathStrokeType(0.75f));
    }

    // 3. Bridged source layer curve
    if (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeSourceWeights.empty()) {
        juce::Path srcPath;
        build_path_for_weights(srcPath, m_bridgeSourceWeights, pw, ph);
        g.setColour(UIConstants::Canvas::wave_bent.withAlpha(0.45f));
        g.strokePath(srcPath, juce::PathStrokeType(0.80f));
    }

    // 4. Cross-Talk composite blend curve
    if (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeMixWeights.empty()) {
        juce::Path mixPath;
        build_path_for_weights(mixPath, m_bridgeMixWeights, pw, ph);
        g.setColour(UIConstants::Canvas::wave_drift_live.withAlpha(m_isContinuousMode ? 0.65f : 0.50f));
        g.strokePath(mixPath, juce::PathStrokeType(0.80f));
    }

    // 5. Target bent curve
    if (!m_targetBentWeights.empty()) {
        juce::Path targetPath;
        build_path_for_weights(targetPath, m_targetBentWeights, pw, ph);
        g.setColour(UIConstants::Canvas::wave_harmonics.withAlpha(0.65f));
        g.strokePath(targetPath, juce::PathStrokeType(0.80f));
    }

    // 6. Harmonic synthesizer ghost preview
    if (m_showHarmonics && m_showHarmonicGhost && !m_harmonicGhostWeights.empty()) {
        juce::Path ghostPath;
        build_path_for_weights(ghostPath, m_harmonicGhostWeights, pw, ph);
        g.setColour(UIConstants::Canvas::wave_bridge_wire.withAlpha(0.65f));
        const float dashLengths[2] = { 5.0f, 3.0f };
        juce::Path dashedGhostStroke;
        juce::PathStrokeType(0.80f).createDashedStroke(dashedGhostStroke, ghostPath, dashLengths, 2);
        g.fillPath(dashedGhostStroke);
    }

    // 7. Live weights stroke
    if (!bentPath.isEmpty()) {
        g.setColour(UIConstants::Canvas::wave_target_bent.withAlpha(0.75f));
        g.strokePath(bentPath, juce::PathStrokeType(0.95f));
    }

    // Legend
    {
        constexpr int legendY = 8;
        g.setFont(juce::FontOptions(10.0f));

        bool showBridge = (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeSourceWeights.empty());
        bool showTarget = (!m_targetBentWeights.empty());
        bool showHarmonic = (m_showHarmonics && m_showHarmonicGhost && !m_harmonicGhostWeights.empty());

        int totalItems = 2 + (showBridge ? 2 : 0) + (showTarget ? 1 : 0) + (showHarmonic ? 1 : 0);
        constexpr int itemW = 78;
        int legendW = totalItems * itemW;
        int legendX = static_cast<int>(pw) - legendW - 8;
        int curX = legendX;

        // Baseline
        g.setColour(UIConstants::Canvas::wave_baseline.withAlpha(0.85f));
        g.fillRect(curX, legendY + 3, 10, 3);
        g.drawText("Baseline W0", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
        curX += itemW;

        if (showBridge) {
            g.setColour(UIConstants::Canvas::wave_bent);
            g.fillRect(curX, legendY + 3, 10, 3);
            g.drawText("Bridge Src", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
            curX += itemW;

            g.setColour(UIConstants::Canvas::wave_drift_live);
            g.fillRect(curX, legendY + 3, 10, 3);
            g.drawText(m_isContinuousMode ? "Cross-Talk" : "Cross-Talk (·)", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
            curX += itemW;
        }

        if (showTarget) {
            g.setColour(UIConstants::Canvas::wave_harmonics);
            g.fillRect(curX, legendY + 3, 10, 3);
            g.drawText("Target Bent", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
            curX += itemW;
        }

        if (showHarmonic) {
            g.setColour(UIConstants::Canvas::wave_bridge_wire);
            g.fillRect(curX, legendY + 3, 10, 3);
            g.drawText("Harmonic (·)", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
            curX += itemW;
        }

        // Live Model
        g.setColour(UIConstants::Canvas::wave_target_bent);
        g.fillRect(curX, legendY + 3, 10, 3);
        g.drawText("Live Model", curX + 13, legendY - 2, 64, 14, juce::Justification::centredLeft);
    }

    // Stats
    g.setColour(UIConstants::Colors::text_white.withAlpha(0.65f));
    g.setFont(juce::FontOptions(10.5f));

    int startDisplayIdx = static_cast<int>(std::round(idxStart));
    int endDisplayIdx = static_cast<int>(std::round(idxEnd));
    juce::String stats = juce::String::formatted(
        "Idx: [%d..%d] / %d  |  Zoom: %.1fx (X) %.1fx (Y)  |  Mean: %.4f",
        startDisplayIdx, endDisplayIdx, totalWeights,
        1.0f / m_viewSpanX, m_zoomY, m_meanVal
    );
    g.drawText(stats, plotArea.reduced(8, 4).toNearestInt(), juce::Justification::bottomRight, true);

    g.setColour(UIConstants::Colors::text_dimmed);
    juce::String hint = "Draw: Left Drag | Pan: 2-finger scroll / Right Drag | Zoom: Pinch / Cmd+Scroll | Double-Click: Reset";
    g.drawText(hint, plotArea.reduced(8, 4).toNearestInt(), juce::Justification::topLeft, true);
}

auto WeightBendingComponent::get_heatmap_colour(float val) const -> juce::Colour {
    float maxAbs = std::max(0.001f, m_displayRange);
    float norm = juce::jlimit(-1.0f, 1.0f, val / maxAbs);
    if (norm < 0.0f) {
        float t = -norm;
        return UIConstants::Canvas::heatmap_zero.interpolatedWith(UIConstants::Canvas::heatmap_negative, t);
    }
    float t = norm;
    return UIConstants::Canvas::heatmap_zero.interpolatedWith(UIConstants::Canvas::heatmap_positive, t);
}

auto WeightBendingComponent::paint_matrix_view(juce::Graphics& g, juce::Rectangle<float> plotArea) -> void {
    float pw = plotArea.getWidth();
    float ph = plotArea.getHeight();

    juce::Graphics::ScopedSaveState sss(g);
    g.reduceClipRegion(plotArea.toNearestInt());

    int rows = m_matrixRows;
    int cols = m_matrixCols;
    if (rows <= 0 || cols <= 0 || m_currentWeights.empty()) {
        return;
    }

    constexpr float marginTop = 26.0f;
    constexpr float marginBottom = 24.0f;
    constexpr float marginLeft = 40.0f;
    constexpr float marginRight = 16.0f;

    float gridW = std::max(10.0f, pw - marginLeft - marginRight);
    float gridH = std::max(10.0f, ph - marginTop - marginBottom);

    float cellW = gridW / static_cast<float>(cols);
    float cellH = gridH / static_cast<float>(rows);
    size_t totalWeights = m_currentWeights.size();

    ensure_palette_lut();
    if (m_matrixImageDirty || !m_matrixImage.isValid() || m_matrixImage.getWidth() != cols || m_matrixImage.getHeight() != rows) {
        if (!m_matrixImage.isValid() || m_matrixImage.getWidth() != cols || m_matrixImage.getHeight() != rows) {
            m_matrixImage = juce::Image(juce::Image::ARGB, cols, rows, false);
        }

        juce::Image::BitmapData bmp(m_matrixImage, juce::Image::BitmapData::writeOnly);
        float maxAbs = std::max(0.001f, m_displayRange);

        for (int r = 0; r < rows; ++r) {
            auto* linePixels = reinterpret_cast<uint32_t*>(bmp.getLinePointer(r));
            for (int c = 0; c < cols; ++c) {
                size_t idx = static_cast<size_t>(r * cols + c);
                float val = (idx < totalWeights) ? m_currentWeights[idx] : 0.0f;
                float norm = juce::jlimit(-1.0f, 1.0f, val / maxAbs);
                int lutIdx = juce::jlimit(0, 511, static_cast<int>((norm + 1.0f) * 255.5f));
                linePixels[c] = m_paletteLut[static_cast<size_t>(lutIdx)];
            }
        }
        m_matrixImageDirty = false;
    }

    auto gridRect = juce::Rectangle<float>(plotArea.getX() + marginLeft, plotArea.getY() + marginTop, gridW, gridH);
    g.setImageResamplingQuality(juce::Graphics::ResamplingQuality::lowResamplingQuality);
    g.drawImage(m_matrixImage, gridRect, juce::RectanglePlacement::stretchToFit);

    if (cellW >= 12.0f && cellH >= 12.0f) {
        g.setColour(UIConstants::Canvas::background.withAlpha(0.45f));
        for (int r = 1; r < rows; ++r) {
            float y = gridRect.getY() + r * cellH;
            g.drawHorizontalLine(static_cast<int>(y), gridRect.getX(), gridRect.getRight());
        }
        for (int c = 1; c < cols; ++c) {
            float x = gridRect.getX() + c * cellW;
            g.drawVerticalLine(static_cast<int>(x), gridRect.getY(), gridRect.getBottom());
        }
    }

    if (m_hoveredRow >= 0 && m_hoveredRow < rows && m_hoveredCol >= 0 && m_hoveredCol < cols) {
        auto cellRect = juce::Rectangle<float>(
            gridRect.getX() + m_hoveredCol * cellW,
            gridRect.getY() + m_hoveredRow * cellH,
            cellW, cellH
        );
        g.setColour(UIConstants::Colors::text_white.withAlpha(0.9f));
        g.drawRect(cellRect, UIConstants::Layout::border_width_thick);
    }

    if (m_showBridgeVisuals && m_bridgeDepth > 0.001f && !m_bridgeSourceWeights.empty()) {
        g.setColour(UIConstants::Canvas::wave_drift_live.withAlpha(0.7f));
        float bridgeBarX = plotArea.getX() + marginLeft - 8.0f;
        g.fillRect(bridgeBarX, plotArea.getY() + marginTop, 4.0f, gridH);
    }

    g.setColour(UIConstants::Colors::text_white.withAlpha(0.4f));
    g.setFont(juce::FontOptions(10.0f));

    int rowStep = std::max(1, rows / 8);
    for (int r = 0; r < rows; r += rowStep) {
        float y = plotArea.getY() + marginTop + r * cellH;
        g.drawText(juce::String(r), static_cast<int>(plotArea.getX() + 2), static_cast<int>(y), static_cast<int>(marginLeft - 6), static_cast<int>(cellH), juce::Justification::centredRight);
    }

    int colStep = std::max(1, cols / 8);
    for (int c = 0; c < cols; c += colStep) {
        float x = plotArea.getX() + marginLeft + c * cellW;
        g.drawText(juce::String(c), static_cast<int>(x), static_cast<int>(plotArea.getY() + marginTop - 16.0f), static_cast<int>(cellW), 14, juce::Justification::centred);
    }

    juce::String shapeStr = "Shape: [";
    for (size_t i = 0; i < m_layerShape.size(); ++i) {
        shapeStr += juce::String(m_layerShape[i]);
        if (i + 1 < m_layerShape.size()) shapeStr += ", ";
    }
    shapeStr += juce::String::formatted("] -> Matrix: %d x %d (%d weights)", rows, cols, static_cast<int>(totalWeights));

    g.setColour(UIConstants::Colors::text_cyan_light.withAlpha(0.85f));
    g.setFont(juce::FontOptions(UIConstants::Typography::font_small));
    g.drawText(shapeStr, static_cast<int>(plotArea.getX() + marginLeft), static_cast<int>(plotArea.getY() + 4), static_cast<int>(gridW), 16, juce::Justification::topLeft);

    if (m_hoveredRow >= 0 && m_hoveredRow < rows && m_hoveredCol >= 0 && m_hoveredCol < cols) {
        size_t idx = static_cast<size_t>(m_hoveredRow * cols + m_hoveredCol);
        if (idx < totalWeights) {
            float val = m_currentWeights[idx];
            float origVal = (idx < m_originalWeights.size()) ? m_originalWeights[idx] : val;
            juce::String cellInfo = juce::String::formatted(
                "Cell [%d, %d] (Idx %d)  |  Current: %+.5f  |  Original: %+.5f  |  Diff: %+.5f",
                m_hoveredRow, m_hoveredCol, static_cast<int>(idx), val, origVal, (val - origVal)
            );

            g.setColour(UIConstants::Canvas::heatmap_positive);
            g.drawText(cellInfo, static_cast<int>(plotArea.getX() + marginLeft), static_cast<int>(plotArea.getBottom() - marginBottom + 4), static_cast<int>(gridW), 16, juce::Justification::topLeft);
        }
    } else {
        g.setColour(UIConstants::Colors::text_dimmed);
        juce::String hint = "Hover to inspect weights | Left Click & Drag: Paint / Bend weight values on 2D matrix";
        g.drawText(hint, static_cast<int>(plotArea.getX() + marginLeft), static_cast<int>(plotArea.getBottom() - marginBottom + 4), static_cast<int>(gridW), 16, juce::Justification::topLeft);
    }

    constexpr float legW = 90.0f;
    constexpr float legH = 10.0f;
    float legX = plotArea.getRight() - marginRight - legW;
    float legY = plotArea.getY() + 6.0f;

    juce::ColourGradient legGrad(
        UIConstants::Canvas::heatmap_negative, legX, legY,
        UIConstants::Canvas::heatmap_positive, legX + legW, legY, false
    );
    legGrad.addColour(0.5, UIConstants::Canvas::heatmap_zero);
    g.setGradientFill(legGrad);
    g.fillRect(legX, legY, legW, legH);
    g.setColour(UIConstants::Canvas::grid_major);
    g.drawRect(legX, legY, legW, legH, UIConstants::Layout::border_width_thin);

    g.setFont(juce::FontOptions(9.0f));
    g.setColour(UIConstants::Colors::text_white.withAlpha(0.7f));
    g.drawText(juce::String(-m_displayRange, 1), static_cast<int>(legX - 24), static_cast<int>(legY - 1), 22, static_cast<int>(legH), juce::Justification::centredRight);
    g.drawText(juce::String(+m_displayRange, 1), static_cast<int>(legX + legW + 3), static_cast<int>(legY - 1), 24, static_cast<int>(legH), juce::Justification::centredLeft);
}

auto WeightBendingComponent::draw_edge_scroll_bars(juce::Graphics& g) -> void {
    auto hArea = get_h_scroll_bar_area();
    auto vArea = get_v_scroll_bar_area();
    auto cornerArea = get_corner_button_area();

    g.setColour(UIConstants::Canvas::scrollbar_bg);
    g.fillRect(hArea);
    g.fillRect(vArea);

    g.setColour(UIConstants::Canvas::scrollbar_border);
    g.drawRect(hArea, UIConstants::Layout::border_width_thin);
    g.drawRect(vArea, UIConstants::Layout::border_width_thin);

    float hThumbX = hArea.getX() + m_viewStartX * hArea.getWidth();
    float hThumbW = std::max(12.0f, m_viewSpanX * hArea.getWidth());
    juce::Rectangle<float> hThumb(hThumbX, hArea.getY() + 2.0f, hThumbW, hArea.getHeight() - 4.0f);

    bool hHover = (m_dragMode == DragMode::HScrollThumb || m_dragMode == DragMode::HScrollLeftHandle || m_dragMode == DragMode::HScrollRightHandle);
    g.setColour(hHover ? UIConstants::Canvas::scrollbar_thumb_hover : UIConstants::Canvas::scrollbar_thumb);
    g.fillRoundedRectangle(hThumb, UIConstants::Layout::corner_radius_small);

    float vSpanNorm = juce::jlimit(0.01f, 1.0f, 1.0f / m_zoomY);
    float vCenterNorm = juce::jlimit(vSpanNorm * 0.5f, 1.0f - vSpanNorm * 0.5f, 0.5f - (m_panY / (2.0f * m_displayRange)));
    float vStartNorm = vCenterNorm - vSpanNorm * 0.5f;

    float vThumbY = vArea.getY() + vStartNorm * vArea.getHeight();
    float vThumbH = std::max(12.0f, vSpanNorm * vArea.getHeight());
    juce::Rectangle<float> vThumb(vArea.getX() + 2.0f, vThumbY, vArea.getWidth() - 4.0f, vThumbH);

    bool vHover = (m_dragMode == DragMode::VScrollThumb || m_dragMode == DragMode::VScrollTopHandle || m_dragMode == DragMode::VScrollBottomHandle);
    g.setColour(vHover ? UIConstants::Canvas::scrollbar_thumb_hover : UIConstants::Canvas::scrollbar_thumb);
    g.fillRoundedRectangle(vThumb, UIConstants::Layout::corner_radius_small);

    g.setColour(UIConstants::Canvas::scrollbar_corner);
    g.fillRect(cornerArea);
    g.setColour(UIConstants::Canvas::grid_major);
    g.drawRect(cornerArea, UIConstants::Layout::border_width_thin);

    g.setColour(UIConstants::Colors::text_dimmed);
    g.setFont(juce::FontOptions(8.5f));
    g.drawText("1:1", cornerArea.toNearestInt(), juce::Justification::centred, false);
}

auto WeightBendingComponent::mouseMagnify(const juce::MouseEvent& e, float scaleFactor) -> void {
    if (m_currentWeights.empty()) return;
    auto plotArea = get_plot_area();
    float mouseXNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - plotArea.getX()) / plotArea.getWidth());

    float newSpanX = juce::jlimit(0.005f, 1.0f, m_viewSpanX / scaleFactor);
    float pivotIdxNorm = m_viewStartX + mouseXNorm * m_viewSpanX;
    m_viewStartX = juce::jlimit(0.0f, 1.0f - newSpanX, pivotIdxNorm - mouseXNorm * newSpanX);
    m_viewSpanX = newSpanX;

    m_zoomY = juce::jlimit(0.5f, 25.0f, m_zoomY * scaleFactor);
    repaint();
}

auto WeightBendingComponent::mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) -> void {
    if (m_currentWeights.empty()) return;
    auto plotArea = get_plot_area();

    bool isZoomModifier = e.mods.isAltDown() || e.mods.isCommandDown() || e.mods.isCtrlDown();

    if (isZoomModifier) {
        float factorX = 1.0f + wheel.deltaX * 3.0f;
        float factorY = 1.0f + wheel.deltaY * 3.0f;

        if (std::abs(wheel.deltaX) < 0.0001f && std::abs(wheel.deltaY) > 0.0001f) {
            factorX = 1.0f + wheel.deltaY * 3.0f;
        }

        float mouseXNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - plotArea.getX()) / plotArea.getWidth());
        float pivotX = m_viewStartX + mouseXNorm * m_viewSpanX;
        float newSpanX = juce::jlimit(0.005f, 1.0f, m_viewSpanX / factorX);
        m_viewStartX = juce::jlimit(0.0f, 1.0f - newSpanX, pivotX - mouseXNorm * newSpanX);
        m_viewSpanX = newSpanX;

        m_zoomY = juce::jlimit(0.5f, 25.0f, m_zoomY * factorY);
    } else {
        float panStepX = -wheel.deltaX * m_viewSpanX * 0.7f;
        m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, m_viewStartX + panStepX);

        float yVisibleSpan = (2.0f * m_displayRange) / m_zoomY;
        float panStepY = wheel.deltaY * yVisibleSpan * 0.7f;
        m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, m_panY + panStepY);
    }

    repaint();
}

auto WeightBendingComponent::mouseDoubleClick(const juce::MouseEvent& /*e*/) -> void {
    reset_view();
}

auto WeightBendingComponent::mouseMove(const juce::MouseEvent& e) -> void {
    if (m_renderMode == RenderMode::Matrix) {
        update_hovered_matrix_cell(e.position);
    }
}

auto WeightBendingComponent::mouseExit(const juce::MouseEvent& /*e*/) -> void {
    if (m_renderMode == RenderMode::Matrix) {
        m_hoveredRow = -1;
        m_hoveredCol = -1;
        repaint();
    }
}

auto WeightBendingComponent::mouseDown(const juce::MouseEvent& e) -> void {
    if (m_renderMode == RenderMode::Matrix) {
        m_dragMode = DragMode::Drawing;
        m_lastMousePos = e.position;
        apply_matrix_drawing(e.position, e.mods.isShiftDown() || e.mods.isRightButtonDown());
        return;
    }

    auto cornerArea = get_corner_button_area();
    if (cornerArea.contains(e.position)) {
        reset_view();
        return;
    }

    auto hArea = get_h_scroll_bar_area();
    if (hArea.contains(e.position)) {
        start_h_scroll_drag(e);
        return;
    }

    auto vArea = get_v_scroll_bar_area();
    if (vArea.contains(e.position)) {
        start_v_scroll_drag(e);
        return;
    }

    if (e.mods.isRightButtonDown() || e.mods.isMiddleButtonDown() || e.mods.isAltDown()) {
        m_dragMode = DragMode::PanCanvas;
        m_lastMousePos = e.position;
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
    } else {
        m_dragMode = DragMode::Drawing;
        m_lastMousePos = e.position;
        setMouseCursor(juce::MouseCursor::CrosshairCursor);
        apply_drawing_segment(e.position.x, e.position.y, e.position.x, e.position.y);
    }
}

auto WeightBendingComponent::mouseDrag(const juce::MouseEvent& e) -> void {
    if (m_renderMode == RenderMode::Matrix) {
        update_hovered_matrix_cell(e.position);
        apply_matrix_drawing(e.position, e.mods.isShiftDown() || e.mods.isRightButtonDown());
        m_lastMousePos = e.position;
        return;
    }

    if (m_dragMode == DragMode::PanCanvas) {
        auto delta = e.position - m_lastMousePos;
        m_lastMousePos = e.position;

        auto plotArea = get_plot_area();
        if (plotArea.getWidth() > 0 && plotArea.getHeight() > 0) {
            float normDeltaX = -delta.x / plotArea.getWidth() * m_viewSpanX;
            m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, m_viewStartX + normDeltaX);

            float yVisibleSpan = (2.0f * m_displayRange) / m_zoomY;
            float normDeltaY = delta.y / plotArea.getHeight() * yVisibleSpan;
            m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, m_panY + normDeltaY);

            repaint();
        }
    } else if (m_dragMode == DragMode::HScrollThumb) {
        auto hArea = get_h_scroll_bar_area();
        float normDelta = (e.position.x - m_dragStartPos.x) / hArea.getWidth();
        m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, m_dragInitialStartX + normDelta);
        repaint();
    } else if (m_dragMode == DragMode::HScrollLeftHandle) {
        auto hArea = get_h_scroll_bar_area();
        float clickNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - hArea.getX()) / hArea.getWidth());
        float currentEnd = m_dragInitialStartX + m_dragInitialSpanX;
        float newStart = std::min(clickNorm, currentEnd - 0.005f);
        m_viewStartX = newStart;
        m_viewSpanX = currentEnd - newStart;
        repaint();
    } else if (m_dragMode == DragMode::HScrollRightHandle) {
        auto hArea = get_h_scroll_bar_area();
        float clickNorm = juce::jlimit(0.0f, 1.0f, (e.position.x - hArea.getX()) / hArea.getWidth());
        float newEnd = std::max(clickNorm, m_dragInitialStartX + 0.005f);
        m_viewSpanX = juce::jlimit(0.005f, 1.0f - m_dragInitialStartX, newEnd - m_dragInitialStartX);
        repaint();
    } else if (m_dragMode == DragMode::VScrollThumb) {
        auto vArea = get_v_scroll_bar_area();
        float deltaNorm = (e.position.y - m_dragStartPos.y) / vArea.getHeight();
        m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, m_dragInitialPanY - deltaNorm * 2.0f * m_displayRange);
        repaint();
    } else if (m_dragMode == DragMode::VScrollTopHandle || m_dragMode == DragMode::VScrollBottomHandle) {
        auto vArea = get_v_scroll_bar_area();
        float deltaY = (e.position.y - m_dragStartPos.y);
        float scale = 1.0f + (m_dragMode == DragMode::VScrollTopHandle ? -deltaY : deltaY) / (vArea.getHeight() * 0.5f);
        m_zoomY = juce::jlimit(0.5f, 25.0f, m_dragInitialZoomY * scale);
        repaint();
    } else if (m_dragMode == DragMode::Drawing) {
        apply_drawing_segment(m_lastMousePos.x, m_lastMousePos.y, e.position.x, e.position.y);
        m_lastMousePos = e.position;
    }
}

auto WeightBendingComponent::mouseUp(const juce::MouseEvent& /*e*/) -> void {
    m_dragMode = DragMode::None;
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
    repaint();
}

auto WeightBendingComponent::update_hovered_matrix_cell(juce::Point<float> pos) -> void {
    auto plotArea = get_plot_area();
    constexpr float marginTop = 26.0f;
    constexpr float marginBottom = 24.0f;
    constexpr float marginLeft = 40.0f;
    constexpr float marginRight = 16.0f;

    float gridX = plotArea.getX() + marginLeft;
    float gridY = plotArea.getY() + marginTop;
    float gridW = std::max(10.0f, plotArea.getWidth() - marginLeft - marginRight);
    float gridH = std::max(10.0f, plotArea.getHeight() - marginTop - marginBottom);

    if (pos.x >= gridX && pos.x <= gridX + gridW && pos.y >= gridY && pos.y <= gridY + gridH && m_matrixCols > 0 && m_matrixRows > 0) {
        int col = static_cast<int>((pos.x - gridX) / (gridW / static_cast<float>(m_matrixCols)));
        int row = static_cast<int>((pos.y - gridY) / (gridH / static_cast<float>(m_matrixRows)));

        col = juce::jlimit(0, m_matrixCols - 1, col);
        row = juce::jlimit(0, m_matrixRows - 1, row);

        if (col != m_hoveredCol || row != m_hoveredRow) {
            m_hoveredCol = col;
            m_hoveredRow = row;
            repaint();
        }
    } else {
        if (m_hoveredCol != -1 || m_hoveredRow != -1) {
            m_hoveredCol = -1;
            m_hoveredRow = -1;
            repaint();
        }
    }
}

auto WeightBendingComponent::apply_matrix_drawing(juce::Point<float> pos, bool invert_or_erase) -> void {
    update_hovered_matrix_cell(pos);
    if (m_hoveredRow >= 0 && m_hoveredRow < m_matrixRows && m_hoveredCol >= 0 && m_hoveredCol < m_matrixCols) {
        size_t idx = static_cast<size_t>(m_hoveredRow * m_matrixCols + m_hoveredCol);
        if (idx < m_currentWeights.size()) {
            float delta = invert_or_erase ? -0.1f * m_displayRange : 0.1f * m_displayRange;
            m_currentWeights[idx] = juce::jlimit(-10.0f, 10.0f, m_currentWeights[idx] + delta);
            recalculate_stats();
            m_matrixImageDirty = true;
            repaint();

            if (onWeightsModified) {
                onWeightsModified(m_currentWeights);
            }
        }
    }
}

auto WeightBendingComponent::ensure_palette_lut() -> void {
    if (!m_paletteLut.empty()) {
        return;
    }
    m_paletteLut.resize(512);

    for (int i = 0; i < 256; ++i) {
        float t = (255 - i) / 255.0f;
        m_paletteLut[static_cast<size_t>(i)] = UIConstants::Canvas::heatmap_zero.interpolatedWith(UIConstants::Canvas::heatmap_negative, t).getARGB();
    }
    for (int i = 256; i < 512; ++i) {
        float t = (i - 256) / 255.0f;
        m_paletteLut[static_cast<size_t>(i)] = UIConstants::Canvas::heatmap_zero.interpolatedWith(UIConstants::Canvas::heatmap_positive, t).getARGB();
    }
}

auto WeightBendingComponent::calculate_matrix_dimensions() -> void {
    int N = static_cast<int>(m_currentWeights.size());
    if (N <= 0) {
        m_matrixRows = 1;
        m_matrixCols = 1;
        return;
    }

    if (m_layerShape.size() >= 2) {
        m_matrixRows = static_cast<int>(m_layerShape[0]);
        int64_t cols = 1;
        for (size_t i = 1; i < m_layerShape.size(); ++i) {
            cols *= m_layerShape[i];
        }
        m_matrixCols = static_cast<int>(cols);
    } else {
        int r = static_cast<int>(std::floor(std::sqrt(static_cast<float>(N))));
        while (r > 1 && (N % r != 0)) {
            --r;
        }
        m_matrixRows = r;
        m_matrixCols = N / r;
    }

    m_matrixRows = std::max(1, m_matrixRows);
    m_matrixCols = std::max(1, m_matrixCols);
}

auto WeightBendingComponent::recalculate_stats() -> void {
    if (m_currentWeights.empty()) {
        m_minVal = -1.0f;
        m_maxVal = 1.0f;
        m_meanVal = 0.0f;
        m_displayRange = 1.5f;
        return;
    }

    m_minVal = *std::min_element(m_currentWeights.begin(), m_currentWeights.end());
    m_maxVal = *std::max_element(m_currentWeights.begin(), m_currentWeights.end());

    double sum = 0.0;
    for (float w : m_currentWeights) {
        sum += w;
    }
    m_meanVal = static_cast<float>(sum / m_currentWeights.size());

    float maxAbs = std::max(std::abs(m_minVal), std::abs(m_maxVal));
    m_displayRange = std::max(1.0f, maxAbs * 1.25f);
}

auto WeightBendingComponent::value_to_y(float val, float height) const -> float {
    float effectiveSpan = (2.0f * m_displayRange) / m_zoomY;
    float yMin = m_panY - effectiveSpan * 0.5f;
    float norm = (val - yMin) / effectiveSpan;
    norm = juce::jlimit(0.0f, 1.0f, norm);
    return (1.0f - norm) * (height - 16.0f) + 8.0f;
}

auto WeightBendingComponent::y_to_value(float y, float height) const -> float {
    float norm = 1.0f - ((y - 8.0f) / (height - 16.0f));
    norm = juce::jlimit(0.0f, 1.0f, norm);
    float effectiveSpan = (2.0f * m_displayRange) / m_zoomY;
    float yMin = m_panY - effectiveSpan * 0.5f;
    return yMin + norm * effectiveSpan;
}

auto WeightBendingComponent::build_path_for_weights(juce::Path& p, const std::vector<float>& weights, float w, float h) -> void {
    if (weights.empty() || w <= 0.0f) return;

    int totalWeights = static_cast<int>(weights.size());
    float startIdxF = m_viewStartX * static_cast<float>(totalWeights - 1);
    float endIdxF = (m_viewStartX + m_viewSpanX) * static_cast<float>(totalWeights - 1);
    int totalVisible = static_cast<int>(std::ceil(endIdxF - startIdxF + 1));
    int numPixels = static_cast<int>(w);
    if (numPixels <= 1 || totalVisible <= 1) return;

    if (totalVisible > numPixels * 2) {
        p.preallocateSpace(numPixels * 4);
        bool first = true;

        for (int px = 0; px < numPixels; ++px) {
            float normX0 = static_cast<float>(px) / static_cast<float>(numPixels);
            float normX1 = static_cast<float>(px + 1) / static_cast<float>(numPixels);

            int i0 = juce::jlimit(0, totalWeights - 1, static_cast<int>(std::floor((m_viewStartX + normX0 * m_viewSpanX) * static_cast<float>(totalWeights - 1))));
            int i1 = juce::jlimit(0, totalWeights - 1, static_cast<int>(std::ceil((m_viewStartX + normX1 * m_viewSpanX) * static_cast<float>(totalWeights - 1))));
            if (i1 < i0) i1 = i0;

            float minVal = weights[static_cast<size_t>(i0)];
            float maxVal = minVal;
            for (int i = i0 + 1; i <= i1; ++i) {
                float v = weights[static_cast<size_t>(i)];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }

            float screenX = normX0 * w;
            float yMin = value_to_y(minVal, h);
            float yMax = value_to_y(maxVal, h);

            if (first) {
                p.startNewSubPath(screenX, yMin);
                if (std::abs(yMax - yMin) > 0.5f) {
                    p.lineTo(screenX, yMax);
                }
                first = false;
            } else {
                p.lineTo(screenX, yMin);
                if (std::abs(yMax - yMin) > 0.5f) {
                    p.lineTo(screenX, yMax);
                }
            }
        }
        return;
    }

    int numPoints = std::min(totalVisible, numPixels);
    p.preallocateSpace(numPoints * 2);
    bool first = true;

    for (int x = 0; x < numPoints; ++x) {
        float normPlotX = static_cast<float>(x) / static_cast<float>(numPoints - 1);
        float weightIdxNorm = m_viewStartX + normPlotX * m_viewSpanX;
        size_t idx = static_cast<size_t>(juce::jlimit(0, totalWeights - 1, static_cast<int>(std::round(weightIdxNorm * (totalWeights - 1)))));

        float y = value_to_y(weights[idx], h);
        float screenX = normPlotX * w;

        if (first) {
            p.startNewSubPath(screenX, y);
            first = false;
        } else {
            p.lineTo(screenX, y);
        }
    }
}

auto WeightBendingComponent::apply_drawing_segment(float x0, float y0, float x1, float y1) -> void {
    if (m_currentWeights.empty()) return;

    auto plotArea = get_plot_area();
    float pw = plotArea.getWidth();
    float ph = plotArea.getHeight();
    if (pw <= 0.0f || ph <= 0.0f) return;

    int totalWeights = static_cast<int>(m_currentWeights.size());

    auto screenToWeight = [&](float screenX, float screenY, float& outIdx, float& outVal) {
        float normPlotX = juce::jlimit(0.0f, 1.0f, (screenX - plotArea.getX()) / pw);
        float weightNormX = m_viewStartX + normPlotX * m_viewSpanX;
        outIdx = weightNormX * static_cast<float>(totalWeights - 1);
        outVal = y_to_value(screenY - plotArea.getY(), ph);
    };

    float idx0, val0, idx1, val1;
    screenToWeight(x0, y0, idx0, val0);
    screenToWeight(x1, y1, idx1, val1);

    float weightsPerPixel = (m_viewSpanX * static_cast<float>(totalWeights)) / pw;
    int brushRadius = (weightsPerPixel > 1.5f) ? static_cast<int>(std::ceil(weightsPerPixel * 0.5f)) : 1;
    brushRadius = juce::jlimit(1, 4, brushRadius);

    int minIdx = static_cast<int>(std::floor(std::min(idx0, idx1))) - brushRadius;
    int maxIdx = static_cast<int>(std::ceil(std::max(idx0, idx1))) + brushRadius;
    minIdx = juce::jlimit(0, totalWeights - 1, minIdx);
    maxIdx = juce::jlimit(0, totalWeights - 1, maxIdx);

    float dx = idx1 - idx0;
    float dy = val1 - val0;
    float lenSq = dx * dx;

    for (int idx = minIdx; idx <= maxIdx; ++idx) {
        float targetVal = val1;
        if (lenSq > 0.0001f) {
            float t = juce::jlimit(0.0f, 1.0f, static_cast<float>(idx - idx0) / dx);
            targetVal = val0 + t * dy;
        }

        float distFromCenter = 0.0f;
        if (lenSq > 0.0001f) {
            if (idx < std::min(idx0, idx1)) {
                distFromCenter = std::min(idx0, idx1) - static_cast<float>(idx);
            } else if (idx > std::max(idx0, idx1)) {
                distFromCenter = static_cast<float>(idx) - std::max(idx0, idx1);
            } else {
                distFromCenter = 0.0f;
            }
        } else {
            distFromCenter = std::abs(static_cast<float>(idx) - idx0);
        }

        if (distFromCenter <= static_cast<float>(brushRadius)) {
            float normDist = distFromCenter / static_cast<float>(brushRadius);
            float falloff = (1.0f - normDist * normDist);
            m_currentWeights[static_cast<size_t>(idx)] = m_currentWeights[static_cast<size_t>(idx)] * (1.0f - falloff) + targetVal * falloff;
        }
    }

    recalculate_stats();
    repaint();

    if (onWeightsModified) {
        onWeightsModified(m_currentWeights);
    }
}

auto WeightBendingComponent::start_h_scroll_drag(const juce::MouseEvent& e) -> void {
    auto hArea = get_h_scroll_bar_area();
    float thumbX = hArea.getX() + m_viewStartX * hArea.getWidth();
    float thumbW = std::max(12.0f, m_viewSpanX * hArea.getWidth());

    m_dragStartPos = e.position;
    m_dragInitialStartX = m_viewStartX;
    m_dragInitialSpanX = m_viewSpanX;

    float clickX = e.position.x;
    constexpr float handleSize = 6.0f;

    if (clickX >= thumbX && clickX <= thumbX + handleSize) {
        m_dragMode = DragMode::HScrollLeftHandle;
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    } else if (clickX >= thumbX + thumbW - handleSize && clickX <= thumbX + thumbW) {
        m_dragMode = DragMode::HScrollRightHandle;
        setMouseCursor(juce::MouseCursor::LeftRightResizeCursor);
    } else if (clickX >= thumbX && clickX <= thumbX + thumbW) {
        m_dragMode = DragMode::HScrollThumb;
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
    } else {
        float clickNorm = (clickX - hArea.getX()) / hArea.getWidth();
        m_viewStartX = juce::jlimit(0.0f, 1.0f - m_viewSpanX, clickNorm - m_viewSpanX * 0.5f);
        m_dragInitialStartX = m_viewStartX;
        m_dragMode = DragMode::HScrollThumb;
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        repaint();
    }
}

auto WeightBendingComponent::start_v_scroll_drag(const juce::MouseEvent& e) -> void {
    auto vArea = get_v_scroll_bar_area();
    float vSpanNorm = juce::jlimit(0.01f, 1.0f, 1.0f / m_zoomY);
    float vCenterNorm = juce::jlimit(vSpanNorm * 0.5f, 1.0f - vSpanNorm * 0.5f, 0.5f - (m_panY / (2.0f * m_displayRange)));
    float vStartNorm = vCenterNorm - vSpanNorm * 0.5f;

    float thumbY = vArea.getY() + vStartNorm * vArea.getHeight();
    float thumbH = std::max(12.0f, vSpanNorm * vArea.getHeight());

    m_dragStartPos = e.position;
    m_dragInitialPanY = m_panY;
    m_dragInitialZoomY = m_zoomY;

    float clickY = e.position.y;
    constexpr float handleSize = 6.0f;

    if (clickY >= thumbY && clickY <= thumbY + handleSize) {
        m_dragMode = DragMode::VScrollTopHandle;
        setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
    } else if (clickY >= thumbY + thumbH - handleSize && clickY <= thumbY + thumbH) {
        m_dragMode = DragMode::VScrollBottomHandle;
        setMouseCursor(juce::MouseCursor::UpDownResizeCursor);
    } else if (clickY >= thumbY && clickY <= thumbY + thumbH) {
        m_dragMode = DragMode::VScrollThumb;
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
    } else {
        float clickNorm = (clickY - vArea.getY()) / vArea.getHeight();
        m_panY = juce::jlimit(-m_displayRange * 2.0f, m_displayRange * 2.0f, (0.5f - clickNorm) * 2.0f * m_displayRange);
        m_dragInitialPanY = m_panY;
        m_dragMode = DragMode::VScrollThumb;
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
        repaint();
    }
}
