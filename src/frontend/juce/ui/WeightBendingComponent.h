#pragma once
#include <JuceHeader.h>
#include <vector>
#include <functional>

class WeightBendingComponent : public juce::Component {
public:
    enum class RenderMode {
        Curve,
        Matrix
    };

    WeightBendingComponent();
    ~WeightBendingComponent() override = default;

    std::function<void(const std::vector<float>&)> onWeightsModified;

    auto set_render_mode(RenderMode mode) -> void;
    auto setRenderMode(RenderMode mode) -> void { set_render_mode(mode); }

    auto get_render_mode() const -> RenderMode { return m_renderMode; }
    auto getRenderMode() const -> RenderMode { return get_render_mode(); }

    auto set_layer_shape(const std::vector<int64_t>& shape) -> void;
    auto setLayerShape(const std::vector<int64_t>& shape) -> void { set_layer_shape(shape); }

    auto set_weights(const std::vector<float>& original, const std::vector<float>& current) -> void;
    auto setWeights(const std::vector<float>& original, const std::vector<float>& current) -> void { set_weights(original, current); }

    auto update_current_weights(const std::vector<float>& current) -> void;
    auto updateCurrentWeights(const std::vector<float>& current) -> void { update_current_weights(current); }

    auto set_target_bent_weights(const std::vector<float>& target) -> void;
    auto setTargetBentWeights(const std::vector<float>& target) -> void { set_target_bent_weights(target); }

    auto set_bridge_weights(const std::vector<float>& source_weights, const std::vector<float>& mix_weights, float depth, bool is_continuous) -> void;
    auto setBridgeWeights(const std::vector<float>& sourceWeights, const std::vector<float>& mixWeights, float depth, bool isContinuous) -> void {
        set_bridge_weights(sourceWeights, mixWeights, depth, isContinuous);
    }

    auto set_harmonic_ghost_weights(const std::vector<float>& ghost_weights, bool show_ghost) -> void;
    auto setHarmonicGhostWeights(const std::vector<float>& ghostWeights, bool showGhost) -> void {
        set_harmonic_ghost_weights(ghostWeights, showGhost);
    }

    auto set_show_bridge_visuals(bool show) -> void;
    auto setShowBridgeVisuals(bool show) -> void { set_show_bridge_visuals(show); }

    auto set_show_harmonics(bool show) -> void;
    auto setShowHarmonics(bool show) -> void { set_show_harmonics(show); }

    auto is_bridge_visuals_enabled() const -> bool { return m_showBridgeVisuals; }
    auto isBridgeVisualsEnabled() const -> bool { return is_bridge_visuals_enabled(); }

    auto is_harmonics_enabled() const -> bool { return m_showHarmonics; }
    auto isHarmonicsEnabled() const -> bool { return is_harmonics_enabled(); }

    auto get_current_weights() const -> const std::vector<float>& { return m_currentWeights; }
    auto getCurrentWeights() const -> const std::vector<float>& { return get_current_weights(); }

    auto get_layer_shape() const -> const std::vector<int64_t>& { return m_layerShape; }
    auto getLayerShape() const -> const std::vector<int64_t>& { return get_layer_shape(); }

    auto get_original_weights() const -> const std::vector<float>& { return m_originalWeights; }
    auto getOriginalWeights() const -> const std::vector<float>& { return get_original_weights(); }

    auto is_currently_drawing() const -> bool;
    auto isCurrentlyDrawing() const -> bool { return is_currently_drawing(); }

    auto reset_view() -> void;
    auto resetView() -> void { reset_view(); }

    auto get_plot_area() const -> juce::Rectangle<float>;
    auto getPlotArea() const -> juce::Rectangle<float> { return get_plot_area(); }

    auto get_h_scroll_bar_area() const -> juce::Rectangle<float>;
    auto getHScrollBarArea() const -> juce::Rectangle<float> { return get_h_scroll_bar_area(); }

    auto get_v_scroll_bar_area() const -> juce::Rectangle<float>;
    auto getVScrollBarArea() const -> juce::Rectangle<float> { return get_v_scroll_bar_area(); }

    auto get_corner_button_area() const -> juce::Rectangle<float>;
    auto getCornerButtonArea() const -> juce::Rectangle<float> { return get_corner_button_area(); }

    auto paint(juce::Graphics& g) -> void override;
    auto mouseMagnify(const juce::MouseEvent& e, float scaleFactor) -> void override;
    auto mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) -> void override;
    auto mouseDoubleClick(const juce::MouseEvent& e) -> void override;
    auto mouseMove(const juce::MouseEvent& e) -> void override;
    auto mouseExit(const juce::MouseEvent& e) -> void override;
    auto mouseDown(const juce::MouseEvent& e) -> void override;
    auto mouseDrag(const juce::MouseEvent& e) -> void override;
    auto mouseUp(const juce::MouseEvent& e) -> void override;

private:
    enum class DragMode {
        None,
        Drawing,
        PanCanvas,
        HScrollThumb,
        HScrollLeftHandle,
        HScrollRightHandle,
        VScrollThumb,
        VScrollTopHandle,
        VScrollBottomHandle
    };

    auto paint_curve_view(juce::Graphics& g, juce::Rectangle<float> plot_area) -> void;
    auto paint_matrix_view(juce::Graphics& g, juce::Rectangle<float> plot_area) -> void;
    auto draw_edge_scroll_bars(juce::Graphics& g) -> void;
    auto get_heatmap_colour(float val) const -> juce::Colour;
    auto update_hovered_matrix_cell(juce::Point<float> pos) -> void;
    auto apply_matrix_drawing(juce::Point<float> pos, bool invert_or_erase) -> void;
    auto ensure_palette_lut() -> void;
    auto calculate_matrix_dimensions() -> void;
    auto recalculate_stats() -> void;
    auto value_to_y(float val, float height) const -> float;
    auto y_to_value(float y, float height) const -> float;
    auto build_path_for_weights(juce::Path& p, const std::vector<float>& weights, float w, float h) -> void;
    auto apply_drawing_segment(float x0, float y0, float x1, float y1) -> void;
    auto start_h_scroll_drag(const juce::MouseEvent& e) -> void;
    auto start_v_scroll_drag(const juce::MouseEvent& e) -> void;

    DragMode m_dragMode{DragMode::None};
    juce::Point<float> m_lastMousePos;
    juce::Point<float> m_dragStartPos;
    float m_dragInitialStartX{0.0f};
    float m_dragInitialSpanX{1.0f};
    float m_dragInitialPanY{0.0f};
    float m_dragInitialZoomY{1.0f};

    std::vector<float> m_originalWeights;
    std::vector<float> m_currentWeights;
    std::vector<float> m_targetBentWeights;
    std::vector<float> m_bridgeSourceWeights;
    std::vector<float> m_bridgeMixWeights;
    std::vector<float> m_harmonicGhostWeights;
    bool m_showHarmonicGhost{false};
    bool m_showBridgeVisuals{false};
    bool m_showHarmonics{false};
    float m_bridgeDepth{0.0f};
    bool m_isContinuousMode{true};

    RenderMode m_renderMode{RenderMode::Curve};
    std::vector<int64_t> m_layerShape;
    int m_matrixRows{1};
    int m_matrixCols{1};
    int m_hoveredRow{-1};
    int m_hoveredCol{-1};

    juce::Image m_matrixImage;
    bool m_matrixImageDirty{true};
    std::vector<uint32_t> m_paletteLut;

    float m_viewStartX{0.0f};
    float m_viewSpanX{1.0f};
    float m_zoomY{1.0f};
    float m_panY{0.0f};

    float m_minVal{-1.0f};
    float m_maxVal{1.0f};
    float m_meanVal{0.0f};
    float m_displayRange{1.5f};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WeightBendingComponent)
};
