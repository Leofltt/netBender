#pragma once
#include <JuceHeader.h>
#include <functional>
#include <vector>

class LatentTerrainComponent : public juce::Component {
public:
    LatentTerrainComponent();
    ~LatentTerrainComponent() override = default;

    std::function<void(float, float)> onCoordsChanged;

    auto set_coords(float x, float y) -> void;
    auto setCoords(float x, float y) -> void { set_coords(x, y); }

    auto get_coord_x() const -> float { return m_x; }
    auto getCoordX() const -> float { return get_coord_x(); }

    auto get_coord_y() const -> float { return m_y; }
    auto getCoordY() const -> float { return get_coord_y(); }

    auto set_enabled_state(bool is_enabled, bool has_bottleneck) -> void;
    auto setEnabledState(bool isEnabled, bool hasBottleneck) -> void { set_enabled_state(isEnabled, hasBottleneck); }

    auto paint(juce::Graphics& g) -> void override;
    auto mouseDown(const juce::MouseEvent& e) -> void override;
    auto mouseDrag(const juce::MouseEvent& e) -> void override;
    auto mouseDoubleClick(const juce::MouseEvent& e) -> void override;

private:
    auto update_background_terrain_image(int w, int h) -> void;
    auto record_trail_point(float px, float py) -> void;
    auto update_from_mouse(const juce::MouseEvent& e) -> void;

    struct Point2D { float x; float y; };

    float m_x{0.0f};
    float m_y{0.0f};
    bool m_isEnabled{false};
    bool m_hasBottleneck{true};

    juce::Image m_backgroundTerrainImage;
    std::vector<Point2D> m_trail;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentTerrainComponent)
};
