#pragma once
#include <JuceHeader.h>

namespace UIConstants {
    namespace Colors {
        // Backgrounds & Surfaces
        inline const juce::Colour background_gradient_top     { 0xff0f0e13 };
        inline const juce::Colour background_gradient_bottom  { 0xff1a1724 };
        inline const juce::Colour surface_dark               { 0xff13111c };
        inline const juce::Colour surface_panel              { 0xff1a1726 };
        inline const juce::Colour surface_card               { 0xff1c1926 };
        inline const juce::Colour popup_background           { 0xff181622 };
        inline const juce::Colour border_subtle              { 0xff2a243d };
        inline const juce::Colour border_medium              { 0xff3d3554 };
        inline const juce::Colour divider_line               { 0xff382f4c };

        // Accents & Themes
        inline const juce::Colour magenta                   { 0xffd946ef };
        inline const juce::Colour violet                    { 0xff7c3aed };
        inline const juce::Colour dark_violet                { 0xff581c87 };
        inline const juce::Colour deep_violet                { 0xff4a1d72 };
        inline const juce::Colour lavender                  { 0xffa78bfa };
        inline const juce::Colour lilac                     { 0xffc084fc };
        inline const juce::Colour indigo                    { 0xff818cf8 };
        inline const juce::Colour indigo_light               { 0xffa5b4fc };
        inline const juce::Colour blue                      { 0xff6366f1 };
        inline const juce::Colour sky_blue                   { 0xff38bdf8 };
        inline const juce::Colour deep_sky_blue               { 0xff0284c7 };
        inline const juce::Colour cyan                      { 0xff06b6d4 };
        inline const juce::Colour dark_cyan                  { 0xff0e7490 };
        inline const juce::Colour flame_orange               { 0xffea580c };
        inline const juce::Colour drift_orange               { 0xffff9933 };
        inline const juce::Colour amber                     { 0xffd97706 };
        inline const juce::Colour amber_light                { 0xfffbbf24 };
        inline const juce::Colour amber_dark                 { 0xff4a3712 };
        inline const juce::Colour amber_bright               { 0xfffde047 };
        inline const juce::Colour ruby                      { 0xff801e2a };
        inline const juce::Colour dark_ruby                  { 0xff5a1018 };
        inline const juce::Colour navy                      { 0xff1e3a5f };
        inline const juce::Colour forest_green               { 0xff1b3a2a };
        inline const juce::Colour crimson                   { 0xff8b2500 };
        inline const juce::Colour fuse_blown                 { 0xff990000 };
        inline const juce::Colour fuse_glowing_green          { 0xff006633 };
        inline const juce::Colour short_circuit_active        { 0xffff3300 };
        inline const juce::Colour amber_accent               { 0xfff59e0b };

        // Buttons & Tabs
        inline const juce::Colour tab_active_bg               { 0xff4a1d72 };
        inline const juce::Colour tab_inactive_bg             { 0xff1c1926 };
        inline const juce::Colour button_secondary_bg         { 0xff1c1926 };

        // Text
        inline const juce::Colour text_white                 { 0xffffffff };
        inline const juce::Colour text_secondary             { 0xffe2e8f0 };
        inline const juce::Colour text_light                 { 0xffcbd5e1 };
        inline const juce::Colour text_muted                 { 0xff94a3b8 };
        inline const juce::Colour text_dimmed                { 0xff64748b };
        inline const juce::Colour text_disabled              { 0xff475569 };
        inline const juce::Colour text_cyan_light             { 0xffccfbf1 };
    }

    namespace Layout {
        // Window dimensions
        constexpr int window_default_width                    = 1100;
        constexpr int window_default_height                   = 640;
        constexpr int window_min_width                        = 1060;
        constexpr int window_min_height                       = 600;
        constexpr int window_max_width                        = 1600;
        constexpr int window_max_height                       = 1000;

        // Spacings & gaps
        constexpr int gap_tiny                               = 3;
        constexpr int gap_small                              = 4;
        constexpr int gap_compact                            = 6;
        constexpr int gap_medium                             = 8;
        constexpr int gap_standard                           = 10;
        constexpr int gap_large                              = 12;
        constexpr int gap_xlarge                             = 14;

        // Heights
        constexpr int header_height                          = 34;
        constexpr int layer_bar_height                        = 32;
        constexpr int status_bar_height                       = 20;
        constexpr int row_height_standard                     = 26;
        constexpr int row_height_compact                      = 22;
        constexpr int row_height_tiny                         = 18;
        constexpr int toggle_height                          = 24;
        constexpr int label_height                           = 16;
        constexpr int label_height_tall                       = 20;
        constexpr int action_button_height                    = 28;
        constexpr int tab_height                             = 26;

        // Side panel
        constexpr int side_panel_width                        = 240;

        // Slider textboxes
        constexpr int slider_box_width_standard                = 55;
        constexpr int slider_box_height_standard               = 18;
        constexpr int slider_box_width_wide                    = 50;
        constexpr int slider_box_height_wide                   = 20;
        constexpr int slider_box_width_compact                 = 45;
        constexpr int slider_box_height_compact                = 18;

        // Borders & Radii
        constexpr float border_width_thin                     = 1.0f;
        constexpr float border_width_thick                    = 1.5f;
        constexpr float corner_radius_small                   = 3.0f;
        constexpr float corner_radius_medium                  = 4.0f;
        constexpr float corner_radius_large                   = 6.0f;

        // Tickbox
        constexpr float tick_box_size                         = 14.0f;
        constexpr float tick_box_inner_inset                   = 3.0f;

        // Editor layout
        constexpr float header_divider_y                      = 52.0f;
        constexpr int editor_timer_hz                         = 60;
    }

    namespace Typography {
        constexpr float font_title                           = 18.0f;
        constexpr float font_heading                         = 13.0f;
        constexpr float font_body                            = 12.0f;
        constexpr float font_small                           = 11.0f;
        constexpr float text_edge_gap                         = 6.0f;
    }

    namespace Header {
        constexpr int save_model_width                        = 125;
        constexpr int fuse_width                             = 88;
        constexpr int short_circuit_width                     = 82;
        constexpr int trigger_combo_width                     = 90;
        constexpr int trigger_label_width                     = 48;
        constexpr int dry_wet_slider_width                     = 85;
        constexpr int dry_wet_label_width                      = 50;
        constexpr int buffer_status_width                     = 44;
        constexpr int buffer_label_width                      = 40;
        constexpr int gpu_button_width                        = 48;
        constexpr int load_button_width                       = 90;
        constexpr int method_label_width                      = 40;
        constexpr int method_combo_width                      = 90;
    }

    namespace LayerBar {
        constexpr int view_weights_width                      = 68;
        constexpr int view_latent_width                       = 64;
        constexpr int curve_button_width                      = 44;
        constexpr int matrix_button_width                     = 44;
        constexpr int category_label_width                    = 40;
        constexpr int category_combo_width                    = 110;
        constexpr int layer_label_width                       = 40;
        constexpr int reset_all_width                         = 125;
        constexpr int reset_layer_width                       = 105;
    }

    namespace Canvas {
        inline const juce::Colour background             { 0xff0e0d13 };
        inline const juce::Colour inner_canvas            { 0xff110f17 };
        inline const juce::Colour border                 { 0xff3a344d };
        inline const juce::Colour grid_minor              { 0xff262136 };
        inline const juce::Colour grid_major              { 0xff4c4363 };
        inline const juce::Colour center_axis             { 0xff1a1727 };

        inline const juce::Colour wave_baseline           { 0xff10b981 };
        inline const juce::Colour wave_bent               { 0xffc26a38 };
        inline const juce::Colour wave_drift_live          { 0xffd99b26 };
        inline const juce::Colour wave_harmonics          { 0xffec4899 };
        inline const juce::Colour wave_bridge_wire         { 0xfffbbf24 };
        inline const juce::Colour wave_target_bent         { 0xffc084fc };

        inline const juce::Colour heatmap_negative        { 0xff6366f1 };
        inline const juce::Colour heatmap_zero            { 0xff16131f };
        inline const juce::Colour heatmap_positive        { 0xff06b6d4 };

        inline const juce::Colour scrollbar_bg            { 0xff14121b };
        inline const juce::Colour scrollbar_border        { 0xff282338 };
        inline const juce::Colour scrollbar_corner        { 0xff1d1929 };
        inline const juce::Colour scrollbar_thumb         { 0xff635580 };
        inline const juce::Colour scrollbar_thumb_hover    { 0xffa855f7 };
        constexpr float scroll_bar_size                    = 14.0f;
    }

    namespace Latent {
        inline const juce::Colour background             { 0xff0e0d13 };
        inline const juce::Colour inner_pad               { 0xff110f17 };
        inline const juce::Colour border                 { 0xff2d253d };
        inline const juce::Colour grid                   { 0xff382f4c };
        inline const juce::Colour crosshair              { 0xff4c3d69 };
        inline const juce::Colour rings                  { 0xff58447d };
        inline const juce::Colour rays                   { 0xff3b3254 };
        inline const juce::Colour active_ray              { 0xff06b6d4 };
        inline const juce::Colour active_trail            { 0xffd946ef };
        inline const juce::Colour inactive               { 0xff64748b };

        inline const juce::Colour terrain_low             { 0xff0b0a12 };
        inline const juce::Colour terrain_mid_low          { 0xff17122a };
        inline const juce::Colour terrain_mid_high         { 0xff2e1065 };
        inline const juce::Colour terrain_high            { 0xff065f46 };
    }
}
