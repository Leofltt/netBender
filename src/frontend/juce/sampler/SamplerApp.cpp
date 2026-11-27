#include <JuceHeader.h>
#include "SamplerMainComponent.h"

class NetBenderSamplerApplication final : public juce::JUCEApplication {
public:
    auto getApplicationName() -> const juce::String override { return "NetBenderSampler"; }
    auto getApplicationVersion() -> const juce::String override { return "0.1.0"; }
    auto moreThanOneInstanceAllowed() -> bool override { return true; }

    auto initialise(const juce::String& command_line) -> void override {
        juce::String initial_bundle = command_line.trim().unquoted();
        m_main_window = std::make_unique<MainWindow>(getApplicationName(), initial_bundle);
    }

    auto shutdown() -> void override { m_main_window.reset(); }

    auto systemRequestedQuit() -> void override { quit(); }

private:
    class MainWindow final : public juce::DocumentWindow {
    public:
        MainWindow(const juce::String& name, const juce::String& initial_bundle)
            : DocumentWindow(name, juce::Colour(0xff0f0e13), DocumentWindow::allButtons) {
            setUsingNativeTitleBar(true);
            setContentOwned(new SamplerMainComponent(initial_bundle), true);
            setResizable(true, true);
            centreWithSize(getWidth(), getHeight());
            setVisible(true);
        }

        auto closeButtonPressed() -> void override {
            JUCEApplication::getInstance()->systemRequestedQuit();
        }
    };

    std::unique_ptr<MainWindow> m_main_window;
};

START_JUCE_APPLICATION(NetBenderSamplerApplication)
