#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <random>
#include <cmath>

//==============================================================================
NetBenderAudioProcessor::NetBenderAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
       m_model_thread(*this)
#endif
{
    addParameter (m_dryWetParam = new juce::AudioParameterFloat (
        juce::ParameterID ("dry_wet", 1), "Dry / Wet",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 1.0f
    ));

    addParameter (m_scaleParam = new juce::AudioParameterFloat (
        juce::ParameterID ("scale", 1), "Layer Scale",
        juce::NormalisableRange<float> (0.0f, 5.0f, 0.01f), 1.0f
    ));

    addParameter (m_offsetParam = new juce::AudioParameterFloat (
        juce::ParameterID ("offset", 1), "Layer Offset",
        juce::NormalisableRange<float> (-2.0f, 2.0f, 0.001f), 0.0f
    ));

    addParameter (m_heatParam = new juce::AudioParameterFloat (
        juce::ParameterID ("heat", 1), "Layer Heat / Volatility",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.0f
    ));

    addParameter (m_memoryParam = new juce::AudioParameterFloat (
        juce::ParameterID ("memory", 1), "Drift Memory / Drag",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.001f), 0.8f
    ));

    addParameter (m_shortCircuitParam = new juce::AudioParameterBool (
        juce::ParameterID ("short_circuit", 1), "Momentary Short Circuit", false
    ));

    addParameter (m_latentXParam = new juce::AudioParameterFloat (
        juce::ParameterID ("latent_x", 1), "Latent Terrain X",
        juce::NormalisableRange<float> (-1.0f, 1.0f, 0.001f), 0.0f
    ));

    addParameter (m_latentYParam = new juce::AudioParameterFloat (
        juce::ParameterID ("latent_y", 1), "Latent Terrain Y",
        juce::NormalisableRange<float> (-1.0f, 1.0f, 0.001f), 0.0f
    ));

    addParameter (m_latentDepthParam = new juce::AudioParameterFloat (
        juce::ParameterID ("latent_depth", 1), "Latent Depth",
        juce::NormalisableRange<float> (0.0f, 2.0f, 0.01f), 0.5f
    ));

    m_backend = BackendFactory::create_default();
}

NetBenderAudioProcessor::~NetBenderAudioProcessor()
{
    m_model_thread.stop();
}

//==============================================================================
auto NetBenderAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) -> void {
    juce::ignoreUnused(samplesPerBlock);
    m_hostSampleRate.store(sampleRate);
    if (m_backend)
        m_backend->set_sample_rate(sampleRate);
    if (m_modelLoaded.load() && m_backend) {
        double modelSr = m_backend->get_sample_rate();
        m_modelSampleRate.store(modelSr);
        m_resamplingActive.store(std::abs(sampleRate - modelSr) > 1.0);
    }
    initBuffers();
}

auto NetBenderAudioProcessor::releaseResources() -> void {
    m_model_thread.stop();
}

auto NetBenderAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const -> bool {
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

auto NetBenderAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) -> void {
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    const juce::ScopedTryLock sl (getCallbackLock());
    if (!sl.isLocked())
    {
        buffer.clear();
        return;
    }

    // Process incoming MIDI triggers (Note On kicks momentary glitch; Note Off releases)
    for (const auto metadata : midiMessages)
    {
        auto msg = metadata.getMessage();
        if (msg.isNoteOn())
        {
            m_bending_manager.trigger_short_circuit(true);
        }
        else if (msg.isNoteOff())
        {
            m_bending_manager.trigger_short_circuit(false);
        }
    }

    // Clear unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    if (!m_modelLoaded.load() || m_model_out <= 0)
    {
        for (auto i = 0; i < buffer.getNumChannels(); ++i)
            buffer.clear(i, 0, buffer.getNumSamples());
        return;
    }

    auto numSamples = buffer.getNumSamples();

    // Sidechain Audio Envelope Follower for Transient Triggering
    if (totalNumInputChannels > 0 && numSamples > 0)
    {
        float maxPeak = 0.0f;
        for (int c = 0; c < totalNumInputChannels; ++c)
        {
            float peak = buffer.getMagnitude(c, 0, numSamples);
            if (peak > maxPeak) maxPeak = peak;
        }
        m_bending_manager.update_audio_envelope(maxPeak);
    }
    

    // 2. Feed incoming audio into dry delay buffers (for all modes) and model inputs (for forward mode)
    int numDryChannels = (int)m_dry_delay_buffers.size();
    for (int c = 0; c < numDryChannels; ++c)
    {
        const float* inPtr = nullptr;
        if (c < totalNumInputChannels)
            inPtr = buffer.getReadPointer(c);
        else if (totalNumInputChannels > 0)
            inPtr = buffer.getReadPointer(0);

        m_dry_delay_buffers[c].put(inPtr, numSamples);
    }

    bool resampling = m_resamplingActive.load();
    double hostSr = m_hostSampleRate.load();
    double modelSr = m_modelSampleRate.load();

    if (m_model_in > 0)
    {
        if (resampling && hostSr > 0.0 && modelSr > 0.0)
        {
            // Resample host input -> model sample rate
            double inRatio = hostSr / modelSr;
            int numModelSamples = (int)std::ceil(numSamples / inRatio);
            if (numModelSamples > 0)
            {
                for (int c = 0; c < m_model_in; ++c)
                {
                    const float* inPtr = nullptr;
                    if (c < totalNumInputChannels)
                        inPtr = buffer.getReadPointer(c);
                    else if (totalNumInputChannels > 0)
                        inPtr = buffer.getReadPointer(0);

                    if (c < (int)m_input_resamplers.size() && c < (int)m_resampled_in_scratch.size())
                    {
                        int targetSamples = std::min(numModelSamples, (int)m_resampled_in_scratch[c].size());

                        m_input_resamplers[c].process(
                            inRatio,
                            inPtr,
                            m_resampled_in_scratch[c].data(),
                            targetSamples
                        );

                        m_in_buffers[c].put(m_resampled_in_scratch[c].data(), targetSamples);
                    }
                }
            }
        }
        else
        {
            // Direct transfer without resampling
            for (int c = 0; c < m_model_in; ++c)
            {
                const float* inPtr = nullptr;
                if (c < totalNumInputChannels)
                    inPtr = buffer.getReadPointer(c);
                else if (totalNumInputChannels > 0)
                    inPtr = buffer.getReadPointer(0);

                m_in_buffers[c].put(inPtr, numSamples);
            }
        }
        
        // Check if we have enough samples to trigger inference and thread is idle
        if (!m_model_thread.is_processing() && m_in_buffers[0].get_available() >= m_bufferSize)
        {
            std::unique_lock<std::mutex> lock(m_staging_mutex, std::try_to_lock);
            if (lock.owns_lock())
            {
                for (int c = 0; c < m_model_in; ++c)
                {
                    m_in_buffers[c].get(m_staging_in[c].data(), m_bufferSize);
                }
                m_model_thread.trigger_compute();
            }
        }
    }
    else
    {
        // Generative model (0 audio inputs): trigger inference whenever output buffer needs filling
        if (!m_model_thread.is_processing() && m_out_buffers[0].get_available() < m_bufferSize * 3)
        {
            m_model_thread.trigger_compute();
        }
    }
    
    // 3. Pull processed samples from output circular buffers (resampling model SR -> host SR if active)
    int n_outs = std::min((int)totalNumOutputChannels, m_model_out);
    if (resampling && hostSr > 0.0 && modelSr > 0.0)
    {
        double outRatio = modelSr / hostSr;
        int requiredModelSamples = (int)std::ceil(numSamples * outRatio) + 2;
        int masterConsumed = 0;

        for (int c = 0; c < n_outs; ++c)
        {
            if (c < (int)m_output_resamplers.size() && c < (int)m_resampled_out_scratch.size())
            {
                int peekCount = std::min(requiredModelSamples, (int)m_resampled_out_scratch[c].size());
                int avail = m_out_buffers[c].peek(m_resampled_out_scratch[c].data(), peekCount);
                if (avail < peekCount)
                {
                    std::fill(m_resampled_out_scratch[c].begin() + avail,
                              m_resampled_out_scratch[c].begin() + peekCount, 0.0f);
                }

                int consumed = m_output_resamplers[c].process(
                    outRatio,
                    m_resampled_out_scratch[c].data(),
                    buffer.getWritePointer(c),
                    numSamples
                );

                if (c == 0) masterConsumed = std::min(consumed, avail);
            }
            else
            {
                buffer.clear(c, 0, numSamples);
            }
        }

        // Lock sample consumption across all channels to maintain strict stereo phase alignment
        for (int c = 0; c < n_outs; ++c)
        {
            if (c < (int)m_output_resamplers.size() && c < (int)m_resampled_out_scratch.size())
            {
                m_out_buffers[c].drop(masterConsumed);
            }
        }
    }
    else
    {
        for (int c = 0; c < n_outs; ++c)
        {
            m_out_buffers[c].get(buffer.getWritePointer(c), numSamples);
        }
    }
    
    // If model is mono (1 out) and host is stereo (2 out), duplicate Left to Right channel
    if (m_model_out == 1 && totalNumOutputChannels >= 2)
    {
        buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
    }
    // Clear any extra output channels beyond model outputs
    for (int c = std::max(n_outs, (m_model_out == 1 ? 2 : 1)); c < totalNumOutputChannels; ++c)
    {
        buffer.clear(c, 0, numSamples);
    }

    // 4. Dry/Wet Blending & Safety Sentry
    float wetGain = m_dryWet.load();
    float dryGain = 1.0f - wetGain;

    if ((int)m_dryBlockScratch.size() < numSamples)
        m_dryBlockScratch.resize(numSamples, 0.0f);

    for (int c = 0; c < (int)totalNumOutputChannels; ++c)
    {
        int dryCh = std::min(c, (int)m_dry_delay_buffers.size() - 1);
        if (dryCh >= 0)
        {
            m_dry_delay_buffers[dryCh].get(m_dryBlockScratch.data(), numSamples);
            float* outPtr = buffer.getWritePointer(c);
            m_safety_sentry.process_channel(outPtr, m_dryBlockScratch.data(), numSamples, wetGain, dryGain);
        }
    }

    // Auto-recovery handling if enabled (lock-free & allocation-free on audio thread)
    m_safety_sentry.tick_recovery([this] {
        m_safety_sentry.reset_fuse();
        m_bending_manager.request_audio_thread_reset_all();
    });
}

//==============================================================================
auto NetBenderAudioProcessor::getAvailableModes() const -> std::vector<juce::String> {
    std::vector<juce::String> modes;
    if (!m_modelLoaded.load() || !m_backend)
        return modes;

    auto backendModes = m_backend->get_plugin_modes();
    for (const auto& m : backendModes)
        modes.emplace_back(juce::String(m));
    return modes;
}

auto NetBenderAudioProcessor::loadModel(const juce::File& file) -> bool {
    m_modelLoaded.store(false);
    m_model_thread.stop();
    
    std::string path = file.getFullPathName().toStdString();
    
    try
    {
        auto newBackend = BackendFactory::create_for_file(path);
        if (!newBackend)
            return false;

        if (file.getFileName().containsIgnoreCase("_coreml.pte") || file.getFileName().containsIgnoreCase("_mps.pte"))
            m_useGpu.store(true);
        else if (file.getFileName().containsIgnoreCase("_xnnpack.pte") || file.getFileName().containsIgnoreCase("_portable.pte"))
            m_useGpu.store(false);

        newBackend->use_gpu(m_useGpu.load());
        m_backend = std::move(newBackend);

        int err = m_backend->load(path, getSampleRate());
        if (err == 0)
        {
            m_modelPath = file.getFullPathName();
            
            // Find default mode: prefer "forward", else "prior", else first available
            auto modes = m_backend->get_plugin_modes();
            if (!modes.empty())
            {
                std::string defaultMode = modes[0];
                for (const auto& m : modes)
                {
                    if (m == "forward")
                    {
                        defaultMode = "forward";
                        break;
                    }
                }
                {
                    std::lock_guard<std::mutex> stateLock(m_state_mutex);
                    m_currentMethod = defaultMode;
                }
                
                auto params = m_backend->get_mode_params(defaultMode);
                if (params.size() >= 4)
                {
                    m_model_in = params[0];
                    m_model_out = params[2];
                }
            }

            // Auto-configure buffer size to backend preferred size or model ratio.
            int prefSize = m_backend->get_preferred_buffer_size();
            if (prefSize > 0)
            {
                m_bufferSize = prefSize;
            }
            else
            {
                int higher_ratio = m_backend->get_higher_ratio();
                int reqSize = 1;
                while (reqSize < higher_ratio)
                    reqSize <<= 1;
                int minBufferForRatio = (higher_ratio >= 2048) ? 4096 : 2048;
                m_bufferSize = std::max(minBufferForRatio, reqSize * 2);
            }
            
            // Reset latent terrain cached matrices and stale layer bending states on new model load
            m_latent_engine.reset();
            m_bending_manager.clear_all_layer_bending();
            {
                std::lock_guard<std::mutex> stateLock(m_state_mutex);
                m_activeLayerName = "";
            }

            // Query model sample rate vs host sample rate
            double hostSr = getSampleRate();
            m_hostSampleRate.store(hostSr > 0.0 ? hostSr : 44100.0);
            double modelSr = m_backend->get_sample_rate();
            m_modelSampleRate.store(modelSr);
            m_resamplingActive.store(std::abs(m_hostSampleRate.load() - modelSr) > 1.0);

            m_modelLoaded.store(true);
            initBuffers();
            return true;
        }
    }
    catch (const std::exception& e)
    {
        DBG("Exception loading model: " << e.what());
    }
    catch (...)
    {
        DBG("Unknown exception loading model");
    }
    
    return false;
}

auto NetBenderAudioProcessor::setCurrentMethod(const juce::String& method) -> void {
    if (!m_backend) return;
    std::string methodStr = method.toStdString();
    auto modes = m_backend->get_plugin_modes();
    bool isValidMode = (std::find(modes.begin(), modes.end(), methodStr) != modes.end())
                    || m_backend->has_method(methodStr);

    if (isValidMode) {
        int prevIn = m_model_in;
        int prevOut = m_model_out;
        {
            std::lock_guard<std::mutex> stateLock(m_state_mutex);
            m_currentMethod = method;
        }
        auto params = m_backend->get_mode_params(methodStr);
        if (params.size() >= 4) {
            m_model_in = params[0];
            m_model_out = params[2];
        }

        if (prevIn != m_model_in || prevOut != m_model_out) {
            initBuffers();
        } else {
            m_pendingClearFifo.store(true);
            if (!m_model_thread.isThreadRunning() && m_backend) {
                m_backend->clear_decode_gen_fifo();
            }
        }
    }
}

auto NetBenderAudioProcessor::setBufferSize(int size) -> void {
    if (size > 0 && (size & (size - 1)) == 0) { // Power of two check
        m_bufferSize = size;
        initBuffers();
    }
}

auto NetBenderAudioProcessor::setUseGpu(bool enabled) -> void {
    if (m_useGpu.load() == enabled && m_modelLoaded.load())
        return;

    m_useGpu.store(enabled);
    if (!m_modelLoaded.load() || m_modelPath.isEmpty()) {
        if (m_backend)
            m_backend->use_gpu(enabled);
        return;
    }

    juce::String savedMethod = getCurrentMethod();
    juce::File targetFile(m_modelPath);

    // For .pte models, automatically swap between sibling *_xnnpack.pte (CPU) and *_coreml.pte (GPU/ANE) if present
    if (m_modelPath.endsWithIgnoreCase(".pte")) {
        juce::String candidatePath = m_modelPath;
        if (enabled && candidatePath.containsIgnoreCase("_xnnpack.pte"))
            candidatePath = candidatePath.replace("_xnnpack.pte", "_coreml.pte", true);
        else if (!enabled && candidatePath.containsIgnoreCase("_coreml.pte"))
            candidatePath = candidatePath.replace("_coreml.pte", "_xnnpack.pte", true);

        juce::File candidateFile(candidatePath);
        if (candidateFile.existsAsFile())
            targetFile = candidateFile;
    }

    if (loadModel(targetFile)) {
        if (savedMethod.isNotEmpty())
            setCurrentMethod(savedMethod);
    }
}

auto NetBenderAudioProcessor::initBuffers() -> void {
    m_model_thread.stop();
    
    const juce::ScopedLock sl (getCallbackLock());
    std::lock_guard<std::mutex> lock(m_staging_mutex);
    
    int n_in = std::max(1, m_model_in);
    int n_out = std::max(1, m_model_out);
    
    m_in_buffers.resize(n_in);
    m_out_buffers.resize(n_out);
    
    m_staging_in.resize(n_in);
    m_staging_out.resize(n_out);
    m_worker_in.resize(n_in);
    m_worker_out.resize(n_out);
    
    // In forward mode, delay is 2 * m_bufferSize (1 buffer input accumulation + 1 buffer model inference transfer)
    double hostSr = m_hostSampleRate.load();
    double modelSr = m_modelSampleRate.load();
    bool resampling = m_resamplingActive.load() && (hostSr > 0.0) && (modelSr > 0.0);
    
    // Scale latency to host sample rate if resampling is active
    int latencySamples = 0;
    if (m_model_in > 0)
    {
        double latencyModel = m_bufferSize * 2.0;
        latencySamples = resampling ? (int)std::round(latencyModel * (hostSr / modelSr)) : (m_bufferSize * 2);
    }
    setLatencySamples(latencySamples);

    int n_dry = std::max(2, std::max(n_in, n_out));
    m_dry_delay_buffers.resize(n_dry);

    // Initialize resamplers and scratch storage
    m_input_resamplers.resize(n_in);
    for (auto& r : m_input_resamplers) r.reset();

    m_output_resamplers.resize(n_out);
    for (auto& r : m_output_resamplers) r.reset();

    int maxScratchSize = std::max(8192, m_bufferSize * 4);
    m_resampled_in_scratch.resize(n_in);
    for (auto& buf : m_resampled_in_scratch)
        buf.assign(maxScratchSize, 0.0f);

    m_resampled_out_scratch.resize(n_out);
    for (auto& buf : m_resampled_out_scratch)
        buf.assign(maxScratchSize, 0.0f);

    m_dryBlockScratch.assign(maxScratchSize, 0.0f);

    for (int i = 0; i < n_in; ++i)
    {
        m_in_buffers[i].init(m_bufferSize);
        m_staging_in[i].assign(m_bufferSize, 0.0f);
        m_worker_in[i].assign(m_bufferSize, 0.0f);
    }

    for (int i = 0; i < n_dry; ++i)
    {
        int dryCap = std::max(m_bufferSize, latencySamples);
        m_dry_delay_buffers[i].init(dryCap);
        // Pre-fill dry delay buffer with latency zeros so dry aligns with wet output
        if (latencySamples > 0)
            m_dry_delay_buffers[i].put(nullptr, latencySamples);
    }
    
    int modelLatencySamples = (m_model_in > 0) ? (m_bufferSize * 2) : 0;
    for (int i = 0; i < n_out; ++i)
    {
        m_out_buffers[i].init(m_bufferSize);
        m_staging_out[i].assign(m_bufferSize, 0.0f);
        m_worker_out[i].assign(m_bufferSize, 0.0f);
        // Pre-fill output buffer with latency cushion in model sample units
        // to prevent starvation during background model inference and align with dry delay buffer
        if (modelLatencySamples > 0)
        {
            m_out_buffers[i].put(nullptr, modelLatencySamples);
        }
    }
    
    m_output_ready.store(false);
    
    if (m_backend)
    {
        m_backend->clear_decode_gen_fifo();
    }
    
    if (m_modelLoaded.load())
    {
        m_model_thread.startThread(juce::Thread::Priority::high);
    }
}

auto NetBenderAudioProcessor::runInference() -> void {
    if (!m_modelLoaded.load() || m_model_out <= 0)
        return;

    std::string activeLayerStr;
    std::string modeStr;
    IBackend::LatentHook userLatentHook;
    {
        std::lock_guard<std::mutex> stateLock(m_state_mutex);
        activeLayerStr = m_activeLayerName.toStdString();
        modeStr = m_currentMethod.toStdString();
        userLatentHook = m_latentHook;
    }

    if (m_pendingClearFifo.exchange(false) && m_backend)
        m_backend->clear_decode_gen_fifo();

    // 1. Update and apply weight bending to backend
    float scaleVal = m_scaleParam ? m_scaleParam->get() : 1.0f;
    float offsetVal = m_offsetParam ? m_offsetParam->get() : 0.0f;
    float heatVal = m_heatParam ? m_heatParam->get() : 0.0f;
    float memVal = m_memoryParam ? m_memoryParam->get() : -1.0f;
    bool shortVal = m_shortCircuitParam ? m_shortCircuitParam->get() : false;
    m_bending_manager.apply_bending(m_backend.get(), activeLayerStr, scaleVal, offsetVal, heatVal, memVal, shortVal);

    // 2. Copy audio inputs from staging buffers to local worker buffers under a brief lock
    {
        std::lock_guard<std::mutex> lock(m_staging_mutex);
        for (int c = 0; c < m_model_in; ++c)
        {
            if (c < (int)m_staging_in.size() && c < (int)m_worker_in.size())
            {
                std::memcpy(m_worker_in[c].data(), m_staging_in[c].data(), m_bufferSize * sizeof(float));
            }
        }
    }
    std::vector<float*> in_ptrs;
    std::vector<float*> out_ptrs;
    
    int numIn = std::min(m_model_in, (int)m_worker_in.size());
    for (int c = 0; c < numIn; ++c)
        in_ptrs.emplace_back(m_worker_in[c].data());
        
    int numOut = std::min(m_model_out, (int)m_worker_out.size());
    for (int c = 0; c < numOut; ++c)
        out_ptrs.emplace_back(m_worker_out[c].data());
        
    if (modeStr == "forward")
    {
        // Compose latent hook: if user hook is set, use it; otherwise if latent terrain is enabled, execute it!
        IBackend::LatentHook activeHook = userLatentHook;
        if (!activeHook && m_latent_engine.is_enabled())
        {
            float lx = m_latentXParam ? m_latentXParam->get() : -999.0f;
            float ly = m_latentYParam ? m_latentYParam->get() : -999.0f;
            float depth = m_latentDepthParam ? m_latentDepthParam->get() : -999.0f;
            activeHook = [this, lx, ly, depth](IBackend::LatentBuffer& buf) {
                m_latent_engine.apply_hook(buf, lx, ly, depth);
            };
        }

        if (m_backend)
            m_backend->perform_forward(in_ptrs, out_ptrs, 1, m_model_out, m_bufferSize, activeHook);
    }
    else if (modeStr == "decode gen")
    {
        // Autonomous neural synthesis: generate latent trajectory with Pad/Drift/Harmonics and decode
        IBackend::LatentHook activeHook = userLatentHook;
        if (!activeHook)
        {
            float lx = m_latentXParam ? m_latentXParam->get() : -999.0f;
            float ly = m_latentYParam ? m_latentYParam->get() : -999.0f;
            float depth = m_latentDepthParam ? m_latentDepthParam->get() : -999.0f;
            activeHook = [this, lx, ly, depth](IBackend::LatentBuffer& buf) {
                m_latent_engine.apply_hook(buf, lx, ly, depth);
            };
        }

        if (m_backend)
            m_backend->perform_decode_gen(out_ptrs, 1, m_model_out, m_bufferSize, activeHook);
    }
    else if (modeStr == "prior" || modeStr == "generate")
    {
        if (m_backend)
            m_backend->perform_prior_decode(out_ptrs, 1, m_model_out, m_bufferSize);
    }
    else
    {
        if (m_backend)
            m_backend->perform(in_ptrs, out_ptrs, modeStr, 1, m_model_out, m_bufferSize);
    }

    // 3. Push computed output directly into output ring buffers
    for (int c = 0; c < m_model_out; ++c)
    {
        if (c < (int)m_out_buffers.size() && c < (int)m_worker_out.size())
        {
            m_out_buffers[c].put(m_worker_out[c].data(), m_bufferSize);
        }
    }
}

//==============================================================================
auto NetBenderAudioProcessor::getStateInformation(juce::MemoryBlock& destData) -> void {
    juce::XmlElement xmlState ("NetBenderSettings");
    xmlState.setAttribute ("modelPath", m_modelPath);
    xmlState.setAttribute ("currentMethod", getCurrentMethod());
    xmlState.setAttribute ("bufferSize", m_bufferSize);
    xmlState.setAttribute ("activeLayer", getActiveLayerName());
    xmlState.setAttribute ("dryWet", (double)getDryWet());
    xmlState.setAttribute ("triggerMode", (int)getTriggerMode());
    xmlState.setAttribute ("envAttack", (double)getEnvelopeAttack());
    xmlState.setAttribute ("envRelease", (double)getEnvelopeRelease());
    xmlState.setAttribute ("transientThresh", (double)getTransientThreshold());

    auto* layersElement = xmlState.createNewChildElement ("Layers");
    auto allStates = m_bending_manager.get_all_layer_states();
    for (const auto& pair : allStates)
    {
        auto* layerEl = layersElement->createNewChildElement ("Layer");
        layerEl->setAttribute ("name", pair.first);
        layerEl->setAttribute ("scale", (double)pair.second.scale);
        layerEl->setAttribute ("offset", (double)pair.second.offset);
        layerEl->setAttribute ("heat", (double)pair.second.heat);
        layerEl->setAttribute ("memory", (double)pair.second.memory);
        layerEl->setAttribute ("driftMode", (int)pair.second.driftMode);
        layerEl->setAttribute ("frozen", pair.second.frozen);
    }

    copyXmlToBinary (xmlState, destData);
}

auto NetBenderAudioProcessor::setStateInformation(const void* data, int sizeInBytes) -> void {
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr)
    {
        if (xmlState->hasTagName ("NetBenderSettings") || xmlState->hasTagName ("NNBendingSettings"))
        {
            m_bufferSize = xmlState->getIntAttribute ("bufferSize", 2048);
            setDryWet ((float)xmlState->getDoubleAttribute ("dryWet", 1.0));
            setActiveLayerName (xmlState->getStringAttribute ("activeLayer"));
            setTriggerMode ((TriggerMode)xmlState->getIntAttribute ("triggerMode", 0));
            setEnvelopeAttack ((float)xmlState->getDoubleAttribute ("envAttack", 0.25));
            setEnvelopeRelease ((float)xmlState->getDoubleAttribute ("envRelease", 0.85));
            setTransientThreshold ((float)xmlState->getDoubleAttribute ("transientThresh", 0.15));

            juce::String path = xmlState->getStringAttribute ("modelPath");
            if (path.isNotEmpty())
            {
                juce::File file(path);
                if (file.existsAsFile())
                    loadModel(file);
            }
            juce::String method = xmlState->getStringAttribute ("currentMethod");
            if (method.isNotEmpty())
                setCurrentMethod(method);

            // Restore per-layer bending states
            auto* layersElement = xmlState->getChildByName ("Layers");
            if (layersElement != nullptr)
            {
                for (auto* layerEl : layersElement->getChildIterator())
                {
                    std::string name = layerEl->getStringAttribute ("name").toStdString();
                    if (!name.empty())
                    {
                        float scale = (float)layerEl->getDoubleAttribute ("scale", 1.0);
                        float offset = (float)layerEl->getDoubleAttribute ("offset", 0.0);
                        double heatVal = layerEl->getDoubleAttribute ("heat", -1.0);
                        if (heatVal < 0.0)
                            heatVal = layerEl->getDoubleAttribute ("jitter", 0.0);
                        float heat = (float)heatVal;
                        float memory = (float)layerEl->getDoubleAttribute ("memory", 0.8);
                        auto driftMode = (DriftMode)layerEl->getIntAttribute ("driftMode", 0);
                        bool frozen = layerEl->getBoolAttribute ("frozen", false);
                        m_bending_manager.restore_layer_state(name, scale, offset, heat, memory, driftMode, frozen);
                    }
                }
            }
        }
    }
}

//==============================================================================
auto NetBenderAudioProcessor::createEditor() -> juce::AudioProcessorEditor* {
    return new NetBenderAudioProcessorEditor (*this);
}

//==============================================================================
// This creates the filter...
auto JUCE_CALLTYPE createPluginFilter() -> juce::AudioProcessor* {
    return new NetBenderAudioProcessor();
}
