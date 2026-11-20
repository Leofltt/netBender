#include "../shared/nn_base.h"
#include "c74_min.h"

class nn: public nn_base<nn, vector_operator<>> {

public:
    MIN_DESCRIPTION{"Interface for deep learning models and live circuit bending"};
    MIN_TAGS{"audio, deep learning, ai, bending"};
    MIN_AUTHOR{"NetBender / Antoine Caillon & Axel Chemla--Romeu-Santos"};
    MIN_RELATED{"netBender.info, mc.netBender~, mcs.netBender~"};

    static std::string get_external_name() {
       return std::string("netBender~");
    } 
    nn(const atoms &args = {}) {
        init_external(args); 
     }

    int get_sample_rate() override {
        return samplerate(); 
    }

    void init_process() override { 
        nn_base<nn>::init_process(); 
    }

    void init_external(const atoms &args) override {
      DEBUG_PRINT("initializing model");
      init_model(); 
      DEBUG_PRINT("initializing downloader");
      init_downloader(); 
      if (!args.size()) { return; } 
      DEBUG_PRINT("initializing inputs & outputs");
      init_inputs_and_outputs(args); 
      DEBUG_PRINT("initializing inlets & outlets");
      init_inlets_and_outlets(); 
      DEBUG_PRINT("initializing process"); 
      init_process(); 
    } 
    void perform(audio_bundle input, audio_bundle output) override;

    message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "netBender~ " << VERSION << " (ExecuTorch / ONNX / Torch "
             << BackendFactory::get_torch_version() << ") - NetBender / ACIDS" << endl;
        return {};
      }};
      
};


void nn::perform(audio_bundle input, audio_bundle output) {
  const int vec_size = static_cast<int>(input.frame_count());
  if (!m_ready.load(std::memory_order_acquire) || !m_engine.is_configured()) {
    return;
  }

  const int in_cap = static_cast<int>(m_in_ptrs_scratch.size());
  const int out_cap = static_cast<int>(m_out_ptrs_scratch.size());
  const int active_in = std::min({std::max(0, m_model_in),
                                  static_cast<int>(input.channel_count()), in_cap});
  const int active_out = std::min({std::max(0, m_model_out),
                                   static_cast<int>(output.channel_count()), out_cap});

  for (int c = 0; c < active_in; ++c) {
    m_in_ptrs_scratch[static_cast<size_t>(c)] = input.samples(c);
  }
  for (int c = active_in; c < in_cap; ++c) {
    m_in_ptrs_scratch[static_cast<size_t>(c)] = nullptr;
  }

  for (int c = 0; c < active_out; ++c) {
    m_out_ptrs_scratch[static_cast<size_t>(c)] = output.samples(c);
  }
  for (int c = active_out; c < out_cap; ++c) {
    m_out_ptrs_scratch[static_cast<size_t>(c)] = nullptr;
  }

  m_engine.process(m_in_ptrs_scratch.data(), std::min(std::max(0, m_model_in), in_cap),
                   m_out_ptrs_scratch.data(), std::min(std::max(0, m_model_out), out_cap),
                   vec_size);

  for (int c = active_out; c < static_cast<int>(output.channel_count()); ++c) {
    auto *out = output.samples(c);
    if (out != nullptr && vec_size > 0) {
      std::fill(out, out + vec_size, 0.0);
    }
  }
}


MIN_EXTERNAL(nn);