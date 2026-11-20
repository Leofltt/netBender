#include "../shared/nn_base.h"
#include "c74_min.h"

long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count);
long simplemc_inputchanged(c74::max::t_object *x, long index, long count);


class mc_nn: public nn_base<mc_nn, mc_operator<>> {

public:
    MIN_DESCRIPTION{"Multi-channel interface for deep learning models and live circuit bending"};
    MIN_TAGS{"audio, deep learning, ai, bending"};
    MIN_AUTHOR{"NetBender / Antoine Caillon & Axel Chemla--Romeu-Santos"};
    MIN_RELATED{"netBender.info, netBender~, mcs.netBender~"};

    static std::string get_external_name() { return "mc.netBender~";} 
    mc_nn(const atoms &args = {}) {
        init_external(args); 
    }

    int get_sample_rate() override {
        return samplerate(); 
    }

    void init_external(const atoms &args) override {
        init_model(); 
        init_downloader(); 
        if (!args.size()) { return; } 
        init_inputs_and_outputs(args);
        init_inlets_and_outlets();
        init_process();
    } 
    void perform(audio_bundle input, audio_bundle output) override;

    // channel handling
    std::vector<int> channel_map;
    std::vector<int> m_configured_channel_map;
    int m_configured_batches{1};
    int m_configured_batches_out{1};
    int n_batches_arg = 0;
    int get_batches(); 
    int get_batches_out(); 

    void init_inputs_and_outputs(const atoms& atoms) override; 
    void init_inlets_and_outlets() override; 
    void init_buffers() override; 
    void init_process() override; 
    bool init_buffers_at_init() override { return true; }
    bool update_channel_map(const long& index, const long& count); 

    message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "mc.netBender~ " << VERSION << " (ExecuTorch / ONNX / Torch "
             << BackendFactory::get_torch_version() << ") - NetBender / ACIDS" << endl;
        c74::max::t_class *c = args[0];
        c74::max::class_addmethod(
            c, (c74::max::method)simplemc_multichanneloutputs,
            "multichanneloutputs", c74::max::A_CANT, 0);
        c74::max::class_addmethod(c, (c74::max::method)simplemc_inputchanged,
                                  "inputchanged", c74::max::A_CANT, 0);
        return {};
      }};

    attribute<int> chans_attr {
      this, 
      "chans", 
      0,
      description{"set a fixed number of output channels"}, 
      setter{
        MIN_FUNCTION {
          if (args.size() == 0)
            return args; 
          int in_chans = args[0];
          if (in_chans > 0) {
            n_batches_arg = in_chans;
            wait_for_buffer_reset = true;
            DEBUG_PRINT("setting out channels to %d", in_chans);
          }
          return args;
        }
      }};
};

int mc_nn::get_batches() {
  if (channel_map.size() > 0) {
    return *std::max_element(channel_map.begin(), channel_map.end());
  } else {
    return 1; 
  }
}

int mc_nn::get_batches_out() {
  if (n_batches_arg > 0) {
    return n_batches_arg; 
  } else {
    return get_batches(); 
  }
}

void mc_nn::init_inputs_and_outputs(const atoms& args)  {
    nn_base::init_inputs_and_outputs(args);
    for (int i(0); i < m_model_in; i++)
        channel_map.push_back(1);
}

void mc_nn::init_inlets_and_outlets() {
  for (int i(0); i < n_inlets; i++) {
    std::string input_label = "";
    try {
      input_label = m_model->get_input_label(m_method, i);
      if (input_label.find("(signal)") == 0) {
        input_label = "(multichannel)" + input_label.substr(8);
      }
    } catch (...) {
      input_label = "(multichannel) model input " + std::to_string(i);
    }
    m_inlets.push_back(std::make_unique<inlet<>>(this, input_label, "multichannelsignal"));
  }

  for (int i(0); i < n_outlets; i++) {
    std::string output_label = "";
    try {
      output_label = m_model->get_output_label(m_method, i);
      if (output_label.find("(signal)") == 0) {
        output_label = "(multichannel)" + output_label.substr(8);
      }
    } catch (...) {
      output_label = "(multichannel) model output " + std::to_string(i);
    }
    m_outlets.push_back(
        std::make_unique<outlet<>>(this, output_label, "multichannelsignal"));
  }
}

void mc_nn::init_buffers() {
  if (!m_ready.load(std::memory_order_acquire)) { return; }
  m_engine.detach_backend();
  const ScopedReconfigureGate fe_gate(m_perform_gate);

  if (channel_map.empty() && n_inlets > 0) {
    channel_map.assign(static_cast<size_t>(n_inlets), 1);
  }

  update_method();
  if (!m_ready.load(std::memory_order_acquire)) {
    m_can_perform.store(false, std::memory_order_release);
    return;
  }

  #ifdef _WIN32
  m_use_thread = false;
  #endif

  m_configured_batches = std::max(1, get_batches());
  m_configured_batches_out = std::max(1, get_batches_out());
  m_configured_channel_map = channel_map;

  m_in_ptrs_scratch.assign(
      static_cast<size_t>(std::max(1, std::max(0, m_model_in) * m_configured_batches)), nullptr);
  m_out_ptrs_scratch.assign(
      static_cast<size_t>(std::max(1, std::max(0, m_model_out) * m_configured_batches)), nullptr);

  configure_streaming_engine(m_configured_batches);

  wait_for_buffer_reset = false;
  buffer_initialised = true;
}

void mc_nn::init_process() {
  nn_base<mc_nn, mc_operator<>>::init_process();
}

void mc_nn::perform(audio_bundle input, audio_bundle output) {
  const int vec_size = static_cast<int>(input.frame_count());
  if (!m_ready.load(std::memory_order_acquire) || !m_engine.is_configured()) {
    return;
  }

  std::fill(m_in_ptrs_scratch.begin(), m_in_ptrs_scratch.end(), nullptr);
  std::fill(m_out_ptrs_scratch.begin(), m_out_ptrs_scratch.end(), nullptr);

  int current_inlet = 0;
  int current_lane = 0;
  const int chan_in = static_cast<int>(input.channel_count());
  for (int c_in = 0; c_in < chan_in; ++c_in) {
    if (current_inlet >= static_cast<int>(m_configured_channel_map.size())) {
      break;
    }
    if (current_inlet < m_model_in && current_lane < m_configured_batches) {
      const int model_in_idx = current_inlet * m_configured_batches + current_lane;
      if (model_in_idx < static_cast<int>(m_in_ptrs_scratch.size())) {
        m_in_ptrs_scratch[static_cast<size_t>(model_in_idx)] = input.samples(c_in);
      }
    }
    ++current_lane;
    if (current_lane >= std::max(1, m_configured_channel_map[static_cast<size_t>(current_inlet)])) {
      ++current_inlet;
      current_lane = 0;
    }
  }

  const int chan_out = static_cast<int>(output.channel_count());
  const int active_batches_out = std::min(m_configured_batches, m_configured_batches_out);
  const int active_outlets = std::min(std::max(0, n_outlets), std::max(0, m_model_out));
  for (int b = 0; b < active_batches_out; ++b) {
    for (int c = 0; c < active_outlets; ++c) {
      const int max_out_idx = c * m_configured_batches_out + b;
      const int model_out_idx = b * m_model_out + c;
      if (max_out_idx < chan_out &&
          model_out_idx < static_cast<int>(m_out_ptrs_scratch.size())) {
        m_out_ptrs_scratch[static_cast<size_t>(model_out_idx)] = output.samples(max_out_idx);
      }
    }
  }

  const int expected_in = std::max(0, m_model_in) * m_configured_batches;
  const int expected_out = std::max(0, m_model_out) * m_configured_batches;
  m_engine.process(m_in_ptrs_scratch.data(), expected_in,
                   m_out_ptrs_scratch.data(), expected_out, vec_size);

  const int total_mapped_out = std::max(0, n_outlets) * m_configured_batches_out;
  for (int idx = 0; idx < chan_out; ++idx) {
    bool is_bound = false;
    if (m_configured_batches_out > 0 && idx < total_mapped_out) {
      const int c = idx / m_configured_batches_out;
      const int b = idx % m_configured_batches_out;
      is_bound = (c < active_outlets) && (b < active_batches_out);
    }
    if (!is_bound) {
      auto *out = output.samples(idx);
      if (out != nullptr && vec_size > 0) {
        std::fill(out, out + vec_size, 0.0);
      }
    }
  }
}

bool mc_nn::update_channel_map(const long& index, const long& count) {
  if (channel_map.empty() && n_inlets > 0) {
    channel_map.assign(static_cast<size_t>(n_inlets), 1);
  }
  if (index < 0 || static_cast<size_t>(index) >= channel_map.size()) {
    return false;
  }
  if (channel_map[static_cast<size_t>(index)] != static_cast<int>(count)) {
    channel_map[static_cast<size_t>(index)] = static_cast<int>(count);
    wait_for_buffer_reset = true;
  }
  return true;
}


long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count) {
  minwrap<mc_nn> *ob = (minwrap<mc_nn> *)(x);
  return ob->m_min_object.get_batches_out();
}

long simplemc_inputchanged(c74::max::t_object *x, long index, long count) {
  minwrap<mc_nn> *ob = (minwrap<mc_nn> *)(x);
  ob->m_min_object.update_channel_map(index, count);
  return true; 
}



MIN_EXTERNAL(mc_nn);