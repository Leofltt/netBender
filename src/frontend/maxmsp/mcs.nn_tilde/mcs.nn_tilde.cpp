#include "../shared/nn_base.h"
#include "c74_min.h"

long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count);
long simplemc_inputchanged(c74::max::t_object *x, long index, long count);


class mcs_nn: public nn_base<mcs_nn, mc_operator<>> {

public:
    MIN_DESCRIPTION{"Multi-channel interface for deep learning models and live circuit bending"};
    MIN_TAGS{"audio, deep learning, ai, bending"};
    MIN_AUTHOR{"NetBender / Antoine Caillon & Axel Chemla--Romeu-Santos"};
    MIN_RELATED{"netBender.info, netBender~, mc.netBender~"};

    static std::string get_external_name() { return "mcs.netBender~";} 
    mcs_nn(const atoms &args = {}) {
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
        wait_for_buffer_reset = true; 
        init_process();
    } 
    void perform(audio_bundle input, audio_bundle output) override;

    void dump_object() override;

    // channel handling
    int get_batches(); 
    int n_mc_inputs() {
      return std::accumulate(channel_map.begin(), channel_map.end(), 0);
    } 
    bool check_inputs(); 
    std::vector<long> channel_map;
    std::vector<long> m_configured_channel_map;
    int m_configured_batches{1};
    int m_configured_out_channels{1};

    void init_inputs_and_outputs(const atoms& atoms) override; 
    void init_inlets_and_outlets() override; 
    void init_buffers() override; 
    void init_process() override; 
    void update_method(std::string method = "") override;
    bool update_channel_map(const long& index, const long& count); 

    int m_out_channels = 0; 
    int m_out_channels_arg = 0; 
    void update_out_channels() {
      if ((m_out_channels_arg == 0) && (m_model_out != m_out_channels)) {
        DEBUG_PRINT("updating out channels to %d", m_model_out);
        m_out_channels = m_model_out; 
        wait_for_buffer_reset = true; 
      }
      // if method is changed, multi channels outputs will stay the same. 
    }

    message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "mcs.netBender~ " << VERSION << " (ExecuTorch / ONNX / Torch "
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
              m_out_channels_arg = in_chans;
              m_out_channels = in_chans;
              wait_for_buffer_reset = true;
              DEBUG_PRINT("setting out channels to %d", in_chans);
            }
            return args;
          }
        }};
};

bool mcs_nn::check_inputs() {
    return true;
}

int mcs_nn::get_batches() {
  return *std::max_element(channel_map.begin(), channel_map.end());
}

void mcs_nn::init_inputs_and_outputs(const atoms &args) {
  bool empty_mode = false; 
  DEBUG_PRINT("parsing inputs & outputs...");
  if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
    auto model_path = std::string(args[0]);
    if (model_path == "void") {
      empty_mode = true;    
      m_model_in = 1; 
      m_model_out = 1; 
    } else {
      try {
        m_path = to_model_path(model_path);
      } catch (std::string &e) {
        error(e);
      }
    }
  }
  
  if (empty_mode) {
    DEBUG_PRINT("empty mode");
    if (args.size() > 1) { // FOUR ARGUMENTS ARE GIVEN
      n_batches = int(args[1]);
    }
    if (args.size() > 2) { // THREE ARGUMENTS ARE GIVEN
      m_buffer_size = int(args[2]);
    }
    channel_map = std::vector<long>(n_batches, 1); 
  } else {
    if (args.size() > 1) { // TWO ARGUMENTS ARE GIVEN
      m_method = std::string(args[1]);
    }
    if (args.size() > 2) { // TWO ARGUMENTS ARE GIVEN
      n_batches = int(args[2]);
    }
    if (args.size() > 3) { // THREE ARGUMENTS ARE GIVEN
      m_buffer_size = int(args[3]);
    }
    channel_map = std::vector<long>(n_batches, 1); 
    DEBUG_PRINT("loading model..."); 
    load_model(m_path);
    if (m_ready.load(std::memory_order_acquire)) {
      update_method();
      m_out_channels = (m_out_channels_arg > 0) ? m_out_channels_arg : m_model_out;
    } 
  }
  if (m_buffer_size == -1) {
    // NO THREAD MODE
    m_buffer_size = DEFAULT_BUFFER_SIZE;
  }
}

bool mcs_nn::update_channel_map(const long& index, const long& count)
{
  if (index < 0 || static_cast<size_t>(index) >= channel_map.size()) {
    return false;
  }
  if (channel_map[static_cast<size_t>(index)] != count) {
    channel_map[static_cast<size_t>(index)] = count;
    wait_for_buffer_reset = true; 
    if (count != m_model_in) {
      return false;
    } else {
      return true;
    }
  }
  return true; 
}


void mcs_nn::init_inlets_and_outlets() {
   
  DEBUG_PRINT("loading model..."); 
  DEBUG_PRINT("n_batches : %d", n_batches); 
  std::string input_label; 
  for (int i(0); i < n_batches; i++) {
    if (m_model_in > 0) {
      input_label = "(multichannel) batch " + std::to_string(i) + "(" + std::to_string(m_model_in) + " dimensions)";
    } else {
      input_label = "(multichannel) batch " + std::to_string(i); 
    }
    m_inlets.push_back(
      std::make_unique<inlet<>>(this, input_label, "multichannelsignal"));
  }

  std::string output_label;
  for (int i(0); i < n_batches; i++) {
    output_label = "(multichannel) batch " + std::to_string(i) + "(" + std::to_string(m_out_channels) + " dimensions)";
    m_outlets.push_back(
        std::make_unique<outlet<>>(this, output_label, "multichannelsignal"));
  }
  n_inlets = n_batches; 
  n_outlets = n_batches; 
}

void mcs_nn::init_buffers() {
  if (!m_ready.load(std::memory_order_acquire)) { return; }
  m_engine.detach_backend();
  const ScopedReconfigureGate fe_gate(m_perform_gate);

  update_method(); 
  if (!m_ready.load(std::memory_order_acquire)) {
    m_can_perform.store(false, std::memory_order_release);
    return;
  }

  if (m_out_channels <= 0) {
    m_can_perform.store(false, std::memory_order_release);
    error("could not retrieve number of output channels");
    return;
  }

  DEBUG_PRINT("initializing buffers...");
  if (m_buffer_size == -1) {
    // NO THREAD MODE
    m_buffer_size = DEFAULT_BUFFER_SIZE;
  }
  DEBUG_PRINT("buffer size : %d", m_buffer_size);
  if (m_buffer_size == 0) {
    m_use_thread = false; 
    m_buffer_size = m_higher_ratio; 
  } else {
    if (m_buffer_size < m_higher_ratio) {
      cerr << "buffer size too small, switching to " << m_buffer_size << endl;
      m_buffer_size = m_higher_ratio;
    } else {
      m_buffer_size = power_ceil(m_buffer_size);
    }
  }

  #ifdef _WIN32
  m_use_thread = false;
  #endif

  m_configured_batches = std::max(1, n_batches);
  m_configured_out_channels = m_out_channels;
  m_configured_channel_map = channel_map;

  m_in_ptrs_scratch.assign(
      static_cast<size_t>(std::max(1, std::max(0, m_model_in) * m_configured_batches)), nullptr);
  m_out_ptrs_scratch.assign(
      static_cast<size_t>(std::max(1, std::max(0, m_model_out) * m_configured_batches)), nullptr);

  configure_streaming_engine(m_configured_batches);

  DEBUG_PRINT("buffers initialized");
  wait_for_buffer_reset = false; 
  buffer_initialised = true; 
}



void mcs_nn::update_method(std::string method) {
  if (!method.empty()) {
      set_method(method);
  }
  if (!m_model || !m_model->is_loaded()) {
    cerr << "no model is set yet" << endl;
    return; 
  }
  if (m_model->has_method(m_method)) {
    auto params = m_model->get_method_params(m_method);
    if (params.empty()) {
      m_ready.store(false, std::memory_order_release);
      m_can_perform.store(false, std::memory_order_release);
      return;
    }
    // input parameters
    m_model_in = params[0];
    m_in_ratio = params[1];
    // output parameters
    m_model_out = params[2];
    m_out_ratio = params[3]; 

    if (m_out_channels == 0) { 
      m_out_channels = m_model_out; 
    } 
    m_can_perform.store(true, std::memory_order_release);
    wait_for_buffer_reset = true; 
  } else {
    cerr << "method " << m_method << " not present in model" << endl;
    m_ready.store(false, std::memory_order_release);
    m_can_perform.store(false, std::memory_order_release);
  }
}

void mcs_nn::init_process() {
  nn_base<mcs_nn, mc_operator<>>::init_process();
}

void mcs_nn::perform(audio_bundle input, audio_bundle output) {
  const int vec_size = static_cast<int>(input.frame_count());
  if (!m_ready.load(std::memory_order_acquire) || !m_engine.is_configured()) {
    return;
  }

  std::fill(m_in_ptrs_scratch.begin(), m_in_ptrs_scratch.end(), nullptr);
  std::fill(m_out_ptrs_scratch.begin(), m_out_ptrs_scratch.end(), nullptr);

  const int chan_in = static_cast<int>(input.channel_count());
  int in_idx = 0;
  for (size_t b = 0; b < m_configured_channel_map.size() && in_idx < chan_in; ++b) {
    const long ch_count = std::max<long>(1, m_configured_channel_map[b]);
    for (long c = 0; c < ch_count && in_idx < chan_in; ++c) {
      if (static_cast<int>(b) < m_configured_batches && static_cast<int>(c) < m_model_in) {
        const int model_in_idx =
            static_cast<int>(c) * m_configured_batches + static_cast<int>(b);
        if (model_in_idx < static_cast<int>(m_in_ptrs_scratch.size())) {
          m_in_ptrs_scratch[static_cast<size_t>(model_in_idx)] = input.samples(in_idx);
        }
      }
      ++in_idx;
    }
  }

  const int chan_out = static_cast<int>(output.channel_count());
  const int active_batches_out = std::min(std::max(0, n_outlets), m_configured_batches);
  const int active_chans_out = std::min(std::max(0, m_model_out), m_configured_out_channels);
  for (int b = 0; b < active_batches_out; ++b) {
    for (int c = 0; c < active_chans_out; ++c) {
      const int max_out_idx = b * m_configured_out_channels + c;
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

  const int total_mapped_out = std::max(0, n_outlets) * m_configured_out_channels;
  for (int idx = 0; idx < chan_out; ++idx) {
    bool is_bound = false;
    if (m_configured_out_channels > 0 && idx < total_mapped_out) {
      const int b = idx / m_configured_out_channels;
      const int c = idx % m_configured_out_channels;
      is_bound = (b < active_batches_out) && (c < active_chans_out);
    }
    if (!is_bound) {
      auto *out = output.samples(idx);
      if (out != nullptr && vec_size > 0) {
        std::fill(out, out + vec_size, 0.0);
      }
    }
  }
}

long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count) {
  minwrap<mcs_nn> *ob = (minwrap<mcs_nn> *)(x);
  return ob->m_min_object.m_out_channels;
}

long simplemc_inputchanged(c74::max::t_object *x, long index, long count) {
  minwrap<mcs_nn> *ob = (minwrap<mcs_nn> *)(x);
  auto is_full = ob->m_min_object.update_channel_map(index, count);
  ob->m_min_object.update_out_channels();

  if (!is_full) {
    auto n_channels = ob->m_min_object.m_model_in; 
    c74::max::object_warn(
        x, (std::string("got " + std::to_string(count) + 
            " for " + std::to_string(n_channels) + 
            " model inputs").c_str())
    );
  }
  return true;
}

void mcs_nn::dump_object() {
  cout << "model_path: " << std::string(m_path) << endl;
  if (m_model) {
    if (m_model->is_loaded()) {
      cout << "input dimension: " << m_model_in << endl;
      cout << "output dimension: " << m_model_out << endl;
    } else {
      cout << "input dimension: no model yet" << endl;
      cout << "output dimension: no model yet" << endl;
    }
  } else {
    cout << "input dimension: no model yet" << endl;
    cout << "output dimension: no model yet" << endl;
  }
  cout << "input ratio: " << std::to_string(m_in_ratio) << endl; 
  cout << "output ratio: " << std::to_string(m_out_ratio) << endl; 
  cout << "methods: ";
  for (auto method: m_model->get_available_methods())
    cout << method << "; ";
  cout << endl; 
  cout << "attributes: ";
  for (auto attribute: m_model->get_settable_attributes())
    cout << attribute << "; ";
  cout << endl;
}

MIN_EXTERNAL(mcs_nn);