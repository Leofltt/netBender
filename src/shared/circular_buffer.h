#pragma once
#include <memory>
#include <algorithm>
#include <cstddef>

template <class in_type, class out_type>
class circular_buffer {
public:
  circular_buffer();
  auto initialize(size_t size) -> void;
  auto empty() -> bool;
  auto full() -> bool;
  auto put(in_type* input_array, int N) -> void;
  auto get(out_type* output_array, int N) -> void;
  auto fill(out_type val) -> void;
  auto reset() -> void;
  auto max_size() -> size_t { return _max_size; }

protected:
  std::unique_ptr<out_type[]> _buffer;
  size_t _max_size = 0;

  int _head = 0;
  int _tail = 0;
  int _count = 0;
  bool _full = false;
};

template <class in_type, class out_type>
circular_buffer<in_type, out_type>::circular_buffer() {}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::initialize(size_t size) -> void {
  _buffer = std::make_unique<out_type[]>(size);
  auto zero_val = static_cast<out_type>(0.0f);
  std::fill(_buffer.get(), _buffer.get() + size, zero_val);
  _max_size = size;
}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::empty() -> bool {
  return (!_full && _head == _tail);
}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::full() -> bool {
  return _full;
}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::fill(out_type /*value*/) -> void {
}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::put(in_type* input_array, int N) -> void {
  if (!_max_size) {
    return;
  }

  while (N--) {
    _buffer[_head] = out_type(*(input_array++));
    _head = (_head + 1) % _max_size;
    if (_full) {
      _tail = (_tail + 1) % _max_size;
    }
    _full = _head == _tail;
  }
}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::get(out_type* output_array, int N) -> void {
  if (!_max_size) {
    return;
  }

  while (N--) {
    if (empty()) {
      *(output_array++) = out_type();
    } else {
      *(output_array++) = _buffer[_tail];
      _tail = (_tail + 1) % _max_size;
      _full = false;
    }
  }
}

template <class in_type, class out_type>
auto circular_buffer<in_type, out_type>::reset() -> void {
  _head = _tail;
  _count = 0;
  _full = false;
}
