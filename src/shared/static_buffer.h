#pragma once
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <cstddef>
#if !defined(NETBENDER_NO_TORCH_HEADERS)
#include <torch/torch.h>
#endif

template <typename data_type> 
class StaticBuffer {
public:
    using BufferData = std::vector<data_type>;
    using BufferShape = std::array<size_t, 2>;

private:
    std::shared_ptr<BufferData> _data; 
    BufferShape _dims;
    double _samplingRate; 
    
public: 
    StaticBuffer() {
        _dims = BufferShape({0, 0});
    }

    StaticBuffer(const size_t dim1, const size_t dim2, const double samplingRate = -1) {
        _dims = BufferShape({dim1, dim2});
        _data = std::make_shared<BufferData>(get_empty_buffer(dim1, dim2));
        _samplingRate = samplingRate;
    }

    StaticBuffer(BufferData data, const double samplingRate) {
        _dims = BufferShape({data.dims(), data[0].dims()});
        _data = std::make_shared<BufferData>(data);
        _samplingRate = samplingRate;
    }

    StaticBuffer(const StaticBuffer& buffer) {
        _dims = buffer._dims; 
        _data = buffer._data;
        _samplingRate = buffer._samplingRate;
    }

    auto dims() -> BufferShape {
        return _dims;
    }

    auto sr() -> double {
        return _samplingRate;
    }

    static auto get_empty_buffer(const size_t dim1, const size_t dim2) -> BufferData {
        return BufferData(dim1 * dim2, 0.f);
    }

    auto clear() -> void {
        _data.get()->clear();
    }

    auto reset() -> void {
        std::fill(_data.get()->begin(), _data.get()->end(), 0.0f);
    }

#if !defined(NETBENDER_NO_TORCH_HEADERS)
    auto to_tensor() -> torch::Tensor {
        auto obj = torch::from_blob(_data.get()->data(), {static_cast<long long>(_dims[0]), static_cast<long long>(_dims[1])}, torch::kFloat);
        return obj;
    }
#endif

    auto at(const size_t dim1, const size_t dim2) -> float& {
        return _data.get()->at(dim1 * _dims[0] + dim2);
    }

    auto put(data_type data, const size_t dim1, const size_t dim2) -> void {
        at(dim1, dim2) = data;
    }
};
