#ifndef LLAMA2CPP_TENSOR_HPP
#define LLAMA2CPP_TENSOR_HPP
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "memory.hpp"

namespace llama2cpp {
class Shape {
   public:
    Shape() : m_shape(), m_stride(), m_num_dims(0) {}

    Shape(std::initializer_list<size_t> shape) : m_shape(shape), m_stride(), m_num_dims(0) { initialize(); }

    Shape(const Shape &shape) : m_shape(shape.shapeVec()), m_stride(shape.strideVec()), m_num_dims(shape.numDims()) {}

    Shape(const size_t dim) : m_shape({dim}), m_stride(), m_num_dims(1) { initialize(); }

    template <typename... ARGS>
    Shape(ARGS... args) : m_shape({args...}), m_stride(), m_num_dims(sizeof...(args)) {
        initialize();
    }

    template <typename... ARGS>
    auto operator()(size_t idx, ARGS... args) const -> const size_t {
        assert(sizeof...(args) < m_shape.size());
        assert(idx < m_shape[(m_num_dims - 1 - sizeof...(ARGS))]);
        return idx * m_stride[(m_num_dims - 1 - sizeof...(ARGS))] + this->operator()(args...);
    }

    auto operator()(size_t idx) const -> const size_t {
        assert(idx < m_shape[m_num_dims - 1]);
        return idx * m_stride[m_num_dims - 1];
    }

    auto operator[](size_t idx) const -> const size_t { return m_shape.at(idx); }

    auto size() const -> const size_t {
        if (m_shape.empty()) {
            return 0;
        }
        return m_shape[0] * m_stride[0];
    }

    auto numDims() const -> const size_t { return m_num_dims; }

    auto numElements() const -> const size_t { return m_shape[0] * m_stride[0]; }

    auto shapeVec() const -> const std::vector<size_t> { return m_shape; }

    auto strideVec() const -> const std::vector<size_t> { return m_stride; }

   private:
    void initialize() {
        m_num_dims = m_shape.size();
        m_stride.resize(m_num_dims, 1);
        m_stride[m_num_dims - 1] = 1;
        for (auto i = m_num_dims - 1; i >= 1; --i) {
            m_stride[i - 1] = m_shape[i] * m_stride[i];
        }
    }
    std::vector<size_t> m_shape;
    std::vector<size_t> m_stride;
    size_t m_num_dims;
};

std::ostream &operator<<(std::ostream &os, const Shape &shape) {
    os << "Shape (";
    auto &vec = shape.shapeVec();
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << vec[i];
    }
    os << ")";
    return os;
}

template <class T>
class TensorView {
   public:
    using value_type = T;                                        ///< datatype
    using reference = value_type &;                              ///< reference type
    using const_reference = const value_type &;                  ///< const reference type
    using pointer = value_type *;                                ///< pointer type
    using const_pointer = const value_type *;                    ///< const pointer type
    using size_type = size_t;                                    ///< size type
    using ptr = typename std::shared_ptr<TensorView<T>>;         ///< shared pointer type
    using unique_ptr = typename std::unique_ptr<TensorView<T>>;  ///< unique pointer type

    TensorView(pointer data, const Shape &shape) : m_data(data), m_shape(shape) {}
    TensorView(const TensorView &view) : m_data(view.data()), m_shape(view.shape()) {}
    TensorView(TensorView &view) : m_data(view.data()), m_shape(view.shape()) {}

    template <typename... ARGS>
    auto operator()(ARGS... args) -> reference {
        assert(m_data != nullptr);
        return *(m_data + m_shape(args...));
    }

    template <typename... ARGS>
    auto operator()(ARGS... args) const -> const_reference {
        assert(m_data != nullptr);
        return *(m_data + m_shape(args...));
    }

    auto operator[](size_t index) -> reference { return *(m_data + index); }

    auto operator[](size_t index) const -> const_reference { return *(m_data + index); }

    // TODO slice API

    auto shape() const -> const Shape & { return m_shape; }

    auto size() const -> const size_t { return m_shape.size(); }

    auto setShape(const Shape &shape) { m_shape = shape; }

    auto data() const -> const_pointer { return m_data; }

    auto data() -> pointer { return m_data; }

    auto setData(pointer p) { m_data = p; }

   private:
    pointer m_data;
    Shape m_shape;
};

template <template <class> class COMPUTE, class T>
class Tensor : public TensorView<T> {
   public:
    using super = TensorView<T>;
    using value_type = super::value_type;                             ///< datatype
    using reference = value_type &;                                   ///< reference type
    using const_reference = const value_type &;                       ///< const reference type
    using pointer = value_type *;                                     ///< pointer type
    using size_type = size_t;                                         ///< size type
    using ptr = typename std::shared_ptr<Tensor<COMPUTE, T>>;         ///< shared pointer type
    using unique_ptr = typename std::unique_ptr<Tensor<COMPUTE, T>>;  ///< unique pointer type

    Tensor() : TensorView<T>(nullptr, Shape()), m_memory({}) {}
    Tensor(const pointer ptr, const Shape &shape) : TensorView<T>(nullptr, shape), m_memory(shape.size()) {
        this->setData(m_memory.data());
        copyFrom(ptr, numElements());
    }

    Tensor(const Shape &shape) : TensorView<T>(nullptr, shape), m_memory(shape.size()) { this->setData(m_memory.data()); }

    Tensor(const Shape &shape, std::vector<value_type> &values) : TensorView<T>(nullptr, shape), m_memory(values) { this->setData(m_memory.data()); }

    void reShape(const Shape &shape) {
        this->setShape(shape);
        m_memory.resize(shape.size());
        this->setData(m_memory.data());
    }

    void copyFrom(const pointer p, size_t num_elements) { m_memory.copyFrom(p, num_elements); }
    void copyFrom(TensorView<value_type> &tensor) {
        m_memory.copyFrom(tensor.data(), tensor.size());
        this->setShape(tensor.shape());
    }

    auto numElements() const -> const size_t { return this->shape().numElements(); }

   private:
    Memory<COMPUTE, value_type> m_memory;
};

}  // namespace llama2cpp
#endif