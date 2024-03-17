#ifndef LLAMA2CPP_TENSOR_HPP
#define LLAMA2CPP_TENSOR_HPP
#include <string>
#include <vector>
#include <llama2cpp/memory.hpp>
#include <iostream>
#include <cassert>

namespace llama2cpp
{
    class Shape
    {
    public:
        Shape() : m_shape(), m_stride(), m_num_dims(0)
        {
        }

        Shape(std::initializer_list<size_t> shape) : m_shape(shape), m_stride(), m_num_dims(0)
        {
            initialize();
        }

        template <typename... ARGS>
        auto operator()(size_t idx, ARGS... args) const -> const size_t
        {
            assert(sizeof...(args) < m_shape.size());
            assert(idx < m_shape[(m_num_dims - 1 - sizeof...(ARGS))]);
            return idx * m_stride[(m_num_dims - 1 - sizeof...(ARGS))] + this->operator()(args...);
        }

        auto operator()(size_t idx) const -> const size_t
        {
            assert(idx < m_shape[m_num_dims - 1]);
            return idx * m_stride[m_num_dims - 1];
        }

        auto size() const -> const size_t
        {
            return m_shape[0] * m_stride[0];
        }

    private:
        void initialize()
        {
            m_num_dims = m_shape.size();
            m_stride.resize(m_num_dims, 1);
            m_stride[m_num_dims - 1] = 1;
            for (auto i = m_num_dims - 1; i >= 1; --i)
            {
                m_stride[i - 1] = m_shape[i] * m_stride[i];
            }
        }
        std::vector<size_t> m_shape;
        std::vector<size_t> m_stride;
        size_t m_num_dims;
    };

    template <template <class> class COMPUTE, class T, size_t DIM>
    class Tensor
    {
    public:
        using value_type = T;                                                 ///< datatype
        using reference = value_type &;                                       ///< reference type
        using const_reference = const value_type &;                           ///< const reference type
        using pointer = value_type *;                                         ///< pointer type
        using size_type = size_t;                                             ///< size type
        using ptr = typename std::shared_ptr<Tensor<COMPUTE, T, DIM>>;        ///< shared pointer type
        using unique_ptr = typename std::unique_ptr<Tensor<COMPUTE, T, DIM>>; ///< unique pointer type
        static constexpr const size_t dimension = DIM;                        ///< dimension of the tensor

        Tensor() : m_shape(), m_memory()
        {
        }

        Tensor(Shape shape) : m_shape(shape), m_memory(shape.size())
        {
        }

        Tensor(Shape shape, std::vector<value_type> &values) : m_shape(shape), m_memory(values)
        {
        }

        template <typename... ARGS>
        auto operator()(ARGS... args) -> reference
        {
            assert(!m_memory.empty());
            return m_memory[m_shape(args...)];
        }

        /**
         * @brief method to access elements of the memory as a tensor
         *
         * @tparam ARGS index type
         * @param args indices
         * @return const reference to the elements of the tensor
         */
        template <typename... ARGS>
        auto operator()(ARGS... args) const -> const_reference
        {
            assert(!m_memory.empty());
            return m_memory[m_shape(args...)];
        }

        auto operator[](size_t index) -> reference
        {
            return m_memory[index];
        }

        auto operator[](size_t index) const -> const_reference
        {
            return m_memory[index];
        }

        auto data() -> pointer { return m_memory.data(); }

        void reShape(const Shape &shape)
        {
            m_shape = shape;
            m_memory.resize(shape.size());
        }

    private:
        Shape m_shape;
        Memory<COMPUTE, T> m_memory;
    };

}
#endif