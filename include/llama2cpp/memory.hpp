#ifndef LLAMA2CPP_MEMORY_HPP
#define LLAMA2CPP_MEMORY_HPP
#include <string>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace llama2cpp
{
    struct XPU
    {
    };

    template <class T>
    struct CPU : public XPU
    {
        using allocator_type = std::allocator<T>;

        static void fill(T *begin_, size_t num_elements, T val)
        {
            std::fill(begin_, begin_ + num_elements, val);
        }

        static void copy(const T *src_, T *dest_, size_t num_elements)
        {
            std::memcpy(dest_, src_, num_elements * sizeof(T));
        }

        static auto get(T *data, size_t index) -> T &
        {
            return *(data + index);
        }
    };

    template <class T>
    struct CUDA : public XPU
    {
        // @TODO implement cuda allocator
    };

    /**
     * @brief A generic memory buffer
     * @TODO make this cuda compatable.
     *
     * @tparam T datatype
     * @tparam Alloc memory allocator for the buffer
     */
    template <template <class> class COMPUTE, class T>
    class Memory
    {
    public:
        using allocator_type = COMPUTE<T>::allocator_type;               // allocator type
        using value_type = T;                                            // datatype
        using reference = value_type &;                                  // reference type
        using const_reference = const value_type &;                      // const reference type
        using pointer = value_type *;                                    // pointer type
        using size_type = size_t;                                        // size type
        using ptr = typename std::shared_ptr<Memory<COMPUTE, T>>;        // shared pointer type
        using unique_ptr = typename std::unique_ptr<Memory<COMPUTE, T>>; // unique pointer type

        /**
         * @brief Construct a new Memory object
         *
         */
        Memory() : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0) {}

        /**
         * @brief Construct a new Memory object
         *
         * @param num_elements number of elements in the memory
         */
        Memory(const size_t num_elements) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0)
        {
            reserve(num_elements);
            // resize(num_elements);
        }

        /**
         * @brief Construct a new Memory object
         *
         * @param scalar initialize memory with scalar value
         * @param num_elements number of elements in the memory
         */
        Memory(const T scalar, const size_t num_elements) : m_alloc()
        {
            // resize(num_elements);
            reserve(num_elements);
            COMPUTE<T>::fill(m_data, m_size, scalar);
        }

        Memory(const std::vector<value_type> &values) : m_alloc(), m_data(nullptr), m_size(0), m_allocated_size(0)
        {
            reserve(values.size());
            COMPUTE<T>::copy(values.data(), m_data, values.size());
            m_size = values.size();
            m_allocated_size = values.size();
        }

        /**
         * @brief Copy Construct a new Memory object
         *
         * @param other
         */
        Memory(const Memory &other) : m_alloc()
        {
            reserve(other.size());
            COMPUTE<T>::copy(other.data(), m_data, other.size());
        }

        // /**
        //  * @brief Move Construct a new Memory object
        //  *
        //  * @param other
        //  */
        // Memory(Memory &&other) : m_alloc()
        // {
        //     reserve(other.size());

        // }

        // /**
        //  * @brief Copy assignment operator
        //  *
        //  * @param other
        //  * @return Memory&
        //  */
        // Memory &operator=(const Memory &other)
        // {
        //     reserve(other.size());
        //     m_data = other.data();
        // }

        // /**
        //  * @brief Move assignment operator
        //  *
        //  * @param other
        //  * @return Memory&
        //  */
        // Memory &operator=(Memory &&other)
        // {
        //     memory_ = Container_t(other.begin(), other.end());
        //     other = Memory();
        // }

        virtual ~Memory()
        {
            m_alloc.deallocate(m_data, m_allocated_size);
        }

        /**
         * @brief resizes the memory buffer
         *
         * @param num_elements
         */
        void resize(const size_t num_elements, value_type val = value_type())
        {
            // TODO need to copy data in resize.
            if (num_elements == m_allocated_size)
            {
                return;
            }
            if (num_elements > m_allocated_size)
            {
                auto temp = m_alloc.allocate(num_elements);
                if (m_data != nullptr)
                {
                    COMPUTE<T>::copy(m_data, temp, m_size);
                }
                COMPUTE<T>::fill(temp + m_size, num_elements - m_size, val);
                if (m_data != nullptr)
                {
                    m_alloc.deallocate(m_data, m_allocated_size);
                }
                m_data = temp;
                m_allocated_size = num_elements;
                m_size = num_elements;
            }
            else // num_elements less than m_size
            {
                m_size = num_elements;
            }
        }

        /**
         * @brief allocates memory for given number of elements
         *
         * @param num_elements
         */
        void reserve(const size_t num_elements)
        {
            if (num_elements > m_allocated_size)
            {
                m_data = m_alloc.allocate(num_elements);
                m_allocated_size = num_elements;
            }
            m_size = num_elements;
        }

        /**
         * @brief access memory at the given index
         *
         * @param index index to the element in the memory
         * @return T& reference to the element pointed by the index in the memory
         */
        auto operator[](size_t index) -> reference
        {
            if (index >= m_size)
            {
                throw std::runtime_error("Array index out of bound");
            }
            return COMPUTE<T>::get(m_data, index);
        }

        auto operator[](size_t index) const -> const_reference
        {
            if (index >= m_size)
            {
                throw std::runtime_error("Array index out of bound");
            }
            return COMPUTE<T>::get(m_data, index);
        }

        /**
         * @brief Current size of the memory
         *
         * @return size_t
         */
        auto size() const -> size_type
        {
            return m_size;
        }

        /**
         * @brief raw pointer to the memory data
         *
         * @return T* pointer
         */
        auto data() -> pointer { return m_data; }

        auto empty() -> bool
        {
            return size() == 0;
        }

        auto copyFrom(pointer data, size_t num_elements)
        {
            COMPUTE<T>::copy(data, m_data, num_elements);
        }

        // TODO: implement iterators.

    private:
        std::allocator<value_type> m_alloc;
        pointer m_data;
        size_type m_size;
        size_type m_allocated_size;
    };

}
#endif