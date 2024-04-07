#include <iostream>
#include <llama2cpp/tensor.hpp>
#include <llama2cpp/ops.hpp>
#include "gtest/gtest.h"

TEST(testTensor, creation)
{
    // EXPECT_EQ(1000, cubic(10));
    llama2cpp::Shape shape = {2, 3};
    std::cout << shape << std::endl;
    std::vector data = {1, 2, 3, 4, 5, 6};
    llama2cpp::Tensor<llama2cpp::CPU, int> tensor(shape, data);
    for (auto i = 0; i < 2; ++i)
    {
        for (auto j = 0; j < 3; ++j)
        {
            EXPECT_EQ(data[i * 3 + j], tensor(i, j));
        }
    }
    // for (auto i = 0; i < 2; ++i)
    // {
    //     for (auto j = 0; j < 3; ++j)
    //         std::cout << tensor(i, j) << " ";
    //     std::cout << std::endl;
    // }
}

TEST(testTensor, matmul)
{
    std::vector<float> w_data = {1, 2, 3, 4, 5, 6};
    llama2cpp::Tensor<llama2cpp::CPU, float> w(llama2cpp::Shape(2UL, 3UL), w_data);

    std::vector<float> x_data = {7, 8, 9};
    llama2cpp::Tensor<llama2cpp::CPU, float> x(llama2cpp::Shape(3UL), x_data);

    llama2cpp::Tensor<llama2cpp::CPU, float> out(llama2cpp::Shape(2UL));

    llama2cpp::matmul(out, x, w);
    // 7*1+8*2+9*3 = 50
    // 7*4+8*5+9*6 = 122
    EXPECT_EQ(out(0), 50.0F);
    EXPECT_EQ(out(1), 122.0F);
}