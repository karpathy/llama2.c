#include <iostream>
#include <llama2cpp/transformer/ops.hpp>
#include <llama2cpp/transformer/tensor.hpp>

#include "gtest/gtest.h"

TEST(testTensor, creation) {
    // EXPECT_EQ(1000, cubic(10));
    llama2cpp::Shape shape = {2, 3};
    std::cout << shape << std::endl;
    std::vector data = {1, 2, 3, 4, 5, 6};
    llama2cpp::Tensor<llama2cpp::CPU, int> tensor(shape, data);
    for (auto i = 0; i < 2; ++i) {
        for (auto j = 0; j < 3; ++j) {
            EXPECT_EQ(data[i * 3 + j], tensor(i, j));
        }
    }
    // for (auto i = 0; i < 2; ++i)
    // {
    //     for (auto j = 0; j < 3; ++j)
    //         std::cout << tensor(i, j) << " ";
    //     std::cout << std::endl;
    // }

    llama2cpp::Shape other_shape = {2, 3};
    EXPECT_EQ(shape, other_shape);
}

TEST(testTensor, matmul) {
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

    llama2cpp::Shape test_shape = {2, 5};
    std::cout << test_shape(1, 0) << std::endl;
}

TEST(testTensor, slice) {
    llama2cpp::Shape shape0(2UL);
    std::vector<float> w_data0(shape0.size());
    w_data0[0] = 1;
    w_data0[1] = 2;
    llama2cpp::Tensor<llama2cpp::CPU, float> tensor0(shape0, w_data0);
    EXPECT_EQ(tensor0(0), 1);
    EXPECT_EQ(tensor0(1), 2);

    llama2cpp::Shape shape(2UL, 3UL, 5UL);  // CHW
    std::vector<float> w_data(shape.size());
    for (auto i = 0; i < shape.size(); ++i) {
        w_data[i] = i;
    }
    llama2cpp::Tensor<llama2cpp::CPU, float> tensor(shape, w_data);
    EXPECT_EQ(tensor(0, 1, 0), 5);
    std::cout << "Original shape: " << shape << std::endl;
    std::cout << "shape.slice(0): " << shape.slice(0) << ":" << shape.slice(0).numElements() << std::endl;
    std::cout << "shape.slice(1): " << shape.slice(1) << ":" << shape.slice(1).numElements() << std::endl;
    std::cout << "shape.slice(0, 0): " << shape.slice(0, 0) << ":" << shape.slice(0, 0).numElements() << std::endl;
    std::cout << "shape.slice(0, 1): " << shape.slice(0, 1) << ":" << shape.slice(0, 1).numElements() << std::endl;
    std::cout << "shape.slice(0, 0, 0): " << shape.slice(0, 0, 0) << ":" << shape.slice(0, 0, 0).numElements() << std::endl;

    std::cout << "-----------------------------\n";
    // Current Shape = {2,3,5}
    // Current Stride = {15,5,1}

    //                                           C  -> {H,W}
    llama2cpp::TensorView<float> s1 = tensor.slice(0);  // TensorView(ptr+0*(15), Shape(3,5)) stride {5,1}
    llama2cpp::TensorView<float> s2 = tensor.slice(1);  // TensorView(ptr+1*(15), Shape(3,5)) stride {5,1}

    //                                           C, H  -> {W}
    llama2cpp::TensorView<float> s3 = tensor.slice(1, 0);  // TensorView(ptr+1*(15)+0*(5), Shape(5)) stride {1}
    llama2cpp::TensorView<float> s4 = tensor.slice(1, 1);  // TensorView(ptr+1*(15)+1*(5), Shape(5)) stride {1}

    //                                           C, H, W  -> {}
    llama2cpp::TensorView<float> s5 = tensor.slice(1, 1, 2);  // TensorView(ptr+1*(15)+1*(5)+2*(1), Shape(1)) stride {}

    std::cout << "-----------------------------\n";
    std::cout << "tensor.shape() :" << tensor.shape() << std::endl;
    std::cout << "tensor.slice(0) :" << s1.shape() << std::endl;
    std::cout << "tensor.slice(1) :" << s2.shape() << std::endl;
    std::cout << "tensor.slice(1,0) :" << s3.shape() << std::endl;
    std::cout << "tensor.slice(1,1) :" << s4.shape() << std::endl;
    std::cout << "tensor.slice(1,1,2) :" << s5.shape() << std::endl;
}