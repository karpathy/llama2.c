#include <iostream>
#include <llama2cpp/tensor.hpp>

int main(int argv, char **argc)
{
    std::cout << "Testing tensors" << std::endl;
    llama2cpp::Shape shape = {2, 3};
    // std::cout << shape(2) << std::endl;
    std::vector data = {1, 2, 3, 4, 5, 6};
    llama2cpp::Tensor<llama2cpp::CPU, int, 1> tensor(shape, data);
    std::cout << "print\n";
    for (auto i = 0; i < 2; ++i)
    {
        for (auto j = 0; j < 3; ++j)
            std::cout << tensor(i,j) << " ";
        std::cout << std::endl;
    }
}