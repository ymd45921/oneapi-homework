#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

#include "my.hpp"

// 简单的图像卷积核函数
void convolution(const std::vector<float>& input, std::vector<float>& output,
                 const std::vector<float>& kernel, int width, int height, int kernelSize) {
    // 初始化 SYCL 环境
    sycl::queue queue(sycl::default_selector_v);

    // 分配设备内存
    sycl::buffer<float, 2> bufferInput(input.data(), sycl::range<2>(width, height));
    sycl::buffer<float, 2> bufferOutput(output.data(), sycl::range<2>(width, height));
    sycl::buffer<float, 1> bufferKernel(kernel.data(), sycl::range<1>(kernelSize));

    // 启动 kernel
    queue.submit([&](sycl::handler& cgh) {
        auto accessorInput = bufferInput.get_access<sycl::access::mode::read>(cgh);
        auto accessorOutput = bufferOutput.get_access<sycl::access::mode::write>(cgh);
        auto accessorKernel = bufferKernel.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<class ConvolutionKernel>(sycl::range<2>(width, height), [=](sycl::item<2> item) {
            int x = item.get_id(0);
            int y = item.get_id(1);

            float sum = 0.0f;
            for (int i = 0; i < kernelSize; ++i) {
                for (int j = 0; j < kernelSize; ++j) {
                    int inputX = x + i - kernelSize / 2;
                    int inputY = y + j - kernelSize / 2;

                    if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                        sum += accessorInput[{(unsigned)inputX, (unsigned)inputY}] * accessorKernel[i * kernelSize + j];
                    }
                }
            }

            accessorOutput[item] = sum;
        });
    });

    // 等待计算完成
    queue.wait();
}

int main() {
    // 图像参数
    const int width = 4;
    const int height = 4;

    // 输入图像
    std::vector<float> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    // 卷积核
    const int kernelSize = 3;
    std::vector<float> kernel = {
        1.0f, 2.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        -1.0f, -2.0f, -1.0f
    };

    // 输出图像
    std::vector<float> output(width * height, 0.0f);

    // 执行并行卷积
    convolution(input, output, kernel, width, height, kernelSize);

    // 打印结果
    std::cout << "Input Image:" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << input[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nConvolution Kernel:" << std::endl;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            std::cout << kernel[i * kernelSize + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nOutput Image:" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << output[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
