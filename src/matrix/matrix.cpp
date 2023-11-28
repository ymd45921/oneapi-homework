#include <sycl/sycl.hpp>
#include <iostream>

constexpr size_t N = 1024;  // 矩阵大小

class MatrixMultiplication;

void matrix_multiply(const std::vector<float>& matA, const std::vector<float>& matB, std::vector<float>& matC) {

    // 初始化 SYCL 环境
    sycl::queue queue(sycl::default_selector{});

    // 分配设备内存
    sycl::buffer<float, 2> bufferA(matA.data(), sycl::range<2>(N, N));
    sycl::buffer<float, 2> bufferB(matB.data(), sycl::range<2>(N, N));
    sycl::buffer<float, 2> bufferC(matC.data(), sycl::range<2>(N, N));

    // 启动 kernel
    queue.submit([&](sycl::handler& cgh) {
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<MatrixMultiplication>(sycl::range<2>(N, N), [=](sycl::item<2> item) {
            size_t row = item.get_id(0);
            size_t col = item.get_id(1);

            float sum = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                sum += accessorA[{row, i}] * accessorB[{i, col}];
            }

            accessorC[item] = sum;
        });
    });

    // 等待计算完成
    queue.wait();
}

int main() {
    // 生成随机矩阵
    std::vector<float> matA(N * N, 1.0f);
    std::vector<float> matB(N * N, 2.0f);
    std::vector<float> matC(N * N, 0.0f);

    // 进行矩阵乘法
    matrix_multiply(matA, matB, matC);

    // 打印结果
    #ifdef DEBUG
        freopen("result.log", "w", stdout);
    #endif
    std::cout << "Result Matrix:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << matC[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
