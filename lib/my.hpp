#ifndef OneAPI_Homework_my_hpp
#define OneAPI_Homework_my_hpp
#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <random>
#include <limits>

namespace my {

    template <typename T>
    struct rand {

        // 使用 C++11 的随机数引擎生成随机数
        std::default_random_engine e;
        std::uniform_real_distribution<T> u;

        rand(T min, T max) : u(min, max) {}

        T operator()() { return u(e); }
    };

    template <typename T>
    struct equal {

        T tolerance;

        equal(T eps = std::numeric_limits<T>::epsilon()) : tolerance(eps) {}

        bool operator()(T a, T b) { return std::abs(a - b) < tolerance; }
    };

    // 使用上面的随机数生成器生成 axb 的随机矩阵，a 和 b 的大小由模板参数指定
    template <typename T, int a, int b>
    void random_matrix(T (&matrix)[a][b], rand<T> &rand) {
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++)
                matrix[i][j] = rand();
        }
    }

    // Host 暴力运算矩阵乘法
    template <typename T, int A, int B, int C>
    void matrix_multiply(const T (&a)[A][B], const T (&b)[B][C], T (&c)[A][C]) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < C; j++) {
                T sum = 0;
                for (int k = 0; k < B; k++)
                    sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        }
    }

    // Host 暴力运算矩阵减法
    template <typename T, int A, int B>
    void matrix_subtract(const T (&a)[A][B], const T (&b)[A][B], T (&c)[A][B]) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < B; j++)
                c[i][j] = a[i][j] - b[i][j];
        }
    }

    // 识别 C 二重数组的大小，并返回用来计算偏移量的函数
    template <typename T, int A, int B>
    constexpr auto get_offset(T (&)[A][B]) {
        return [=](int i, int j) { return i * B + j; };
    }

    // 使用上面的浮点数 equal 函数检查两个矩阵是否相等
    template <typename T, int A, int B>
    bool check_matrix_equal(const T (&a)[A][B], const T (&b)[A][B], equal<T> &eq) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < B; j++)
                if (!eq(a[i][j], b[i][j]))
                    return false;
        }
        return true;
    }

    // 将矩阵输出到文件
    template <typename T, int A, int B>
    void print_matrix(std::ostream &os, const T (&matrix)[A][B]) {
        for (int i = 0; i < A; i++) {
            for (int j = 0; j < B; j++)
                os << matrix[i][j] << " ";
            os << "\n";
        }
    }

    struct device_info {
        std::string name;
        sycl::device device;
    };

    struct platform_info {
        std::string name;
        sycl::platform platform;
        std::vector<device_info> devices;
    };

    // 查询所有可以用来创建 sycl 队列的设备
    std::vector<platform_info> get_devices() {
        auto platforms = sycl::platform::get_platforms();
        std::vector<platform_info> system(platforms.size()); 
        std::transform(platforms.begin(), platforms.end(), system.begin(),
                       [](sycl::platform &p) { 
                            auto name = p.get_info<sycl::info::platform::name>();
                            auto _devices = p.get_devices();
                            std::vector<device_info> devices(_devices.size());
                            std::transform(_devices.begin(), _devices.end(), devices.begin(),
                                           [](sycl::device &d) {
                                                return device_info{d.get_info<sycl::info::device::name>(), d};
                                           });
                            return platform_info{name, p, devices};
                       });
        return system;
    }

    // 将设备信息输出到流
    std::ostream &operator<<(std::ostream &os, const platform_info &info) {
        os << "Platform: " << info.name << "\n";
        for (auto &device : info.devices)
            os << "    Device: " << device.name << "\n";
        return os;
    }
}

#endif /* OneAPI_Homework_my_hpp */