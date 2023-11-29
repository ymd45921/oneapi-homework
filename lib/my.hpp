#ifndef OneAPI_Homework_my_hpp
#define OneAPI_Homework_my_hpp
#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <limits>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

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

    union pixel_rgba {
        struct {
            unsigned char r, g, b, a;
        };
        unsigned char data[4];
    };

    std::ostream &operator<<(std::ostream &os, const pixel_rgba &pixel) {
        if (!os.dec) {  // 将 rgba 输出为 #RRGGBBAA 的形式
            os << "#";
            // 备份当前 os 的输出格式和填充字符
            auto flags = os.flags();
            auto fill = os.fill();
            os << std::hex << std::setw(2) << std::setfill('0');
            for (auto &c : pixel.data) os << static_cast<int>(c);
            // 恢复 os 的输出格式和填充字符
            os.flags(flags), os.fill(fill);
        } else {
            os << "rgba(" << static_cast<int>(pixel.r) << ", " << static_cast<int>(pixel.g) << ", "
               << static_cast<int>(pixel.b) << ", " << static_cast<double>(pixel.a) / 255 << ")";
        }
        return os;
    }

    // 定义一个 stb_image 的 RAII 包装类
    class image {
        int width, height, channels;
        unsigned char *data;

    public:
        enum class channel {
            undefined = 0,
            grey = 1,
            grey_alpha = 2,
            rgb = 3,
            rgba = 4
        };

        explicit image(const char *filename, channel req = channel::undefined) {
            data = stbi_load(filename, &width, &height, &channels, static_cast<int>(req));
            if (!data)
                throw std::runtime_error("Failed to load image");
        }

        ~image() { stbi_image_free(data); }

        int get_width() const { return width; }

        int get_height() const { return height; }

        channel get_channels() const { return static_cast<channel>(channels); }

        const unsigned char *get_raw() const { return data; }

        std::pair<std::shared_ptr<pixel_rgba[]>, std::size_t> get_rgba() const {
            auto size = width * height;
            std::shared_ptr<pixel_rgba[]> pixels(new pixel_rgba[size]);
            for (int i = 0; i < size; i++) {
                if (channels < 3) {
                    pixels[i].r = pixels[i].g = pixels[i].b = data[i * channels];
                    pixels[i].a = channels == 2 ? data[i * channels + 1] : 255;
                } else {
                    pixels[i].r = data[i * channels];
                    pixels[i].g = data[i * channels + 1];
                    pixels[i].b = data[i * channels + 2];
                    pixels[i].a = channels == 4 ? data[i * channels + 3] : 255;
                }
            }
            return {pixels, size};
        }
    };

    // 定义一个卷积核
    template <typename T, int size>
    struct kernel {
        T data[size * size];

        constexpr kernel() = default;

        constexpr kernel(std::initializer_list<T> list) {
            std::copy(list.begin(), list.end(), data);
        }

        // 用随机数填充卷积核
        void random() {
            rand<T> rand(-1, 1);
            std::generate(data, data + size * size, rand);
        }

        // 将卷积核输出到流
        friend std::ostream &operator<<(std::ostream &os, const kernel &k) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++)
                    os << k.data[i * size + j] << " ";
                os << "\n";
            }
            return os;
        }
    };

    // 定义一个高斯卷积核，使用模板元编程计算高斯卷积核的值
    template <typename T, int size>
    struct gaussian_kernel : kernel<T, size> {
        constexpr gaussian_kernel() {
            static_assert(size % 2 == 1, "size must be odd");
            static_assert(size >= 3, "size must be greater than or equal to 3");
            static_assert(size <= 7, "size must be less than or equal to 7");
            static_assert(std::is_floating_point<T>::value, "T must be floating point");
            static_assert(std::numeric_limits<T>::is_iec559, "T must be IEEE 754 floating point");

            constexpr auto sigma = (size - 1) / 6.0f;
            constexpr auto sigma2 = sigma * sigma;
            constexpr auto sigma4 = sigma2 * sigma2;
            constexpr auto sigma6 = sigma4 * sigma2;
            constexpr auto coeff = 1.0f / (2.0f * M_PI * sigma6);

            constexpr auto offset = size / 2;
            [[maybe_unused]] constexpr auto offset2 = offset * offset;

            constexpr auto get_offset = [](int i, int j) { return (i + offset) * size + j + offset; };

            constexpr auto get_value = [=](int i, int j) {
                auto x = i - offset;
                auto y = j - offset;
                return coeff * std::exp(-(x * x + y * y) / (2.0f * sigma2));
            };

            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++)
                    this->data[get_offset(i, j)] = get_value(i, j);
            }

            // 归一化
            auto sum = std::accumulate(this->data, this->data + size * size, 0.0f);
            std::transform(this->data, this->data + size * size, this->data,
                           [=](T v) { return v / sum; });
        }
    };
}

#endif /* OneAPI_Homework_my_hpp */