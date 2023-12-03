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
#include <optional>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace my {

    template <typename T>
    struct rand {

        std::default_random_engine e;
        std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>, std::uniform_int_distribution<T>> u;

        rand(T min, T max) : u(min, max) {}

        T operator()() { return u(e); }
    };

    template <typename T>
    int sgn(T val) { return (T(0) < val) - (val < T(0)); }

    template <typename T>
    struct equal {

        T tolerance;

        equal(T eps = std::numeric_limits<T>::epsilon()) : tolerance(eps) {}

        bool operator()(T a, T b) { return std::abs(a - b) < tolerance; }

        int sgn(T x) { return this->operator()(x, T(0)) ? 0 : my::sgn(x); }
    };

    // 使用上面的随机数生成器生成长度为 n 的随机向量；数字类型由模板参数指定
    template <typename T>
    std::vector<T> random_vector(int n, std::optional<rand<T>> rand = std::nullopt) {
        if (!rand.has_value()) {
            rand = my::rand<T>(-n, n);
        }
        std::vector<T> vector(n);
        std::generate(vector.begin(), vector.end(), *rand);
        return vector;
    }

    // 识别 C 二重数组的大小，并返回用来计算偏移量的函数
    template <typename T, int A, int B>
    constexpr auto get_offset(T (&)[A][B]) {
        return [=](int i, int j) { return i * B + j; };
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

    using image_data_rgba = std::vector<pixel_rgba>;

    bool operator==(const pixel_rgba &a, const pixel_rgba &b) {
        return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
    }

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

    pixel_rgba make_pixel_rgba(float r, float g, float b, float a = 255) {
        pixel_rgba pixel;
        pixel.r = static_cast<unsigned char>(std::clamp(std::round(r), 0.f, 255.f));
        pixel.g = static_cast<unsigned char>(std::clamp(std::round(g), 0.f, 255.f));
        pixel.b = static_cast<unsigned char>(std::clamp(std::round(b), 0.f, 255.f));
        pixel.a = static_cast<unsigned char>(std::clamp(std::round(a), 0.f, 255.f));   
        return pixel;
    }

    pixel_rgba make_pixel_rgba(int r, int g, int b, int a = 255) {
        pixel_rgba pixel;
        pixel.r = static_cast<unsigned char>(std::clamp(r, 0, 255));
        pixel.g = static_cast<unsigned char>(std::clamp(g, 0, 255));
        pixel.b = static_cast<unsigned char>(std::clamp(b, 0, 255));
        pixel.a = static_cast<unsigned char>(std::clamp(a, 0, 255));   
        return pixel;
    }

    // 定义一个 stb_image 的 RAII 包装类
    class image {

        int width, height, fact_channels;
        unsigned char *data;

    public:
        enum class channel {
            undefined = 0,
            grey = 1,
            grey_alpha = 2,
            rgb = 3,
            rgba = 4
        };

    private:
        channel data_channels;

    public:
        // hint: 即使指明了 req，stbi_load 返回的通道数仍然是实际的通道数
        // ? 即使指明了 req，stbi_load 也不一定会返回对应的通道数，因此需要使用 stbi__convert_format 进行转换
        explicit image(const char *filename, channel req = channel::undefined) {
            data = stbi_load(filename, &width, &height, &fact_channels, static_cast<int>(req));
            if (!data)
                throw std::runtime_error("Failed to load image");
            data_channels = req == channel::undefined ? static_cast<channel>(fact_channels) : req;
        }

        image(int width, int height, channel req = channel::rgba)
            : width(width), height(height), fact_channels(static_cast<int>(req)), data_channels(req) {
            if (req == channel::undefined)
                throw std::runtime_error("Invalid new image channels");
            data = (unsigned char *)stbi__malloc(width * height * fact_channels);
        }

        image(const pixel_rgba *pixels, int width, int height)
            : width(width), height(height), fact_channels(4), data_channels(channel::rgba) {
            data = (unsigned char *)stbi__malloc(width * height * fact_channels);
            memcpy(data, pixels, width * height * sizeof(pixel_rgba));
        }

        ~image() { stbi_image_free(data); }

        int get_width() const { return width; }

        int get_height() const { return height; }

        channel get_channels() const { return data_channels; }

        const unsigned char *get_raw() const { return data; }

        void save_png(const char *filename) const {
            int channels = static_cast<int>(data_channels);
            if (!stbi_write_png(filename, width, height, channels, data, width * channels))
                throw std::runtime_error("Failed to write image");
        }

        int get_index(int x, int y) const { return y * width + x; }

        int get_offset(int x, int y) const { return get_index(x, y) * (int)data_channels; }

        // hint: 很简单的道理 —— wxh 的图片其实有 h 行 w 列，写成二维数组是 [h][w]；sycl::range 也只是二维数组布局
        // 但是作为图片，我们更习惯于 [w][h] 的布局。wxh 仍然具有意义
        image_data_rgba get_data_rgba() const {
            auto size = width * height;
            auto channels = (int)data_channels;
            image_data_rgba pixels(size);
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    auto i = get_index(x, y), j = i;
                    if (channels < 3) {
                        pixels[j].r = pixels[i].g = pixels[i].b = data[i * channels];
                        pixels[j].a = channels == 2 ? data[i * channels + 1] : 255;
                    } else {
                        pixels[j].r = data[i * channels];
                        pixels[j].g = data[i * channels + 1];
                        pixels[j].b = data[i * channels + 2];
                        pixels[j].a = channels == 4 ? data[i * channels + 3] : 255;
                    }
                }
            }
            return pixels;
        }
    };

    // 定义一个卷积核
    template <typename T, int size>
    struct kernel {
        T data[size * size]{};

        constexpr kernel() = default;

        constexpr kernel(std::initializer_list<T> list) {
            std::fill(data, data + size * size, T(0));
            std::copy(list.begin(), list.end(), data);
        }

        constexpr kernel(T _fill) {
            std::fill(data, data + size * size, _fill);
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

        // 归一化卷积核，如果卷积核的和为 0，则不进行归一化
        void normalize() {
            auto sum = std::accumulate(data, data + size * size, 0.0f);
            if (sum != 0.0f)
                std::transform(data, data + size * size, data,
                               [=](T v) { return v / sum; });
        }

        void fill(T value) {
            std::fill(data, data + size * size, value);
        }

        constexpr int get_size() const { return size; }

        constexpr T *get_data() { return data; }
    };

    // 定义一个高斯卷积核，使用模板元编程计算高斯卷积核的值
    template <typename T, int size>
    struct gaussian_kernel : kernel<T, size> {
        explicit constexpr gaussian_kernel(const T sigma = (size - 1) / (T)6.0) {
            static_assert(size % 2 == 1, "size must be odd");
            static_assert(size >= 3, "size must be greater than or equal to 3");
            static_assert(std::is_floating_point<T>::value, "T must be floating point");
            static_assert(std::numeric_limits<T>::is_iec559, "T must be IEEE 754 floating point");

            const auto sigma2 = sigma * sigma;
            const auto sigma4 = sigma2 * sigma2;
            const auto sigma6 = sigma4 * sigma2;
            const auto coeff = 1.0f / (2.0f * M_PI * sigma6);

            const auto offset = size / 2;
            [[maybe_unused]] const auto offset2 = offset * offset;

            const auto get_offset = [](int i, int j) { return (i + offset) * size + j + offset; };

            const auto get_value = [=](int i, int j) {
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

    // 定义一个锐化卷积核，使用模板元编程计算锐化卷积核的值
    // info: 仅实现了 3x3 的拉普拉斯锐化卷积核
    template <typename T, int size>
    struct sharpen_kernel : kernel<T, size> {
        explicit constexpr sharpen_kernel(const T alpha = 1) {
            static_assert(size % 2 == 1, "size must be odd");
            static_assert(size >= 3, "size must be equal to 3");
            static_assert(std::is_floating_point<T>::value, "T must be floating point");
            static_assert(std::numeric_limits<T>::is_iec559, "T must be IEEE 754 floating point");

            const auto offset = size / 2;
            [[maybe_unused]] const auto offset2 = offset * offset;

            const auto get_offset = [](int i, int j) { return (i + offset) * size + j + offset; };

            const auto get_value = [=](int i, int j) {
                if (i == 0 && j == 0)
                    return (T)1 + alpha * 4;
                else if (i == 0 || j == 0)
                    return -alpha;
                else return (T)0;
            };

            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++)
                    this->data[get_offset(i, j)] = get_value(i, j);
            }
        }
    };

    // 尽量选择指定名称的设备
    class device_selector {
        // sycl::device_selector 的已经被 SYCL2020 弃用；无需再继承自 sycl::device_selector

        std::string name;

    public:
        explicit device_selector(const std::string &name) : name(name) {}

        int operator()(const sycl::device &device) const {  // override
            auto &&current = device.get_info<sycl::info::device::name>();
            if (current == name)
                return std::numeric_limits<int>::max();
            int rating = 0;
            if (current.find(name) != std::string::npos) rating += 1000;
            if (device.is_gpu()) rating += 100;
            if (device.is_cpu()) rating += 10;
            if (device.is_accelerator()) rating += 1;
            return rating;    
        }
    };

    void print_platforms(std::optional<std::string> filename = std::nullopt) {
        auto info = my::get_devices();
        std::ofstream ofs_device(filename.value_or("device.txt"));
        std::cout << "Found " << info.size() << " platforms.\n";
        for (auto &platform : info) {
                using my::operator<<;
                ofs_device << platform;
                std::cout << platform;
        }
        std::cout << std::endl;
    }

    const sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
}

#endif /* OneAPI_Homework_my_hpp */