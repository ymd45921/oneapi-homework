#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <random>
#include <limits>

constexpr auto matrix_unit_size = 128;
constexpr auto matrix_size = matrix_unit_size << 3;
constexpr auto M = matrix_size >> 3;
constexpr auto N = matrix_size >> 2;
constexpr auto P = matrix_size >> 1;       

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
 
signed main() {

    float a_host[M][N], b_host[N][P], c_host[M][P];
    my::rand<float> rand_float(0, 1);
    random_matrix(a_host, rand_float);
    random_matrix(b_host, rand_float);

    float (*c_out)[P] = new float[M][P], c_dev[M][P];

    auto info = my::get_devices();
    std::cout << "Found " << info.size() << " platforms.\n";
    // 输出所有可用的设备信息到文件
    std::ofstream ofs_device("devices.txt");
    for (auto &platform : info) {
        using my::operator<<;
        ofs_device << platform;
        std::cout << platform;
    }

    try {
        sycl::queue q(sycl::default_selector_v);
        std::cout << "Running on device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";

        sycl::buffer<float, 2> a_buf(sycl::range(M, N));
        sycl::buffer<float, 2> b_buf(sycl::range(N, P));
        sycl::buffer c_buf((float *)c_out, sycl::range(M, P));
        std::cout << "Problem size: " << "c[" << M << "][" << P << "] = a[" 
                  << M << "][" << N << "] * b[" << N << "][" << P << "]\n";

        q.submit([&](sycl::handler &h) {
            // 获得了对 Device 上的 a_buf 的访问权
            sycl::accessor a(a_buf, h, sycl::write_only);
            auto p_a_host = (float *)a_host;
            auto offset = my::get_offset(a_host);
            h.parallel_for(sycl::range(M, N), [=](sycl::id<2> index) {
                a[index] = p_a_host[offset(index[0], index[1])];
            });
        });
        q.submit([&](sycl::handler &h) {
            // 获得了对 Device 上的 b_buf 的访问权
            sycl::accessor b(b_buf, h, sycl::write_only);
            auto p_b_host = (float *)b_host;
            auto offset = my::get_offset(b_host);
            h.parallel_for(sycl::range(N, P), [=](sycl::id<2> index) {
                b[index] = p_b_host[offset(index[0], index[1])];
            });
        });

        q.submit([&](sycl::handler &h) {
            sycl::accessor a(a_buf, h, sycl::read_only);
            sycl::accessor b(b_buf, h, sycl::read_only);
            sycl::accessor c(c_buf, h, sycl::write_only, sycl::no_init);

            int a_width = a.get_range()[1];

            h.parallel_for(sycl::range(M, P), [=](sycl::id<2> index) {
                float sum = 0;
                auto row = index[0], col = index[1];
                for (int i = 0; i < a_width; i++) {
                    sum += a[row][i] * b[i][col];
                }
                c[index] = sum;
            });
        });
    } catch (sycl::exception const &e) {
        std::cout << "An exception is caught for matrix multiplication.\n";
        std::terminate();
    }

    std::cout << "Running on host...\n";
    my::matrix_multiply(a_host, b_host, c_host);
    std::memcpy(c_dev, c_out, sizeof(c_dev));

    my::equal eq(1e-4f);
    if (my::check_matrix_equal(c_host, c_dev, eq)) {
        std::cout << "Matrix multiplication successfully completed on device.\n";
    } else {
        std::cout << "Matrix multiplication failed on device.\n";
        std::ofstream ofs("matrix.txt");
        ofs << "Device output:\n";
        my::print_matrix(ofs, c_dev);
        ofs << "Host output:\n";
        my::print_matrix(ofs, c_host);
        float c_diff[M][P];
        my::matrix_subtract(c_host, c_dev, c_diff);
        ofs << "Difference:\n";
        my::print_matrix(ofs, c_diff);
        ofs.close();
    }
    delete[] c_out;
    return 0;
}