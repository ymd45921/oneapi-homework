#include <sycl/sycl.hpp>
#include <iostream>
#include <my.hpp>

constexpr auto matrix_unit_size = 128;
constexpr auto matrix_size = matrix_unit_size << 3;
constexpr auto M = matrix_size >> 3;
constexpr auto N = matrix_size >> 2;
constexpr auto P = matrix_size >> 1;       

 
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