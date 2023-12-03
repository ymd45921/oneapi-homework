#include <sycl/sycl.hpp>
#include <iostream>

#include "my.hpp"
#include "my/mat.hpp"

constexpr auto matrix_unit_size = 256;
constexpr auto matrix_size = matrix_unit_size << 3;
constexpr auto M = matrix_size >> 3;
constexpr auto N = matrix_size >> 2;
constexpr auto P = matrix_size >> 1;       

double kernel(sycl::queue &q, sycl::buffer<float, 2> &a_buf, sycl::buffer<float, 2> &b_buf, 
            sycl::buffer<float, 2> &c_buf) {
    auto mm = q.submit([&](sycl::handler &h) {
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

        mm.wait();
        auto start = mm.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = mm.get_profiling_info<sycl::info::event_profiling::command_end>();
        return (end - start) * 1e-6;
}
 
signed main() {

    float a_host[M][N], b_host[N][P], c_host[M][P];
    my::rand<float> rand_float(0, 1);
    random_matrix(a_host, rand_float);
    random_matrix(b_host, rand_float);

    float (*c_out)[P] = new float[M][P], c_dev[M][P];
    my::print_platforms();

    double kernel_duration = 0;
    try {
        sycl::queue q(my::device_selector("Intel(R)"), my::prop_list);
        std::cout << "Running on device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";

        sycl::buffer<float, 2> a_buf(sycl::range(M, N));
        sycl::buffer<float, 2> b_buf(sycl::range(N, P));
        sycl::buffer c_buf((float *)c_out, sycl::range(M, P));
        std::cout << "Problem size: " << "c[" << M << "][" << P << "] = a[" 
                  << M << "][" << N << "] * b[" << N << "][" << P << "]\n";

        q.submit([&](sycl::handler &h) {
            sycl::accessor a(a_buf, h, sycl::write_only);
            auto p_a_host = (float *)a_host;
            auto offset = my::get_offset(a_host);
            h.parallel_for(sycl::range(M, N), [=](sycl::id<2> index) {
                a[index] = p_a_host[offset(index[0], index[1])];
            });
        });
        q.submit([&](sycl::handler &h) {
            sycl::accessor b(b_buf, h, sycl::write_only);
            auto p_b_host = (float *)b_host;
            auto offset = my::get_offset(b_host);
            h.parallel_for(sycl::range(N, P), [=](sycl::id<2> index) {
                b[index] = p_b_host[offset(index[0], index[1])];
            });
        });

        auto start = std::chrono::high_resolution_clock::now();
        kernel_duration = kernel(q, a_buf, b_buf, c_buf);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Device duration: " << duration << " ms\n";
    } catch (sycl::exception const &e) {
        std::cout << "An exception is caught for matrix multiplication.\n";
        std::terminate();
    }
    std::cout << "Kernel duration: " << kernel_duration << " ms\n" << std::endl;

    std::cout << "Running on host...\n";
    auto host_start = std::chrono::high_resolution_clock::now();
    my::matrix_multiply(a_host, b_host, c_host);
    auto host_end = std::chrono::high_resolution_clock::now();
    auto host_duration = std::chrono::duration_cast<std::chrono::milliseconds>(host_end - host_start).count();
    std::cout << "Host duration: " << host_duration << " ms\n" << std::endl;

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