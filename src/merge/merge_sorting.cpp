#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

#include "my.hpp"

// Host 端进行的归并排序
void host_merge_sort(std::vector<int>& data, std::vector<int> &tmp, size_t left, size_t right) {
    if (left >= right - 1) return;
    size_t mid = left + (right - left) / 2;
    host_merge_sort(data, tmp, left, mid);
    host_merge_sort(data, tmp, mid, right);
    size_t i = left, j = mid, k = left;
    while (i < mid && j < right) {
        tmp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];
    }
    while (i < mid) {
        tmp[k++] = data[i++];
    }
    while (j < right) {
        tmp[k++] = data[j++];
    }
    for (k = left; k < right; ++k) {
        data[k] = tmp[k];
    }
}

template <typename T>
double parallel_merge_sort(sycl::queue& queue, sycl::buffer<T>& buffer, size_t size) {
    double kernel_duration = 0;
    for (auto stride = 2; stride <= size; stride *= 2) {
        auto event = queue.submit([&](sycl::handler& cgh) {
            auto data = buffer.template get_access<sycl::access::mode::read_write>(cgh);
            auto groups = (size + stride - 1) / stride;
            cgh.parallel_for(sycl::range<1>{groups}, [=](sycl::item<1> item) {
                size_t idx = item.get_id(0);
                size_t left = idx * stride;
                size_t mid = std::min(left + stride / 2, size);
                size_t right = std::min(left + stride, size);

                std::vector<T> tmp(right - left);
                size_t i = left, j = mid, k = 0;

                while (i < mid && j < right) {
                    tmp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];
                }
                while (i < mid) {
                    tmp[k++] = data[i++];
                }
                while (j < right) {
                    tmp[k++] = data[j++];
                }
                for (k = 0, i = left; i < right; ++i, ++k) {
                    data[i] = tmp[k];
                }
            });
        });
        event.wait();
        auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        kernel_duration += (end - start) * 1e-6;
    }
    return kernel_duration;
}

int main() {

    // Initialize data
    constexpr auto size = 1 << 14;
    auto data = my::random_vector<int>(size, my::rand(0, size * 2));
    std::ofstream fout("merge_sorting.txt");
    for (auto& x : data) { fout << x << " "; }
    fout << std::endl;
    std::vector<int> data_copy(size), tmp(size), data_host(size);
    std::copy(data.begin(), data.end(), data_copy.begin());
    std::copy(data.begin(), data.end(), data_host.begin());

    try {
        sycl::queue queue(sycl::default_selector_v, my::prop_list);
        sycl::buffer<int> buffer(data.data(), sycl::range<1>(size));
        auto device_start = std::chrono::high_resolution_clock::now();
        auto kernel_duration = praallel_merge_sort(queue, buffer, size);
        auto device_end = std::chrono::high_resolution_clock::now();
        auto device_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(device_end - device_start).count() * 1e-6;
        std::cout << "Device duration: " << device_duration << " ms\n";
        std::cout << "Kernel duration: " << kernel_duration << " ms\n";
    } catch (sycl::exception& e) {
        std::cout << e.what() << std::endl;
    }

    auto host_start = std::chrono::high_resolution_clock::now();
    host_merge_sort(data_host, tmp, 0, size);
    auto host_end = std::chrono::high_resolution_clock::now();
    auto host_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(host_end - host_start).count() * 1e-6;
    std::cout << "Host duration: " << host_duration << " ms\n";

    auto host_stl_start = std::chrono::high_resolution_clock::now();
    std::stable_sort(data_copy.begin(), data_copy.end());
    auto host_stl_end = std::chrono::high_resolution_clock::now();
    auto host_stl_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(host_stl_end - host_stl_start).count() * 1e-6;
    std::cout << "Host STL duration: " << host_stl_duration << " ms\n";

    std::cout << "Result: " << (data == data_copy ? "Correct" : "Wrong") << std::endl;
    for (auto& x : data) { fout << x << " "; }
    fout << std::endl;
    for (auto& x : data_copy) { fout << x << " "; }
    fout.close();

    return 0;
}
