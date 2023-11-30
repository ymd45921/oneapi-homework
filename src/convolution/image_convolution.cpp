#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

#include "my.hpp"

// #define coordinate_fix

template <typename T, int N>
my::image_data_rgba host_convolution(int width, int height, const my::image_data_rgba &input, 
                                     const my::kernel<T, N>& kernel, bool normalize = true) {
    my::image_data_rgba output(width * height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y){
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
            float sum_weight = 0.0f;
            const auto kernel_size = kernel.get_size(), kernel_offset = kernel_size / 2;
            auto id = y * width + x;
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int inputX = x + i - kernel_offset;
                    int inputY = y + j - kernel_offset;
                    int inputID = inputY * width + inputX;

                    if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                        auto weight = kernel.data[i * kernel_size + j];
                        sum_r += input[inputID].r * weight;
                        sum_g += input[inputID].g * weight;
                        sum_b += input[inputID].b * weight;
                        sum_a += input[inputID].a * weight;
                        sum_weight += weight;
                    }
                }
            }
            if (normalize)
                sum_r /= sum_weight, sum_g /= sum_weight, sum_b /= sum_weight, sum_a /= sum_weight;
            output[id] = my::make_pixel_rgba(sum_r, sum_g, sum_b, sum_a);
        #ifdef coordinate_fix
            output[id] = my::make_pixel_rgba(x * 255.f / width, y * 255.f / height, 0.f, 255.f);
        #endif
        }
    }
    return output;
}

int main() {
    // 图像参数
    // todo: 使用 sycl 提供的图像类和 host 图像类
    constexpr auto filename = workspace_root "img/IMG_2881.JPG";
    my::image img(filename, my::image::channel::rgba);
    if (img.get_channels() != my::image::channel::rgba) {
        std::cout << "Warning: image channel is not rgba" << std::endl;
    } else {
        std::cout << "Image channel is rgba" << std::endl;
    }
    auto width = img.get_width(), height = img.get_height();
    // hint: 已经从 stbi 布局转换为二维数组布局
    auto input = img.get_data_rgba();
    std::cout << "Image size: " << width << " * " << height << std::endl;

    // 卷积核
    // my::gaussian_kernel<float, 19> kernel;
    // my::kernel<float, 11> kernel(1.f);
    my::sharpen_kernel<float, 3> kernel(12);
    kernel.normalize();

    // 输出图像
    my::image_data_rgba output(width * height);

    // 打印信息
    std::cout << "\nInput Image:" << std::endl;
    std::cout << "  Path: " << filename << std::endl;
    std::cout << "  Width: " << width << std::endl;
    std::cout << "  Height: " << height << std::endl;
    std::cout << "  Channels: " << (int)img.get_channels() << std::endl;
    using my::operator<<;
    std::cout << "\nConvolution Kernel:" << std::endl;
    std::cout << "  Size: " << kernel.get_size() << std::endl;
    if (kernel.get_size() < 10) std::cout << kernel;
    else std::cout << "  Too large to print." << std::endl;

    auto start_device_timer = std::chrono::steady_clock::now();
    // 执行并行卷积
    try {
        sycl::queue queue(sycl::default_selector_v);
        std::cout << "\nRunning on device: "
                  << queue.get_device().get_info<sycl::info::device::name>() << "\n";
        
        // hint: sycl::buffer<T, 2> 等同于 T[][]；而 wxh 的图像数据应当使用二维数组 [h][w] 表示
        sycl::buffer<my::pixel_rgba, 2> buffer_input(input.data(), sycl::range<2>(height, width));
        sycl::buffer<my::pixel_rgba, 2> buffer_output(output.data(), sycl::range<2>(height, width));
        sycl::buffer<float, 2> buffer_kernel(kernel.get_data(), sycl::range<2>(kernel.get_size(), kernel.get_size()));

        queue.submit([&](sycl::handler& cgh) {
            auto accessor_input = buffer_input.get_access<sycl::access::mode::read>(cgh);
            auto accessor_output = buffer_output.get_access<sycl::access::mode::write>(cgh);
            auto accessor_kernel = buffer_kernel.get_access<sycl::access::mode::read>(cgh);

            // hint: sycl::handle::parallel_for 的第一个类型参数是用户指定的 kernel 名称，程序内需要保证唯一
            cgh.parallel_for<class ConvolutionKernel>(sycl::range<2>(height, width), [=](sycl::item<2> item) {
                
                int y = item.get_id(0);
                int x = item.get_id(1);

                float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f, sum_a = 0.0f;
                float sum_weight = 0.0f;
                const auto kernel_size = kernel.get_size(), kernel_offset = kernel_size / 2;
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int inputX = x + i - kernel_offset;
                        int inputY = y + j - kernel_offset;
                        auto inputCoord = sycl::id<2>{(unsigned)inputY, (unsigned)inputX};

                        if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                            auto weight = accessor_kernel[{(unsigned)i, (unsigned)j}];
                            sum_r += accessor_input[inputCoord].r * weight;
                            sum_g += accessor_input[inputCoord].g * weight;
                            sum_b += accessor_input[inputCoord].b * weight;
                            sum_a += accessor_input[inputCoord].a * weight;
                            sum_weight += weight;
                        }
                    }
                }
                sum_r /= sum_weight, sum_g /= sum_weight, sum_b /= sum_weight, sum_a /= sum_weight;
                accessor_output[item] = my::make_pixel_rgba(sum_r, sum_g, sum_b, sum_a);
            #ifdef coordinate_fix
                accessor_output[item] = my::make_pixel_rgba((float)x / width * 255.f, (float)y / height * 255.f, 0.f, 255.f);
            #endif
            });
        });
        queue.wait();

    } catch (sycl::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    auto end_device_timer = std::chrono::steady_clock::now();

    // 打印结果
    my::image out_img(output.data(), width, height);
    constexpr auto out_file = "out.png";
    out_img.save_png(out_file);
    std::cout << "\nOutput Image:" << std::endl;
    std::cout << "  Path: " << out_file << std::endl;
    std::cout << "  Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_device_timer - start_device_timer).count() << "ms" << std::endl;

    std::cout << "\nHost Convolution processing..." << std::endl;
    auto start_host_timer = std::chrono::steady_clock::now();
    auto host_output = host_convolution(width, height, input, kernel, true);
    auto end_host_timer = std::chrono::steady_clock::now();
    my::image host_out_img(host_output.data(), width, height);
    constexpr auto host_out_file = "host_out.png";
    host_out_img.save_png(host_out_file);
    std::cout << "  Path: " << host_out_file << std::endl;
    std::cout << "  Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_host_timer - start_host_timer).count() << "ms" << std::endl;

    using my::operator==;
    std::cout << std::endl;
    if (std::all_of(output.begin(), output.end(), 
                    [&](auto &a) { return a == host_output[&a - output.data()]; })) {
        std::cout << "Host Convolution result is the same as Device Convolution result." << std::endl;
    } else {
        std::cout << "Host Convolution result is not the same as Device Convolution result." << std::endl;
    }
    return 0;
}
