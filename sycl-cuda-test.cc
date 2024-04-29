
#include <iostream>
#include <array>
#include <span>

#include <sycl/sycl.hpp>

int main() {
    for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        std::cout << device.get_info<sycl::info::device::name>() << std::endl;
    }
    auto dat = std::array<double, 3>{1,2,3};
    { 
        auto buf = sycl::buffer<double, 1>{dat.data(), 3};
        auto que = sycl::queue();
        que.submit([&](auto& h) noexcept {
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            h.parallel_for({dat.size()}, [=](auto idx) noexcept {
                acc[idx] *= 3.0;
            });
        });
    }
    for (auto item : dat) {
        std::cout << item << std::endl;
    }
    return 0;
}
