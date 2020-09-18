#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"
#include "build_info.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iterator>
#include <vector>


template <typename T>
constexpr uint32_t to_uint32(T n)
{
    return static_cast<uint32_t>(n);
}


[[noreturn]] inline void fatal(std::string_view msg)
{
    std::cerr << msg << "\n";
    std::exit(-1);
}


#define ASSUME(expr) (!!(expr) ? ((void)0) : fatal(#expr))


template <typename T>
constexpr T clamp(T value, T low, T high)
{
    return value < low ? low : (value > high ? high : value);
}


template <typename Container, typename WeightFunctor>
auto choose_best(const Container &candidates, WeightFunctor f)
{
    auto iter = std::begin(candidates);
    auto end_iter = std::end(candidates);
    int max_weight = std::numeric_limits<int>::min();
    auto best_one = *iter;
    while (iter != end_iter) {
        int weight = f(*iter);
        if (weight > max_weight) {
            max_weight = weight;
            best_one = *iter;
        }
        ++iter;
    }
    return best_one;
}


inline vk::UniqueShaderModule load_shader(vk::Device device, std::string_view path)
{
    std::ifstream input(path.data(), std::ios::binary);
    std::vector<unsigned char> contents(std::istreambuf_iterator<char>(input.rdbuf()), {});

    auto sm = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo{}
                                                  .setCodeSize(contents.size() / 4)
                                                  .setPCode(reinterpret_cast<const uint32_t *>(contents.data())));
    return std::move(sm);
}
