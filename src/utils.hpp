#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"
#include "build_info.hpp"
#include "debugbreak.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iterator>
#include <vector>
#include <algorithm>


template <typename T>
constexpr uint32_t to_uint32(T n)
{
    return static_cast<uint32_t>(n);
}


[[noreturn]] inline void fatal(std::string_view msg)
{
    std::cerr << msg << "\n";
#ifndef NDEBUG
    debug_break();
#endif
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
    ASSUME(input);
    std::vector<unsigned char> contents(std::istreambuf_iterator<char>(input.rdbuf()), {});

    auto sm = device.createShaderModuleUnique(vk::ShaderModuleCreateInfo{}
                                                  .setCodeSize(contents.size())
                                                  .setPCode(reinterpret_cast<const uint32_t *>(contents.data())));
    return std::move(sm);
}


// For now just a super-simple, non-scaling implementation.
class FramebufferCache
{
public:
    FramebufferCache(vk::Device device, size_t max_elem_count) : _device(device), _max_elem_count(max_elem_count) {}
    vk::Framebuffer get_or_create(const vk::FramebufferCreateInfo &ci)
    {
        ++_current_lookup_time;

        // Serve from cache
        for (auto &entry : _cache) {
            if (entry.key == ci) {
                entry.last_lookup_time = _current_lookup_time;
                return entry.value.get();
            }
        }
        // Create new element, cache it
        CachedElement new_entry;
        new_entry.key = ci;
        new_entry.value = _device.createFramebufferUnique(ci);
        vk::Framebuffer new_fb = new_entry.value.get();
        new_entry.last_lookup_time = _current_lookup_time;
        _cache.push_back(std::move(new_entry));

        // Purge oldest element if necessary
        if (_cache.size() > _max_elem_count) {
            auto iter =
                std::min_element(_cache.begin(), _cache.end(), [](const CachedElement &e1, const CachedElement &e2) {
                    return e1.last_lookup_time < e2.last_lookup_time;
                });
            _cache.erase(iter);
        }
        return new_fb;
    }

    size_t size() const { return _cache.size(); }

private:
    vk::Device _device;
    size_t _max_elem_count = 10;
    struct CachedElement
    {
        vk::FramebufferCreateInfo key;
        vk::UniqueFramebuffer value;
        uint64_t last_lookup_time = 0;
    };
    std::vector<CachedElement> _cache;
    uint64_t _current_lookup_time = 0;
};
