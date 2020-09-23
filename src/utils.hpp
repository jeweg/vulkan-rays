#pragma once
#include "vulkan/vulkan.hpp"
#include "build_info.hpp"
#include "debugbreak.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iterator>
#include <vector>
#include <algorithm>
#include <sstream>

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


inline std::string format_properties_to_string(vk::PhysicalDevice phys_device, vk::Format format)
{
    vk::FormatProperties format_props = phys_device.getFormatProperties(format);

    std::array<std::ostringstream, 3> oss;
    std::array<bool, 3> oss_used = {false};
    std::array<VkFormatFeatureFlags, 3> features = {
        static_cast<VkFormatFeatureFlags>(format_props.linearTilingFeatures),
        static_cast<VkFormatFeatureFlags>(format_props.optimalTilingFeatures),
        static_cast<VkFormatFeatureFlags>(format_props.bufferFeatures)};

    bool first = true;
    for (const auto &feature_bit : {
             std::make_pair("VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT),
             std::make_pair("VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT", VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT),
             std::make_pair("VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT", VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT),
             std::make_pair("VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT", VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT),
             std::make_pair("VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT", VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT",
                 VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT),
             std::make_pair("VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT", VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT),
             std::make_pair("VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT", VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT", VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT", VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT),
             std::make_pair("VK_FORMAT_FEATURE_BLIT_SRC_BIT", VK_FORMAT_FEATURE_BLIT_SRC_BIT),
             std::make_pair("VK_FORMAT_FEATURE_BLIT_DST_BIT", VK_FORMAT_FEATURE_BLIT_DST_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT),
             std::make_pair("VK_FORMAT_FEATURE_TRANSFER_SRC_BIT", VK_FORMAT_FEATURE_TRANSFER_SRC_BIT),
             std::make_pair("VK_FORMAT_FEATURE_TRANSFER_DST_BIT", VK_FORMAT_FEATURE_TRANSFER_DST_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT", VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT),
             std::make_pair("VK_FORMAT_FEATURE_DISJOINT_BIT", VK_FORMAT_FEATURE_DISJOINT_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT", VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT),
             std::make_pair(
                 "VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG",
                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG),
             std::make_pair(
                 "VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR",
                 VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR),
             std::make_pair(
                 "VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT", VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT),
         }) {
        for (size_t i = 0; i < 3; ++i) {
            if (features[i] & feature_bit.second) {
                if (oss_used[i]) { oss[i] << " | "; }
                oss[i] << feature_bit.first;
                oss_used[i] = true;
            }
        }
    }

    return "linear tiling: " + oss[0].str() + "\noptimal tiling: " + oss[1].str() + "\nbuffer: " + oss[2].str();
}
