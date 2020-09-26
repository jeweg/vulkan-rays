#pragma once
#include "build_info.hpp"
#include "debugbreak.h"
#include "vk_mem_alloc.h"
#include "vulkan/vulkan.hpp"
#include "glm/vec2.hpp"
#include "glm/gtc/type_ptr.hpp"
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


inline constexpr void check_vk_result(VkResult result)
{
    ASSUME(result == VK_SUCCESS);
}


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


struct VmaImage
{
    VmaImage() = default;
    VmaImage(VmaAllocator vma_allocator, VmaMemoryUsage usage, const VkImageCreateInfo &ci) :
        vma_allocator(vma_allocator)
    {
        VmaAllocationCreateInfo alloc_info = {0};
        alloc_info.usage = usage;
        ASSUME(vmaCreateImage(vma_allocator, &ci, &alloc_info, &image, &vma_allocation, nullptr) == VK_SUCCESS);
    }
    VmaImage(const VmaImage &) = delete;
    VmaImage &operator=(const VmaImage &) = delete;
    VmaImage(VmaImage &&other) noexcept { *this = std::move(other); }
    VmaImage &operator=(VmaImage &&other) noexcept
    {
        if (image && image != other.image) { vmaDestroyImage(vma_allocator, image, vma_allocation); }

        vma_allocator = std::move(other.vma_allocator);
        vma_allocation = std::move(other.vma_allocation);
        image = std::move(other.image);

        // Leave other in a valid moved-from state
        other.vma_allocation = VK_NULL_HANDLE;
        other.image = VK_NULL_HANDLE;
        return *this;
    }

    ~VmaImage() noexcept
    {
        if (image) { vmaDestroyImage(vma_allocator, image, vma_allocation); }
    }

    operator vk::Image() const { return image; }
    explicit operator VkImage() const { return image; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
    VmaAllocation vma_allocation = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
};


struct VmaBuffer
{
    VmaBuffer() = default;
    VmaBuffer(
        VmaAllocator vma_allocator, VmaMemoryUsage usage, bool persistently_mapped, const VkBufferCreateInfo &ci) :
        vma_allocator(vma_allocator)
    {
        VmaAllocationCreateInfo alloc_ci = {0};
        alloc_ci.usage = usage;
        if (persistently_mapped) { alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT; }
        ASSUME(vmaCreateBuffer(vma_allocator, &ci, &alloc_ci, &buffer, &vma_allocation, &alloc_info) == VK_SUCCESS);
    }
    VmaBuffer(const VmaBuffer &) = delete;
    VmaBuffer &operator=(const VmaBuffer &) = delete;
    VmaBuffer(VmaBuffer &&other) noexcept { *this = std::move(other); }
    VmaBuffer &operator=(VmaBuffer &&other) noexcept
    {
        if (buffer && buffer != other.buffer) { vmaDestroyBuffer(vma_allocator, buffer, vma_allocation); }

        vma_allocator = std::move(other.vma_allocator);
        vma_allocation = std::move(other.vma_allocation);
        alloc_info = std::move(other.alloc_info);
        buffer = std::move(other.buffer);

        // Leave other in a valid moved-from state
        other.vma_allocation = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        return *this;
    }

    ~VmaBuffer() noexcept
    {
        if (buffer) { vmaDestroyBuffer(vma_allocator, buffer, vma_allocation); }
    }

    void *mapped_data() { return alloc_info.pMappedData; }

    operator vk::Buffer() const { return buffer; }
    explicit operator VkBuffer() const { return buffer; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
    VmaAllocation vma_allocation = VK_NULL_HANDLE;
    VmaAllocationInfo alloc_info;
    VkBuffer buffer = VK_NULL_HANDLE;
};


inline void foo()
{
    // Data from https://www.graphics.cornell.edu/online/box/data.html
    // Y is up. The floor is at y = 0.
    // Every 3 consecutive components make up a vertex, 4 of those a quadliteral.
    // The order and number of quadliterals is
    // 1x floor (holes skipped), 1x light, 1x ceiling, 1x back wall, 1x right all, 1x left wall,
    // 5x short block, 5x tall block.
    static constexpr float COORDS[] = {
        552.8f, 0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   559.2f, 549.6f, 0.0f,   559.2f, 343.0f, 548.8f,
        227.0f, 343.0f, 548.8f, 332.0f, 213.0f, 548.8f, 332.0f, 213.0f, 548.8f, 227.0f, 556.0f, 548.8f, 0.0f,   556.0f,
        548.8f, 559.2f, 0.0f,   548.8f, 559.2f, 0.0f,   548.8f, 0.0f,   549.6f, 0.0f,   559.2f, 0.0f,   0.0f,   559.2f,
        0.0f,   548.8f, 559.2f, 556.0f, 548.8f, 559.2f, 0.0f,   0.0f,   559.2f, 0.0f,   0.0f,   0.0f,   0.0f,   548.8f,
        0.0f,   0.0f,   548.8f, 559.2f, 552.8f, 0.0f,   0.0f,   549.6f, 0.0f,   559.2f, 556.0f, 548.8f, 559.2f, 556.0f,
        548.8f, 0.0f,   130.0f, 165.0f, 65.0f,  82.0f,  165.0f, 225.0f, 240.0f, 165.0f, 272.0f, 290.0f, 165.0f, 114.0f,
        290.0f, 0.0f,   114.0f, 290.0f, 165.0f, 114.0f, 240.0f, 165.0f, 272.0f, 240.0f, 0.0f,   272.0f, 130.0f, 0.0f,
        65.0f,  130.0f, 165.0f, 65.0f,  290.0f, 165.0f, 114.0f, 290.0f, 0.0f,   114.0f, 82.0f,  0.0f,   225.0f, 82.0f,
        165.0f, 225.0f, 130.0f, 165.0f, 65.0f,  130.0f, 0.0f,   65.0f,  240.0f, 0.0f,   272.0f, 240.0f, 165.0f, 272.0f,
        82.0f,  165.0f, 225.0f, 82.0f,  0.0f,   225.0f, 423.0f, 330.0f, 247.0f, 265.0f, 330.0f, 296.0f, 314.0f, 330.0f,
        456.0f, 472.0f, 330.0f, 406.0f, 423.0f, 0.0f,   247.0f, 423.0f, 330.0f, 247.0f, 472.0f, 330.0f, 406.0f, 472.0f,
        0.0f,   406.0f, 472.0f, 0.0f,   406.0f, 472.0f, 330.0f, 406.0f, 314.0f, 330.0f, 456.0f, 314.0f, 0.0f,   456.0f,
        314.0f, 0.0f,   456.0f, 314.0f, 330.0f, 456.0f, 265.0f, 330.0f, 296.0f, 265.0f, 0.0f,   296.0f, 265.0f, 0.0f,
        296.0f, 265.0f, 330.0f, 296.0f, 423.0f, 330.0f, 247.0f, 423.0f, 0.0f,   247.0f,
    };

    auto make_quad = [&](size_t quad_index) {
        std::array<glm::vec3, 4> vertices;
        for (size_t i = 0; i < 4; ++i) { vertices[i] = glm::make_vec3(COORDS + quad_index * 4 * 3 + i * 3); }
    };
}
