#pragma once

#include "utils.hpp"
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"
#include <array>

struct FramebufferKey
{
    vk::FramebufferCreateFlags flags;
    vk::RenderPass render_pass;
    vk::Extent2D extent;
    uint32_t layers = 1;
    std::vector<vk::ImageView> attachments;

    friend bool operator==(const FramebufferKey &a, const FramebufferKey &b)
    {
        return a.render_pass == b.render_pass && a.layers == b.layers && a.extent == b.extent && a.flags == b.flags
               && a.attachments == b.attachments;
    }
    friend bool operator!=(const FramebufferKey &a, const FramebufferKey &b) { return !(a == b); }
};

class Device
{
public:
    enum class Queue : uint32_t
    {
        Graphics = 0,
        Compute,
        Transfer,
        Present,
        AsyncTransfer,
        AsyncCompute,
        ELEM_COUNT
    };

    Device(vk::Instance instance, vk::PhysicalDevice physical_device, VkSurfaceKHR surface);
    ~Device();
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    vk::Instance get_instance() const { return _instance; }
    vk::PhysicalDevice get_physical_device() const { return _physical_device; }
    vk::Device get_device() const { return _device.get(); }
    vk::Device get() const { return _device.get(); }
    VmaAllocator get_vma_allocator() const { return _vma_allocator; }

    uint32_t get_family_index(Queue) const;
    vk::Queue get_queue(Queue) const;

    void run_commands(Queue, std::function<void(vk::CommandBuffer)> body);


    vk::Framebuffer get_framebuffer(const FramebufferKey &);

private:
    void init_queue_families(VkSurfaceKHR);
    void init_vma(vk::Instance instance);

    // (vk::Queue, family_index) pairs.
    std::array<uint32_t, to_uint32(Queue::ELEM_COUNT)> _queue_families;

    vk::Instance _instance;
    vk::PhysicalDevice _physical_device;
    vk::UniqueDevice _device;
    vk::UniqueCommandPool _general_command_pool;

    VmaAllocator _vma_allocator = VK_NULL_HANDLE;

    std::unique_ptr<class FramebufferCache> _framebuffer_cache;
};