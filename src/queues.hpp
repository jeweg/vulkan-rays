#pragma once
#include "utils.hpp"
#include "vulkan/vulkan.hpp"
#include <utility>


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

class Queues
{
public:
    Queues(vk::PhysicalDevice pd, VkSurfaceKHR surface);

    // Retrieves and stores the queue objects.
    void update(vk::Device);

    std::vector<vk::DeviceQueueCreateInfo> get_create_info_list() const;

    // Note that update() must have been called prior to calling this.
    uint32_t get_family_index(Queue) const;
    // Note that update() must have been called prior to calling this.
    vk::Queue get_queue(Queue) const;

private:
    // (vk::Queue, family_index) pairs.
    std::array<std::pair<vk::Queue, uint32_t>, to_uint32(Queue::ELEM_COUNT)> _queues;
};
