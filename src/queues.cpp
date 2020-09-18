#include "queues.hpp"
#include "utils.hpp"

#include <vector>
#include <algorithm>


Queues::Queues(vk::PhysicalDevice pd, VkSurfaceKHR surface)
{
    for (auto &q : _queues) { q.second = -1; }

    struct QueueBit
    {
        enum : uint8_t
        {
            G = 1, // Graphics
            C = 2, // Compute
            T = 4, // Transfer
            P = 8 // Present
        };
    };

    //=========================================================================
    // Choose queue families for the various types of queues.
    // We keep Graphics, Compute, Transfer, Present closely together,
    // while the async queues would like to be exclusive.

    // Get all family caps
    std::vector<vk::QueueFamilyProperties2> fprops = pd.getQueueFamilyProperties2();
    std::vector<uint8_t> queue_fam_masks;
    queue_fam_masks.reserve(fprops.size());
    for (size_t fam_index = 0; fam_index < fprops.size(); ++fam_index) {
        const auto &fam_props = fprops[fam_index].queueFamilyProperties;
        uint8_t mask = 0;
        if (fam_props.queueFlags & vk::QueueFlagBits::eGraphics) { mask |= QueueBit::G; }
        if (fam_props.queueFlags & vk::QueueFlagBits::eCompute) { mask |= QueueBit::C; }
        if (fam_props.queueFlags & vk::QueueFlagBits::eTransfer) { mask |= QueueBit::T; }
        if (pd.getSurfaceSupportKHR(fam_index, surface)) { mask |= QueueBit::P; }
        // if (glfwGetPhysicalDevicePresentationSupport(instance, pd, fam_index)) { mask |= QueueBit::P; }
        queue_fam_masks.push_back(mask);
    }

    // Helpers
    auto find_family = [&](uint8_t desired_caps, uint8_t undesired_caps) -> uint32_t {
        for (size_t fam_index = 0; fam_index < queue_fam_masks.size(); ++fam_index) {
            if ((queue_fam_masks[fam_index] & desired_caps) != desired_caps) { continue; }
            if ((~queue_fam_masks[fam_index] & undesired_caps) != undesired_caps) { continue; }
            return to_uint32(fam_index);
        }
        return -1;
    };
    auto assign_if_unassigned = [&](std::initializer_list<Queue> qs, uint32_t family_index) {
        for (Queue q : qs) {
            auto &fi = _queues[static_cast<size_t>(q)].second;
            if (fi == -1) { fi = family_index; }
        }
    };
    auto still_unassigned = [this](Queue q) { return get_family_index(q) == -1; };

    // Keep async-compute as exclusive as possible.
    assign_if_unassigned({Queue::AsyncCompute}, find_family(QueueBit::C, QueueBit::G | QueueBit::T | QueueBit::P));
    assign_if_unassigned({Queue::AsyncCompute}, find_family(QueueBit::C, QueueBit::G | QueueBit::T));
    assign_if_unassigned({Queue::AsyncCompute}, find_family(QueueBit::C, QueueBit::G));
    assign_if_unassigned({Queue::AsyncCompute}, find_family(QueueBit::C, 0));

    // Keep async-transfer as exclusive as possible.
    assign_if_unassigned({Queue::AsyncTransfer}, find_family(QueueBit::T, QueueBit::G | QueueBit::C | QueueBit::P));
    assign_if_unassigned({Queue::AsyncTransfer}, find_family(QueueBit::T, QueueBit::G | QueueBit::C));
    assign_if_unassigned({Queue::AsyncTransfer}, find_family(QueueBit::T, QueueBit::G));
    assign_if_unassigned({Queue::AsyncTransfer}, find_family(QueueBit::T, 0));

    // Attempt to keep the rest together on the same queue family.
    assign_if_unassigned(
        {Queue::Graphics, Queue::Compute, Queue::Transfer, Queue::Present},
        find_family(QueueBit::G | QueueBit::C | QueueBit::T | QueueBit::P, 0));
    assign_if_unassigned(
        {Queue::Graphics, Queue::Transfer, Queue::Present}, find_family(QueueBit::G | QueueBit::T | QueueBit::P, 0));
    assign_if_unassigned(
        {Queue::Graphics, Queue::Compute, Queue::Transfer}, find_family(QueueBit::G | QueueBit::C | QueueBit::T, 0));

    // Anything still unassigned uses their first supported family.
    assign_if_unassigned({Queue::Graphics}, find_family(QueueBit::G, 0));
    assign_if_unassigned({Queue::Compute}, find_family(QueueBit::C, 0));
    assign_if_unassigned({Queue::Transfer}, find_family(QueueBit::T, 0));
    assign_if_unassigned({Queue::Present}, find_family(QueueBit::P, 0));
}


std::vector<vk::DeviceQueueCreateInfo> Queues::get_create_info_list() const
{
    std::vector<uint32_t> family_indices;
    for (uint32_t queue_enum_elem = 0; queue_enum_elem < to_uint32(Queue::ELEM_COUNT); ++queue_enum_elem) {
        Queue q = static_cast<Queue>(queue_enum_elem);
        family_indices.push_back(get_family_index(q));
    }
    std::sort(family_indices.begin(), family_indices.end());
    family_indices.erase(std::unique(family_indices.begin(), family_indices.end()), family_indices.end());

    static const float queue_prio = 0.5f;
    std::vector<vk::DeviceQueueCreateInfo> result;
    std::vector<float> prios = {0.5f};
    for (uint32_t fi : family_indices) {
        result.push_back(
            vk::DeviceQueueCreateInfo{}.setQueueCount(1).setQueueFamilyIndex(fi).setPQueuePriorities(&queue_prio));
    }
    return result;
}


void Queues::update(vk::Device device)
{
    for (auto &pair : _queues) { pair.first = device.getQueue(pair.second, 0); }
}


uint32_t Queues::get_family_index(Queue q) const
{
    return _queues[static_cast<size_t>(q)].second;
}


vk::Queue Queues::get_queue(Queue q) const
{
    return _queues[static_cast<size_t>(q)].first;
}
