#include "device.hpp"
#include "utils.hpp"
#include <vector>


// For now just a super-simple, non-scaling implementation.
class FramebufferCache
{
public:
    FramebufferCache(const Device &device, size_t max_elem_count) : _device(device), _max_elem_count(max_elem_count) {}

    vk::Framebuffer get_or_create(const FramebufferKey &key)
    {
        ++_current_lookup_time;

        // Serve from cache
        for (auto &entry : _cache) {
            if (entry.key == key) {
                entry.last_lookup_time = _current_lookup_time;
                return entry.value.get();
            }
        }
        // Create new element, cache it
        CachedElement new_entry;
        new_entry.key = key;

        new_entry.value = _device.get().createFramebufferUnique(vk::FramebufferCreateInfo{}
                                                                    .setFlags(key.flags)
                                                                    .setAttachments(key.attachments)
                                                                    .setWidth(key.extent.width)
                                                                    .setHeight(key.extent.height)
                                                                    .setRenderPass(key.render_pass)
                                                                    .setLayers(key.layers));
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
    const Device &_device;
    size_t _max_elem_count = 10;
    struct CachedElement
    {
        FramebufferKey key;
        vk::UniqueFramebuffer value;
        uint64_t last_lookup_time = 0;
    };
    std::vector<CachedElement> _cache;
    uint64_t _current_lookup_time = 0;
};


Device::Device(vk::Instance instance, vk::PhysicalDevice physical_device, VkSurfaceKHR surface) :
    _instance(instance), _physical_device(physical_device)
{
    ASSUME(physical_device);

    std::vector<const char *> device_extensions = {
        "VK_KHR_swapchain", "VK_KHR_get_memory_requirements2", "VK_KHR_dedicated_allocation"};

    init_queue_families(surface);

    // Find unique used family index and create DeviceQueueCreateInfo objects for them.
    // We'll use one queue per family.
    std::vector<uint32_t> family_indices;
    for (uint32_t queue_enum_elem = 0; queue_enum_elem < to_uint32(Queue::ELEM_COUNT); ++queue_enum_elem) {
        Queue q = static_cast<Queue>(queue_enum_elem);
        family_indices.push_back(get_family_index(q));
    }
    std::sort(family_indices.begin(), family_indices.end());
    family_indices.erase(std::unique(family_indices.begin(), family_indices.end()), family_indices.end());
    static const float queue_prio = 0.5f;
    std::vector<vk::DeviceQueueCreateInfo> device_queue_cis;
    std::vector<float> prios = {0.5f};
    for (uint32_t fi : family_indices) {
        device_queue_cis.push_back(
            vk::DeviceQueueCreateInfo{}.setQueueCount(1).setQueueFamilyIndex(fi).setPQueuePriorities(&queue_prio));
    }

    _device = physical_device.createDeviceUnique(
        vk::DeviceCreateInfo{}.setPEnabledExtensionNames(device_extensions).setQueueCreateInfos(device_queue_cis));

    init_vma(instance);
    _general_command_pool =
        _device->createCommandPoolUnique(vk::CommandPoolCreateInfo{}
                                             .setQueueFamilyIndex(get_family_index(Queue::Graphics))
                                             .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer));

    _framebuffer_cache = std::make_unique<FramebufferCache>(*this, 10);
}


Device::~Device()
{
    vmaDestroyAllocator(_vma_allocator);
}


void Device::init_queue_families(VkSurfaceKHR surface)
{
    // Choose queue families for the various types of queues.
    // We keep Graphics, Compute, Transfer, Present closely together,
    // while the async queues would like to be exclusive.

    // Get all family caps in an easily digestable form
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

    std::fill(_queue_families.begin(), _queue_families.end(), -1);

    std::vector<vk::QueueFamilyProperties2> fprops = _physical_device.getQueueFamilyProperties2();
    std::vector<uint8_t> queue_fam_masks;
    queue_fam_masks.reserve(fprops.size());
    for (size_t fam_index = 0; fam_index < fprops.size(); ++fam_index) {
        const auto &fam_props = fprops[fam_index].queueFamilyProperties;
        uint8_t mask = 0;
        if (fam_props.queueFlags & vk::QueueFlagBits::eGraphics) { mask |= QueueBit::G; }
        if (fam_props.queueFlags & vk::QueueFlagBits::eCompute) { mask |= QueueBit::C; }
        if (fam_props.queueFlags & vk::QueueFlagBits::eTransfer) { mask |= QueueBit::T; }
        if (_physical_device.getSurfaceSupportKHR(static_cast<uint32_t>(fam_index), surface)) { mask |= QueueBit::P; }
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
            uint32_t &fi = _queue_families[static_cast<size_t>(q)];
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


void Device::init_vma(vk::Instance instance)
{
    VmaVulkanFunctions vma_funcs = {0};

    vma_funcs.vkGetPhysicalDeviceProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties;
    vma_funcs.vkGetPhysicalDeviceMemoryProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties;
    vma_funcs.vkAllocateMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory;
    vma_funcs.vkFreeMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory;
    vma_funcs.vkMapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory;
    vma_funcs.vkUnmapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory;
    vma_funcs.vkFlushMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges;
    vma_funcs.vkInvalidateMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges;
    vma_funcs.vkBindBufferMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory;
    vma_funcs.vkBindImageMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory;
    vma_funcs.vkGetBufferMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements;
    vma_funcs.vkGetImageMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements;
    vma_funcs.vkCreateBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer;
    vma_funcs.vkDestroyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer;
    vma_funcs.vkCreateImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage;
    vma_funcs.vkDestroyImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage;
    vma_funcs.vkCmdCopyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer;
#if VMA_DEDICATED_ALLOCATION
    vma_funcs.vkGetBufferMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2KHR;
    vma_funcs.vkGetImageMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2KHR;
#endif

    VmaAllocatorCreateInfo ci = {};
    ci.instance = instance;
    ci.physicalDevice = _physical_device;
    ci.device = _device.get();
    ci.pVulkanFunctions = &vma_funcs;
    ci.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;

    check_vk_result(vmaCreateAllocator(&ci, &_vma_allocator));
}


uint32_t Device::get_family_index(Queue queue) const
{
    return _queue_families[static_cast<size_t>(queue)];
}


vk::Queue Device::get_queue(Queue queue) const
{
    return _device->getQueue(get_family_index(queue), 0);
}


void Device::run_commands(Queue queue, std::function<void(vk::CommandBuffer)> body)
{
    auto cbs = _device->allocateCommandBuffersUnique(
        vk::CommandBufferAllocateInfo{}.setCommandPool(_general_command_pool.get()).setCommandBufferCount(1));
    ASSUME(cbs.size() == 1);
    auto &cb = cbs.front();

    cb->begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    body(cb.get());
    cb->end();

    auto submit_info = vk::SubmitInfo{}.setCommandBuffers(cb.get());
    get_queue(queue).submit({submit_info}, {});

    // TODO: we might want to support this using fences. And also make it optional?
    _device->waitIdle();
}


vk::Framebuffer Device::get_framebuffer(const FramebufferKey &key)
{
    return _framebuffer_cache->get_or_create(key);
}
