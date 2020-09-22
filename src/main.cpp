#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "utils.hpp"
#include "queues.hpp"
#include "build_info.hpp"
#include "debugbreak.h"

#define GLM_FORCE_SWIZZLE
#include "glm/vec2.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <string_view>
#include <iostream>
#include <chrono>
#include <optional>

// We avoid swapchain resizing issues here by
// using a fixed size.
constexpr int W = 1200;
constexpr int H = 800;


VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_types,
    VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData,
    void *)
{
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(message_severity)) << "]"
              << "(" << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagBitsEXT>(message_types)) << ") "
              << pCallbackData->pMessage << "\n";

    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { debug_break(); }

    return VK_TRUE;
}


void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW error " << std::to_string(error) << ": " << description << "\n";
}


// Our surface gets created by GLFW so we manage it in a custom RAII class.
// TODO: There's probably a way to use vulkan.hpp's unique handle system for this as well.
struct SurfaceDestroyer
{
    vk::Instance instance;
    VkSurfaceKHR surface;
    SurfaceDestroyer(vk::Instance instance, VkSurfaceKHR surface) : instance(instance), surface(surface) {}
    ~SurfaceDestroyer() { vkDestroySurfaceKHR(instance, surface, nullptr); }
};


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
    VmaImage(VmaImage &&other) noexcept :
        vma_allocator(other.vma_allocator), vma_allocation(other.vma_allocation), image(other.image)
    {
        other.vma_allocation = VK_NULL_HANDLE;
        other.image = VK_NULL_HANDLE;
    }
    VmaImage &operator=(VmaImage &&other) noexcept
    {
        vma_allocator = other.vma_allocator;
        vma_allocation = other.vma_allocation;
        image = other.image;
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
    VmaBuffer(VmaBuffer &&other) noexcept :
        vma_allocator(other.vma_allocator),
        vma_allocation(other.vma_allocation),
        alloc_info(other.alloc_info),
        buffer(other.buffer)
    {
        other.vma_allocation = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
    }
    VmaBuffer &operator=(VmaBuffer &&other) noexcept
    {
        vma_allocator = other.vma_allocator;
        vma_allocation = other.vma_allocation;
        alloc_info = other.alloc_info;
        buffer = other.buffer;
        other.vma_allocation = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        return *this;
    }

    void *mapped_data() { return alloc_info.pMappedData; }

    ~VmaBuffer() noexcept
    {
        if (buffer) { vmaDestroyBuffer(vma_allocator, buffer, vma_allocation); }
    }

    operator vk::Buffer() const { return buffer; }
    explicit operator VkBuffer() const { return buffer; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
    VmaAllocation vma_allocation = VK_NULL_HANDLE;
    VmaAllocationInfo alloc_info;
    VkBuffer buffer = VK_NULL_HANDLE;
};


struct VmaAllocatorGuard
{
    VmaAllocatorGuard(const VmaAllocatorCreateInfo &ci) { vmaCreateAllocator(&ci, &vma_allocator); }
    ~VmaAllocatorGuard() { vmaDestroyAllocator(vma_allocator); }
    operator VmaAllocator() const { return vma_allocator; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
};

vk::UniqueCommandPool g_one_shot_graphics_command_pool;

// TODO: combine device, queues, command pools into one context object.
// We tend to need these together too much and they all hinge on the vk::Device anyway.
void RunSingleTimeCommands(vk::Device device, const Queues &queues, std::function<void(vk::CommandBuffer)> inner_body)
{
    auto command_buffers =
        device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{}
                                                .setCommandPool(g_one_shot_graphics_command_pool.get())
                                                .setCommandBufferCount(1));
    ASSUME(command_buffers.size() == 1);
    command_buffers.front()->begin(
        vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    inner_body(command_buffers.front().get());

    auto submit_info = vk::SubmitInfo{}.setCommandBuffers(command_buffers.front().get());
    queues.get_queue(Queue::Graphics).submit({submit_info}, {});
    device.waitIdle();
}

bool g_mouse_dragging = false;
glm::vec2 g_mouse_pos;

class MouseDragger
{
public:
    bool is_dragging() const { return _is_dragging; }
    bool has_delta() const { return is_dragging() && _latest_pos.has_value(); }

    glm::vec2 get_delta() const
    {
        ASSUME(has_delta());
        glm::vec2 delta(0);
        if (_last_reported_pos) { delta = _latest_pos.value() - _last_reported_pos.value(); }
        _last_reported_pos = _latest_pos;
        return delta;
    }

    void start_dragging()
    {
        if (!_is_dragging) {
            _is_dragging = true;
            _latest_pos.reset();
            _last_reported_pos.reset();
        }
    }
    void stop_dragging() { _is_dragging = false; }
    void update(double xpos, double ypos)
    {
        if (_is_dragging) { _latest_pos.emplace(xpos, ypos); }
    }

private:
    bool _is_dragging = false;
    mutable std::optional<glm::vec2> _last_reported_pos;
    std::optional<glm::vec2> _latest_pos;
};

MouseDragger g_left_mouse_dragger;
MouseDragger g_right_mouse_dragger;
float g_wheel_dragger = 0;

static void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    g_left_mouse_dragger.update(xpos, ypos);
    g_right_mouse_dragger.update(xpos, ypos);
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) { g_left_mouse_dragger.start_dragging(); }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) { g_right_mouse_dragger.start_dragging(); }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) { g_left_mouse_dragger.stop_dragging(); }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) { g_right_mouse_dragger.stop_dragging(); }
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    g_wheel_dragger = yoffset;
}

constexpr size_t NUM_FRAMES_IN_FLIGHT = 3;

struct PerFrame
{
    vk::UniqueCommandPool command_pool;
    vk::UniqueCommandBuffer command_buffer;

    vk::UniqueImageView swapchain_image_view;
    vk::UniqueSemaphore image_available_for_rendering_sema;
    vk::UniqueSemaphore rendering_finished_sema;
    vk::UniqueFence finished_fence;

    vk::Device device;
    vk::SwapchainKHR swapchain;

    Queues queues;

    PerFrame(vk::Device device, vk::SwapchainKHR swapchain, Queues queues) :
        device(device), swapchain(swapchain), queues(queues)
    {
        command_pool = device.createCommandPoolUnique(vk::CommandPoolCreateInfo{}
                                                          .setQueueFamilyIndex(queues.get_family_index(Queue::Graphics))
                                                          .setFlags(vk::CommandPoolCreateFlagBits::eTransient));
        image_available_for_rendering_sema = device.createSemaphoreUnique(vk::SemaphoreCreateInfo{});
        rendering_finished_sema = device.createSemaphoreUnique(vk::SemaphoreCreateInfo{});
        finished_fence = device.createFenceUnique(vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));
    }

    // Much simplified workflow for the graphics piipeline.
    // Just one command buffer per frame submit.

    vk::CommandBuffer begin_frame()
    {
        device.waitForFences({finished_fence.get()}, true, -1);
        device.resetFences({finished_fence.get()});
        // ASSUME(!command_buffer);
        device.resetCommandPool(command_pool.get(), {});

        auto command_buffers = device.allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo{}.setCommandPool(command_pool.get()).setCommandBufferCount(1));
        ASSUME(command_buffers.size() == 1);
        command_buffer = std::move(command_buffers.front());

        command_buffer->begin(vk::CommandBufferBeginInfo{});

        return command_buffer.get();
    }


    void end_frame()
    {
        command_buffer->end();

        // Acquire image for rendering
        uint32_t image_index = device.acquireNextImageKHR(swapchain, -1, image_available_for_rendering_sema.get(), {});

        auto submit_info =
            vk::SubmitInfo{}
                .setCommandBuffers(command_buffer.get())
                .setWaitSemaphores(image_available_for_rendering_sema.get())
                .setSignalSemaphores(rendering_finished_sema.get())
                .setWaitDstStageMask(vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput));
        queues.get_queue(Queue::Graphics).submit({submit_info}, finished_fence.get());

        // Present
        queues.get_queue(Queue::Present)
            .presentKHR(vk::PresentInfoKHR{}
                            .setWaitSemaphores(rendering_finished_sema.get())
                            .setSwapchains(swapchain)
                            .setImageIndices(image_index));
    }
};
// std::array<PerFrame, NUM_FRAMES_IN_FLIGHT> per_frames;
std::vector<PerFrame> per_frames;


int main()
{
    try {
        //----------------------------------------------------------------------
        // Vulkan instance

        vk::DynamicLoader dl;
        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
            dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        std::vector<const char *> instance_extensions = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_get_surface_capabilities2"};
#if defined(_WIN32)
        instance_extensions.push_back("VK_KHR_win32_surface");
#elif !(defined(__APPLE__) || defined(__MACH__))
        // TODO: How do we choose between xcb and xlib? Does glfw require one? Not sure.
        instance_extensions.add("VK_KHR_xcb_surface");
        // extensions_ptrs.push_back("VK_KHR_xlib_surface");
#else
#    error Unsupported platform!
#endif

        uint32_t glfw_exts_count = 0;
        const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_exts_count);
        for (uint32_t i = 0; i < glfw_exts_count; ++i) { instance_extensions.push_back(glfw_exts[i]); }

        vk::ApplicationInfo appInfo("Test", 1, "Custom", 1, VK_API_VERSION_1_1);
        std::vector<const char *> layer_names = {"VK_LAYER_KHRONOS_validation"};
        vk::UniqueInstance instance = vk::createInstanceUnique(vk::InstanceCreateInfo{}
                                                                   .setPApplicationInfo(&appInfo)
                                                                   .setPEnabledExtensionNames(instance_extensions)
                                                                   .setPEnabledLayerNames(layer_names));
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

        auto debug_utils_messenger = instance->createDebugUtilsMessengerEXTUnique(
            vk::DebugUtilsMessengerCreateInfoEXT{}
                .setMessageSeverity(
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
                .setMessageType(
                    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
                    | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
                .setPfnUserCallback(&debug_utils_callback));

        //----------------------------------------------------------------------
        // Use GLFW3 to create a window and a corresponding Vulkan surface.

        ASSUME(glfwInit());
        glfwSetErrorCallback(&glfwErrorCallback);
        ASSUME(glfwVulkanSupported());

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        GLFWwindow *glfw_window = glfwCreateWindow(1200, 800, "Vulkan Rays", nullptr, nullptr);
        if (!glfw_window) {
            glfwTerminate();
            fatal("Failed to create window");
        }

        VkSurfaceKHR surface;
        if (VkResult err = glfwCreateWindowSurface(instance.get(), glfw_window, nullptr, &surface)) {
            glfwTerminate();
            fatal("Failed to create window surface");
        }
        SurfaceDestroyer surface_destroyer(instance.get(), surface);

        //----------------------------------------------------------------------
        // Physical and logical device.

        // Choose first physical device
        std::vector<vk::PhysicalDevice> physical_devices = instance->enumeratePhysicalDevices();
        ASSUME(!physical_devices.empty());

        vk::PhysicalDevice phys_device = physical_devices.front();

        std::vector<const char *> device_extensions = {
            "VK_KHR_swapchain", "VK_KHR_get_memory_requirements2", "VK_KHR_dedicated_allocation"};

        Queues queues(phys_device, surface);
        vk::UniqueDevice device =
            phys_device.createDeviceUnique(vk::DeviceCreateInfo{}
                                               .setPEnabledExtensionNames(device_extensions)
                                               .setQueueCreateInfos(queues.get_create_info_list()));
        queues.update(device.get());

        //----------------------------------------------------------------------
        // Init Vulkan-Memory-Allocator

        VmaVulkanFunctions vma_funcs = {0};

        vma_funcs.vkGetPhysicalDeviceProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties;
        vma_funcs.vkGetPhysicalDeviceMemoryProperties =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties;
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
        VmaAllocatorCreateInfo allocator_info = {};
        allocator_info.instance = instance.get();
        allocator_info.physicalDevice = phys_device;
        allocator_info.device = device.get();
        allocator_info.pVulkanFunctions = &vma_funcs;
        allocator_info.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
        VmaAllocatorGuard vma_allocator(allocator_info);

        //----------------------------------------------------------------------
        // Surface format and swap chain

        auto surface_info = vk::PhysicalDeviceSurfaceInfo2KHR{}.setSurface(surface);

        auto surface_format =
            choose_best(phys_device.getSurfaceFormats2KHR(surface_info), [](vk::SurfaceFormat2KHR sf) {
                if (sf.surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
                    && sf.surfaceFormat.format == vk::Format::eB8G8R8A8Srgb) {
                    return 1000;
                } else if (
                    sf.surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear
                    && sf.surfaceFormat.format == vk::Format::eB8G8R8A8Unorm) {
                    return 100;
                } else if (sf.surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                    return 50;
                }
                return 0;
            });

        auto surface_caps = phys_device.getSurfaceCapabilities2KHR(surface_info);

        auto create_swap_chain = [&](uint32_t w, uint32_t h) -> vk::UniqueSwapchainKHR {
            auto caps = surface_caps.surfaceCapabilities;

            vk::Extent2D extent;
            if (caps.currentExtent.width == -1) {
                extent.width = clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width);
                extent.height = clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height);
            } else {
                extent = caps.currentExtent;
            }
            // ASSUME(VK_TRUE == phys_device.getSurfaceSupportKHR(queues.get_family_index(Queue::Present),
            // surface));

            auto present_mode = choose_best(phys_device.getSurfacePresentModesKHR(surface), [](vk::PresentModeKHR pm) {
                switch (pm) {
                case vk::PresentModeKHR::eMailbox: return 1000;
                case vk::PresentModeKHR::eFifo: return 100;
                case vk::PresentModeKHR::eFifoRelaxed: return 10;
                default: return 0;
                }
            });

            // Use identity if available, current otherwise.
            vk::SurfaceTransformFlagBitsKHR pre_transform =
                caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity ?
                    vk::SurfaceTransformFlagBitsKHR::eIdentity :
                    caps.currentTransform;

            auto swapchain_ci = vk::SwapchainCreateInfoKHR{}
                                    .setSurface(surface)
                                    .setMinImageCount(NUM_FRAMES_IN_FLIGHT)
                                    .setImageFormat(surface_format.surfaceFormat.format)
                                    .setImageColorSpace(surface_format.surfaceFormat.colorSpace)
                                    .setImageExtent(extent)
                                    .setImageArrayLayers(1)
                                    .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                                    .setImageSharingMode(vk::SharingMode::eExclusive)
                                    .setPreTransform(pre_transform)
                                    .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                                    .setPresentMode(present_mode)
                                    .setClipped(true);
            std::array<uint32_t, 2> family_indices = {
                queues.get_family_index(Queue::Graphics), queues.get_family_index(Queue::Present)};
            if (family_indices.front() != family_indices.back()) {
                // If the graphics and present queues are from different queue families,
                // we either have to explicitly transfer ownership of images between the
                // queues, or we have to create the swapchain with imageSharingMode
                // as VK_SHARING_MODE_CONCURRENT
                swapchain_ci.setImageSharingMode(vk::SharingMode::eConcurrent);
                swapchain_ci.setQueueFamilyIndices(family_indices);
            }

            return std::move(device->createSwapchainKHRUnique(swapchain_ci));
        };

        vk::UniqueSwapchainKHR swapchain = create_swap_chain(1200, 800);

        //----------------------------------------------------------------------
        // Per-frame resources
        // one per frame in flight per queue family.

        for (size_t i = 0; i < NUM_FRAMES_IN_FLIGHT; ++i) {
            per_frames.emplace_back(device.get(), swapchain.get(), queues);
        }

        {
            std::vector<vk::Image> images = device->getSwapchainImagesKHR(swapchain.get());

            for (size_t i = 0; i < images.size(); ++i) {
                auto image_view = device->createImageViewUnique(
                    vk::ImageViewCreateInfo{}
                        .setImage(images[i])
                        .setViewType(vk::ImageViewType::e2D)
                        .setSubresourceRange(vk::ImageSubresourceRange{}
                                                 .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                 .setBaseMipLevel(0)
                                                 .setLevelCount(1)
                                                 .setBaseArrayLayer(0)
                                                 .setLayerCount(1))
                        .setFormat(surface_format.surfaceFormat.format));
                per_frames[i].swapchain_image_view = std::move(image_view);
            }
        }

        //----------------------------------------------------------------------
        // Create offscreen image

        constexpr vk::Format IMAGE_FORMAT = vk::Format::eR32G32B32A32Sfloat;
        // std::cerr << format_properties_to_string(phys_device, IMAGE_FORMAT) << "\n";

        VmaImage image = VmaImage(
            vma_allocator,
            VMA_MEMORY_USAGE_GPU_ONLY,
            vk::ImageCreateInfo{}
                .setImageType(vk::ImageType::e2D)
                .setExtent({W, H, 1})
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(IMAGE_FORMAT)
                .setTiling(vk::ImageTiling::eOptimal)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setSharingMode(
                    queues.get_family_index(Queue::Graphics) != queues.get_family_index(Queue::Compute) ?
                        vk::SharingMode::eConcurrent :
                        vk::SharingMode::eExclusive));

        vk::UniqueImageView image_view =
            device->createImageViewUnique(vk::ImageViewCreateInfo{}
                                              .setImage(image)
                                              .setFormat(IMAGE_FORMAT /*vk::Format::eR8G8B8A8Uint*/)
                                              .setViewType(vk::ImageViewType::e2D)
                                              .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                       .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                                       .setBaseMipLevel(0)
                                                                       .setLayerCount(1)
                                                                       .setLevelCount(1)
                                                                       .setBaseArrayLayer(0)));

        vk::UniqueSampler image_sampler = device->createSamplerUnique(vk::SamplerCreateInfo{});

        //----------------------------------------------------------------------
        // Shared pools and caches

        g_one_shot_graphics_command_pool = device->createCommandPoolUnique(
            vk::CommandPoolCreateInfo{}
                .setQueueFamilyIndex(queues.get_family_index(Queue::Graphics))
                .setFlags(
                    vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer));

        vk::UniquePipelineCache pipeline_cache = device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo{});

        FramebufferCache fb_cache(device.get(), 10);

        vk::UniqueDescriptorPool descriptor_pool = device->createDescriptorPoolUnique(
            vk::DescriptorPoolCreateInfo{}.setMaxSets(100).setPoolSizes(std::initializer_list<vk::DescriptorPoolSize>{
                vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageImage).setDescriptorCount(100),
                vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(100)}));

        //----------------------------------------------------------------------
        // Graphics pipeline

        struct
        {
            vk::UniquePipeline pipeline;
            vk::UniquePipelineLayout layout;
            vk::UniqueDescriptorSetLayout dsl;
            vk::UniqueRenderPass render_pass;
        } graphics_pipeline;

        {
            graphics_pipeline.dsl =
                device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{}.setBindings(
                    vk::DescriptorSetLayoutBinding{}
                        .setBinding(0)
                        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eFragment)));

            graphics_pipeline.layout = device->createPipelineLayoutUnique(
                vk::PipelineLayoutCreateInfo{}.setSetLayouts(graphics_pipeline.dsl.get()));

            vk::UniqueShaderModule vertex_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/vs.vert.spirv");
            vk::UniqueShaderModule fragment_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/fs.frag.spirv");

            graphics_pipeline.render_pass = device->createRenderPassUnique(
                vk::RenderPassCreateInfo{}
                    .setAttachments(vk::AttachmentDescription{}
                                        .setFormat(surface_format.surfaceFormat.format)
                                        .setSamples(vk::SampleCountFlagBits::e1)
                                        .setLoadOp(vk::AttachmentLoadOp::eClear)
                                        .setStoreOp(vk::AttachmentStoreOp::eStore)
                                        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                        .setInitialLayout(vk::ImageLayout::eUndefined)
                                        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR))
                    //.setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal))
                    .setSubpasses(vk::SubpassDescription{}
                                      .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                      .setColorAttachments(vk::AttachmentReference{}.setAttachment(0).setLayout(
                                          vk::ImageLayout::eColorAttachmentOptimal)))
                    .setDependencies(
                        vk::SubpassDependency{}
                            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                            .setDstSubpass(0)
                            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                            .setSrcAccessMask({})
                            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                            .setDstAccessMask(
                                vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite)));

            std::vector<vk::PipelineShaderStageCreateInfo> shader_stages = {
                vk::PipelineShaderStageCreateInfo{}
                    .setStage(vk::ShaderStageFlagBits::eVertex)
                    .setModule(vertex_shader.get())
                    .setPName("main"),
                vk::PipelineShaderStageCreateInfo{}
                    .setStage(vk::ShaderStageFlagBits::eFragment)
                    .setModule(fragment_shader.get())
                    .setPName("main")};
            auto vertex_input_state = vk::PipelineVertexInputStateCreateInfo{};
            auto input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo{}
                                            .setTopology(vk::PrimitiveTopology::eTriangleList)
                                            .setPrimitiveRestartEnable(false);
            auto dyn_state = vk::PipelineDynamicStateCreateInfo{};
            auto viewport_state = vk::PipelineViewportStateCreateInfo{}
                                      .setViewports(vk::Viewport(0, 0, W, H, 0, 1))
                                      .setScissors(vk::Rect2D({0, 0}, {W, H}));
            auto rasterization_state = vk::PipelineRasterizationStateCreateInfo{}
                                           .setPolygonMode(vk::PolygonMode::eFill)
                                           .setCullMode(vk::CullModeFlagBits::eNone)
                                           .setFrontFace(vk::FrontFace::eClockwise)
                                           .setDepthClampEnable(false)
                                           .setRasterizerDiscardEnable(false)
                                           .setDepthBiasEnable(false)
                                           .setLineWidth(1.f);
            auto multisample_state = vk::PipelineMultisampleStateCreateInfo{}
                                         .setRasterizationSamples(vk::SampleCountFlagBits::e1)
                                         .setSampleShadingEnable(false);
            auto depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo{}
                                           .setDepthTestEnable(true)
                                           .setDepthWriteEnable(true)
                                           .setDepthCompareOp(vk::CompareOp::eLessOrEqual)
                                           .setBack(vk::StencilOpState{}
                                                        .setFailOp(vk::StencilOp::eKeep)
                                                        .setPassOp(vk::StencilOp::eKeep)
                                                        .setCompareOp(vk::CompareOp::eAlways))
                                           .setFront(vk::StencilOpState{}
                                                         .setFailOp(vk::StencilOp::eKeep)
                                                         .setPassOp(vk::StencilOp::eKeep)
                                                         .setCompareOp(vk::CompareOp::eAlways));
            auto color_blend_state = vk::PipelineColorBlendStateCreateInfo{}.setAttachments(
                vk::PipelineColorBlendAttachmentState{}.setColorWriteMask(
                    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB
                    | vk::ColorComponentFlagBits::eA));

            graphics_pipeline.pipeline = device->createGraphicsPipelineUnique(
                pipeline_cache.get(),
                vk::GraphicsPipelineCreateInfo{}
                    .setLayout(graphics_pipeline.layout.get())
                    .setRenderPass(graphics_pipeline.render_pass.get())
                    .setStages(shader_stages)
                    .setPVertexInputState(&vertex_input_state)
                    .setPInputAssemblyState(&input_assembly_state)
                    .setPDynamicState(&dyn_state)
                    .setPViewportState(&viewport_state)
                    .setPRasterizationState(&rasterization_state)
                    .setPMultisampleState(&multisample_state)
                    .setPDepthStencilState(&depth_stencil_state)
                    .setPColorBlendState(&color_blend_state));
        }
        std::vector<vk::DescriptorSet> ds =
            device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                               .setDescriptorPool(descriptor_pool.get())
                                               .setDescriptorSetCount(1)
                                               .setSetLayouts(graphics_pipeline.dsl.get()));
        device->updateDescriptorSets(
            vk::WriteDescriptorSet{}
                .setDstSet(ds.front())
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDstBinding(0)
                .setDescriptorCount(1)
                .setImageInfo(vk::DescriptorImageInfo{}
                                  .setImageLayout(vk::ImageLayout::eGeneral)
                                  .setImageView(image_view.get())
                                  .setSampler(image_sampler.get())),
            {});


        //----------------------------------------------------------------------
        // Compute pipeline

        struct ComputePipeline
        {
            vk::UniquePipeline pipeline;
            vk::UniquePipelineLayout layout;
            vk::UniqueDescriptorSetLayout dsl;

            // TODO: there's still something wrong here.
            // If I move the view_transform to the end of the struct
            // declaration, it doesn't work.
            // Here I attempt to pad everything to 16-byte boundaries for compatibility
            // with GLSL, but apparently this isn't enough.
            // The pragma doesn't make a difference.
            //#pragma pack(push, 1)
            struct PushConstants
            {
                glm::mat4 view_to_world_transform = glm::mat4(1);
                uint32_t progression_index = 0;
                uint32_t dummy1[3];
                float delta_time = 0.f;
                uint32_t dummy2[3];
            } push_constants;
            //#pragma pack(pop)

            struct Sphere
            {
                glm::vec4 center_and_radius;
                glm::vec4 albedo_and_roughness;
                glm::vec4 emissive_and_ior;
                glm::vec4 specular_and_coefficient;
            };

            struct UBO
            {
                std::array<Sphere, 6> spheres;
            };
            UBO *ubo_data = nullptr;
            VmaBuffer ubo;
        } compute_pipeline;

        {
            compute_pipeline.ubo = std::move(VmaBuffer(
                vma_allocator,
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                true, // Automatically persistently mapped
                vk::BufferCreateInfo{}
                    .setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
                    .setSharingMode(vk::SharingMode::eExclusive)
                    .setSize(sizeof(ComputePipeline::UBO))));
            compute_pipeline.ubo_data = static_cast<ComputePipeline::UBO *>(compute_pipeline.ubo.mapped_data());

            vk::UniqueShaderModule compute_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/compute.comp.spirv");

            auto dslb_0 = vk::DescriptorSetLayoutBinding{}
                              .setBinding(0)
                              .setDescriptorType(vk::DescriptorType::eStorageImage)
                              .setDescriptorCount(1)
                              .setStageFlags(vk::ShaderStageFlagBits::eCompute);
            auto dslb_1 = vk::DescriptorSetLayoutBinding{}
                              .setBinding(1)
                              .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                              .setDescriptorCount(1)
                              .setStageFlags(vk::ShaderStageFlagBits::eCompute);
            compute_pipeline.dsl =
                device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{}.setBindings(
                    std::initializer_list<vk::DescriptorSetLayoutBinding>{dslb_0, dslb_1}));

            auto foo = sizeof(ComputePipeline::PushConstants);

            compute_pipeline.layout = device->createPipelineLayoutUnique(
                vk::PipelineLayoutCreateInfo{}
                    .setPushConstantRanges(vk::PushConstantRange{}
                                               .setOffset(0)
                                               .setSize(sizeof(ComputePipeline::PushConstants))
                                               .setStageFlags(vk::ShaderStageFlagBits::eCompute))
                    .setSetLayouts(compute_pipeline.dsl.get()));

            compute_pipeline.pipeline = device->createComputePipelineUnique(
                pipeline_cache.get(),
                vk::ComputePipelineCreateInfo{}
                    .setStage(vk::PipelineShaderStageCreateInfo{}
                                  .setStage(vk::ShaderStageFlagBits::eCompute)
                                  .setModule(compute_shader.get())
                                  .setPName("main"))
                    .setLayout(compute_pipeline.layout.get()));
        }

        std::vector<vk::DescriptorSet> compute_ds =
            device->allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                               .setDescriptorPool(descriptor_pool.get())
                                               .setDescriptorSetCount(1)
                                               .setSetLayouts(compute_pipeline.dsl.get()));
        device->updateDescriptorSets(
            std::initializer_list<vk::WriteDescriptorSet>{
                vk::WriteDescriptorSet{}
                    .setDstSet(compute_ds.front())
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setDstBinding(0)
                    .setDescriptorCount(1)
                    .setImageInfo(vk::DescriptorImageInfo{}
                                      .setImageLayout(vk::ImageLayout::eGeneral)
                                      .setImageView(image_view.get())
                                      .setSampler(image_sampler.get())),
                vk::WriteDescriptorSet{}
                    .setDstSet(compute_ds.front())
                    .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                    .setDstBinding(1)
                    .setDescriptorCount(1)
                    .setBufferInfo(vk::DescriptorBufferInfo{}
                                       .setBuffer(compute_pipeline.ubo)
                                       .setOffset(0)
                                       .setRange(sizeof(ComputePipeline::UBO)))},
            {});

        //----------------------------------------------------------------------
        // Define world

        {
            {
                ComputePipeline::Sphere &sphere = compute_pipeline.ubo_data->spheres[0];
                sphere.center_and_radius = glm::vec4(0, 0, 0, 1);
                sphere.albedo_and_roughness = glm::vec4(1, 0.7, 0, 0.3);
                sphere.emissive_and_ior = glm::vec4(3.0, 2.7, 0.8, 1);
                sphere.specular_and_coefficient = glm::vec4(1, 1, 1, 0.6);
            }
            {
                ComputePipeline::Sphere &sphere = compute_pipeline.ubo_data->spheres[1];
                sphere.center_and_radius = glm::vec4(0, -16.4, 0, 15.4);
                sphere.albedo_and_roughness = glm::vec4(0.7, 0.3, 0.3, 0.4);
                sphere.emissive_and_ior = glm::vec4(0, 0, 0, 1);
                sphere.specular_and_coefficient = glm::vec4(1, 0.8, 0.4, 0.6);
            }
            {
                ComputePipeline::Sphere &sphere = compute_pipeline.ubo_data->spheres[2];
                sphere.center_and_radius = glm::vec4(1.21, -0.47, 1.54, 0.7);
                sphere.albedo_and_roughness = glm::vec4(0.1, 0.4, 0.9, 0.9);
                sphere.emissive_and_ior = glm::vec4(0, 0, 0, 1);
                sphere.specular_and_coefficient = glm::vec4(1, 1, 1, 0.0);
            }
            {
                ComputePipeline::Sphere &sphere = compute_pipeline.ubo_data->spheres[3];
                sphere.center_and_radius = glm::vec4(-2.1, 0.64, 0.2, 0.59);
                sphere.albedo_and_roughness = glm::vec4(0.86, 0, 0, 0.5);
                sphere.emissive_and_ior = glm::vec4(0, 0, 0, 1);
                sphere.specular_and_coefficient = glm::vec4(1, 1, 1, 0.5);
            }
            {
                ComputePipeline::Sphere &sphere = compute_pipeline.ubo_data->spheres[4];
                sphere.center_and_radius = glm::vec4(-1.42, -0.63, -0.36, 0.45);
                sphere.albedo_and_roughness = glm::vec4(0.8, 0.8, 0.8, 0.5);
                sphere.emissive_and_ior = glm::vec4(0, 0, 0, 1);
                sphere.specular_and_coefficient = glm::vec4(1, 1, 1, 0.5);
            }
            {
                ComputePipeline::Sphere &sphere = compute_pipeline.ubo_data->spheres[5];
                sphere.center_and_radius = glm::vec4(-0.58, -0.76, -1.53, 0.33);
                sphere.albedo_and_roughness = glm::vec4(0.1, 0.7, 0.2, 0.5);
                sphere.emissive_and_ior = glm::vec4(0, 0, 0, 1);
                sphere.specular_and_coefficient = glm::vec4(1, 1, 1, 0.5);
            }
        }

        //----------------------------------------------------------------------
        // Render loop

        glfwSetKeyCallback(glfw_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {}
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        });

        glfwSetCursorPosCallback(glfw_window, &cursor_position_callback);
        glfwSetMouseButtonCallback(glfw_window, &mouse_button_callback);
        glfwSetScrollCallback(glfw_window, &scroll_callback);

        // The camera state
        float eye_angle_h = 0;
        float eye_angle_v = 0;
        float eye_dist = 7;

        uint64_t global_frame_number = 0;
        std::chrono::high_resolution_clock clock;
        auto start_time = std::chrono::high_resolution_clock::now();
        uint32_t progression_index = 0;
        glm::mat4 last_rendered_view_transform;


        while (!glfwWindowShouldClose(glfw_window)) {
            uint32_t mod_frame_number = static_cast<uint32_t>(global_frame_number % NUM_FRAMES_IN_FLIGHT);
            auto this_time = std::chrono::high_resolution_clock::now();
            uint64_t delta_time_mus =
                std::chrono::duration_cast<std::chrono::microseconds>(this_time - start_time).count();
            float delta_time_s = delta_time_mus / 1000000.0f;

            glfwPollEvents();

            PerFrame &this_frame = per_frames[mod_frame_number];
            auto cmd_buffer = this_frame.begin_frame();

            // Compute dispatch

            cmd_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlagBits::eByRegion,
                {},
                {},
                vk::ImageMemoryBarrier{}
                    .setImage(image)
                    .setOldLayout(vk::ImageLayout::eUndefined)
                    .setNewLayout(vk::ImageLayout::eGeneral)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderWrite)
                    .setSubresourceRange(vk::ImageSubresourceRange{}
                                             .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                             .setBaseMipLevel(0)
                                             .setLayerCount(1)
                                             .setLevelCount(1)
                                             .setBaseArrayLayer(0)));

            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline.pipeline.get());

            //----------------------------------------------------------------------
            // Update push constants
            {
                auto &m = compute_pipeline.push_constants.view_to_world_transform;

                if (g_right_mouse_dragger.has_delta()) { eye_dist += g_right_mouse_dragger.get_delta().y * 0.02; }
                eye_dist -= g_wheel_dragger;
                g_wheel_dragger = 0;
                eye_dist = std::min(eye_dist, 20.f);
                eye_dist = std::max(eye_dist, 1.f);

                if (g_left_mouse_dragger.has_delta()) {
                    auto delta = g_left_mouse_dragger.get_delta();
                    eye_angle_h += delta.x * 0.009;
                    eye_angle_v += delta.y * 0.009;
                }

                constexpr float PI_OVER_2 = 1.57079632679f;
                eye_angle_v = std::min(eye_angle_v, PI_OVER_2 * 0.95f);
                eye_angle_v = std::max(eye_angle_v, -PI_OVER_2 * 0.95f);

                glm::vec3 eye(0, 0, eye_dist);
                eye = glm::rotateX(eye, eye_angle_v);
                eye = glm::rotateY(eye, eye_angle_h);
                compute_pipeline.push_constants.view_to_world_transform =
                    glm::inverse(glm::lookAt(glm::vec3(eye), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)));

                if (last_rendered_view_transform != compute_pipeline.push_constants.view_to_world_transform) {
                    // Progression must start anew.
                    progression_index = 0;
                }
                last_rendered_view_transform = compute_pipeline.push_constants.view_to_world_transform;
            }
            compute_pipeline.push_constants.delta_time = delta_time_s;
            compute_pipeline.push_constants.progression_index = progression_index;
            cmd_buffer.pushConstants<ComputePipeline::PushConstants>(
                compute_pipeline.layout.get(), vk::ShaderStageFlagBits::eCompute, 0, compute_pipeline.push_constants);

            //----------------------------------------------------------------------

            cmd_buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, compute_pipeline.layout.get(), 0, compute_ds.front(), {});
            cmd_buffer.dispatch(W, H, 1);

            // Graphics dispatch

            cmd_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eFragmentShader,
                vk::DependencyFlagBits::eByRegion,
                {},
                {},
                vk::ImageMemoryBarrier{}
                    .setImage(image)
                    .setOldLayout(vk::ImageLayout::eGeneral)
                    .setNewLayout(vk::ImageLayout::eGeneral)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
                    .setSubresourceRange(vk::ImageSubresourceRange{}
                                             .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                             .setBaseMipLevel(0)
                                             .setLayerCount(1)
                                             .setLevelCount(1)
                                             .setBaseArrayLayer(0)));

            cmd_buffer.beginRenderPass(
                vk::RenderPassBeginInfo{}
                    .setRenderPass(graphics_pipeline.render_pass.get())
                    .setRenderArea(vk::Rect2D({0, 0}, {W, H}))
                    .setClearValues(vk::ClearValue{}.setColor(vk::ClearColorValue{}.setFloat32({0, 0, 0, 0})))
                    .setFramebuffer(fb_cache.get_or_create(vk::FramebufferCreateInfo{}
                                                               .setRenderPass(graphics_pipeline.render_pass.get())
                                                               .setWidth(W)
                                                               .setHeight(H)
                                                               .setLayers(1)
                                                               .setAttachments(this_frame.swapchain_image_view.get()))),
                vk::SubpassContents::eInline);
            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.pipeline.get());
            cmd_buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, graphics_pipeline.layout.get(), 0, ds.front(), {});
            cmd_buffer.draw(3, 1, 0, 0);
            cmd_buffer.endRenderPass();


            this_frame.end_frame();
            ++global_frame_number;
            ++progression_index;
        }
        device->waitIdle();

        g_one_shot_graphics_command_pool.reset();

        per_frames.clear();
    } catch (vk::SystemError &err) {
        std::cerr << "vk::SystemError: " << err.what() << "\n";
        exit(-1);
    } catch (std::exception &err) {
        std::cerr << "std::exception: " << err.what() << "\n";
        exit(-1);
    } catch (...) {
        std::cerr << "Unknown exception!\n";
        exit(-1);
    }

    return 0;
}
