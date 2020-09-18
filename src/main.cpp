#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "utils.hpp"
#include "queues.hpp"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <string_view>
#include <iostream>


VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_types,
    VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData,
    void *)
{
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(message_severity)) << "]"
              << "(" << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagBitsEXT>(message_types)) << ") "
              << pCallbackData->pMessage << "\n";
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
        GLFWwindow *glfw_window = glfwCreateWindow(1200, 800, "Example", nullptr, nullptr);
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
        VmaAllocator vma_allocator = VK_NULL_HANDLE;
        vmaCreateAllocator(&allocator_info, &vma_allocator);

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
            // ASSUME(VK_TRUE == phys_device.getSurfaceSupportKHR(queues.get_family_index(Queue::Present), surface));

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
                                    .setMinImageCount(2)
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

            vk::UniqueSwapchainKHR sc = device->createSwapchainKHRUnique(swapchain_ci);
            return std::move(sc);
        };

        vk::UniqueSwapchainKHR swapchain = create_swap_chain(1200, 800);

        //----------------------------------------------------------------------
        // Render loop

        glfwSetKeyCallback(glfw_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {}
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        });

        uint64_t global_frame = 0;
        while (!glfwWindowShouldClose(glfw_window)) { glfwPollEvents(); }
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
