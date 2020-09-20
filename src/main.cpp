#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "utils.hpp"
#include "queues.hpp"
#include "build_info.hpp"
#include "debugbreak.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <string_view>
#include <iostream>

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
    VmaImage(VmaAllocator vma_allocator, VmaMemoryUsage usage, const VkImageCreateInfo &ci) :
        vma_allocator(vma_allocator)
    {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = usage;
        ASSUME(vmaCreateImage(vma_allocator, &ci, &alloc_info, &image, &vma_allocation, nullptr) == VK_SUCCESS);
    }

    ~VmaImage() noexcept { vmaDestroyImage(vma_allocator, image, vma_allocation); }

    operator vk::Image() const { return image; }
    explicit operator VkImage() const { return image; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
    VmaAllocation vma_allocation = VK_NULL_HANDLE;
    VkImage image;
};


struct VmaAllocatorGuard
{
    VmaAllocatorGuard(const VmaAllocatorCreateInfo &ci) { vmaCreateAllocator(&ci, &vma_allocator); }
    ~VmaAllocatorGuard() { vmaDestroyAllocator(vma_allocator); }
    operator VmaAllocator() const { return vma_allocator; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
};


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

        std::cerr << format_properties_to_string(phys_device, vk::Format::eR8G8B8A8Uint) << "\n";

        uint32_t family_indices[] = {queues.get_family_index(Queue::Graphics), queues.get_family_index(Queue::Compute)};
        VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        img_ci.imageType = VK_IMAGE_TYPE_2D;
        img_ci.extent.width = W;
        img_ci.extent.height = H;
        img_ci.extent.depth = 1;
        img_ci.mipLevels = 1;
        img_ci.arrayLayers = 1;
        img_ci.format = VK_FORMAT_R8G8B8A8_UINT;
        img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img_ci.usage = 0; // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
        img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
        if (queues.get_family_index(Queue::Graphics) != queues.get_family_index(Queue::Compute)) {
            img_ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            img_ci.queueFamilyIndexCount = 2;
        } else {
            img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            img_ci.queueFamilyIndexCount = 1;
        }
        img_ci.pQueueFamilyIndices = family_indices;
        img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT;

        VmaImage image = VmaImage(vma_allocator, VMA_MEMORY_USAGE_GPU_ONLY, img_ci);

        //----------------------------------------------------------------------
        // Graphics pipeline

        vk::UniquePipelineCache pipeline_cache = device->createPipelineCacheUnique(vk::PipelineCacheCreateInfo{});

        vk::UniqueRenderPass render_pass;
        vk::UniquePipeline graphics_pipeline;

        {
            vk::UniqueShaderModule vertex_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/vs.vert.spirv");
            vk::UniqueShaderModule fragment_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/fs.frag.spirv");

            render_pass = device->createRenderPassUnique(
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

            vk::UniquePipelineLayout pipeline_layout =
                device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{});

            graphics_pipeline = device->createGraphicsPipelineUnique(
                pipeline_cache.get(),
                vk::GraphicsPipelineCreateInfo{}
                    .setLayout(pipeline_layout.get())
                    .setRenderPass(render_pass.get())
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

        FramebufferCache fb_cache(device.get(), 10);

        //----------------------------------------------------------------------
        // Compute pipeline

        vk::UniquePipeline compute_pipeline;

        {
            vk::UniqueShaderModule compute_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/compute.comp.spirv");

            vk::UniqueDescriptorSetLayout dsl =
                device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{}.setBindings(
                    vk::DescriptorSetLayoutBinding{}
                        .setBinding(0)
                        .setDescriptorType(vk::DescriptorType::eStorageImage)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eCompute)));

            vk::UniquePipelineLayout pipeline_layout =
                device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{}.setSetLayouts(dsl.get()));

            compute_pipeline = device->createComputePipelineUnique(
                pipeline_cache.get(),
                vk::ComputePipelineCreateInfo{}
                    .setStage(vk::PipelineShaderStageCreateInfo{}
                                  .setStage(vk::ShaderStageFlagBits::eCompute)
                                  .setModule(compute_shader.get())
                                  .setPName("main"))
                    .setLayout(pipeline_layout.get()));
        }


        //----------------------------------------------------------------------
        // Render loop

        glfwSetKeyCallback(glfw_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
            if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {}
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        });

        uint64_t global_frame_number = 0;
        while (!glfwWindowShouldClose(glfw_window)) {
            uint32_t mod_frame_number = static_cast<uint32_t>(global_frame_number % NUM_FRAMES_IN_FLIGHT);
            glfwPollEvents();

            // std::cerr << "frame " << global_frame_number << " (mod: " << mod_frame_number << ")\n";

            PerFrame &this_frame = per_frames[mod_frame_number];
            auto cmd_buffer = this_frame.begin_frame();

            cmd_buffer.beginRenderPass(
                vk::RenderPassBeginInfo{}
                    .setRenderPass(render_pass.get())
                    .setRenderArea(vk::Rect2D({0, 0}, {W, H}))
                    .setClearValues(vk::ClearValue{}.setColor(vk::ClearColorValue{}.setFloat32({0, 0.7, 0, 1})))
                    .setFramebuffer(fb_cache.get_or_create(vk::FramebufferCreateInfo{}
                                                               .setRenderPass(render_pass.get())
                                                               .setWidth(W)
                                                               .setHeight(H)
                                                               .setLayers(1)
                                                               .setAttachments(this_frame.swapchain_image_view.get()))),
                vk::SubpassContents::eInline);

            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.get());
            cmd_buffer.draw(3, 1, 0, 0);
            cmd_buffer.endRenderPass();

            this_frame.end_frame();
            ++global_frame_number;
        }
        device->waitIdle();

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
