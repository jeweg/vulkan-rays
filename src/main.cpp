//#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"

#include "utils.hpp"
#include "device.hpp"
#include "swap_chain.hpp"
#include "build_info.hpp"
#include "debugbreak.h"

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_glfw.h"

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

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

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
bool g_window_resized = false;

static void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (!ImGui::GetIO().WantCaptureMouse) {
        g_left_mouse_dragger.update(xpos, ypos);
        g_right_mouse_dragger.update(xpos, ypos);
    }
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    if (!ImGui::GetIO().WantCaptureMouse) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) { g_left_mouse_dragger.start_dragging(); }
        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) { g_right_mouse_dragger.start_dragging(); }
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) { g_left_mouse_dragger.stop_dragging(); }
        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) { g_right_mouse_dragger.stop_dragging(); }
    }
}

static void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    if (!ImGui::GetIO().WantCaptureMouse) { g_wheel_dragger = static_cast<float>(yoffset); }
}


static void window_resized_callback(GLFWwindow *window, int width, int height)
{
    g_window_resized = true;
}


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
        std::vector<const char *> layer_names = {
            "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor",
            //"VK_LAYER_LUNARG_api_dump",
        };
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

        // We're initialializing Vulkan ourselves.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

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

        Device device(instance.get(), phys_device, surface);

        SwapChain swap_chain(device, surface, W, H);

        //----------------------------------------------------------------------
        // Create offscreen image

        constexpr vk::Format IMAGE_FORMAT = vk::Format::eR32G32B32A32Sfloat;

        VmaImage image = VmaImage(
            device.get_vma_allocator(),
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
                    device.get_family_index(Device::Queue::Graphics)
                            != device.get_family_index(Device::Queue::Compute) ?
                        vk::SharingMode::eConcurrent :
                        vk::SharingMode::eExclusive));

        vk::UniqueImageView image_view = device.get().createImageViewUnique(
            vk::ImageViewCreateInfo{}
                .setImage(image)
                .setFormat(IMAGE_FORMAT /*vk::Format::eR8G8B8A8Uint*/)
                .setViewType(vk::ImageViewType::e2D)
                .setSubresourceRange(vk::ImageSubresourceRange{}
                                         .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                         .setBaseMipLevel(0)
                                         .setLayerCount(1)
                                         .setLevelCount(1)
                                         .setBaseArrayLayer(0)));

        vk::UniqueSampler image_sampler = device.get().createSamplerUnique(vk::SamplerCreateInfo{});

        //----------------------------------------------------------------------
        // Shared pools and caches

        vk::UniquePipelineCache pipeline_cache = device.get().createPipelineCacheUnique(vk::PipelineCacheCreateInfo{});

        vk::UniqueDescriptorPool descriptor_pool = device.get().createDescriptorPoolUnique(
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
                device.get().createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{}.setBindings(
                    vk::DescriptorSetLayoutBinding{}
                        .setBinding(0)
                        .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eFragment)));

            graphics_pipeline.layout = device.get().createPipelineLayoutUnique(
                vk::PipelineLayoutCreateInfo{}.setSetLayouts(graphics_pipeline.dsl.get()));

            vk::UniqueShaderModule vertex_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/vs.vert.spirv");
            vk::UniqueShaderModule fragment_shader =
                load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/fs.frag.spirv");

            graphics_pipeline.render_pass = device.get().createRenderPassUnique(
                vk::RenderPassCreateInfo{}
                    .setAttachments(vk::AttachmentDescription{}
                                        .setFormat(swap_chain.get_format())
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
                                // TODO: probably no read access necessary here!
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

            graphics_pipeline.pipeline = device.get().createGraphicsPipelineUnique(
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
            device.get().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                    .setDescriptorPool(descriptor_pool.get())
                                                    .setDescriptorSetCount(1)
                                                    .setSetLayouts(graphics_pipeline.dsl.get()));
        device.get().updateDescriptorSets(
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
                device.get_vma_allocator(),
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
                device.get().createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{}.setBindings(
                    std::initializer_list<vk::DescriptorSetLayoutBinding>{dslb_0, dslb_1}));

            auto foo = sizeof(ComputePipeline::PushConstants);

            compute_pipeline.layout = device.get().createPipelineLayoutUnique(
                vk::PipelineLayoutCreateInfo{}
                    .setPushConstantRanges(vk::PushConstantRange{}
                                               .setOffset(0)
                                               .setSize(sizeof(ComputePipeline::PushConstants))
                                               .setStageFlags(vk::ShaderStageFlagBits::eCompute))
                    .setSetLayouts(compute_pipeline.dsl.get()));

            compute_pipeline.pipeline = device.get().createComputePipelineUnique(
                pipeline_cache.get(),
                vk::ComputePipelineCreateInfo{}
                    .setStage(vk::PipelineShaderStageCreateInfo{}
                                  .setStage(vk::ShaderStageFlagBits::eCompute)
                                  .setModule(compute_shader.get())
                                  .setPName("main"))
                    .setLayout(compute_pipeline.layout.get()));
        }

        std::vector<vk::DescriptorSet> compute_ds =
            device.get().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                    .setDescriptorPool(descriptor_pool.get())
                                                    .setDescriptorSetCount(1)
                                                    .setSetLayouts(compute_pipeline.dsl.get()));
        device.get().updateDescriptorSets(
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

        // ----------------------------------------------------------------------
        // More setup for GLFW and ImGui

        glfwSetKeyCallback(glfw_window, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
            if (!ImGui::GetIO().WantCaptureKeyboard) {
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { glfwSetWindowShouldClose(window, GLFW_TRUE); }
            }
        });
        glfwSetCursorPosCallback(glfw_window, &cursor_position_callback);
        glfwSetMouseButtonCallback(glfw_window, &mouse_button_callback);
        glfwSetScrollCallback(glfw_window, &scroll_callback);
        glfwSetFramebufferSizeCallback(glfw_window, &window_resized_callback);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        {
            ImGui_ImplVulkan_InitInfo ci = {0};
            ci.Instance = instance.get();
            ci.PhysicalDevice = phys_device;
            ci.Device = device.get();
            ci.QueueFamily = device.get_family_index(Device::Queue::Graphics);
            ci.Queue = device.get_queue(Device::Queue::Graphics);
            ci.PipelineCache = pipeline_cache.get();
            ci.DescriptorPool = descriptor_pool.get();
            ci.MinImageCount = swap_chain.get_num_frames_in_flight();
            ci.ImageCount = ci.MinImageCount;

            ci.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
            ci.CheckVkResultFn = [](VkResult result) { ASSUME(result == VK_SUCCESS); };
            ASSUME(ImGui_ImplVulkan_Init(&ci, graphics_pipeline.render_pass.get()));

            device.run_commands(
                Device::Queue::Graphics, [](vk::CommandBuffer cb) { ImGui_ImplVulkan_CreateFontsTexture(cb); });

            ASSUME(ImGui_ImplGlfw_InitForVulkan(glfw_window, true));
        }

        //----------------------------------------------------------------------
        // Render loop

        // The camera state
        float eye_angle_h = 0;
        float eye_angle_v = 0;
        float eye_dist = 7;

        auto start_time = std::chrono::high_resolution_clock::now();
        uint32_t progression_index = 0;
        glm::mat4 last_rendered_view_transform;

        while (!glfwWindowShouldClose(glfw_window)) {
            auto this_time = std::chrono::high_resolution_clock::now();
            uint64_t delta_time_mus =
                std::chrono::duration_cast<std::chrono::microseconds>(this_time - start_time).count();
            float delta_time_s = delta_time_mus / 1000000.0f;
            glfwPollEvents();

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // ImGui::ShowDemoWindow();
            /*
            ImGui::Begin("Controls", nullptr, 0);
            static float f1;
            ImGui::SliderFloat("Exposure", &f1, 0.1f, 20.0f, "%.2f");
            ImGui::End();
            */

            SwapChain::FrameImage frame = swap_chain.begin_next_frame();
            vk::CommandBuffer cmd_buffer = frame.get_cmd_buffer(Device::Queue::Graphics);

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

                if (g_right_mouse_dragger.has_delta()) { eye_dist += g_right_mouse_dragger.get_delta().y * 0.02f; }
                eye_dist -= g_wheel_dragger;
                g_wheel_dragger = 0;
                eye_dist = std::min(eye_dist, 20.f);
                eye_dist = std::max(eye_dist, 1.f);

                if (g_left_mouse_dragger.has_delta()) {
                    auto delta = g_left_mouse_dragger.get_delta();
                    eye_angle_h += delta.x * 0.009f;
                    eye_angle_v += delta.y * 0.009f;
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


            cmd_buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, compute_pipeline.layout.get(), 0, compute_ds.front(), {});
            cmd_buffer.dispatch(W, H, 1);

            //----------------------------------------------------------------------
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

            FramebufferKey fb_key;
            fb_key.render_pass = graphics_pipeline.render_pass.get();
            fb_key.extents = glm::ivec2(W, H);
            fb_key.attachments.push_back(frame.get_image_view());

            cmd_buffer.beginRenderPass(
                vk::RenderPassBeginInfo{}
                    .setRenderPass(graphics_pipeline.render_pass.get())
                    .setRenderArea(vk::Rect2D({0, 0}, {W, H}))
                    .setClearValues(vk::ClearValue{}.setColor(vk::ClearColorValue{}.setFloat32({0, 1, 0, 0})))
                    .setFramebuffer(device.get_framebuffer(fb_key)),
                vk::SubpassContents::eInline);
            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.pipeline.get());
            cmd_buffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics, graphics_pipeline.layout.get(), 0, ds.front(), {});
            cmd_buffer.draw(3, 1, 0, 0);

            // Render to internal data structures, then render those to to Vulkan.
            ImGui::Render();
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd_buffer);

            cmd_buffer.endRenderPass();

            ++progression_index;
        }
        device.get().waitIdle();

        ImGui_ImplVulkan_DestroyFontUploadObjects();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();

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
