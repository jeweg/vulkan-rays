#pragma once
#include "utils.hpp"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_glfw.h"
#include "vulkan/vulkan.hpp"


struct GuiHandler
{
    // RAII per-frame
    class Frame
    {
    public:
        Frame(vk::CommandBuffer cmd_buffer) : _cmd_buffer(cmd_buffer)
        {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
        }
        Frame(const Frame &) = delete;
        Frame &operator=(const Frame &) = delete;
        ~Frame() { render(); }
        void render()
        {
            if (!_rendered) {
                ImGui::Render();
                ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _cmd_buffer);
                _rendered = true;
            }
        }

    private:
        vk::CommandBuffer _cmd_buffer;
        bool _rendered = false;
    };


    GuiHandler()
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
    }

    ~GuiHandler()
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    // Reentrant!
    void init(
        vk::Instance instance,
        Device &device,
        Window &window,
        vk::DescriptorPool descriptor_pool,
        vk::PipelineCache pipeline_cache,
        vk::RenderPass render_pass)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();

        ASSUME(ImGui_ImplGlfw_InitForVulkan(window.get_glfw_window(), true));

        ImGui_ImplVulkan_InitInfo ci = {0};
        ci.Instance = instance;
        ci.PhysicalDevice = device.get_physical_device();
        ci.Device = device.get();
        ci.QueueFamily = device.get_family_index(Device::Queue::Graphics);
        ci.Queue = device.get_queue(Device::Queue::Graphics);
        ci.PipelineCache = pipeline_cache;
        ci.DescriptorPool = descriptor_pool;
        ci.MinImageCount = window.get_swap_chain().get_num_frames_in_flight();
        ci.ImageCount = ci.MinImageCount;

        ci.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        ci.CheckVkResultFn = [](VkResult result) { ASSUME(result == VK_SUCCESS); };
        ASSUME(ImGui_ImplVulkan_Init(&ci, render_pass));

        device.run_commands(
            Device::Queue::Graphics, [](vk::CommandBuffer cb) { ImGui_ImplVulkan_CreateFontsTexture(cb); });
    }


    Frame new_frame(vk::CommandBuffer command_buffer) { return Frame(command_buffer); }
};
