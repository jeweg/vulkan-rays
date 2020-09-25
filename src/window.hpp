#pragma once

#include "swap_chain.hpp"
#include "vulkan/vulkan.hpp"
#include <memory>

class Device;
struct GLFWwindow;

class Window
{
public:
    Window(vk::Instance vk_instance, uint32_t w, uint32_t h, const char *title);
    ~Window();

    vk::SurfaceKHR get_surface() const;
    GLFWwindow *get_glfw_window();
    bool should_close();

    vk::Extent2D get_extent() const;
    uint32_t get_width() const;
    uint32_t get_height() const;

    void make_swap_chain(const Device &device, bool want_vsync = true, bool want_limiter = true);

    SwapChain &get_swap_chain();

    void destroy_swap_chain();

private:
    void on_resized(int w, int h);
    static void window_resized_callback(GLFWwindow *window, int w, int h);

    GLFWwindow *_glfw_window = nullptr;
    vk::Instance _vk_instance;
    VkSurfaceKHR _vk_surface = VK_NULL_HANDLE;
    std::unique_ptr<SwapChain> _swap_chain;
    uint32_t _width = -1;
    uint32_t _height = -1;
};
