#include "window.hpp"
#include "device.hpp"
#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"


Window::Window(vk::Instance vk_instance, uint32_t w, uint32_t h, const char *title) :
    _vk_instance(vk_instance), _width(w), _height(h)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    _glfw_window = glfwCreateWindow(w, h, title, nullptr, nullptr);
    ASSUME(_glfw_window);
    glfwSetWindowUserPointer(_glfw_window, this);
    check_vk_result(glfwCreateWindowSurface(_vk_instance, _glfw_window, nullptr, &_vk_surface));

    glfwSetFramebufferSizeCallback(_glfw_window, &window_resized_callback);
}

Window::~Window()
{
    if (_vk_surface) { vkDestroySurfaceKHR(_vk_instance, _vk_surface, nullptr); }
    if (_glfw_window) { glfwDestroyWindow(_glfw_window); }
}

vk::SurfaceKHR Window::get_surface() const
{
    return _vk_surface;
}
GLFWwindow *Window::get_glfw_window()
{
    return _glfw_window;
}
bool Window::should_close()
{
    return glfwWindowShouldClose(_glfw_window);
}

vk::Extent2D Window::get_extent() const
{
    return vk::Extent2D(_width, _height);
}
uint32_t Window::get_width() const
{
    return _width;
}
uint32_t Window::get_height() const
{
    return _height;
}

void Window::make_swap_chain(const Device &device, bool want_vsync, bool want_limiter)
{
    _swap_chain = std::make_unique<SwapChain>(device, _vk_surface, _width, _height, want_vsync, want_limiter);
}

SwapChain &Window::get_swap_chain()
{
    ASSUME(_swap_chain);
    return *_swap_chain;
}

void Window::destroy_swap_chain()
{
    _swap_chain.reset();
}

void Window::on_resized(int w, int h)
{
    _width = w;
    _height = h;
    // We might not wanna resize right away, just take notice of the fact that
    // stuff has to be recreated. We only want to actually recreate once we need it.
    // The size might change multiple times before we actually need the swap chain again.
    // But we can do let SwapChain worry about its own laziness.
    if (_swap_chain) { _swap_chain->set_extent(w, h); }
}

void Window::window_resized_callback(GLFWwindow *window, int w, int h)
{
    ASSUME(glfwGetWindowUserPointer(window));
    static_cast<Window *>(glfwGetWindowUserPointer(window))->on_resized(w, h);
}
