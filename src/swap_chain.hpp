#pragma once

#include "device.hpp"
#include "vulkan/vulkan.hpp"
#include <vector>
#include <memory>


class SwapChain
{
public:
    SwapChain(
        const Device &device,
        vk::SurfaceKHR surface,
        uint32_t width,
        uint32_t height,
        bool want_vsync = true,
        bool want_limiter = true);
    ~SwapChain() noexcept;
    SwapChain(const SwapChain &) = delete;
    SwapChain &operator=(const SwapChain &) = delete;

    vk::Format get_format() const;
    uint32_t get_num_frames_in_flight() const;
    vk::Extent2D get_extent() const { return _extent; }
    void set_extent(uint32_t width, uint32_t height);


    // RAII class returned from begin_next_frame.
    // We can use it to query data about the current frame's resources.
    // The exposed command buffer has begin() called on it already.
    // When FrameImage goes out of scope, the command buffer is ended
    // and the frame is submitted for presentation.
    class FrameImage
    {
    public:
        FrameImage(SwapChain &, uint32_t frame_index, uint32_t swapchain_image_index);
        ~FrameImage() noexcept;
        FrameImage(FrameImage &&) = default;
        FrameImage &operator=(FrameImage &&) = default;
        FrameImage(FrameImage &) = delete;
        FrameImage &operator=(FrameImage &) = delete;

        bool is_valid() const;
        vk::CommandBuffer get_cmd_buffer(Device::Queue);
        vk::ImageView get_image_view();

    private:
        friend class SwapChain;
        SwapChain &_base_swap_chain;
        uint32_t _frame_data_index;
        uint32_t _swapchain_image_index;
    };

    // Begins frame, returns a RAII object with information that
    // will submit and present the frame when it goes out of scope.
    FrameImage begin_next_frame();


private:
    // Interface for FrameImage
    friend FrameImage;
    vk::CommandBuffer get_cmd_buffer(const FrameImage &, Device::Queue);
    vk::ImageView get_image_view(const FrameImage &);
    void end_current_frame(FrameImage &);

private:
    void recreate();

private:
    struct FrameData;

    const Device &_device;
    vk::SurfaceKHR _surface;
    uint32_t _num_frames_in_flight = 0;
    vk::UniqueSwapchainKHR _swapchain;
    vk::SurfaceFormatKHR _surface_format;
    vk::PresentModeKHR _present_mode;
    std::vector<std::unique_ptr<FrameData>> _frame_sequence;
    std::vector<vk::UniqueImageView> _swapchain_image_views;
    uint64_t _frame_counter = 0;
    uint32_t _current_frame_index = -1;
    vk::Extent2D _extent;
    bool _recreate_on_begin_next_frame = false;
};