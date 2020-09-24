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
        VkSurfaceKHR surface,
        uint32_t width,
        uint32_t height,
        bool want_vsync = true,
        bool want_limiter = true);
    ~SwapChain() noexcept;
    SwapChain(const SwapChain &) = delete;
    SwapChain &operator=(const SwapChain &) = delete;

    vk::Format get_format() const;

    uint32_t get_num_frames_in_flight() const;

    class FrameImage
    {
    public:
        FrameImage(SwapChain &, uint32_t frame_index);
        ~FrameImage() noexcept;
        FrameImage(FrameImage &&) = default;
        FrameImage &operator=(FrameImage &&) = default;
        FrameImage(FrameImage &) = delete;
        FrameImage &operator=(FrameImage &) = delete;

        vk::CommandBuffer get_cmd_buffer(Device::Queue);
        vk::ImageView get_image_view();

    private:
        friend class SwapChain;
        SwapChain &_base_swap_chain;
        uint32_t _frame_index; // Stored here and in base for sanity checking.
    };

    // Begins frame, returns a RAII object with information that
    // will submit and present the frame when it goes out of scope.
    FrameImage begin_next_frame();

    void begin_next_frame2(vk::CommandBuffer *out_cb, vk::ImageView *out_iv);
    void end_current_frame2();

private:
    // Interface for FrameImage
    friend FrameImage;
    vk::CommandBuffer get_cmd_buffer(const FrameImage &, Device::Queue);
    vk::ImageView get_image_view(const FrameImage &);
    void end_current_frame(const FrameImage &);

private:
    struct FrameData;

    const Device &_device;
    uint32_t _num_frames_in_flight = 0;
    vk::UniqueSwapchainKHR _swapchain;
    vk::SurfaceFormatKHR _surface_format;
    std::vector<std::unique_ptr<FrameData>> _frame_sequence;
    uint64_t _frame_counter = 0;
    uint32_t _current_frame_index = -1;
};