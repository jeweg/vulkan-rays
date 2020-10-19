#include "swap_chain.hpp"
#include "device.hpp"

#include <iostream>


// Note that the swapchain image views cannot be part of
// this struct because the index to use depends on the
// result of acquireNextImageKHR, but that call requires us to
// specify the semaphore. We can't just round-robin (modulo) the
// frame index, that approach tends to go out of sync when the
// swapchain gets recreated. We can get out of that bind by separating
// the swapchain image views into their own array. We index that array
// by the number from acquireNextImageKHR, while the frame data sequence
// is indexed in a round-robin style -- that way there's no harm in
// the indices getting out of sync.
struct SwapChain::FrameData
{
    FrameData() = default;
    FrameData(const FrameData &) = delete;
    FrameData &operator=(const FrameData &) = delete;

    vk::UniqueCommandPool command_pool;
    vk::UniqueCommandBuffer command_buffer;

    vk::UniqueSemaphore image_available_for_rendering_sema;
    vk::UniqueSemaphore rendering_finished_sema;
    vk::UniqueFence finished_fence;
};


SwapChain::SwapChain(
    const Device &device, vk::SurfaceKHR surface, uint32_t width, uint32_t height, bool want_vsync, bool want_limiter) :
    _device(device), _surface(surface)
{
    auto surface_format = choose_best(
        device.get_physical_device().getSurfaceFormats2KHR(vk::PhysicalDeviceSurfaceInfo2KHR{}.setSurface(surface)),
        [](vk::SurfaceFormat2KHR sf) {
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
    _surface_format = surface_format.surfaceFormat;

    _present_mode =
        choose_best(device.get_physical_device().getSurfacePresentModesKHR(surface), [&](vk::PresentModeKHR pm) {
            // We assign some default weights first to have a reasonable fallback order
            // and then assign a much higher weight to the mode or modes that suit the
            // requested behavior obviously best.
            int weight = 0;
            switch (pm) {
            case vk::PresentModeKHR::eFifo: weight = 10; break;
            case vk::PresentModeKHR::eFifoRelaxed: weight = 9; break;
            case vk::PresentModeKHR::eMailbox: weight = 8; break;
            default:
            case vk::PresentModeKHR::eImmediate: weight = 7; break;
            }

            if (want_vsync && !want_limiter) {
                if (vk::PresentModeKHR::eMailbox == pm) {
                    weight = 1000;
                } else if (vk::PresentModeKHR::eFifoRelaxed == pm) {
                    weight = 100;
                }
            } else if (!want_vsync && !want_limiter) {
                if (vk::PresentModeKHR::eImmediate == pm) {
                    weight = 1000;
                } else if (vk::PresentModeKHR::eFifoRelaxed == pm) {
                    weight = 100;
                }
            } else if (want_vsync && want_limiter) {
                if (vk::PresentModeKHR::eFifo == pm) {
                    weight = 1000;
                } else if (vk::PresentModeKHR::eMailbox == pm) {
                    weight = 100;
                }
            } else if (!want_vsync && want_limiter) {
                if (vk::PresentModeKHR::eFifoRelaxed == pm) {
                    weight = 1000;
                } else if (vk::PresentModeKHR::eImmediate == pm) {
                    weight = 100;
                }
            }
            return weight;
        });

    std::cerr << "Selected present mode " << to_string(_present_mode) << "\n";

    _num_frames_in_flight = 1;
    // TODO: if (is_android or perhaps is_mobile) { _num_frames_in_flight = 3; }

    _extent.setWidth(width);
    _extent.setHeight(height);
    _recreate_on_begin_next_frame = true;
}


SwapChain::~SwapChain() noexcept = default;


void SwapChain::recreate()
{
    const auto &surface_caps = _device.get_physical_device()
                                   .getSurfaceCapabilities2KHR(vk::PhysicalDeviceSurfaceInfo2KHR{}.setSurface(_surface))
                                   .surfaceCapabilities;

    // Use identity if available, current otherwise.
    vk::SurfaceTransformFlagBitsKHR pre_transform =
        surface_caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity ?
            vk::SurfaceTransformFlagBitsKHR::eIdentity :
            surface_caps.currentTransform;

    vk::Extent2D extent;
    if (surface_caps.currentExtent.width == -1) {
        extent.width = clamp(_extent.width, surface_caps.minImageExtent.width, surface_caps.maxImageExtent.width);
        extent.height = clamp(_extent.height, surface_caps.minImageExtent.height, surface_caps.maxImageExtent.height);
    } else {
        extent = surface_caps.currentExtent;
    }

    _device.get().waitIdle();
    _frame_sequence.clear();
    _swapchain_image_views.clear();

    if (extent.width == 0 || extent.height == 0) {
        // We cannot create a zero-sized swapchain, yet when e.g. the window is minimized both
        // min and max image extent (from the surface caps) are (0, 0), and we must honor that as well.
        // The only solution is to not create a new swapchain. We'll destroy our resources to
        // indicate this state.
        _swapchain.reset();
        return;
    }

    auto swapchain_ci = vk::SwapchainCreateInfoKHR{}
                            .setSurface(_surface)
                            .setMinImageCount(surface_caps.minImageCount)
                            .setImageFormat(_surface_format.format)
                            .setImageColorSpace(_surface_format.colorSpace)
                            .setImageExtent(extent)
                            .setImageArrayLayers(1)
                            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                            .setImageSharingMode(vk::SharingMode::eExclusive)
                            .setPreTransform(pre_transform)
                            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                            .setPresentMode(_present_mode)
                            .setClipped(true)
                            .setOldSwapchain(_swapchain.get());

    std::array<uint32_t, 2> family_indices = {
        _device.get_family_index(Device::Queue::Graphics), _device.get_family_index(Device::Queue::Present)};
    if (family_indices.front() != family_indices.back()) {
        // If the graphics and present queues are from different queue families,
        // we either have to explicitly transfer ownership of images between the
        // queues, or we have to create the swapchain with imageSharingMode
        // as VK_SHARING_MODE_CONCURRENT
        swapchain_ci.setImageSharingMode(vk::SharingMode::eConcurrent);
        swapchain_ci.setQueueFamilyIndices(family_indices);
    }
    _swapchain = _device.get().createSwapchainKHRUnique(swapchain_ci);
    _extent = swapchain_ci.imageExtent;

    //----------------------------------------------------------------------
    // Initialize per-frame data <_num_frames_in_flight> times

    _frame_sequence.reserve(_num_frames_in_flight);
    for (size_t i = 0; i < _num_frames_in_flight; ++i) {
        auto fd = std::make_unique<FrameData>();

        fd->command_pool = _device.get().createCommandPoolUnique(
            vk::CommandPoolCreateInfo{}
                .setQueueFamilyIndex(_device.get_family_index(Device::Queue::Graphics))
                .setFlags(vk::CommandPoolCreateFlagBits::eTransient));

        fd->image_available_for_rendering_sema = _device.get().createSemaphoreUnique(vk::SemaphoreCreateInfo{});
        fd->rendering_finished_sema = _device.get().createSemaphoreUnique(vk::SemaphoreCreateInfo{});
        fd->finished_fence =
            _device.get().createFenceUnique(vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));
        _frame_sequence.push_back(std::move(fd));
    }

    std::vector<vk::Image> swapchain_images = _device.get().getSwapchainImagesKHR(_swapchain.get());
    for (vk::Image swapchain_image : swapchain_images) {
        _swapchain_image_views.push_back(std::move(_device.get().createImageViewUnique(
            vk::ImageViewCreateInfo{}
                .setImage(swapchain_image)
                .setViewType(vk::ImageViewType::e2D)
                .setSubresourceRange(vk::ImageSubresourceRange{}
                                         .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                         .setBaseMipLevel(0)
                                         .setLevelCount(1)
                                         .setBaseArrayLayer(0)
                                         .setLayerCount(1))
                .setFormat(_surface_format.format))));
    }
}


vk::Format SwapChain::get_format() const
{
    return _surface_format.format;
}


uint32_t SwapChain::get_num_frames_in_flight() const
{
    return _num_frames_in_flight;
}


SwapChain::FrameImage::FrameImage(SwapChain &base, uint32_t frame_data_index, uint32_t swapchain_image_index) :
    _base_swap_chain(base), _frame_data_index(frame_data_index), _swapchain_image_index(swapchain_image_index)
{}


SwapChain::FrameImage::~FrameImage() noexcept
{
    try {
        _base_swap_chain.end_current_frame(*this);
    } catch (vk::SystemError &err) {
        std::cerr << "vk::SystemError: " << err.what() << "\n";
        ASSUME(false);
    } catch (std::exception &err) {
        std::cerr << "std::exception: " << err.what() << "\n";
        ASSUME(false);
    } catch (...) {
        std::cerr << "Unknown exception!\n";
        ASSUME(false);
    }
}


bool SwapChain::FrameImage::is_valid() const
{
    return _frame_data_index != -1;
}


uint32_t SwapChain::get_image_count() const
{
    return to_uint32(_swapchain_image_views.size());
}


vk::CommandBuffer SwapChain::FrameImage::get_cmd_buffer(Device::Queue queue)
{
    ASSUME(is_valid());
    return _base_swap_chain.get_cmd_buffer(*this, queue);
}


vk::ImageView SwapChain::FrameImage::get_image_view()
{
    ASSUME(is_valid());
    return _base_swap_chain.get_image_view(*this);
}


vk::CommandBuffer SwapChain::get_cmd_buffer(const FrameImage &frame_image, Device::Queue queue)
{
    // Check sanity -- otherwise the frame control has gone out of whack.
    ASSUME(frame_image._frame_data_index == _current_frame_index);
    FrameData &fd = *_frame_sequence[_current_frame_index];
    return fd.command_buffer.get();
}


vk::ImageView SwapChain::get_image_view(const FrameImage &frame_image)
{
    // Check sanity -- otherwise the frame control has gone out of whack.
    ASSUME(frame_image._frame_data_index == _current_frame_index);
    return _swapchain_image_views[frame_image._swapchain_image_index].get();
}


SwapChain::FrameImage SwapChain::begin_next_frame()
{
    _current_frame_index = static_cast<uint32_t>(_frame_counter % _num_frames_in_flight);
    ++_frame_counter;

    if (_recreate_on_begin_next_frame) {
        recreate();
        _recreate_on_begin_next_frame = false;
    }
    if (!_swapchain) {
        // Return a special invalid FrameImage object to indicate that we
        // can't render right now. Perhaps the window is minimized.
        return FrameImage(*this, -1, -1);
    }

    FrameData *fd = _frame_sequence[_current_frame_index].get();
    _device.get().waitForFences({fd->finished_fence.get()}, true, -1);

    _device.get().resetFences({fd->finished_fence.get()});
    _device.get().resetCommandPool(fd->command_pool.get(), {});

    uint32_t swapchain_image_index = -1;
    do {
        if (_recreate_on_begin_next_frame) {
            recreate();
            _recreate_on_begin_next_frame = false;
        }
        if (!_swapchain) { return FrameImage(*this, -1, -1); }

        // Update helper ptr b/c the sequence as possibly rebuilt.
        fd = _frame_sequence[_current_frame_index].get();

        try {
            swapchain_image_index = _device.get().acquireNextImageKHR(
                _swapchain.get(), -1, fd->image_available_for_rendering_sema.get(), {});
        } catch (const vk::OutOfDateKHRError &) {
            // Loop around, recreate swapchain.
            _recreate_on_begin_next_frame = true;
        }
    } while (_recreate_on_begin_next_frame);

    ASSUME(swapchain_image_index != -1);

    auto command_buffers = _device.get().allocateCommandBuffersUnique(
        vk::CommandBufferAllocateInfo{}.setCommandPool(fd->command_pool.get()).setCommandBufferCount(1));
    ASSUME(command_buffers.size() == 1);
    fd->command_buffer = std::move(command_buffers.front());
    fd->command_buffer->begin(vk::CommandBufferBeginInfo{});

    return FrameImage(*this, _current_frame_index, swapchain_image_index);
}


void SwapChain::end_current_frame(FrameImage &frame_image)
{
    // Acquire swapchain image, submit the command buffer, present the image.

    if (!frame_image.is_valid()) { return; }

    // Check sanity -- otherwise the frame control has gone out of whack.
    ASSUME(frame_image._frame_data_index == _current_frame_index);
    FrameData &fd = *_frame_sequence[_current_frame_index];

    fd.command_buffer->end();

    vk::PipelineStageFlags dst_stage_mask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    auto submit_info = vk::SubmitInfo{}
                           .setCommandBuffers(fd.command_buffer.get())
                           .setWaitSemaphores(fd.image_available_for_rendering_sema.get())
                           .setSignalSemaphores(fd.rendering_finished_sema.get())
                           .setWaitDstStageMask(dst_stage_mask);
    _device.get_queue(Device::Queue::Graphics).submit({submit_info}, fd.finished_fence.get());

    try {
        _device.get_queue(Device::Queue::Present)
            .presentKHR(vk::PresentInfoKHR{}
                            .setWaitSemaphores(fd.rendering_finished_sema.get())
                            .setSwapchains(_swapchain.get())
                            .setImageIndices(frame_image._swapchain_image_index));
    } catch (const vk::OutOfDateKHRError &) {
        // Swapchain got invalidated.
        _recreate_on_begin_next_frame = true;
        return;
    }
}


void SwapChain::set_extent(uint32_t width, uint32_t height)
{
    // Only mark it invalidated.
    // We only really need the recreated resources when begin_next_frame is called, right?
    // We'll update _extent right away, though, since it's visible through the API.
    _extent.setWidth(width);
    _extent.setHeight(height);
    _recreate_on_begin_next_frame = true;
}
