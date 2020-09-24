#include "swap_chain.hpp"
#include "device.hpp"

#include <iostream>


struct SwapChain::FrameData
{
    FrameData() = default;
    FrameData(const FrameData &) = delete;
    FrameData &operator=(const FrameData &) = delete;

    vk::UniqueCommandPool command_pool;
    vk::UniqueCommandBuffer command_buffer;

    vk::UniqueImageView swapchain_image_view;
    vk::UniqueSemaphore image_available_for_rendering_sema;
    vk::UniqueSemaphore rendering_finished_sema;
    vk::UniqueFence finished_fence;
};


SwapChain::SwapChain(
    const Device &device, VkSurfaceKHR surface, uint32_t width, uint32_t height, bool want_vsync, bool want_limiter) :
    _device(device)
{
    auto surface_info = vk::PhysicalDeviceSurfaceInfo2KHR{}.setSurface(surface);
    auto surface_format =
        choose_best(device.get_physical_device().getSurfaceFormats2KHR(surface_info), [](vk::SurfaceFormat2KHR sf) {
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
    const auto &caps = device.get_physical_device().getSurfaceCapabilities2KHR(surface_info).surfaceCapabilities;

    vk::Extent2D extent;
    if (caps.currentExtent.width == -1) {
        extent.width = clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    } else {
        extent = caps.currentExtent;
    }

    auto present_mode =
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

    // Use identity if available, current otherwise.
    vk::SurfaceTransformFlagBitsKHR pre_transform =
        caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity ?
            vk::SurfaceTransformFlagBitsKHR::eIdentity :
            caps.currentTransform;

    _num_frames_in_flight = 2;
    // TODO: if (is_android or perhaps is_mobile) { _num_frames_in_flight = 3; }

    auto swapchain_ci = vk::SwapchainCreateInfoKHR{}
                            .setSurface(surface)
                            .setMinImageCount(_num_frames_in_flight)
                            .setImageFormat(_surface_format.format)
                            .setImageColorSpace(_surface_format.colorSpace)
                            .setImageExtent(extent)
                            .setImageArrayLayers(1)
                            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                            .setImageSharingMode(vk::SharingMode::eExclusive)
                            .setPreTransform(pre_transform)
                            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
                            .setPresentMode(present_mode)
                            .setClipped(true);
    std::array<uint32_t, 2> family_indices = {
        device.get_family_index(Device::Queue::Graphics), device.get_family_index(Device::Queue::Present)};
    if (family_indices.front() != family_indices.back()) {
        // If the graphics and present queues are from different queue families,
        // we either have to explicitly transfer ownership of images between the
        // queues, or we have to create the swapchain with imageSharingMode
        // as VK_SHARING_MODE_CONCURRENT
        swapchain_ci.setImageSharingMode(vk::SharingMode::eConcurrent);
        swapchain_ci.setQueueFamilyIndices(family_indices);
    }
    _swapchain = device.get().createSwapchainKHRUnique(swapchain_ci);

    //----------------------------------------------------------------------
    // Initialize per-frame data <_num_frames_in_flight> times

    std::vector<vk::Image> swapchain_images = device.get().getSwapchainImagesKHR(_swapchain.get());
    _frame_sequence.reserve(_num_frames_in_flight);
    for (size_t i = 0; i < _num_frames_in_flight; ++i) {
        auto fd = std::make_unique<FrameData>();

        fd->command_pool = device.get().createCommandPoolUnique(
            vk::CommandPoolCreateInfo{}
                .setQueueFamilyIndex(device.get_family_index(Device::Queue::Graphics))
                .setFlags(vk::CommandPoolCreateFlagBits::eTransient));

        fd->image_available_for_rendering_sema = device.get().createSemaphoreUnique(vk::SemaphoreCreateInfo{});
        fd->rendering_finished_sema = device.get().createSemaphoreUnique(vk::SemaphoreCreateInfo{});
        fd->finished_fence =
            device.get().createFenceUnique(vk::FenceCreateInfo{}.setFlags(vk::FenceCreateFlagBits::eSignaled));

        fd->swapchain_image_view = device.get().createImageViewUnique(
            vk::ImageViewCreateInfo{}
                .setImage(swapchain_images[i])
                .setViewType(vk::ImageViewType::e2D)
                .setSubresourceRange(vk::ImageSubresourceRange{}
                                         .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                         .setBaseMipLevel(0)
                                         .setLevelCount(1)
                                         .setBaseArrayLayer(0)
                                         .setLayerCount(1))
                .setFormat(_surface_format.format));

        _frame_sequence.push_back(std::move(fd));
    }
}


SwapChain::~SwapChain() noexcept = default;


vk::Format SwapChain::get_format() const
{
    return _surface_format.format;
}


uint32_t SwapChain::get_num_frames_in_flight() const
{
    return _num_frames_in_flight;
}


SwapChain::FrameImage::FrameImage(SwapChain &base, uint32_t frame_index) :
    _base_swap_chain(base), _frame_index(frame_index)
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


vk::CommandBuffer SwapChain::FrameImage::get_cmd_buffer(Device::Queue queue)
{
    return _base_swap_chain.get_cmd_buffer(*this, queue);
}


vk::ImageView SwapChain::FrameImage::get_image_view()
{
    return _base_swap_chain.get_image_view(*this);
}


vk::CommandBuffer SwapChain::get_cmd_buffer(const FrameImage &frame_image, Device::Queue queue)
{
    // Check sanity -- otherwise the frame control has gone out of whack.
    ASSUME(frame_image._frame_index == _current_frame_index);
    FrameData &fd = *_frame_sequence[_current_frame_index];

    return fd.command_buffer.get();
}


vk::ImageView SwapChain::get_image_view(const FrameImage &frame_image)
{
    // Check sanity -- otherwise the frame control has gone out of whack.
    ASSUME(frame_image._frame_index == _current_frame_index);
    FrameData &fd = *_frame_sequence[_current_frame_index];
    return fd.swapchain_image_view.get();
}


SwapChain::FrameImage SwapChain::begin_next_frame()
{
    _current_frame_index = static_cast<uint32_t>(_frame_counter % _num_frames_in_flight);
    ++_frame_counter;

    FrameData &fd = *_frame_sequence[_current_frame_index];

    _device.get().waitForFences({fd.finished_fence.get()}, true, -1);
    _device.get().resetFences({fd.finished_fence.get()});
    _device.get().resetCommandPool(fd.command_pool.get(), {});

    auto command_buffers = _device.get().allocateCommandBuffersUnique(
        vk::CommandBufferAllocateInfo{}.setCommandPool(fd.command_pool.get()).setCommandBufferCount(1));
    ASSUME(command_buffers.size() == 1);
    fd.command_buffer = std::move(command_buffers.front());
    fd.command_buffer->begin(vk::CommandBufferBeginInfo{});

    return FrameImage(*this, _current_frame_index);
}


void SwapChain::end_current_frame(const FrameImage &frame_image)
{
    // Acquire swapchain image, submit the command buffer, present the image.

    // Check sanity -- otherwise the frame control has gone out of whack.
    ASSUME(frame_image._frame_index == _current_frame_index);
    FrameData &fd = *_frame_sequence[_current_frame_index];

    fd.command_buffer->end();

    uint32_t image_index = -1;
    try {
        image_index =
            _device.get().acquireNextImageKHR(_swapchain.get(), -1, fd.image_available_for_rendering_sema.get(), {});
    } catch (const vk::OutOfDateKHRError &) {
        // Swapchain got invalidated.
        // TODO: recreate internal objects
        std::cerr << "broken swapchain!\n";
        return;
    }

    auto submit_info =
        vk::SubmitInfo{}
            .setCommandBuffers(fd.command_buffer.get())
            .setWaitSemaphores(fd.image_available_for_rendering_sema.get())
            .setSignalSemaphores(fd.rendering_finished_sema.get())
            .setWaitDstStageMask(vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput));
    _device.get_queue(Device::Queue::Graphics).submit({submit_info}, fd.finished_fence.get());

    _device.get_queue(Device::Queue::Present)
        .presentKHR(vk::PresentInfoKHR{}
                        .setWaitSemaphores(fd.rendering_finished_sema.get())
                        .setSwapchains(_swapchain.get())
                        .setImageIndices(image_index));
}
