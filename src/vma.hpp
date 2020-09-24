#pragma once

#include "vk_mem_alloc.h"


namespace vma {

struct VmaAllocator
{
    Vma


    VmaAllocatorGuard(const VmaAllocatorCreateInfo &ci)
    {
        vmaCreateAllocator(&ci, &vma_allocator);
    }
    ~VmaAllocatorGuard() { vmaDestroyAllocator(vma_allocator); }
    operator VmaAllocator() const { return vma_allocator; }

    VmaAllocator vma_allocator = VK_NULL_HANDLE;
};

}
