cmake_minimum_required(VERSION 3.0)
project(vulkan-rays CXX) 

find_package(Vulkan REQUIRED)
add_subdirectory(3rd-party/glfw)

add_executable(vulkan-rays
	src/main.cpp
	src/queues.hpp
	src/queues.cpp
	src/utils.hpp
)
target_compile_features(vulkan-rays PRIVATE cxx_std_17)
target_link_libraries(vulkan-rays PRIVATE Vulkan::Vulkan glfw)
target_include_directories(vulkan-rays PRIVATE 3rd-party/vma/src)
