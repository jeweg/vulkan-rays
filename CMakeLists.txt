cmake_minimum_required(VERSION 3.0)
project(vulkan-rays CXX) 

find_package(Vulkan REQUIRED)
add_subdirectory(3rd-party/glfw)

set(sources
	src/main.cpp
	src/queues.hpp
	src/queues.cpp
	src/utils.hpp
)

set(glsl_sources
	shaders/vs.vert
	shaders/fs.frag
	shaders/compute.comp
)

set(spirv_files)
foreach(glsl_source IN LISTS glsl_sources)
	set(shader_stage)
    # Compatible w/ https://github.com/google/shaderc/tree/master/glslc#311-shader-stage-specification
	if (glsl_source MATCHES ".*\\.vert$")
		set(shader_stage vertex)
	elseif (glsl_source MATCHES ".*\\.frag$")
		set(shader_stage fragment)
	elseif (glsl_source MATCHES ".*\\.comp$")
		set(shader_stage compute)
	elseif (glsl_source MATCHES ".*\\.tesc$")
		set(shader_stage tesscontrol)
	elseif (glsl_source MATCHES ".*\\.tese$")
		set(shader_stage tesseval)
	elseif (glsl_source MATCHES ".*\\.geom$")
		set(shader_stage geometry)
	endif()
	set(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${glsl_source})
	set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${glsl_source}.spirv)
	set(command "glslc -fshader-stage=${shader_stage} ${input_file} -o ${output_file}")
	add_custom_command(
			OUTPUT ${output_file}
			COMMAND glslc -fshader-stage=${shader_stage} "${input_file}" -o "${output_file}"
			MAIN_DEPENDENCY ${input_file}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)
	list(APPEND spirv_files ${output_file})
endforeach()
add_custom_command(
		OUTPUT fragment.spv
		COMMAND "glslc -fshader-stage=fragment fragmentshader.glsl -o fragment.spv" #glslc is on the system path on my computer, so I am not currently worried about `find_package`ing it.
		MAIN_DEPENDENCY fragmentshader.glsl
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set(build_info_hpp_in "\
#pragma once
#include <string_view>
namespace build_info {
constexpr std::string_view PROJECT_SOURCE_DIR{\"${PROJECT_SOURCE_DIR}\"}\;
constexpr std::string_view PROJECT_BINARY_DIR{\"${PROJECT_BINARY_DIR}\"}\;
}
")
set(build_into_template_path "${CMAKE_CURRENT_BINARY_DIR}/build_info.hpp.in")
file(WRITE "${build_into_template_path}" ${build_info_hpp_in})
configure_file("${build_into_template_path}" "build_info.hpp")

add_executable(vulkan-rays
	${sources}
	${glsl_sources}
	${spirv_files}
)
target_compile_features(vulkan-rays PRIVATE cxx_std_17)
target_link_libraries(vulkan-rays PRIVATE Vulkan::Vulkan glfw)
target_include_directories(vulkan-rays PRIVATE
	3rd-party/vma/src 
	3rd-party/debugbreak 
	"${CMAKE_CURRENT_BINARY_DIR}"
)
