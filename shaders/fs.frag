#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform sampler2D tex_sampler;
layout(location = 0) in vec2 in_tex_coords;
layout(location = 0) out vec4 out_color;

void main(){
	// Flip the image here to be consistent with the ray tracer's assumptions
	out_color = vec4(texture(tex_sampler, vec2(in_tex_coords.x, 1.f - in_tex_coords.y)).xyz, 1);
}