#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform sampler2D tex_sampler;
layout(location = 0) in vec2 in_tex_coords;
layout(location = 0) out vec4 out_color;

void main(){
	bool b1 = mod(gl_FragCoord.x, 100) < 50;
	bool b2 = mod(gl_FragCoord.y, 100) < 50;
	if (b1 == b2) {
		out_color = vec4(1, 0, 0, 1);
	} else {
		out_color = vec4(texture(tex_sampler, in_tex_coords).xyz, 1);
		//out_color = vec4(in_tex_coords.xy, 0, 1);
	}
}