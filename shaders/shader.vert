#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_pos_light_space;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 light_view_projection; 
    vec3 view_position; float _pad0;
    vec3 ambient_color; float _pad1;
	vec3 sun_direction; float _pad2;
	vec3 sun_color; float _pad3;
    float ambient_intensity;
    uint point_lights_count;
	uint spot_lights_count;
    uint render_mode;
};

layout(binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 m_color;
	float m_shininess;
};

void main() {
    vec4 world_position = model * vec4(v_position, 1.0f);
    vec4 normal = model * vec4(v_normal, 0.0f);

    gl_Position = view_projection * world_position;

    f_position = world_position.xyz;
    f_normal = normal.xyz;
    f_uv = v_uv;

    f_pos_light_space = light_view_projection * world_position;

    // for (uint i = 0; i < 2; ++i) {
    //     f_pos_spot_light_space[i] = spot_light_matrices[i] * world_position;
    // }
}
