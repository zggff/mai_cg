#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position; float _pad0;
    vec3 ambient_color; float _pad1;
	vec3 sun_direction; float _pad2;
	vec3 sun_color; float _pad3;
    float ambient_intensity;
    uint point_lights_count;
	uint spot_lights_count;
};

layout(binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 color;
	float shininess;
};

struct PointLight {
	vec4 position_radius;
	vec3 color;
};

struct SpotLight {
	vec4 position_radius;
	vec4 direction_angle;
	vec3 f_color;
};

layout(binding = 2, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(binding = 3, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};



void main() {
    vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(view_position - f_position);
    vec3 half_vector = normalize(view_dir - sun_direction);

    float sun_shade = max(0.0f, -dot(sun_direction, normal));
    vec3 sun_specular = pow(
                            max(0.0f, 
                                dot(normal, 
                                    half_vector)),
                            shininess) * sun_color;

    vec3 sun = sun_shade * sun_color * 
                               (color + sun_specular);

    vec3 ambient = ambient_color * ambient_intensity * color;

    final_color = vec4(ambient + sun, 0);
}
