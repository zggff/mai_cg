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
	vec3 m_color;
	float shininess;
};

layout (binding = 2) uniform sampler2D albedo_texture;

struct PointLight {
	vec4 position_intensity;
	vec3 color;
};

struct SpotLight {
	vec4 position_intensity;
	vec4 direction_angle;
	vec3 color;
};

layout(binding = 3, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(binding = 4, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};



void main() {
    vec4 texel = texture(albedo_texture, f_uv);
    vec3 model_color = texel.rgb;

	if (shininess > 1) {
		final_color = vec4(model_color, 0);
		return;
	}
    vec3 norm = normalize(f_normal);
	vec3 to_camera = normalize(view_position - f_position);
    vec3 half_vector = normalize(to_camera - sun_direction);

    float sun_shade = max(0.0f, -dot(sun_direction, norm));
    vec3 sun_specular = pow(
                            max(0.0f, 
                                dot(norm, 
                                    half_vector)),
                            shininess) * sun_color;

    vec3 sun = sun_shade * sun_color * 
                               (model_color + sun_specular);

    vec3 ambient = ambient_color * ambient_intensity * model_color;

    vec3 color = ambient + sun * shininess;

	for (uint i = 0; i < point_lights_count; ++i) {
		PointLight light = point_lights[i];
		vec3 position = light.position_intensity.xyz;
		float intensity = light.position_intensity.w;

		vec3 to_light = normalize(position - f_position);
		vec3 half_vector = normalize(to_light + to_camera);

		float diff = max(dot(norm, to_light), 0.0);

		float spec = 0.0;
		if (diff > 0.0) {
			spec = pow(max(dot(norm, half_vector), 0.0), shininess);
		}

		float distance = length(position - f_position);
		float attenuation = 1.0 / (distance * distance);

		vec3 diffuse = diff * shininess * light.color;
		vec3 specular = spec * shininess * light.color;

		vec3 spot_color = attenuation * intensity * (diffuse + specular);

		color += spot_color;
	}

	for (uint i = 0; i < spot_lights_count; ++i) {
		SpotLight light = spot_lights[i];
		vec3 position = light.position_intensity.xyz;
		float intensity = light.position_intensity.w;
		vec3 light_direction = light.direction_angle.xyz;
		float light_angle = light.direction_angle.w;

		vec3 to_light = normalize(position - f_position);
		float spot_angle = -dot(light_direction, to_light);
		if (spot_angle > light_angle) {
			vec3 half_vector = normalize(to_light + to_camera);
			float diff = max(dot(norm, to_light), 0.0);
			float spec = 0.0;
			if (diff > 0.0) {
				spec = pow(max(dot(norm, half_vector), 0.0), shininess);
			}

			float distance = length(position - f_position);
			float attenuation = 1.0 / (distance * distance);

			vec3 diffuse = diff * shininess * light.color;
			vec3 specular = spec * shininess * light.color;

			vec3 spot_color = attenuation * intensity * (diffuse + specular);

			color += spot_color;


		}
	}

	final_color = vec4(color, 0);
}
