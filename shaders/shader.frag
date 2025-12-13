#version 450

#define PI 3.1415926538

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_pos_light_space;

layout (location = 0) out vec4 final_color;

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

layout (binding = 2) uniform sampler2D albedo_texture;
layout (binding = 3) uniform sampler2D specular_texture;
layout (binding = 4) uniform sampler2D emissive_texture;

layout (set = 1, binding = 0) uniform sampler2DShadow shadowMap;

struct PointLight {
	vec3 position;
	float intensity;
	vec3 color;
};

struct SpotLight {
	vec3 position;
	float intensity;
	vec3 direction;
	float angle;
	vec3 color;
};

layout(binding = 5) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(binding = 6) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

float calculateShadow(vec4 lightSpacePos, vec3 normal, vec3 lightDir) {
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;

    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || 
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0;
    }

    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);

    float shadow = texture(shadowMap, vec3(projCoords.xy, projCoords.z - bias));

    return shadow; 
}

vec3 calculateBlinnPhong(vec3 lightDir,
						 vec3 lightColor, 
						 vec3 normal,
						 vec3 viewDir,
						 vec3 albedo,
						 float shininess) {
    float diff = max(dot(normal, lightDir), 0.0);
    if (diff == 0.0) return vec3(0.0);
    vec3 diffuse = albedo * lightColor * diff;
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), shininess);
    vec3 specular = lightColor * spec;
    float rimFactor = 1.0 - max(dot(normal, viewDir), 0.0);
    rimFactor = pow(rimFactor, 4.0);
    vec3 rim = lightColor * rimFactor * 0.3 * albedo;

    return diffuse + specular;
}

void main() {
	vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(view_position - f_position);

    vec2 texCoords = f_uv;
    if (render_mode == 1) {
        texCoords = sin(PI * f_uv / 2);
    }
    if (render_mode == 2) {
        texCoords = vec2(1 - f_uv.x, 1 - f_uv.y);
    }

    vec3 model_color = texture(albedo_texture, texCoords).rgb;
	float shininess = texture(specular_texture, texCoords).r * 128.0; // Scale shininess
    vec3 emissive_color = texture(emissive_texture, texCoords).rgb;

	vec3 color = ambient_intensity * ambient_color * model_color * 0.1;

	vec3 sun_dir = normalize(-sun_direction);
	float shadowFactor = calculateShadow(f_pos_light_space, normal, sun_dir);
    
    vec3 sun_contrib = calculateBlinnPhong(sun_dir, sun_color, normal, view_dir, model_color, shininess);
    color += sun_contrib * shadowFactor;

	color += emissive_color;
	
	for (uint i = 0; i < point_lights_count; ++i) {
		PointLight light = point_lights[i];
        vec3 light_vec = light.position - f_position;
        float distance = length(light_vec);
        vec3 ldir = normalize(light_vec);
        float light_falloff = light.intensity / (distance * distance + 0.0001);
        color += calculateBlinnPhong(ldir, light.color, normal, view_dir, model_color, shininess) * light_falloff;
	}

	// Spot lights (no shadow)
	for (uint i = 0; i < spot_lights_count; ++i) {
		SpotLight light = spot_lights[i];
		vec3 position = light.position;
		float intensity = light.intensity;
		vec3 light_direction = normalize(light.direction);
		float light_angle = light.angle;

		vec3 to_light = normalize(position - f_position);
		float spot_angle = dot(light_direction, -to_light);
		
		// Check if within spotlight cone
		if (spot_angle > light_angle) {
			vec3 half_vector = normalize(to_light + view_dir);
			float diff = max(dot(normal, to_light), 0.0);
			float spec = 0.0;
			if (diff > 0.0) {
				spec = pow(max(dot(normal, half_vector), 0.0), shininess);
			}

			float distance = length(position - f_position);
			float attenuation = 1.0 / (distance * distance);
			
			// Apply spotlight falloff
			float spot_factor = pow(clamp((spot_angle - light_angle) / (1.0 - light_angle), 0.0, 1.0), 2.0);

			vec3 diffuse = diff * model_color * light.color;
			vec3 specular = spec * light.color;

			vec3 spot_color = attenuation * intensity * spot_factor * (diffuse + specular);

			color += spot_color;
		}
	}

	
	final_color = vec4(color, 1.0);
}
