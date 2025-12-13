#include "veekay/types.hpp"
#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

#include <iostream>

/*
Матрица камеры рассчитывается с помощью матрицы Look-At. Должны быть реализованы следующие компоненты освещения: рассеянное, направленное и точечные источники света. Точечные источники света должны терять свою интенсивность по закону обратных квадратов
*/

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

using veekay::vec2, veekay::vec3, veekay::mat4;

struct SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec3 view_position;
	float _pad0;
	vec3 ambient_color;
	float _pad1;
	vec3 sun_direction;
	float _pad2;
	vec3 sun_color;
	float _pad3;
	float ambient_intensity;
	uint32_t point_lights_count;
	uint32_t spot_lights_count;
	uint32_t render_mode;
};

struct ModelUniforms {
	mat4 model;
	float shininess;
};

struct Mesh {
	veekay::graphics::Buffer *vertex_buffer;
	veekay::graphics::Buffer *index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	float shininess;
	VkDescriptorSet descriptor_set;
	// Model(VkDevice device);
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;
	constexpr static float mouse_sensitivity = 0.003;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	float pitch;
	float yaw;

	void rotate(veekay::vec2 rotation) {
		yaw += rotation.x * mouse_sensitivity;
		pitch -= rotation.y * mouse_sensitivity;
		float max_pitch = M_PI / 2 * 0.99;
		pitch = std::min(std::max(pitch, -max_pitch), max_pitch);
	}
	veekay::mat4 view() const {
		using namespace veekay;

		auto rotation_y = mat4::rotation({0.0, 1.0, 0.0}, yaw);
		auto rotation_x = mat4::rotation({1.0, 0.0, 0.0}, pitch);

		auto t = veekay::mat4::translation(-position);

		auto r = rotation_x * rotation_y;

		return t * mat4::transpose(r);
	}

	veekay::mat4 view_projection(float aspect_ratio) const {
		auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
		return view() * projection;
	}

	veekay::mat4 orthographic_matrix(float left, float right, float bottom, float top, float zNear, float zFar) const {
		veekay::mat4 result{};
		result[0][0] = 2.0f / (right - left);
		result[1][1] = 2.0f / (bottom - top); // Fixed: removed the duplicate assignment
		result[2][2] = 1.0f / (zNear - zFar);
		result[3][3] = 1.0f;

		result[3][0] = -(right + left) / (right - left);
		result[3][1] = -(bottom + top) / (bottom - top);
		result[3][2] = zNear / (zNear - zFar);

		return result;
	}

	veekay::mat4 ortho_view_projection(float left, float right, float bottom, float top, float zNear, float zFar) const {
		auto projection = this->orthographic_matrix(left, right, bottom, top, zNear, zFar);
		return this->view() * projection;
	}

	veekay::vec3 front() const {
		mat4 view = this->view();
		vec3 front = {view[0][2], view[1][2], view[2][2]};
		return front;
	}

	void look_at(vec3 target) {
		using namespace veekay;
		vec3 direction = vec3::normalized(target - this->position);

		this->yaw = std::atan2(direction.x, direction.z);
		this->pitch = -std::asin(direction.y);
	}
};

struct PointLight {
	veekay::vec3 position;
	float intensity;
	veekay::vec3 color;
};

struct SpotLight {
	veekay::vec3 position;
	float intensity;
	veekay::vec3 direction;
	float angle; // Косинус угла
	veekay::vec3 color;
	float _pad0;
};

struct ShadowPushConstants {
	veekay::mat4 model_matrix;
	veekay::mat4 light_view_proj;
};

// NOTE: Scene objects
inline namespace {
Camera camera{
	.position = {0.0f, -0.5f, -3.0f}};

std::vector<Model> models;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;
} // namespace

// NOTE: Vulkan objects
inline namespace {
VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkShaderModule shadow_vertex_shader_module;

VkDescriptorPool descriptor_pool;
VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorSetLayout descriptor_set_layout_shadow;

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

veekay::graphics::Buffer *scene_uniforms_buffer;
veekay::graphics::Buffer *model_uniforms_buffer;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 16;

veekay::graphics::Buffer *point_lights_buffer;
veekay::graphics::Buffer *spot_lights_buffer;

std::vector<Mesh> meshes;

veekay::graphics::Texture *missing_texture;
veekay::graphics::Texture *default_emissive;
veekay::graphics::Texture *default_specular;

VkSampler missing_texture_sampler;

std::vector<veekay::graphics::Texture *> textures;
VkSampler texture_sampler;

VkFormat os_color_image_format;		  // Формат пикселей
VkImage os_color_image;				  // Объект изображения
VkDeviceMemory os_color_image_memory; // Память изображения
VkImageView os_color_image_view;	  // Логическое изображение

/* Глубинный компонент изображения */
VkFormat os_depth_image_format;
VkImage os_depth_image;
VkDeviceMemory os_depth_image_memory;
VkImageView os_depth_image_view;

constexpr static uint32_t shadow_map_size = 4096;
constexpr static uint32_t max_shadow_casting_spots = 0;

VkPipelineLayout shadow_pipeline_layout;
VkPipeline shadow_pipeline;

VkImage shadow_image;
VkDeviceMemory shadow_image_memory;
VkImageView shadow_image_view;
VkSampler shadow_sampler;
VkDescriptorSet descriptor_set_shadow;

} // namespace
//

VkDescriptorSet descriptorWithTexture(VkDevice &device, veekay::graphics::Texture *texture = nullptr,
									  veekay::graphics::Texture *specular = nullptr, veekay::graphics::Texture *emissive = nullptr) {
	VkDescriptorSet descriptor_set;
	VkDescriptorSetAllocateInfo info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set\n";
		veekay::app.running = false;
		return nullptr;
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
			{
				.buffer = spot_lights_buffer->buffer,
				.offset = 0,
				.range = max_spot_lights * sizeof(SpotLight),
			},
		};

		VkDescriptorImageInfo image_infos[] = {
			{
				.sampler = texture ? texture_sampler : missing_texture_sampler,
				.imageView = texture ? texture->view : missing_texture->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			{
				.sampler = texture_sampler,
				.imageView = specular ? specular->view : default_specular->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			{
				.sampler = texture_sampler,
				.imageView = emissive ? emissive->view : default_emissive->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},

		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 5,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],

			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
							   write_infos, 0, nullptr);
	}
	return descriptor_set;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	// TODO: Scaling and rotation

	// auto r = veekay::mat4::rotation({})
	auto s = veekay::mat4::scaling(scale);
	auto t = veekay::mat4::translation(position);

	return s * t;
}

veekay::mat4 look_at_matrix(const veekay::vec3 &eye, const veekay::vec3 &target, const veekay::vec3 &world_up) {
	veekay::vec3 forward = veekay::vec3::normalized(eye - target);
	veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(world_up, forward));
	veekay::vec3 up = veekay::vec3::cross(forward, right);

	veekay::mat4 result;
	result[0][0] = right.x;
	result[0][1] = up.x;
	result[0][2] = forward.x;
	result[0][3] = 0.0f;
	result[1][0] = right.y;
	result[1][1] = up.y;
	result[1][2] = forward.y;
	result[1][3] = 0.0f;
	result[2][0] = right.z;
	result[2][1] = up.z;
	result[2][2] = forward.z;
	result[2][3] = 0.0f;
	result[3][0] = -veekay::vec3::dot(right, eye);
	result[3][1] = -veekay::vec3::dot(up, eye);
	result[3][2] = -veekay::vec3::dot(forward, eye);
	result[3][3] = 1.0f;

	return result;
}

veekay::mat4 orthographic_matrix(float left, float right, float bottom, float top, float zNear, float zFar) {
	veekay::mat4 result{};
	result[0][0] = 2.0f / (right - left);
	result[1][1] = 2.0f / (bottom - top);
	result[2][2] = 1.0f / (zNear - zFar);
	result[3][3] = 1.0f;

	result[3][0] = -(right + left) / (right - left);
	result[3][1] = -(bottom + top) / (bottom - top);
	result[3][2] = zNear / (zNear - zFar);

	return result;
}

bool useLookAt = true;

float angle = 0;
float radius = 5;

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char *path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char *>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void load_texture(VkCommandBuffer cmd, std::string path) {
	uint32_t width, height;
	std::vector<uint8_t> pixels;
	lodepng::decode(pixels, width, height, path);
	textures.push_back(new veekay::graphics::Texture(
		cmd, width, height,
		VK_FORMAT_R8G8B8A8_UNORM, // 8 бит на каждый канал цвета
		pixels.data()));
}
void load_texture_color(VkCommandBuffer cmd, veekay::vec4 color) {
	textures.push_back(new veekay::graphics::Texture(
		cmd, 1, 1,
		VK_FORMAT_R32G32B32A32_SFLOAT, // 8 бит на каждый канал цвета
		&color));
}

#define ASSET_TILES 0
#define ASSET_EARTH 1
#define ASSET_EARTH_SPECULAR 2
#define ASSET_BLACK 3
#define ASSET_RED 4
#define ASSET_GREEN 5

void load_textures(VkCommandBuffer cmd, VkDevice &device) {
	load_texture(cmd, "./assets/tiles.png");
	load_texture(cmd, "./assets/earth.png");
	load_texture(cmd, "./assets/earth gray.png");
	load_texture_color(cmd, {0.0, 0.0, 0.0, 1.0}); // ASSET_BLACK
	load_texture_color(cmd, {1.0, 0.0, 0.0, 1.0}); // ASSET_RED
	load_texture_color(cmd, {0.0, 1.0, 0.0, 1.0}); // ASSET_GREEN
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,				  // Фильтрация если плотность текселей меньше
			.minFilter = VK_FILTER_LINEAR,				  // Фильтрация если плотность больше
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, // Фильтрация мип-мапов
			// Что делать, если по какой-то из осей вышли за границы текстурных коорд-т
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.anisotropyEnable = true,	 // Включить анизотропную фильтрацию?
			.maxAnisotropy = 16.0f,		 // Кол-во сэмплов анизотропной фильтрации
			.minLod = 0.0f,				 // Минимальный уровень мипа
			.maxLod = VK_LOD_CLAMP_NONE, // Максимальный уровень мипа (тут бескоченость)
		};
		if (vkCreateSampler(device, &info, nullptr, &texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000,
			0xffff00ff,
			0xffff00ff,
			0xff000000,
		};
		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
														VK_FORMAT_B8G8R8A8_UNORM,
														pixels);
	}

	veekay::vec4 white = {1.0f, 1.0f, 1.0f, 1.0f};
	default_specular = new veekay::graphics::Texture(
		cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &white);
	veekay::vec4 black = {0.0f, 0.0f, 0.0f, 0.0f};
	default_emissive = new veekay::graphics::Texture(
		cmd, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, &black);
}

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage &image, VkDeviceMemory &imageMemory) {
	VkImageCreateInfo imageInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.extent.width = width,
		.extent.height = height,
		.extent.depth = 1,
		.mipLevels = 1,
		.arrayLayers = 1,
		.format = format,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.usage = usage,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	vkCreateImage(veekay::app.vk_device, &imageInfo, nullptr, &image);
	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(veekay::app.vk_device, image, &memRequirements);

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);

	uint32_t memoryTypeIndex = UINT32_MAX;
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
			memoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = memoryTypeIndex;

	vkAllocateMemory(veekay::app.vk_device, &allocInfo, nullptr, &imageMemory);
	vkBindImageMemory(veekay::app.vk_device, image, imageMemory, 0);
}

void createImageView(VkImage image, VkFormat format, VkImageView &imageView, VkImageAspectFlags aspectFlags) {
	VkImageViewCreateInfo viewInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange.aspectMask = aspectFlags,
		.subresourceRange.baseMipLevel = 0,
		.subresourceRange.levelCount = 1,
		.subresourceRange.baseArrayLayer = 0,
		.subresourceRange.layerCount = 1,
	};

	vkCreateImageView(veekay::app.vk_device, &viewInfo, nullptr, &imageView);
}

void insertImageBarrier(VkCommandBuffer cmd, VkImage image,
						VkAccessFlags srcAccess, VkAccessFlags dstAccess,
						VkImageLayout oldLayout, VkImageLayout newLayout,
						VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = srcAccess,
		.dstAccessMask = dstAccess,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
		.subresourceRange.baseMipLevel = 0,
		.subresourceRange.levelCount = 1,
		.subresourceRange.baseArrayLayer = 0,
		.subresourceRange.layerCount = 1,
	};
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void initialize(VkCommandBuffer cmd) {
	VkDevice &device = veekay::app.vk_device;
	VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

	load_textures(cmd, device);
	{
		VkSamplerCreateInfo samplerInfo{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR, // линейная фильтрация
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.mipLodBias = 0.0f,
			.maxAnisotropy = 1.0f,
			.minLod = 0.0f,
			.maxLod = 1.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
			.compareEnable = VK_TRUE,
			.compareOp = VK_COMPARE_OP_LESS,
		};

		vkCreateSampler(device, &samplerInfo, nullptr, &shadow_sampler);
	}

	createImage(shadow_map_size, shadow_map_size, VK_FORMAT_D32_SFLOAT,
				VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				shadow_image, shadow_image_memory);
	createImageView(shadow_image, VK_FORMAT_D32_SFLOAT, shadow_image_view, VK_IMAGE_ASPECT_DEPTH_BIT);

	// NOTE: Declare clockwise triangle order as front-facing
	//       Discard triangles that are facing away
	//       Fill triangles, don't draw lines instaed
	VkPipelineRasterizationStateCreateInfo raster_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	// NOTE: How many bytes does a vertex take?
	VkVertexInputBindingDescription buffer_binding{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	// NOTE: Declare vertex attributes
	VkVertexInputAttributeDescription attributes[] = {
		{
			.location = 0,						  // NOTE: First attribute
			.binding = 0,						  // NOTE: First vertex buffer
			.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
			.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, uv),
		},
	};

	// NOTE: Describe inputs
	VkPipelineVertexInputStateCreateInfo input_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &buffer_binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	// NOTE: Every three vertices make up a triangle,
	//       so our vertex buffer contains a "list of triangles"
	VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	// NOTE: Use 1 sample per pixel
	VkPipelineMultisampleStateCreateInfo sample_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = false,
		.minSampleShading = 1.0f,
	};

	// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
	VkPipelineDepthStencilStateCreateInfo depth_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	VkViewport viewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(veekay::app.window_width),
		.height = static_cast<float>(veekay::app.window_height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	VkRect2D scissor{
		.offset = {0, 0},
		.extent = {veekay::app.window_width, veekay::app.window_height},
	};

	// NOTE: Let rasterizer draw on the entire window
	VkPipelineViewportStateCreateInfo viewport_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

		.viewportCount = 1,
		.pViewports = &viewport,

		.scissorCount = 1,
		.pScissors = &scissor,
	};

	{
		VkDescriptorSetLayoutBinding bindings[1];

		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		bindings[0].pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.bindingCount = 1;
		info.pBindings = bindings;
		vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_shadow);
	}

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
							  VK_COLOR_COMPONENT_G_BIT |
							  VK_COLOR_COMPONENT_B_BIT |
							  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},

				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				}};

			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 100,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
									   &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 4,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},

				{
					.binding = 5,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 6,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
											&descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time

		VkDescriptorSetLayout layouts[] = {descriptor_set_layout, descriptor_set_layout_shadow};
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 2,
			.pSetLayouts = layouts,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
								   nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
									  1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	{ // NOTE: shadow pipeline
		shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
		VkPipelineShaderStageCreateInfo vertStage{};
		vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertStage.module = shadow_vertex_shader_module;
		vertStage.pName = "main";

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(ShadowPushConstants);

		VkPipelineLayoutCreateInfo shadowLayoutInfo{};
		shadowLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		shadowLayoutInfo.pushConstantRangeCount = 1;
		shadowLayoutInfo.pPushConstantRanges = &pushConstantRange;
		shadowLayoutInfo.setLayoutCount = 0;
		vkCreatePipelineLayout(device, &shadowLayoutInfo, nullptr, &shadow_pipeline_layout);

		VkPipelineRasterizationStateCreateInfo shadowRasterInfo = raster_info;
		shadowRasterInfo.depthBiasEnable = VK_TRUE;
		shadowRasterInfo.depthBiasConstantFactor = 1.25f;
		shadowRasterInfo.depthBiasSlopeFactor = 1.75f;

		VkPipelineDepthStencilStateCreateInfo shadowDepthInfo = depth_info;
		shadowDepthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

		VkVertexInputAttributeDescription shadowAttribute = {
			.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)};
		VkPipelineVertexInputStateCreateInfo shadowInputState = input_state_info;
		shadowInputState.vertexAttributeDescriptionCount = 1;
		shadowInputState.pVertexAttributeDescriptions = &shadowAttribute;

		VkViewport shadowViewport{};
		shadowViewport.width = (float)shadow_map_size;
		shadowViewport.height = (float)shadow_map_size;
		shadowViewport.minDepth = 0.0f;
		shadowViewport.maxDepth = 1.0f;
		VkRect2D shadowScissor{};
		shadowScissor.extent = {shadow_map_size, shadow_map_size};

		VkPipelineViewportStateCreateInfo shadowViewportState = viewport_info;
		shadowViewportState.viewportCount = 1;
		shadowViewportState.pViewports = &shadowViewport;
		shadowViewportState.scissorCount = 1;
		shadowViewportState.pScissors = &shadowScissor;

		VkPipelineRenderingCreateInfoKHR renderingInfo{};
		renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
		renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
		renderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		renderingInfo.colorAttachmentCount = 0;

		VkGraphicsPipelineCreateInfo shadowInfo{};
		shadowInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		shadowInfo.stageCount = 1;
		shadowInfo.pStages = &vertStage;
		shadowInfo.pVertexInputState = &shadowInputState;
		shadowInfo.pInputAssemblyState = &assembly_state_info;
		shadowInfo.pViewportState = &shadowViewportState;
		shadowInfo.pRasterizationState = &shadowRasterInfo;
		shadowInfo.pMultisampleState = &sample_info;
		shadowInfo.pDepthStencilState = &shadowDepthInfo;
		shadowInfo.pColorBlendState = nullptr;
		shadowInfo.layout = shadow_pipeline_layout;
		shadowInfo.renderPass = VK_NULL_HANDLE;
		shadowInfo.pNext = &renderingInfo;

		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &shadowInfo, nullptr, &shadow_pipeline);
	}

	{
		VkDescriptorSetLayout layouts[] = {descriptor_set_layout_shadow};
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptor_pool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = layouts;

		vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set_shadow);

		VkDescriptorImageInfo imageInfos[1 + max_shadow_casting_spots];
		VkWriteDescriptorSet writes[1 + max_shadow_casting_spots];

		// Направленный свет
		imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		imageInfos[0].imageView = shadow_image_view;
		imageInfos[0].sampler = shadow_sampler;

		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].pNext = nullptr;
		writes[0].dstSet = descriptor_set_shadow;
		writes[0].dstBinding = 0;
		writes[0].dstArrayElement = 0;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[0].descriptorCount = 1;
		writes[0].pImageInfo = &imageInfos[0];
		writes[0].pBufferInfo = nullptr;
		writes[0].pTexelBufferView = nullptr;

		vkUpdateDescriptorSets(device, 1, writes, 0, nullptr);
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	point_lights_buffer = new veekay::graphics::Buffer(
		max_point_lights * sizeof(PointLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	spot_lights_buffer = new veekay::graphics::Buffer(
		max_spot_lights * sizeof(SpotLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0};

		meshes.push_back(Mesh());

		meshes.back().vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		meshes.back().index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		meshes.back().indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0,
			1,
			2,
			2,
			3,
			0,
			4,
			5,
			6,
			6,
			7,
			4,
			8,
			9,
			10,
			10,
			11,
			8,
			12,
			13,
			14,
			14,
			15,
			12,
			16,
			17,
			18,
			18,
			19,
			16,
			20,
			21,
			22,
			22,
			23,
			20,
		};

		meshes.push_back(Mesh());
		meshes.back().vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		meshes.back().index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		meshes.back().indices = uint32_t(indices.size());
	}
	// NOTE: Sphere mesh initialization
	{
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		const int vertex_cnt = 100;
		const float radius = 1.0;

		for (int i = 0; i <= vertex_cnt; ++i) {
			float stackAngle = M_PI / 2 - i * M_PI / vertex_cnt; // from pi/2 to -pi/2
			float xy = radius * cosf(stackAngle);
			float y = radius * sinf(stackAngle);

			for (int j = 0; j <= vertex_cnt; ++j) {
				float sectorAngle = j * 2 * M_PI / vertex_cnt; // from 0 to 2pi

				float x = xy * cosf(sectorAngle);
				float z = xy * sinf(sectorAngle);

				vec3 pos({x, y, z});
				vec3 normal = vec3::normalized(pos);
				vec2 uv({(float)j / vertex_cnt, 1 - (float)i / vertex_cnt});

				vertices.push_back({pos, normal, uv});
			}
		}

		// Indices
		for (int i = 0; i < vertex_cnt; ++i) {
			int k1 = i * (vertex_cnt + 1);
			int k2 = k1 + vertex_cnt + 1;

			for (int j = 0; j < vertex_cnt; ++j, ++k1, ++k2) {
				if (i != 0) {
					indices.push_back(k1); // triangle 1
					indices.push_back(k2);
					indices.push_back(k1 + 1);
				}

				if (i != (vertex_cnt - 1)) {
					indices.push_back(k1 + 1); // triangle 2
					indices.push_back(k2);
					indices.push_back(k2 + 1);
				}
			}
		}
		meshes.push_back(Mesh());
		meshes.back().vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		meshes.back().index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		meshes.back().indices = uint32_t(indices.size());
	}

	// NOTE: Add models to scene
	models.emplace_back(Model{
		.mesh = meshes[0],
		.transform = Transform{
			.position = {0.0, 2.0, 0.0}},
		.shininess = 0.1,
		.descriptor_set = descriptorWithTexture(device, textures[ASSET_TILES]),
	});

	models.emplace_back(Model{
		.mesh = meshes[1],
		.transform = Transform{
			.position = {-1.0f, 0.5f, 0.5f},
		},
		.shininess = 0.4,
		.descriptor_set = descriptorWithTexture(device, nullptr),
	});
	// earth
	models.emplace_back(Model{
		.mesh = meshes[2],
		.transform = Transform{
			.position = {1.0f, 0.0f, 0.5f},
		},
		.shininess = 0.4,
		.descriptor_set = descriptorWithTexture(device, textures[ASSET_EARTH], textures[ASSET_EARTH_SPECULAR]),
	});

	point_lights.push_back(PointLight{
		.color = vec3({0.0, 1.0, 0.0}),
		.intensity = 1,
		.position = vec3({(float)(0.0 + radius * std::sin(angle)),
						  -2,
						  (float)(0.0 + radius * std::cos(angle))})

	});

	spot_lights.push_back(SpotLight{
		.color = vec3({1.0, 0.0, 0.0}),
		.intensity = 1,
		.position = vec3({2,
						  -1,
						  2}),
		.direction = {-2, 0, -2},
		.angle = static_cast<float>(std::cos(M_PI / 4)),
	});

	spot_lights.push_back(SpotLight{
		.color = vec3({1.0, 1.0, 0.0}),
		.intensity = 5,
		.angle = static_cast<float>(std::cos(M_PI / 16)),
	});

	models.emplace_back(Model{
		.mesh = meshes[2],
		.transform = Transform{
			.scale = {0.1, 0.1, 0.1},
			.position = {2, -1, 2},
		},
		.shininess = 1.0,
		.descriptor_set = descriptorWithTexture(device, textures[ASSET_RED], textures[ASSET_BLACK], textures[ASSET_RED]),
	});

	models.emplace_back(Model{
		.mesh = meshes[2],
		.transform = Transform{
			.scale = {0.1, 0.1, 0.1},
			.position = {
				float(0.0 + radius * std::sin(angle)),
				-2,
				float(0.0 + radius * std::cos(angle))},
		},
		.shininess = 1.0,
		.descriptor_set = descriptorWithTexture(device, textures[ASSET_GREEN], textures[ASSET_BLACK], textures[ASSET_GREEN]),
	});

	insertImageBarrier(cmd, shadow_image,
					   0, 0,
					   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
					   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice &device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	vkDestroySampler(device, texture_sampler, nullptr);

	vkDestroySampler(device, shadow_sampler, nullptr);
	vkDestroyImageView(device, shadow_image_view, nullptr);
	vkFreeMemory(device, shadow_image_memory, nullptr);
	vkDestroyImage(device, shadow_image, nullptr);

	for (auto &texture : textures) {
		delete texture;
	}
	delete missing_texture;
	delete default_emissive;
	delete default_specular;

	for (auto &mesh : meshes) {
		delete mesh.index_buffer;
		delete mesh.vertex_buffer;
	}

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;
	delete spot_lights_buffer;
	delete point_lights_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(device, descriptor_set_layout_shadow, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);

	vkDestroyPipeline(device, shadow_pipeline, nullptr);
	vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);

	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);
}

int mode = 0;

float sun_pitch = 2;
float sun_yaw = 0;
void update(double time) {
	ImGui::Begin("Controls:");
	ImGui::Checkbox("use look at for camera", &useLookAt);

	ImGui::SliderAngle("sunPitch", &sun_pitch, 0, 360);
	ImGui::SliderAngle("sunYaw", &sun_yaw, 0, 360);

	ImGui::SliderFloat("radius", &radius, 1, 10);
	ImGui::InputInt("mode", &mode, 1, 1);
	ImGui::End();

	if (!ImGui::IsWindowHovered()) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::right)) {
			auto move_delta = mouse::cursorDelta();
			camera.rotate(move_delta);

			auto view = camera.view();

			veekay::vec3 right = {view[0][0], view[1][0], view[2][0]};
			veekay::vec3 up = {view[0][1], view[1][1], view[2][1]};
			veekay::vec3 front = {view[0][2], view[1][2], view[2][2]};

			if (keyboard::isKeyDown(keyboard::Key::w))
				camera.position += front * 0.1f;

			if (keyboard::isKeyDown(keyboard::Key::s))
				camera.position -= front * 0.1f;

			if (keyboard::isKeyDown(keyboard::Key::d))
				camera.position += right * 0.1f;

			if (keyboard::isKeyDown(keyboard::Key::a))
				camera.position -= right * 0.1f;

			if (keyboard::isKeyDown(keyboard::Key::space))
				camera.position -= up * 0.1f;

			if (keyboard::isKeyDown(keyboard::Key::left_shift))
				camera.position += up * 0.1f;
		}
	}

	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.view_position = camera.position,
		.ambient_color = {1.0, 1.0, 1.0},
		.ambient_intensity = 0.8,
		.sun_color = {0.5, 0.5, 0.5},
		.sun_direction = {},
		.point_lights_count = (uint32_t)point_lights.size(),
		.spot_lights_count = (uint32_t)spot_lights.size(),
		.render_mode = (uint32_t)mode,
	};

	vec3 light_pos_norm = {sin(sun_yaw) * cos(sun_pitch),
					  sin(sun_pitch),
					  cos(sun_yaw) * cos(sun_pitch)};

	auto sun_camera = Camera();
	sun_camera.position = -light_pos_norm * 20;
	sun_camera.look_at({0, 0, 0});
	sun_camera.far_plane = 100.0f;
	sun_camera.near_plane = 1.0f;
	sun_camera.fov = 100;
	scene_uniforms.sun_direction = sun_camera.front();

	veekay::vec3 light_dir = veekay::vec3::normalized(scene_uniforms.sun_direction);
	veekay::vec3 light_pos = -light_dir * 20.0f;

	veekay::mat4 light_view = look_at_matrix(light_pos, {0, 0, 0}, {0, 1, 0});

	float ortho_size = 50.0f;
	float z_near = 1.0f;
	float z_far = 50.0f;

	// матрица проекции - ортогональная "коробка" 50 на 50
	// ось Y вниз, так как вулкан
	veekay::mat4 light_proj = orthographic_matrix(-ortho_size, ortho_size, -ortho_size, ortho_size, z_near, z_far);

	// итоговая матрица света
	veekay::mat4 light_view_projection = light_view * light_proj;

	scene_uniforms.light_view_projection = sun_camera.ortho_view_projection(-ortho_size, ortho_size, -ortho_size, ortho_size, z_near, z_far);
	scene_uniforms.light_view_projection = light_view_projection;

	{
		models[4].transform.position = {
			float(0.0 + radius * std::sin(time)),
			-2,
			float(0.0 + radius * std::cos(time))};
		point_lights[0].position = {
			float(0.0 + radius * std::sin(time)),
			-2,
			float(0.0 + radius * std::cos(time))};
		spot_lights[1].direction = camera.front();
		spot_lights[1].position = camera.position;
	}

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model &model = models[i];
		ModelUniforms &uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.shininess = model.shininess;
	}

	*(SceneUniforms *)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t model_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms &uniforms = model_uniforms[i];

		char *const pointer = static_cast<char *>(model_uniforms_buffer->mapped_region) + i * model_alignment;
		*reinterpret_cast<ModelUniforms *>(pointer) = uniforms;
	}

	const size_t point_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(PointLight));

	for (size_t i = 0, n = point_lights.size(); i < n; ++i) {
		const PointLight &uniforms = point_lights[i];

		char *const pointer = static_cast<char *>(point_lights_buffer->mapped_region) + i * point_alignment;
		*reinterpret_cast<PointLight *>(pointer) = uniforms;
	}

	const size_t spot_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(SpotLight));

	for (size_t i = 0, n = spot_lights.size(); i < n; ++i) {
		const SpotLight &uniforms = spot_lights[i];

		char *const pointer = static_cast<char *>(spot_lights_buffer->mapped_region) + i * spot_alignment;
		*reinterpret_cast<SpotLight *>(pointer) = uniforms;
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	SceneUniforms *scene_data = (SceneUniforms *)scene_uniforms_buffer->mapped_region;
	veekay::mat4 light_VP = scene_data->light_view_projection;

	auto vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(
		veekay::app.vk_device, "vkCmdBeginRenderingKHR");
	auto vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(
		veekay::app.vk_device, "vkCmdEndRenderingKHR");

	{
        insertImageBarrier(cmd, shadow_image,
                           0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

        VkRenderingAttachmentInfoKHR depthAttachment{};
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
        depthAttachment.imageView = shadow_image_view; 
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; 
        depthAttachment.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfoKHR renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
        renderingInfo.renderArea = {{0, 0}, {shadow_map_size, shadow_map_size}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 0;
        renderingInfo.pDepthAttachment = &depthAttachment;

        if (vkCmdBeginRenderingKHR) {
            vkCmdBeginRenderingKHR(cmd, &renderingInfo);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

            VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
            VkBuffer current_index_buffer = VK_NULL_HANDLE;
            VkDeviceSize zero_offset = 0;

            for (const auto& model : models) {
                const Mesh& mesh = model.mesh;

                if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                    current_vertex_buffer = mesh.vertex_buffer->buffer;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
                }
                if (current_index_buffer != mesh.index_buffer->buffer) {
                    current_index_buffer = mesh.index_buffer->buffer;
                    vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
                }

                ShadowPushConstants push;
                push.model_matrix = model.transform.matrix();
                push.light_view_proj = light_VP;


                vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 
                                 0, sizeof(ShadowPushConstants), &push);
                vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
            }

            vkCmdEndRenderingKHR(cmd);
        }
        
        insertImageBarrier(cmd, shadow_image,
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.offset = {0, 0},
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model &model = models[i];
		const Mesh &mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &model.descriptor_set, 1, &offset);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &descriptor_set_shadow, 0, nullptr);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
