#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform Constants {
    mat4 model;
    mat4 lightViewProj;
} push;

void main() {
    gl_Position = push.lightViewProj * push.model * vec4(inPosition, 1.0);
}
