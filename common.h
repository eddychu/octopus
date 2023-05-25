#pragma once
#include <glm/glm.hpp>
#include <glm/ext.hpp>
// resolution
#define WIDTH 500
#define HEIGHT 500
struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
