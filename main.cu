#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <chrono>
#include <vector>
#include "common.h"

struct Mesh
{
    std::vector<Vertex> vertices;
};
// Mesh load_mesh(const char *filename)
// {
//     Assimp::Importer importer;
//     const aiScene *scene = importer.ReadFile(filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
//     // check for errors
//     if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
//     {
//         std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
//     }

//     auto mesh = scene->mMeshes[0];
//     std::vector<Vertex> vertices;

//     // for (unsigned int i = 0; i < mesh->mNumVertices; i++)
//     // {
//     //     Vertex vertex;
//     //     // process vertex positions, normals and texture coordinates
//     //     glm::vec3 vector;
//     //     // positions
//     //     vector.x = mesh->mVertices[i].x;
//     //     vector.y = mesh->mVertices[i].y;
//     //     vector.z = mesh->mVertices[i].z;
//     //     vertex.position = vector;
//     //     // normals
//     //     if (mesh->mNormals)
//     //     {
//     //         vector.x = mesh->mNormals[i].x;
//     //         vector.y = mesh->mNormals[i].y;
//     //         vector.z = mesh->mNormals[i].z;
//     //         vertex.normal = vector;
//     //     }
//     //     // texture coordinates
//     //     if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
//     //     {
//     //         glm::vec2 vec;
//     //         // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't
//     //         // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
//     //         vec.x = mesh->mTextureCoords[0][i].x;
//     //         vec.y = mesh->mTextureCoords[0][i].y;
//     //         vertex.uv = vec;
//     //     }
//     //     else
//     //         vertex.uv = glm::vec2(0.0f, 0.0f);

//     //     vertices.push_back(vertex);
//     // }

//     // process indices
//     for (unsigned int i = 0; i < mesh->mNumFaces; i++)
//     {
//         aiFace face = mesh->mFaces[i];
//         // retrieve all indices of the face and store them in the indices vector
//         for (unsigned int j = 0; j < face.mNumIndices; j += 1)
//         {
//             glm::vec3 position, normal, uv;
//             position.x = mesh->mVertices[face.mIndices[j]].x;
//             position.y = mesh->mVertices[face.mIndices[j]].y;
//             position.z = mesh->mVertices[face.mIndices[j]].z;

//             normal.x = mesh->mNormals[face.mIndices[j]].x;
//             normal.y = mesh->mNormals[face.mIndices[j]].y;
//             normal.z = mesh->mNormals[face.mIndices[j]].z;

//             uv.x = mesh->mTextureCoords[0][face.mIndices[j]].x;
//             uv.y = mesh->mTextureCoords[0][face.mIndices[j]].y;

//             vertices.push_back({position, normal, uv});
//         }
//     }

//     return {vertices};
// }

#include <cmath>

// __device__ void depth_interpolation(vec3_t p, vec3_t p0, vec3_t p1, vec3_t p2, float *out)
// {
//     float d0 = sqrt((p0.x - p.x) * (p0.x - p.x) + (p0.y - p.y) * (p0.y - p.y));
//     float d1 = sqrt((p1.x - p.x) * (p1.x - p.x) + (p1.y - p.y) * (p1.y - p.y));
//     float d2 = sqrt((p2.x - p.x) * (p2.x - p.x) + (p2.y - p.y) * (p2.y - p.y));

//     float d0_inv = 1 / d0;
//     float d1_inv = 1 / d1;
//     float d2_inv = 1 / d2;

//     float w0 = d0_inv / (d0_inv + d1_inv + d2_inv);
//     float w1 = d1_inv / (d0_inv + d1_inv + d2_inv);
//     float w2 = d2_inv / (d0_inv + d1_inv + d2_inv);

//     *out = w0 * p0.z + w1 * p1.z + w2 * p2.z;
// }

__device__ void perspective_divide(glm::vec4 *p)
{
    p->x /= p->w;
    p->y /= p->w;
    p->z /= p->w;
}

__device__ void ndc_to_viewport(glm::vec4 *p)
{
    p->x = (p->x + 1.0f) * WIDTH / 2.0f;
    p->y = (1.0f - p->y) * HEIGHT / 2.0f;
}

__device__ void draw_pixel(unsigned char *image, int i, int j, glm::vec3 c)
{
    int idx = (j * WIDTH + i) * 3;
    int r = c.x * 255;
    int g = c.y * 255;
    int b = c.z * 255;
    image[idx] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}

__device__ void draw_line(unsigned char *image, float x0, float y0, float x1, float y1, glm::vec3 c)
{
    int dx = x1 - x0;
    int dy = y1 - y0;

    float steps = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);

    float x_inc = dx / steps;
    float y_inc = dy / steps;

    for (int i = 0; i < int(steps); i++)
    {
        int x = int(x0 + i * x_inc);
        int y = int(y0 + i * y_inc);
        draw_pixel(image, x, y, c);
    }
}

__global__ void draw_triangle(Vertex *vertices, unsigned char *img, glm::mat4 *mvp)
{
    int i = blockIdx.x * 3;

    Vertex v0 = vertices[i];
    Vertex v1 = vertices[i + 1];
    Vertex v2 = vertices[i + 2];

    glm::vec4 p0 = *mvp * glm::vec4(v0.position, 1.0f);
    glm::vec4 p1 = *mvp * glm::vec4(v1.position, 1.0f);
    glm::vec4 p2 = *mvp * glm::vec4(v2.position, 1.0f);

    perspective_divide(&p0);
    perspective_divide(&p1);
    perspective_divide(&p2);

    ndc_to_viewport(&p0);
    ndc_to_viewport(&p1);
    ndc_to_viewport(&p2);

    draw_line(img, p0.x, p0.y, p1.x, p1.y, v0.normal);
    draw_line(img, p1.x, p1.y, p2.x, p2.y, v1.normal);
    draw_line(img, p2.x, p2.y, p0.x, p0.y, v2.normal);
}

__global__ void fill_depth_buffer(float *depth_buffer)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    int idx = i * WIDTH + j;
    depth_buffer[idx] = 1.0f;
}

int main()
{
    // Triangle vertic
    // Mesh mesh = load_mesh("test.obj");

    Mesh mesh = {
        {
            {{-0.5f, -0.5f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.f, 0.f}},
            {{0.5f, -0.5f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.f, 0.f}},
            {{0.0f, 0.5f, -1.0f}, {0.0f, 0.0f, 1.0f}, {0.5f, 1.f}},
        }};

    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                            (float)WIDTH / (float)HEIGHT,
                                            0.1f, 100.0f);

    glm::mat4 model = glm::mat4(1.0f);

    glm::mat4 mvp = projection * view * model;

    Vertex *d_vertices;
    cudaMalloc((void **)&d_vertices, sizeof(Vertex) * mesh.vertices.size());
    cudaMemcpy(d_vertices, mesh.vertices.data(), sizeof(Vertex) * mesh.vertices.size(), cudaMemcpyHostToDevice);

    glm::mat4 *d_mvp;
    cudaMalloc((void **)&d_mvp, sizeof(glm::mat4));
    cudaMemcpy(d_mvp, &mvp, sizeof(glm::mat4), cudaMemcpyHostToDevice);

    auto t1 = std::chrono::high_resolution_clock::now();

    // output image
    unsigned char img[WIDTH * HEIGHT * 3] = {
        0};

    unsigned char *d_img;

    cudaMalloc((void **)&d_img, WIDTH * HEIGHT * 3 * sizeof(unsigned char));

    draw_triangle<<<1, 1>>>(d_vertices, d_img, d_mvp);
    cudaDeviceSynchronize();
    cudaMemcpy(img, d_img, WIDTH * HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // time stamp end
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> exe_time = t2 - t1;
    std::cout << "execution time : " << exe_time.count() << " ms" << std::endl;

    stbi_write_png("test.png", WIDTH, HEIGHT, 3, img, 0);

    return 0;
}