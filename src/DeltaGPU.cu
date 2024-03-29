#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "DeltaGPU.cuh"
#include "Camera.cuh"

__global__ void ray_trace_kernel(Model model, Camera camera, float* frame_buffer) {
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= camera.px_width * camera.px_height) return;

    int h_pixel = ray_id % camera.px_width;
    int v_pixel = ray_id / camera.px_width;

    frame_buffer[ray_id] = camera.ray_trace(model, h_pixel, v_pixel);
}

__host__ DeltaGPU::DeltaGPU(const char* model_file_name) {
    num_rays = 0;
    std::vector<Facet> facets;
    Facet* facets_dev;
    int n_facets = construct_model(facets, model_file_name);
    cudaMalloc(&facets_dev, n_facets * sizeof(Facet));
    cudaMemcpy(facets_dev, facets.data(), n_facets * sizeof(Facet), cudaMemcpyHostToDevice);
    model = Model(facets_dev, n_facets);
}

__host__ DeltaGPU::~DeltaGPU() {
    cudaFree(model.facets);
}

__host__ int DeltaGPU::construct_model(std::vector<Facet>& facets, const char* model_file_name) const {
    int n_facets = -1;
    float3 vertex;
    std::vector<float3> vertices;
    float3 normal;

    std::ifstream model_file;
    model_file.open(model_file_name);
    std::string word;
    model_file >> word;
    //ASCII File
    if (word == "solid") {
        std::getline(model_file, word);
        model_file >> word;
	while (model_file) {
	    if (word == "facet") {
                ++n_facets;
	    } else if (word == "normal") {
		model_file >> normal.x;
		model_file >> normal.y;
		model_file >> normal.z;
	    } else if (word == "vertex") {
		for (int nv = 0; nv < 3; ++nv) {
                    model_file >> vertex.x;
                    model_file >> vertex.y;
                    model_file >> vertex.z;
                vertices.clear();
	    }
            model_file >> word;
	}
    //Binary File
    } else {
        model_file.close();
	model_file.open(model_file_name, std::ifstream::binary);

	//Read Header
	{char[80] buffer;
        model_file.read(buffer, 80);}

	//If header read succesfully, continue reading file
	if (model_file) {
	    char[4] buffer4;
	    char[2] buffer2;

            //Read no. facets
            model_file.read(buffer4, 4);
            n_facets = (unsigned int) buffer4;

    	    //Read facets
            for (int n = 0; n < n_facets; ++n) {
                model_file.read(buffer4, 4); normal.x = (float) buffer4;
                model_file.read(buffer4, 4); normal.y = (float) buffer4;
                model_file.read(buffer4, 4); normal.z = (float) buffer4;
		for (int v = 0; v < 3; ++v) {
                    model_file.read(buffer4, 4); vertex.x = (float) buffer4;
                    model_file.read(buffer4, 4); vertex.y = (float) buffer4;
                    model_file.read(buffer4, 4); vertex.z = (float) buffer4;
		    vertices.push_back(vertex);
		}
		model_file.read(buffer2, 2);
		facets.push_back(Facet(n, vertices[0], vertices[1], vertices[2], normal));
		vertices.clear();
            }
        }
    }
    model_file.close();
    return n_facets;
}

__host__ void DeltaGPU::allocate_framebuffer(float* frame_buffer, int num_rays_in) {
    if (num_rays_in != num_rays) {
        if (num_rays != 0) {
	    cudaFree(frame_buffer);
	}
	num_rays = num_rays_in;
	cudaMalloc(&frame_buffer, num_rays * sizeof(float));
    }
}

__host__ void DeltaGPU::ray_trace(const int px_width, const int px_height, float* position,
		                  float* target, float* up, float* frame_buffer) {

    float3 posn(position[0], position[1], position[2]);
    float3 targ(target[0], target[1], target[2]);
    float3 u(up[0], up[1], up[2]);
    Camera camera(px_width, px_height, posn, targ, u);
    allocate_framebuffer(frame_buffer_dev, px_width * px_height);
    ray_trace_kernel<<<blocks_per_grid, threads_per_block>>>(model, camera, frame_buffer_dev);
    cudaMemcpy(frame_buffer_dev, frame_buffer, num_rays * sizeof(float), cudaMemcpyDeviceToHost);
}
