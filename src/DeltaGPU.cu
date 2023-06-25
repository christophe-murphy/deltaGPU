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
    Facet* facets, facets_dev;
    int n_facets = construct_model(facets, model_file_name);
    cudaMalloc(&facets_dev, n_facets * sizeof(Facet));
    cudaMemcpy(facets_dev, facets, n_facets * sizeof(Facet), cudaMemcpyHostToDevice);
    model = Model(facets_dev, n_facets);
}

__host__ DeltaGPU::~DeltaGPU() {
    cudaFree(model.facets);
}

__host__ int DeltaGPU::construct_model(Facet* facets, const char* model_file_name) const {
    //Stuff for reading stl file
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
