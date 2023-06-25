#pragma once

#include "Facet.cuh"
#include "Model.cuh"

class DeltaGPU {
    public:
        __host__ DeltaGPU(const char* model_file_name);
	__host__ ~DeltaGPU();
	__host__ int construct_model(Facet* facets, const char* model_file_name) const;
	__host__ void allocate_framebuffer(float* frame_buffer);
	__host__ void ray_trace(const int px_width, const int px_height, float* position,
	                        float* target, float* up, float* frame_buffer);
    private:
        Model model;
	float* frame_buffer_dev;
	int num_rays;
