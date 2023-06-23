#pragma once

class camera {
    public:
	__host__ __device__ Camera(const int px_width, const int px_height, const float3& posn,
			           const float3& target, const float3& up);
	__global__ float ray_trace(const Model& model, const int h_pixel, const int v_pixel);
