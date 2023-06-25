#pragma once

#include "Model.cuh"
#inlcude "Ray.cuh"

class Camera {
    public:
	__device__ Camera(const int px_width_in, const int px_height_in,
			           const float3 posn, const float3 target, const float3 up);
	__device__ float ray_trace(const Model& model, const int h_pixel, const int v_pixel) const;

    private:
	__device__ Ray start_ray(const int u_px, const int v_px) const;

	int px_width;
	int px_height;
	float3 position;
	float3 top_left_pos;
	float3 horizontal_dir;
	float3 vertical_dir;
};

inline __device__ Ray Camera::start_ray(const int u_px, const int v_px) const {
    float u = ((float) u_px + 0.5) / (float) px_width;
    float v = ((float) v_px + 0.5) / (float) px_height;
    float3 direction = top_left_pos + horizontal_dir*u - vertical_dir*v - position;
    direction *= rnorm3d(direction);
    return Ray(position, direction);
}
