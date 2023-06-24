#pragma once

class camera {
    public:
	__host__ __device__ Camera(const int px_width_in, const int px_height_in,
			           const float3& posn_in, const float3& target, const float3& up);
	__global__ float ray_trace(const Model& model, const int h_pixel, const int v_pixel);

    private:
	__device__ Ray start_ray(const int u_px, const int v_px) const;
	int px_width;
	int px_height;
	float view_width;
	float view_height;
	float3 posn;
	float3 top_left_pos;
	float3 horizontal_dir;
	float3 vertical_dir;
	float3 forward_dir;
	const float vfov = 40.0;
};

inline __device__ Ray Camera::start_ray(const int u_px, const int v_px) const {
    float u = ((float) u_px + 0.5) / (float) px_width;
    float v = ((float) v_px + 0.5) / (float) px_height;
    return Ray(top_left_pos + horizontal_dir*u - vertical_dir*v, -forward_dir);
}
