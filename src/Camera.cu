#include "Camera.cuh"
#include "Constants.cuh"

__device__ Camera::Camera(const int px_width_in, const int px_height_in,
		                   const float3 posn, const float3 target, const float3 up) {
    float view_height = 2 * tan_vfov;
    float view_width = view_height * (float) px_width / (float) px_height;
    position = posn;
    float3 forward_dir = position - target;
    forward_dir *= rnorm3d(forward_dir);
    horizontal_dir = cross(up, forward_dir);
    horizontal_dir *= rnorm3d(horizontal_dir);
    vertical_dir = cross(forward_dir, horizontal_dir);
    horizontal_dir *= view_width;
    vertical_dir *= view_height;
    top_left_pos = position - horizontal_dir/2 + vertical_dir/2 - forward_dir;
}

__device__ float Camera::ray_trace(const Model& model, const int h_pixel, const in v_pixel) const {
    Ray ray = start_ray(h_pixel, v_pixel);
    float3 hit_point, normal;
    bool hit = Model.intersect(ray, hit_point, normal);
    float shading = abs(dot(normal, ray.dir));
    return shading;
}
