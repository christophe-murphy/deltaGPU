#include "Camera.cuh"

__host__ __device__ Camera::Camera(const int px_width_in, const int px_height_in,
		                   const float3& posn_in, const float3& target, const float3& up) {
    float h = tan(degrees_to_radians(vfov));
    view_height = 2 * h;
    view_width = view_height * (float) px_width / (float) px_height;
    posn = posn_in;
    forward_dir = posn - target;
    forward_dir *= rnorm3d(forward_dir);
    horizontal_dir = cross(up, forward_dir);
    horizontal_dir *= rnorm3d(horizontal_dir);
    vertical_dir = cross(forward_dir, horizontal_dir);
    horizontal_dir *= view_width;
    vertical_dir *= view_height;
    top_left_pos = posn - horizontal_dir/2 + vertical_dir/2 - forward_dir;
}


