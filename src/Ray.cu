#include "Ray.cuh"

__device__ Ray::Ray(const float3& pos_in, const float3& dir_in, const int id_in)
    : orig(orig_in), dir(dir_in), id(id_in) {}
