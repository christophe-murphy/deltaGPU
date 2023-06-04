#include "Ray.cuh"

__device__ Ray::Ray(const float pos_in[3], const float dir_in[3], const int id_in)
    : pos(pos_in), dir(dir_in), id(id_in) {}
