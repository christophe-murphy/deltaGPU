#pragma once

#include "Ray.cuh"

class Facet {
    public:
        __device__ Facet(const int id_in, const float3 vertex0_in,
			 const float3 vertex1_in, const float3 vertex2_in);
	__device__ bool intersect(Ray& ray, float3& int_point) const;

	const int id;
	const float3 vertex0, vertex1, vertex2;
};
