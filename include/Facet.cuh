#pragma once

#include "Ray.cuh"

class Facet {
    public:
        __device__ Facet(const int id_in, const float vertex0_in[3],
			 const float vertex1_in[3], const float vertex2_in[3]);
	__device__ bool intersect(Ray &ray) const;

	int id;
	float vertex0[3];
	float vertex1[3];
	float vertex2[3];
};
