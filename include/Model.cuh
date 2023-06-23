#pragma once

#include "Facet.cuh"

class Model {
    public:
        __host__ __device__ Model(const Facet* facets_in, const size_t n_facets_in);
	__device__ bool intersect(Ray& ray, float3& int_point) const;

    private:
	const Facet* facets;
        const size_t n_facets;
};
