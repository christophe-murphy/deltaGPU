#include "Model.cuh"

__device__ Model::Model(const Facet* facets_in) : facets(facets_in), n_facets(n_facets_in) {}

__device__ bool Model::intersect(Ray& ray, float3& int_point, float3& facet_norm) const {
    for(int f; f < n_facets; ++f) {
	bool hit = facets[f].intersect(ray, int_point);
	if(hit) {
	    facet_norm = facets[f].normal;
	    return hit;
	}
    }
}
