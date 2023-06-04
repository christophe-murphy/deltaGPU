#include "Facet.cuh"

__device__ Facet::Facet(const int id, const float vertex0[3], const float vertex1[3],
			 const float vertex2[3])
    : 
__device__ Facet::Facet(const int id_in, const float vertex0_in[3],
		        const float vertex1_in[3]], const float vertex2_in[3])
    : id(id_in), vertex0(vertex0_in), vertex1(vertex1_in), vertex2(vertex2_in) {}

__device__ bool Facet::intersect(Ray &ray) {
}
