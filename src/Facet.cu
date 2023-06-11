#include "Facet.cuh"
#include "Constants.cuh"

__device__ Facet::Facet(const int id_in, const float3 vertex0_in,
		        const float3 vertex1_in, const float3 vertex2_in)
    : id(id_in), vertex0(vertex0_in), vertex1(vertex1_in), vertex2(vertex2_in) {}

__device__ bool Facet::intersect(Ray& ray, float3& int_point) const {
    //Möller–Trumbore intersection algorithm
    const float3 edge1 = vertex1 - vertex0;
    const float3 edge2 = vertex2 - vertex0;
    const float3 h = cross(ray.dir, edge2);
    const float a = dot(edge1, h);

    if(a > -eps and a < eps) return false; //Ray parallel to facet

    float f = 1 / a;
    float3 s = ray.orig - vertex0;
    float u = f * dot(s, h);
    if(u < 0. or u > 1.) return false; //Miss

    float3 q = cross(s, edge1);
    float v = f * dot(ray.dir, q);
    if(v < 0. or (u + v) > 1.) return false; //Miss

    float t = f * dot(edge2, q);
    if(t > eps) {
	int_point = ray.orig + ray.dir * t; //Intersect
	return true;
    } else return false; //Miss facet but intersect one or two vertices

}
