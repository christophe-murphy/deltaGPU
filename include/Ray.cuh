#pragma once

class Ray {
    public:
        __device__ Ray(const float3& orig_in, const float3& dir_in, const int id_in);

	int id;
	float3 orig, dir;
};
