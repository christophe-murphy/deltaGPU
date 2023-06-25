#pragma once

class Ray {
    public:
        __device__ Ray(const float3& orig_in, const float3& dir_in);

	float3 orig, dir;
};
