#pragma once

class Ray {
    public:
        __device__ Ray(const float pos_in[3], const float dir_in[3], const int id_in);

	int id;
	float pos[3], dir[3];
};
