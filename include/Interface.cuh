#include "DeltaGPU.cuh"

extern "C" {
    void load_model(DeltaGPU* session, const char* model_file_name);
    void ray_trace(DeltaGPU* session, const int px_width, const int px_height,
                   float* position, float* target, float* up, float* frame_buffer);
}
