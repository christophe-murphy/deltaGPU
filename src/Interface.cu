#include "Interface.cuh"

extern "C" {
void load_model(DeltaGPU* session, const char* model_file_name) {
    DeltaGPU* session = new DeltaGPU(model_file_name);
}

void ray_trace(DeltaGPU* session, const int px_width, const int px_height, float* position,
               float* target, float* up, float* frame_buffer) {
    DeltaGPU.ray_trace(px_width, px_height, position, target, up, frame_buffer);
}

