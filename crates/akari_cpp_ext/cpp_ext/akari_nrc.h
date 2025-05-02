#pragma once
#include <cstdint>
namespace nn {
class NRC;
extern "C" {
NRC *akr_create_nrc(bool input_cuda_buffer, uint32_t input_dim, uint32_t output_dim, uint32_t batch_size, const char *config);
void akr_nrc_train_inference(NRC *nrc, uint64_t count, const float *input, float *output);
void akr_nrc_set_learning_rate(NRC *nrc, float learning_rate);
float akr_nrc_get_learning_rate(NRC *nrc);
uint64_t akr_nrc_param_count(NRC *nrc);
void akr_nrc_save_params(NRC *nrc, float *params);
void akr_nrc_load_params(NRC *nrc, const float *params);
void akr_nrc_inference(NRC *nrc, uint64_t count, const float *input, float *output);
float akr_nrc_train(NRC *nrc, uint64_t count, uint64_t n_iters, const float *input, const float *target);
void akr_destroy_nrc(NRC *nrc);
void akr_nrc_reset_optimizer(NRC *nrc);
}
}// namespace nn