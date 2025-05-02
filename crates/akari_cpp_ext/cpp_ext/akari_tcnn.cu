#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#include <optional>
#include "akari_nrc.h"
#define CUDA_CHECK_ABORT(x)                                                                 \
    do {                                                                                    \
        cudaError_t _result = x;                                                            \
        if (_result != cudaSuccess) {                                                       \
            fprintf(stderr, FILE_LINE " " #x " failed: %s\n", cudaGetErrorString(_result)); \
            abort();                                                                        \
        }                                                                                   \
    } while (0)
namespace nn {
__global__ void collect_batch(
    size_t batch_size,
    tcnn::default_rng_t rng,
    size_t input_dims,
    size_t output_dims,
    size_t count,
    const float *inputs,
    const float *targets,
    float *training_inputs,
    float *training_targets) {
    const uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= batch_size) {
        return;
    }
    rng.advance(tid);
    auto idx = rng.next_uint() % count;
    for (auto i = 0; i < input_dims; i++) {
        training_inputs[tid * input_dims + i] = inputs[idx * input_dims + i];
    }
    for (auto i = 0; i < output_dims; i++) {
        training_targets[tid * output_dims + i] = targets[idx * output_dims + i];
    }
}
#define AKR_ASSERT(x) ([&]() { if (!(x)) { fprintf(stderr, "assertion failed %s\n", #x); abort(); } })()
using model_t = tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>;

__global__ void ema_step_half_precision(
    const uint32_t n_elements,
    const float ema_decay,
    const float ema_debias_old,
    const float ema_debias_new,
    const tcnn::network_precision_t *__restrict__ weights,
    tcnn::network_precision_t *__restrict__ weights_ema) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) return;

    float filtered_val = ((float)weights_ema[i] * ema_decay * ema_debias_old + (float)weights[i] * (1 - ema_decay)) * ema_debias_new;
    weights_ema[i] = (tcnn::network_precision_t)filtered_val;
}
struct EMAModel {
    std::shared_ptr<model_t> model;
    tcnn::GPUMemory<tcnn::network_precision_t> ema_weights;
    tcnn::network_precision_t *model_params = nullptr;
    size_t iteration = 0;
    float m_ema_decay = 0.99f;
    EMAModel(uint32_t n_input_dims, uint32_t n_output_dims, const tcnn::json &config) {
        model = std::make_shared<model_t>(n_input_dims, n_output_dims, config.value("encoding", tcnn::json::object()), config.value("network", tcnn::json::object()));
        auto n_params = model->n_params();
        ema_weights.resize(n_params * 2);
        model->set_params(ema_weights.data(), ema_weights.data(), ema_weights.data() + n_params);
        model_params = model->params();
        AKR_ASSERT(model_params);
        m_ema_decay = config.value("ema_decay", 0.99f);
    }
    void update_model_parameters(cudaStream_t stream, const std::shared_ptr<model_t> &outer) {
        iteration++;
        AKR_ASSERT(outer->n_params() == model->n_params());
        float ema_debias_old = 1 - (float)std::pow(m_ema_decay, iteration - 1);
        float ema_debias_new = 1.0f / (1 - (float)std::pow(m_ema_decay, iteration));
        tcnn::linear_kernel(ema_step_half_precision, 1, stream,
                            model->n_params(),
                            m_ema_decay,
                            ema_debias_old,
                            ema_debias_new,
                            outer->params(),
                            model_params);
    }
};
struct NRC {
    bool is_input_cuda_buffer;
    size_t input_dims, output_dims, batch_size;
    tcnn::TrainableModel model{};
    float *cuda_input_buffer = nullptr;
    float *cuda_output_or_target_buffer = nullptr;
    size_t cuda_buffer_size = 0;// the input is of size input_dim * cuda_buffer_size, the output is of size output_dim * cuda_buffer_size
    tcnn::GPUMatrix<float> training_input_matrix{}, training_target_matrix{};
    tcnn::GPUMatrix<float> inference_input_matrix{}, inference_output_matrix{};
    tcnn::default_rng_t rng{1337};
    cudaStream_t stream;
    std::shared_ptr<EMAModel> ema_model;
    nlohmann::json config;
    NRC(bool is_input_cuda_buffer, uint32_t input_dims, uint32_t output_dims, uint32_t batch_size, const char *config)
        : is_input_cuda_buffer(is_input_cuda_buffer),
          input_dims(input_dims),
          output_dims(output_dims),
          batch_size(batch_size),
          training_input_matrix(input_dims, batch_size),
          training_target_matrix(output_dims, batch_size),
          inference_input_matrix(input_dims, batch_size),
          inference_output_matrix(output_dims, batch_size) {
        this->config = tcnn::json::parse(config);
        ema_model = std::make_shared<EMAModel>(input_dims, output_dims, this->config);
        // auto dumped_json = json_config.dump(2);
        // printf("Initializing NRC with config: %s\n", dumped_json.c_str());
        model = tcnn::create_from_config(input_dims, output_dims, this->config);
        printf("is_input_cuda_buffer: %d\n", is_input_cuda_buffer);
        printf("loss: %s\n", model.loss->name().c_str());
        printf("optimizer: %s\n", model.optimizer->name().c_str());
        printf("learning rate: %f\n", model.optimizer->learning_rate());
        // abort();
        CUDA_CHECK_ABORT(cudaStreamCreate(&stream));
        CUDA_CHECK_ABORT(cudaDeviceSynchronize());
    }
    void reset_optimizer() {
        model.optimizer.reset(
            tcnn::create_optimizer<tcnn::network_precision_t>(config.value("optimizer", tcnn::json::object())));
    }
    void ensure_cuda_buffer_size(size_t count) {
        if (cuda_buffer_size < count) {
            if (cuda_input_buffer) {
                CUDA_CHECK_ABORT(cudaFree(cuda_input_buffer));
            }
            if (cuda_output_or_target_buffer) {
                CUDA_CHECK_ABORT(cudaFree(cuda_output_or_target_buffer));
            }
            cuda_buffer_size = count;
            CUDA_CHECK_ABORT(cudaMalloc(&cuda_input_buffer, input_dims * cuda_buffer_size * sizeof(float)));
            CUDA_CHECK_ABORT(cudaMalloc(&cuda_output_or_target_buffer, output_dims * cuda_buffer_size * sizeof(float)));
        }
    }
    float train(uint64_t count, uint64_t n_iters, float *input, float *target) {
        CUDA_CHECK_ABORT(cudaStreamSynchronize(stream));
        std::optional<tcnn::GPUMatrix<float>> input_matrix, target_matrix;
        if (is_input_cuda_buffer) {
            input_matrix.emplace(input, input_dims, count);
            target_matrix.emplace(target, output_dims, count);
        } else {
            ensure_cuda_buffer_size(count);
            CUDA_CHECK_ABORT(cudaDeviceSynchronize());
            CUDA_CHECK_ABORT(cudaMemcpyAsync(cuda_input_buffer, input, input_dims * count * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_ABORT(cudaMemcpyAsync(cuda_output_or_target_buffer, target, output_dims * count * sizeof(float), cudaMemcpyHostToDevice, stream));
            input_matrix.emplace(cuda_input_buffer, input_dims, count);
            target_matrix.emplace(cuda_output_or_target_buffer, output_dims, count);
            // generate_random_uniform<uint32_t>()
        }

        auto avg_loss = 0.0f;
        for (auto it = 0; it < n_iters; it++) {
            tcnn::linear_kernel(collect_batch,
                                1, stream, batch_size,
                                rng,
                                input_dims, output_dims, count,
                                input_matrix->data(), target_matrix->data(),
                                training_input_matrix.data(), training_target_matrix.data());
            auto ctx = model.trainer->training_step(stream, training_input_matrix, training_target_matrix);
            ema_model->update_model_parameters(stream, model.network);
            avg_loss += model.trainer->loss(stream, *ctx);
            rng.advance(batch_size);
        }
        CUDA_CHECK_ABORT(cudaStreamSynchronize(stream));
        CUDA_CHECK_ABORT(cudaDeviceSynchronize());
        return avg_loss / n_iters;
    }
    void train_inference(uint64_t count, float *input, float *output) {
        inference_impl(count, input, output, [&](cudaStream_t stream, const tcnn::GPUMatrix<float> &input, tcnn::GPUMatrix<float> &output) {
            model.network->inference(stream, input, output);
        });
    }
    void inference(uint64_t count, float *input, float *output) {
        inference_impl(count, input, output, [&](cudaStream_t stream, const tcnn::GPUMatrix<float> &input, tcnn::GPUMatrix<float> &output) {
            ema_model->model->inference(stream, input, output);
        });
    }
    template<class F>
    void inference_impl(uint64_t count, float *input, float *output, F &&inference_func) {
        CUDA_CHECK_ABORT(cudaStreamSynchronize(stream));
        std::optional<tcnn::GPUMatrix<float>> input_matrix, target_matrix;
        if (is_input_cuda_buffer) {
            input_matrix.emplace(input, input_dims, count);
            target_matrix.emplace(output, output_dims, count);
        } else {
            ensure_cuda_buffer_size(count);
            CUDA_CHECK_ABORT(cudaMemcpyAsync(cuda_input_buffer, input, input_dims * count * sizeof(float), cudaMemcpyHostToDevice, stream));
            input_matrix.emplace(cuda_input_buffer, input_dims, count);
            target_matrix.emplace(cuda_output_or_target_buffer, output_dims, count);
        }
        auto n_full_batches = count / batch_size;
        auto remaining_cols = count - n_full_batches * batch_size;
        if (n_full_batches > 0) {
            auto input_batch = input_matrix->slice_cols(0, n_full_batches * batch_size);
            auto output_batch = target_matrix->slice_cols(0, n_full_batches * batch_size);
            inference_func(stream, input_batch, output_batch);
        }
        if (remaining_cols > 0) {
            auto left_over_input = input_matrix->slice_cols(n_full_batches * batch_size, remaining_cols);
            CUDA_CHECK_ABORT(cudaMemcpyAsync(inference_input_matrix.data(),
                                             left_over_input.data(),
                                             input_dims * remaining_cols * sizeof(float),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
            inference_func(stream, inference_input_matrix, inference_output_matrix);
            auto left_over_output = target_matrix->slice_cols(n_full_batches * batch_size, remaining_cols);
            CUDA_CHECK_ABORT(cudaMemcpyAsync(left_over_output.data(),
                                             inference_output_matrix.data(),
                                             output_dims * remaining_cols * sizeof(float),
                                             cudaMemcpyDeviceToDevice,
                                             stream));
        }
        if (!is_input_cuda_buffer) {
            CUDA_CHECK_ABORT(cudaMemcpyAsync(output, cuda_output_or_target_buffer, output_dims * count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK_ABORT(cudaStreamSynchronize(stream));
        CUDA_CHECK_ABORT(cudaDeviceSynchronize());
    }
    void save_params(float *params) {
        auto cnt = model.network->n_params();
        tcnn::network_precision_t *host_params = new tcnn::network_precision_t[cnt];
        CUDA_CHECK_ABORT(cudaMemcpy(host_params, model.network->params(), cnt * sizeof(tcnn::network_precision_t), cudaMemcpyDeviceToHost));
        for (auto i = 0; i < cnt; i++) {
            params[i] = (float)host_params[i];
        }
        delete[] host_params;
    }
    void load_params(const float *params) {
        auto cnt = model.network->n_params();
        tcnn::network_precision_t *host_params = new tcnn::network_precision_t[cnt];
        for (auto i = 0; i < cnt; i++) {
            host_params[i] = (tcnn::network_precision_t)params[i];
        }
        CUDA_CHECK_ABORT(cudaMemcpy(model.network->params(), host_params, cnt * sizeof(tcnn::network_precision_t), cudaMemcpyHostToDevice));
        delete[] host_params;
    }
    ~NRC() {
        CUDA_CHECK_ABORT(cudaDeviceSynchronize());
        if (cuda_input_buffer) {
            CUDA_CHECK_ABORT(cudaFree(cuda_input_buffer));
        }
        if (cuda_output_or_target_buffer) {
            CUDA_CHECK_ABORT(cudaFree(cuda_output_or_target_buffer));
        }
        CUDA_CHECK_ABORT(cudaStreamDestroy(stream));
    }
};
template<typename F>
decltype(auto) catch_and_abort(F &&f) {
    try {
        return f();
    } catch (const std::exception &e) {
        fprintf(stderr, FILE_LINE " %s\n", e.what());
        abort();
    }
}
#define CATCH_AND_ABORT(x) catch_and_abort([&] { return x; })
NRC *akr_create_nrc(bool input_cuda_buffer, uint32_t input_dim, uint32_t output_dim, uint32_t batch_size, const char *config) {
    return CATCH_AND_ABORT(new NRC(input_cuda_buffer, input_dim, output_dim, batch_size, config));
}
uint64_t akr_nrc_param_count(NRC *nrc) {
    return CATCH_AND_ABORT(nrc->model.network->n_params());
}
void akr_nrc_save_params(NRC *nrc, float *params) {
    CATCH_AND_ABORT(nrc->save_params(params));
}
void akr_nrc_load_params(NRC *nrc, const float *params) {
    CATCH_AND_ABORT(nrc->load_params(params));
}
void akr_nrc_inference(NRC *nrc, uint64_t count, const float *input, float *output) {
    CATCH_AND_ABORT(nrc->inference(count, const_cast<float *>(input), output));
}
void akr_nrc_train_inference(NRC *nrc, uint64_t count, const float *input, float *output) {
    CATCH_AND_ABORT(nrc->train_inference(count, const_cast<float *>(input), output));
}
void akr_nrc_set_learning_rate(NRC *nrc, float learning_rate) {
    CATCH_AND_ABORT(nrc->model.optimizer->set_learning_rate(learning_rate));
}
float akr_nrc_get_learning_rate(NRC *nrc) {
    return CATCH_AND_ABORT(nrc->model.optimizer->learning_rate());
}
float akr_nrc_train(NRC *nrc, uint64_t count, uint64_t n_iters, const float *input, const float *target) {
    return CATCH_AND_ABORT(nrc->train(count, n_iters, const_cast<float *>(input), const_cast<float *>(target)));
}
void akr_destroy_nrc(NRC *nrc) {
    CATCH_AND_ABORT(delete nrc);
    CUDA_CHECK_ABORT(cudaDeviceSynchronize());
}
void akr_nrc_reset_optimizer(NRC *nrc) {
    CATCH_AND_ABORT(nrc->reset_optimizer());
}
}// namespace nn