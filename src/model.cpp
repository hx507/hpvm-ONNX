#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "model_def.hpp"

template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

auto memory_info =
    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX runner");
Ort::SessionOptions session_option;
Ort::Session*  sessions[kNumStage];

void init_model(char* model_file_name, int stage) {
  session_option.SetIntraOpNumThreads(1);
  session_option.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  sessions[stage] = new Ort::Session(env, model_file_name, session_option);
}

void do_inference_at_stage(int stage, size_t inp_shape_sz, void* inp_shape,
                           size_t inp_sz, void* inp, size_t out_shape_sz,
                           void* out_shape, size_t out_sz, void* out,
                           const char** input_names,
                           const char** output_names) {
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)inp, inp_sz,
                                                      (int64_t*)inp_shape, inp_shape_sz);
  auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)out, out_sz,
                                                       (int64_t*)out_shape, out_shape_sz);

  Ort::RunOptions run_options{};
  sessions[stage]->Run(run_options, input_names, &input_tensor, 1, output_names,
                       &output_tensor, 1);
}
// std::array<float, 10> do_inference(
// std::array<float, width * height>& input_image) {
// std::array<float, 10> results{};

// auto input_tensor = Ort::Value::CreateTensor<float>(
// memory_info, input_image.data(), input_image.size(), input_shape.data(),
// input_shape.size());

// auto output_tensor = Ort::Value::CreateTensor<float>(
// memory_info, results.data(), results.size(), output_shape.data(),
// output_shape.size());

// Ort::RunOptions run_options{};
// sessions[0]->Run(run_options, input_names, &input_tensor, 1, output_names,
//&output_tensor, 1);
// softmax(results);
// return results;
//}
