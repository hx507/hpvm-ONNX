#include "model.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

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
std::vector<Ort::Session*> sessions;

void init_all_stages() {
  assert(sessions.empty() && "Only initializing stages one single time");
  for (const auto& stage : stages) {
    session_option.SetIntraOpNumThreads(1);
    session_option.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessions.push_back(
        new Ort::Session(env, stage.model_file_name, session_option));
  }
}

void do_inference_with_session(Ort::Session* sess, size_t inp_shape_sz,
                               const int64_t* inp_shape, size_t inp_sz,
                               float* inp, size_t out_shape_sz,
                               const int64_t* out_shape, size_t out_sz,
                               float* out, const char** input_names,
                               const char** output_names) {
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, inp, inp_sz,
                                                      inp_shape, inp_shape_sz);
  auto output_tensor = Ort::Value::CreateTensor<float>(
      memory_info, (float*)out, out_sz, out_shape, out_shape_sz);

  Ort::RunOptions run_options{};
  sess->Run(run_options, input_names, &input_tensor, 1, output_names,
            &output_tensor, 1);
}

void do_inference_at_stage(int s, size_t inp_sz, float* inp, size_t out_sz,
                           float* out) {
  assert(s >= 0 && s < kNumStage &&
         "Trying to inference at an undefined stage");

  const auto& stage = stages[s];
  do_inference_with_session(sessions[s], data_pair(stage.input_shape), inp_sz,
                            inp, data_pair(stage.output_shape), out_sz, out,
                            (const char**)stage.input_names.data(),
                            (const char**)stage.output_names.data());
#ifdef HPVM_STAGE_DEBUG
  stage.dbg_callback(s, out);
#endif
}
