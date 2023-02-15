#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
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

constexpr const int width = 28;
constexpr const int height = 28;
const char* input_names[] = {"Input3"};
const char* output_names[] = {"Plus214_Output_0"};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage:: ./run [model file]";
    return 1;
  }
  char* model_file_name = argv[1];

  // Environment
  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX runner");
  Ort::SessionOptions session_option;
  session_option.SetIntraOpNumThreads(1);
  session_option.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, model_file_name, session_option);

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // Data
  std::array<float, width * height> input_image{};
  std::array<int64_t, 4> input_shape{1, 1, width, height};
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_image.data(), input_image.size(), input_shape.data(),
      input_shape.size());

  std::array<float, 10> results{};
  std::array<int64_t, 2> output_shape{1, 10};
  auto output_tensor = Ort::Value::CreateTensor<float>(
      memory_info, results.data(), results.size(), output_shape.data(),
      output_shape.size());

  // Run
  Ort::RunOptions run_options;
  session.Run(run_options, input_names, &input_tensor, 1, output_names,
              &output_tensor, 1);
  softmax(results);
  int64_t result = std::distance(
      results.begin(), std::max_element(results.begin(), results.end()));
  return 0;
}
