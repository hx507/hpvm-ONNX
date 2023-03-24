#include <stdint.h>

#include <array>
#include <vector>

/**
#define TCB_SPAN_NO_EXCEPTIONS  // need an exception less implementation of
                                // std::span
#include "span.hpp"
namespace std {
using tcb::span;
};
**/

// TODO: testing parameters for mnist, remove me
constexpr const int width = 28;
constexpr const int height = 28;

struct OnnxStage {
  const char* model_file_name;
  std::vector<const char*> input_names, output_names;
  std::vector<int64_t> input_shape, output_shape;
};

const std::vector<OnnxStage> stages = {
    // Stage 1
    {.model_file_name = "../data/mnist.onnx",
     .input_names = {"Input3"},
     .output_names = {"Plus214_Output_0"},
     .input_shape = {1, 1, width, height},
     .output_shape = {1, 10}},

    // Stage 2
    {.model_file_name = "../data/mnist.onnx",
     .input_names = {"Input3"},
     .output_names = {"Plus214_Output_0"},
     .input_shape = {1, 1, width, height},
     .output_shape = {1, 10}},

    // Stage 3
    {.model_file_name = "../data/mnist.onnx",
     .input_names = {"Input3"},
     .output_names = {"Plus214_Output_0"},
     .input_shape = {1, 1, width, height},
     .output_shape = {1, 10}},
};

const int kNumStage = stages.size();

void init_all_stages();

// old------------------------

const char* input_names[] = {"Input3"};
const char* output_names[] = {"Plus214_Output_0"};
constexpr std::array<int64_t, 4> input_shape{1, 1, width, height};
constexpr std::array<int64_t, 2> output_shape{1, 10};

void init_model(char* model_file_name, int stage);

void do_inference_at_stage(int stage, size_t inp_shape_sz,
                           const int64_t* inp_shape, size_t inp_sz, float* inp,
                           size_t out_shape_sz, const int64_t* out_shape,
                           size_t out_sz, float* out, const char** input_names,
                           const char** output_names);
