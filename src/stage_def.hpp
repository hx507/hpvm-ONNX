#pragma once
#include "model.hpp"

// TODO: testing parameters for mnist, remove me
constexpr const int width = 28;
constexpr const int height = 28;
std::vector<OnnxStage> stages = {
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
