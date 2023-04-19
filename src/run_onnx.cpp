#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "heterocc.h"
#include "model.hpp"
#include "stage_def.hpp"

using std::array;
using std::cout;
using std::distance;
using std::max_element;

void debug_mnist(const std::string& s, float* out) {
  array<float, 10> local_res;
  for (int i = 0; i < local_res.size(); i++) local_res[i] = out[i];
  int result = distance(local_res.begin(),
                        max_element(local_res.begin(), local_res.end()));
  cout << "Result at stage " << s << ": " << result << "\n";
}

// clang-format off
void pipeline_kernel(array<float, 10> *result, size_t result_size,
        array<float, width * height>* input_image_1,size_t input_image_size_1,
        array<float, width * height>* input_image_2,size_t input_image_size_2,
        array<float, width * height>* input_image_3,size_t input_image_size_3) {  // clang-format on
  // Run
  void* Section = __hetero_section_begin();

  {  // -----------------------------------------
    void* T = __hetero_task_begin(1, input_image_1, input_image_size_1, 1,
                                  result, result_size);
    do_inference_at_stage("Stage 1", data_pair(*input_image_1),
                          data_pair(*result));
    __hetero_task_end(T);
  }
  {  // -----------------------------------------
    void* T = __hetero_task_begin(2, result, result_size, input_image_2,
                                  input_image_size_2, 1, result, result_size);
    do_inference_at_stage("Stage 2", data_pair(*input_image_2),
                          data_pair(*result));
    __hetero_task_end(T);
  }
  {  // -----------------------------------------
    void* T = __hetero_task_begin(2, result, result_size, input_image_3,
                                  input_image_size_3, 1, result, result_size);
    do_inference_at_stage("Stage 3", data_pair(*input_image_3),
                          data_pair(*result));
    __hetero_task_end(T);
  }

  __hetero_section_end(Section);
}

int main(int argc, char** argv) {
  // Init
  if (argc < 2) {
    std::cout << "Usage:: ./run [model file].onnx";
    return 1;
  }

  init_all_stages();
  for (const auto& [stage_name, stage] : stages)
    stages[stage_name].dbg_callback = debug_mnist;

  // Global data & buffers
  array<float, 10> result;  // shared output used to specify dependency
  array<float, width * height> input_image[kNumStage];
  for (int i = 0; i < kNumStage; i++) input_image[i].fill(0);

  // Launch kernel
  void* DFG =
      __hetero_launch((void*)pipeline_kernel, 4, edge_pair(result),
                      edge_pair(input_image[0]), edge_pair(input_image[1]),
                      edge_pair(input_image[2]), 1, edge_pair(result));
  __hetero_wait(DFG);
  std::cout << "Done! "
            << "\n";
  return 0;
}
