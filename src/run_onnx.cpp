#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "heterocc.h"
#include "model_def.hpp"

#define data_pair(X) X.size(), X.data()
#define edge_pair(X) (void*)&(X), sizeof(X)

void dummy(int* result, size_t __) {
  // Run
  void* Section = __hetero_section_begin();

  void* T1 = __hetero_task_begin(1, result, __, 1, result, __);
  {
    // Data
    std::array<float, width * height> input_image{};
    std::array<float, 10> results;
    // int64_t result = 0;
    for (int i = 0; i < width * height; i++) input_image[i] = *result;
    do_inference_at_stage(0, data_pair(input_shape), data_pair(input_image),
                          data_pair(output_shape), data_pair(results),
                          input_names, output_names);
    *result = std::distance(results.begin(),
                            std::max_element(results.begin(), results.end()));
    std::cout << "Result T1: " << *result << "\n";
  }
  __hetero_task_end(T1);

  void* T2 = __hetero_task_begin(1, result, __, 1, result, __);
  {
    // Data
    std::array<float, width * height> input_image{};
    std::array<float, 10> results;
    // int64_t result = 0;
    for (int i = 0; i < width * height; i++) input_image[i] = *result;
    do_inference_at_stage(0, data_pair(input_shape), data_pair(input_image),
                          data_pair(output_shape), data_pair(results),
                          input_names, output_names);
    *result = std::distance(results.begin(),
                            std::max_element(results.begin(), results.end()));
    std::cout << "Result T2: " << *result << "\n";
  }
  __hetero_task_end(T2);

  void* T3 = __hetero_task_begin(1, result, __, 1, result, __);
  {
    // Data
    std::array<float, width * height> input_image{};
    std::array<float, 10> results;
    // int64_t result = 0;
    for (int i = 0; i < width * height; i++) input_image[i] = *result;
    do_inference_at_stage(0, data_pair(input_shape), data_pair(input_image),
                          data_pair(output_shape), data_pair(results),
                          input_names, output_names);
    *result = std::distance(results.begin(),
                            std::max_element(results.begin(), results.end()));
    std::cout << "Result T3: " << *result << "\n";
  }
  __hetero_task_end(T3);

  __hetero_section_end(Section);
}

int main(int argc, char** argv) {
  // Init
  if (argc < 2) {
    std::cout << "Usage:: ./run [model file]";
    return 1;
  }
  char* model_file_name = argv[1];
  for (int i = 0; i < kNumStage; i++)
    init_model(model_file_name, i);  // use the same model for all stages

  // make hpvm compiler happy
  int result = 0;  // used for dependency
  void* DFG = __hetero_launch((void*)dummy, 1, &result, sizeof(result), 1,
                              &result, sizeof(result));
  __hetero_wait(DFG);
  std::cout << "Done! "
            << "\n";
  return 0;
}
