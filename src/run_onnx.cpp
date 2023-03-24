#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "heterocc.h"
#include "model_def.hpp"

#define data_pair(X) (X).size(), (X).data()
#define edge_pair(X) (void*)&(X), sizeof(X)

using std::array;
using std::cout;
using std::distance;
using std::max_element;

void dummy(int* result, size_t __, array<float, width * height>* input_image,
           size_t ___) {
  // Run
  void* Section = __hetero_section_begin();

  // -----------------------------------------
  void* T1 =
      __hetero_task_begin(1, input_image, ___, 1, result, __);
  {
    array<float, 10> local_res;
    do_inference_at_stage(0, data_pair(input_shape),  // clang-format off
                             data_pair(*input_image),
                             data_pair(output_shape), 
                             data_pair(local_res),
                             input_names, output_names);  // clang-format on
    *result = distance(local_res.begin(),
                       max_element(local_res.begin(), local_res.end()));
    cout << "Result T1: " << *result << "\n";
  }
  __hetero_task_end(T1);

  // -----------------------------------------
  void* T2 =
      __hetero_task_begin(2, result, __, input_image, ___, 1, result, __);
  {
    array<float, 10> local_res;
    do_inference_at_stage(1, data_pair(input_shape),  // clang-format off
                             data_pair(*input_image),
                             data_pair(output_shape), 
                             data_pair(local_res),
                             input_names, output_names);  // clang-format on
    *result = distance(local_res.begin(),
                       max_element(local_res.begin(), local_res.end()));
    cout << "Result T2: " << *result << "\n";
  }
  __hetero_task_end(T2);

  // -----------------------------------------
  void* T3 =
      __hetero_task_begin(2, result, __, input_image, ___, 1, result, __);
  {
    array<float, 10> local_res;
    do_inference_at_stage(2, data_pair(input_shape),  // clang-format off
                             data_pair(*input_image),
                             data_pair(output_shape), 
                             data_pair(local_res),
                             input_names, output_names);  // clang-format on
    // TODO inspect_result_at_stage(stage, input, result);
    *result = distance(local_res.begin(),
                       max_element(local_res.begin(), local_res.end()));
    cout << "Result T3: " << *result << "\n";
  }
  __hetero_task_end(T3);

  __hetero_section_end(Section);
}

int main(int argc, char** argv) {
  // Init
  if (argc < 2) {
    std::cout << "Usage:: ./run [model file].onnx";
    return 1;
  }
  char* model_file_name = argv[1];
  // TODO have an easy&organized way to list shapes/files/names/result_inspection_function
  // (header files)
  for (int i = 0; i < kNumStage; i++) // TODO: also specify io shape/names
    init_model(model_file_name,
               i);  // use the same model for all stages for now

  // Global data & buffers
  int result = 0;  // shared output used to specify dependency
  array<float, width * height> input_image{};
  input_image.fill(0);

  // Launch kernel
  void* DFG = __hetero_launch((void*)dummy, 2, edge_pair(result),
                              edge_pair(input_image), 1, edge_pair(result));
  __hetero_wait(DFG);
  std::cout << "Done! "
            << "\n";
  return 0;
}
