#include <assert.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "heterocc.h"
#include "model_def.hpp"

#define data_pair(X) X.size(), (void*)X.data()

void dummy(void* _, size_t __) {
  void* Section = __hetero_section_begin();
  void* Wrapper = __hetero_task_begin(1, _, __, 1, _, __);

  // Data
  std::array<float, width * height> input_image{};
  // Run
  std::array<float, 10> results;

  do_inference_at_stage(0, data_pair(input_shape), data_pair(input_image),
                        data_pair(output_shape), data_pair(results),
                        input_names, output_names);
  int64_t result = std::distance(
      results.begin(), std::max_element(results.begin(), results.end()));
  std::cout << "Result: " << result << "\n";

  __hetero_task_end(Wrapper);
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
  void* DFG = __hetero_launch((void*)dummy, 1, nullptr, 0, 1, nullptr, 0);
  __hetero_wait(DFG);
  std::cout << "Done! "
            << "\n";
  return 0;
}
