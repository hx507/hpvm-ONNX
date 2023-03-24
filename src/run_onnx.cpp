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

void dummy(int* result, size_t __, array<float, width * height>* input_image,
           size_t ___) {
  // Run
  void* Section = __hetero_section_begin();

  // -----------------------------------------
  void* T1 = __hetero_task_begin(1, input_image, ___, 1, result, __);
  {
    array<float, 10> local_res;
    do_inference_at_stage(0, data_pair(*input_image), data_pair(local_res));
    *result = distance(local_res.begin(),
                       max_element(local_res.begin(), local_res.end()));
    cout << "Result T1: " << *result << "\n";
  }
  __hetero_task_end(T1);

  __hetero_section_end(Section);
}

void debug_mnist(int s, float* out) {
  array<float, 10> local_res;
  for (int i = 0; i < local_res.size(); i++) local_res[i] = out[i];
  int result = distance(local_res.begin(),
                        max_element(local_res.begin(), local_res.end()));
  cout << "Result at stage " << s << ": " << result << "\n";
}

int main(int argc, char** argv) {
  // Init
  if (argc < 2) {
    std::cout << "Usage:: ./run [model file].onnx";
    return 1;
  }

  init_all_stages();
  auto dbg = [&](int s, float* out) {
    array<float, 10> local_res;
    for (int i = 0; i < local_res.size(); i++) local_res[i] = out[i];
    int result = distance(local_res.begin(),
                          max_element(local_res.begin(), local_res.end()));
    cout << "Result at stage " << s << ": " << result << "\n";
  };
  for (int i = 0; i < kNumStage; i++) stages[i].dbg_callback = dbg;

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
