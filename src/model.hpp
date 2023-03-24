#pragma once
#include <stdint.h>

#include <array>
#include <functional>
#include <vector>

#define HPVM_STAGE_DEBUG

/**
#define TCB_SPAN_NO_EXCEPTIONS  // need an exception less implementation of
                                // std::span
#include "span.hpp"
namespace std {
using tcb::span;
};
**/

// Compile time asserts
template <bool b>
class ClassStaticAssert;
template <>
class ClassStaticAssert<true> {
  static const bool value = true;
};
#define STATIC_ASSERT(e) (ClassStaticAssert<e>())

// short hands to handle data pairs
#define data_pair(X) (X).size(), (X).data()
#define edge_pair(X) (void*)&(X), sizeof(X)

struct OnnxStage {
  const char* model_file_name;
  std::vector<char*> input_names, output_names;
  std::vector<int64_t> input_shape, output_shape;
  std::function<void(int stage, float* out)> dbg_callback{};
  // possibly add per-stage session parameter tuning here
};

extern std::vector<OnnxStage> stages;
const extern int kNumStage;

void init_all_stages();
void do_inference_at_stage(int stage, size_t inp_sz, float* inp, size_t out_sz,
                           float* out);
