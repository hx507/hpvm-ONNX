#include <stdint.h>

#include <array>
#include <vector>

constexpr const int width = 28;
constexpr const int height = 28;
constexpr const char* input_names[] = {"Input3"};
constexpr const char* output_names[] = {"Plus214_Output_0"};
constexpr std::array<int64_t, 4> input_shape{1, 1, width, height};
constexpr std::array<int64_t, 2> output_shape{1, 10};

void init_model(char* model_file_name);
std::array<float, 10> do_inference(
    std::array<float, width * height>& input_image);
