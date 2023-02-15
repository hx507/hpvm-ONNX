add_executable(run_onnx run_onnx.cpp)

target_link_libraries(run_onnx PRIVATE onnxruntime)
