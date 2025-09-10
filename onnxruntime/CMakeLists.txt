cmake_minimum_required(VERSION 3.15)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

# Path to ONNX Runtime
set(ONNXRUNTIME_DIR "C:/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Create executable with proper target name
add_executable(onnx_inference main.cpp)

# Link against ONNX Runtime
target_link_libraries(onnx_inference onnxruntime)
