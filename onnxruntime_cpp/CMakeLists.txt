
# cd path\to\project
# cmake -S . -B build
# cmake --build build --config Release
cmake_minimum_required(VERSION 3.15)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

# --- ONNX Runtime ---
# Adjust this path if ONNX Runtime is installed somewhere else
set(ONNXRUNTIME_DIR "C:/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# --- OpenCV ---
set(OpenCV_DIR "D:/TOOLS/opencv/build")   # root of OpenCV build folder
include_directories(${OpenCV_DIR}/include)
link_directories(${OpenCV_DIR}/x64/vc16/lib)

# Add both source files
add_executable(onnx_inference
    detect.cpp
    oop_anomaly_detector.cpp
)

add_custom_command(TARGET onnx_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world4120.dll"
        $<TARGET_FILE_DIR:onnx_inference>)

# --- Link libraries ---
target_link_libraries(onnx_inference
    onnxruntime
    opencv_world4120   # change to match your OpenCV version (e.g., opencv_world470.lib)
)
