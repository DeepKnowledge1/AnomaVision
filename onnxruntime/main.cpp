/*

install onnxruntime:
1. Download the ONNX Runtime package from the official GitHub releases page.
Choose the version that matches your system architecture (e.g., onnxruntime-win-x64-*.zip for Windows 64-bit).

https://github.com/microsoft/onnxruntime/releases?q=onnxruntime-win-x64&expanded=true

2. Extract the downloaded ZIP file to a directory of your choice.
3. In your CMakeLists.txt, set the ONNXRUNTIME_DIR variable to point to the extracted directory.
4. Ensure your CMakeLists.txt includes the ONNX Runtime include and lib directories.
5. Link your executable against the onnxruntime library.
6. Compile your project using CMake and your preferred build system (e.g., Visual Studio, Make).

*/

/*
How to compile in Vs Code (Windows) with CMake Tools extension :
1. Open your project folder in VS Code.
2. Ensure you have the CMake Tools extension installed.
3. Create a CMakeLists.txt file in the root of your project if you haven't already.
4. open the terminal in VS Code (Ctrl + `).
5. Run the following commands in the terminal:
cmake -S . -B build    # Configure the project and generate build files
cmake --build build --config Release  # Build the project in Release mode
6. After building, you can run your executable from the terminal:
debug\onnx_inference.exe
7. Make sure to place your model.onnx file in the same directory as your executable or provide the correct path in the code.

*/


#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
        Ort::SessionOptions session_options;
        // session_options.SetIntraOpNumThreads(1);



        session_options.SetIntraOpNumThreads(4); // Use more CPU cores
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        // For even better performance:
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();



        const wchar_t* model_path = L"model.onnx";
        Ort::Session session(env, model_path, session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Get input info
        size_t num_inputs = session.GetInputCount();
        std::cout << "Number of inputs: " << num_inputs << std::endl;

        for (size_t i = 0; i < num_inputs; i++) {
            const char* input_name = session.GetInputName(i, allocator);
            auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Input " << i << " Name: " << input_name << std::endl;
            std::cout << "Input " << i << " Shape: [";
            for (size_t j = 0; j < input_shape.size(); j++) {
                std::cout << input_shape[j];
                if (j < input_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Get output info
        size_t num_outputs = session.GetOutputCount();
        std::cout << "Number of outputs: " << num_outputs << std::endl;

        std::vector<const char*> output_names;
        if (num_outputs >= 1) output_names.push_back("output_0");
        if (num_outputs >= 2) output_names.push_back("output_1");

        for (size_t i = 0; i < num_outputs; i++) {
            auto output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Output " << i << " Name: " << output_names[i] << std::endl;
            std::cout << "Output " << i << " Shape: [";
            for (size_t j = 0; j < output_shape.size(); j++) {
                std::cout << output_shape[j];
                if (j < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Prepare input tensor (same for all iterations)
        const char* first_input_name = session.GetInputName(0, allocator);
        auto expected_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        size_t total_elements = 1;
        for (auto dim : expected_shape) {
            if (dim > 0) total_elements *= dim;
        }

        std::vector<float> input_tensor_values(total_elements, 1.0f);
        std::vector<int64_t> input_dims;
        for (auto dim : expected_shape) {
            input_dims.push_back(dim > 0 ? dim : 1);
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_dims.data(), input_dims.size()
        );

        const char* input_names[] = {first_input_name};

        // Timing variables
        const int num_iterations = 10;
        std::vector<double> inference_times;
        inference_times.reserve(num_iterations);

        std::cout << "\n=== Running " << num_iterations << " inference iterations ===" << std::endl;

        // Warm-up run (not counted in timing)
        std::cout << "Performing warm-up run..." << std::endl;
        auto warmup_outputs = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names.data(), num_outputs
        );

        // Timed inference loop
        for (int i = 0; i < num_iterations; i++) {
            // Simulate different input data (optional - you can remove this if using same data)
            for (size_t j = 0; j < input_tensor_values.size(); j++) {
                input_tensor_values[j] = static_cast<float>(rand()) / RAND_MAX;
            }

            // Create new tensor for this iteration (if using different data)
            Ort::Value current_input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_dims.data(), input_dims.size()
            );

            // Start timing
            auto start_time = std::chrono::high_resolution_clock::now();

            // Run inference
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, input_names, &current_input_tensor, 1,
                output_names.data(), num_outputs
            );

            // End timing
            auto end_time = std::chrono::high_resolution_clock::now();

            // Calculate duration in milliseconds
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double inference_time_ms = duration.count() / 1000.0;
            inference_times.push_back(inference_time_ms);

            // Print progress every 10 iterations
            std::cout << "Completed " << (i + 1) << "/" << num_iterations
                         << " iterations. Latest time: " << inference_time_ms << " ms" << std::endl;

        }

        // Calculate statistics
        double total_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0);
        double avg_time = total_time / num_iterations;

        std::sort(inference_times.begin(), inference_times.end());
        double min_time = inference_times.front();
        double max_time = inference_times.back();
        double median_time = inference_times[num_iterations / 2];

        // Calculate percentiles
        double p95_time = inference_times[static_cast<int>(num_iterations * 0.95)];
        double p99_time = inference_times[static_cast<int>(num_iterations * 0.99)];

        // Print timing results
        std::cout << "\n=== TIMING RESULTS ===" << std::endl;
        std::cout << "Total iterations: " << num_iterations << std::endl;
        std::cout << "Total time: " << total_time << " ms" << std::endl;
        std::cout << "Average time: " << avg_time << " ms" << std::endl;
        std::cout << "Median time: " << median_time << " ms" << std::endl;
        std::cout << "Min time: " << min_time << " ms" << std::endl;
        std::cout << "Max time: " << max_time << " ms" << std::endl;
        std::cout << "95th percentile: " << p95_time << " ms" << std::endl;
        std::cout << "99th percentile: " << p99_time << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 / avg_time) << " inferences/second" << std::endl;

        // Show sample output from last inference
        std::cout << "\n=== SAMPLE OUTPUT (from last inference) ===" << std::endl;
        auto final_outputs = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names.data(), num_outputs
        );

        for (size_t i = 0; i < final_outputs.size(); i++) {
            auto shape = final_outputs[i].GetTensorTypeAndShapeInfo().GetShape();
            float* output_arr = final_outputs[i].GetTensorMutableData<float>();

            size_t total_elements = 1;
            for (auto dim : shape) {
                if (dim > 0) total_elements *= dim;
            }

            std::cout << "Output " << i << " (" << output_names[i] << "):" << std::endl;
            std::cout << "  Shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  First few values: ";
            for (size_t j = 0; j < std::min(total_elements, size_t(5)); j++) {
                std::cout << output_arr[j] << " ";
            }
            std::cout << std::endl;

            if (i == 0) {
                std::cout << "  -> This is likely the anomaly score" << std::endl;
            } else if (i == 1) {
                std::cout << "  -> This is likely the anomaly map" << std::endl;
            }
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}
