#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <array>
#include <memory>

namespace fs = std::filesystem;

// -------------------- Config --------------------
/**
 * Holds all configurable parameters for the anomaly detection pipeline,
 * including model path, input/output directories, visualization settings,
 * and preprocessing parameters.
 */
struct Config {
    std::string model;                              // Path to the ONNX model file
    fs::path images_dir;                           // Directory containing input images
    fs::path out_dir;                              // Output directory for results
    bool save_viz = false;                         // Whether to save visualization images
    double alpha = 0.5;                           // Alpha blending factor for overlay visualization
    float thresh = 13.0f;                         // Anomaly threshold for classification
    std::array<float,3> mean{0.485f,0.456f,0.406f}; // ImageNet normalization mean values (RGB)
    std::array<float,3> std{0.229f,0.224f,0.225f};  // ImageNet normalization standard deviation values (RGB)

    bool parse(int argc, char** argv);
};

// -------------------- Result --------------------
/**
 * Contains all output information from the anomaly detection process,
 * including anomaly score, generated heatmap, binary mask, and timing information.
 */
struct Result {
    float score = 0;                               // Overall anomaly score for the image
    cv::Mat heatmap;                              // Anomaly heatmap (normalized float values)
    cv::Mat mask;                                 // Binary mask indicating anomalous regions
    bool anomalous = false;                       // Boolean flag: true if image is classified as anomalous
    int pixels = 0;                               // Number of anomalous pixels detected
    float ratio = 0;                              // Ratio of anomalous pixels to total pixels
    double time_ms = 0;                           // Processing time in milliseconds
};

// -------------------- ImageList --------------------
/**
 * Provides functionality to scan directories and return lists of image file paths
 * for batch processing.
 */
class ImageList {
public:
    static std::vector<fs::path> list(const fs::path& dir);
};

// -------------------- Preprocessor --------------------
/**
 * Handles resizing, normalization, and format conversion of input images
 * to match the expected input format of the ONNX model.
 */
class Preprocessor {
public:

    Preprocessor(int H, int W, const std::array<float,3>& mean, const std::array<float,3>& stdev);

    std::vector<float> preprocess(const cv::Mat& bgr) const;

private:
    int H_, W_;                                   // Target image dimensions
    std::array<float,3> mean_;                   // Normalization mean values
    std::array<float,3> stdev_;                  // Normalization standard deviation values
};

// -------------------- ONNXModel --------------------
/**
 *Provides a high-level interface for loading and running ONNX models,
 * handling session management and input/output tensor operations.
 */
class ONNXModel {
public:

    ONNXModel(const std::string& model_path, Ort::Env& env, Ort::SessionOptions& opts);

    // Accessors for model metadata
    const std::string& input_name() const;         // Get input tensor name
    const std::vector<int64_t>& input_shape() const; // Get input tensor shape
    const std::vector<std::string>& output_names() const; // Get output tensor names

    std::vector<Ort::Value> run(float* input_data, size_t input_count);

private:

    void fetch_io();

    Ort::Env& env_;                              // Reference to ONNX Runtime environment
    std::unique_ptr<Ort::Session> session_;      // ONNX inference session
    Ort::AllocatorWithDefaultOptions alloc_;     // Memory allocator for tensors
    std::string input_name_;                     // Cached input tensor name
    std::vector<int64_t> input_shape_;          // Cached input tensor shape
    std::vector<std::string> output_names_;      // Cached output tensor names
};

// -------------------- Postprocessor --------------------
/**
 * Handles conversion of raw neural network outputs into meaningful
 * anomaly detection results including thresholding and mask generation.
 */
class Postprocessor {
public:

    Postprocessor(float threshold);

    void process(std::vector<Ort::Value>& outputs, Result& r);

private:
    float threshold_;                            // Threshold for anomaly classification
};

// -------------------- Visualizer --------------------
/**
 * Provides functionality to create overlay visualizations, draw annotations,
 * and generate visual representations of anomaly detection results.
 */
class Visualizer {
public:

    Visualizer(double alpha);


    cv::Mat overlay(const cv::Mat& orig, const cv::Mat& heat) const;


    void annotate(cv::Mat& img, const Result& r, double t_ms) const;

    void draw_boundaries(cv::Mat& img, const cv::Mat& mask) const;


    static std::string to_fixed(double v, int prec);

private:
    double alpha_;                               // Alpha blending factor for overlays
};

// -------------------- App --------------------
/**
 *
 * Coordinates all components (preprocessing, inference, postprocessing, visualization)
 * and manages the complete workflow from input images to final results.
 */
class App {
public:
    App(const Config& cfg);

    int run();

private:
    Config cfg_;                                 // Application configuration
    Ort::Env env_;                              // ONNX Runtime environment
    Ort::SessionOptions opts_;                   // ONNX Runtime session options
    Ort::AllocatorWithDefaultOptions alloc_;    // ONNX Runtime memory allocator

    // Component instances (initialized during construction)
    std::unique_ptr<ONNXModel> model_;          // Neural network model wrapper
    std::unique_ptr<Preprocessor> preprocessor_; // Image preprocessing pipeline
    std::unique_ptr<Postprocessor> postprocessor_; // Output post-processing pipeline
    std::unique_ptr<Visualizer> visualizer_;     // Visualization generator

    int H_{0}, W_{0};                           // Model input dimensions (height, width)
};
