#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

namespace fs = std::filesystem;

// Configuration class to hold all parameters
class Config {
public:
    std::string model_path;
    fs::path images_dir;
    fs::path output_dir;
    bool save_visualizations = false;
    double overlay_alpha = 0.5;
    float anomaly_threshold = 13.0f;
    std::vector<float> normalization_mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> normalization_std = {0.229f, 0.224f, 0.225f};

    bool parseArgs(int argc, char** argv) {
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " <model.onnx> <images_dir> [--save_viz out_dir] [--alpha 0.5] [--thresh 0.5]\n";
            return false;
        }

        model_path = argv[1];
        images_dir = argv[2];

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--save_viz" && i + 1 < argc) {
                save_visualizations = true;
                output_dir = argv[++i];
            } else if (arg == "--alpha" && i + 1 < argc) {
                overlay_alpha = std::stod(argv[++i]);
            } else if (arg == "--thresh" && i + 1 < argc) {
                anomaly_threshold = std::stof(argv[++i]);
            }
        }

        return validatePaths();
    }

private:
    bool validatePaths() {
        if (!fs::exists(model_path)) {
            std::cerr << "ERROR: Model file does not exist: " << model_path << std::endl;
            return false;
        }

        if (!fs::exists(images_dir)) {
            std::cerr << "ERROR: Images directory does not exist: " << images_dir << std::endl;
            return false;
        }

        if (save_visualizations && !output_dir.empty()) {
            if (!fs::exists(output_dir)) {
                fs::create_directories(output_dir);
            }
        }

        return true;
    }
};

// Image preprocessing utility class
class ImagePreprocessor {
public:
    static std::vector<float> preprocess(const cv::Mat& img_bgr,
                                       const std::vector<int64_t>& target_shape,
                                       const std::vector<float>& mean,
                                       const std::vector<float>& std) {
        int target_h = 224, target_w = 224;
        if (target_shape.size() >= 4) {
            target_h = (int)target_shape[2];
            target_w = (int)target_shape[3];
        }

        cv::Mat img_rgb, resized;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
        cv::resize(img_rgb, resized, cv::Size(target_w, target_h));
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);

        std::vector<float> output;
        output.reserve(3 * target_h * target_w);

        for (int c = 0; c < 3; ++c) {
            cv::Mat ch = channels[c];
            for (int i = 0; i < ch.rows; ++i) {
                float* ptr = ch.ptr<float>(i);
                for (int j = 0; j < ch.cols; ++j) {
                    float normalized = (ptr[j] - mean[c]) / std[c];
                    output.push_back(normalized);
                }
            }
        }

        return output;
    }
};

// Results container
struct DetectionResult {
    float anomaly_score = 0.0f;
    cv::Mat heatmap;
    cv::Mat anomaly_mask;
    bool is_anomalous = false;
    double inference_time_ms = 0.0;
    int anomalous_pixels = 0;
    float anomaly_ratio = 0.0f;
};

// ONNX model wrapper class
class ONNXAnomalyDetector {
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<int64_t> input_shape_;
    std::string input_name_;
    std::vector<std::string> output_names_;

public:
    bool initialize(const std::string& model_path) {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AnomalyDetector");

            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
            std::wstring wmodel_path(model_path.begin(), model_path.end());
            session_ = std::make_unique<Ort::Session>(*env_, wmodel_path.c_str(), session_options);
#else
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
#endif

            // Get input info
            Ort::AllocatorWithDefaultOptions allocator;
            input_name_ = session_->GetInputName(0, allocator);
            auto input_shape_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            input_shape_ = std::vector<int64_t>(input_shape_info.begin(), input_shape_info.end());

            // Get output names
            size_t num_outputs = session_->GetOutputCount();
            for (size_t i = 0; i < num_outputs; ++i) {
                char* name = session_->GetOutputName(i, allocator);
                output_names_.emplace_back(name);
                allocator.Free(name);
            }

            std::cout << "Model initialized successfully with " << num_outputs << " outputs" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize model: " << e.what() << std::endl;
            return false;
        }
    }
// Modified detect method with detailed timing:

DetectionResult detect(const cv::Mat& image, const Config& config) {
    DetectionResult result;

    try {
        auto total_start = std::chrono::high_resolution_clock::now();

        // Preprocessing timing
        auto prep_start = std::chrono::high_resolution_clock::now();
        auto input_data = ImagePreprocessor::preprocess(image, input_shape_,
                                                      config.normalization_mean,
                                                      config.normalization_std);
        auto prep_end = std::chrono::high_resolution_clock::now();
        double prep_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start).count() / 1000.0;

        // Create tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> dims = input_shape_;
        if (dims[0] <= 0) dims[0] = 1;

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), dims.data(), dims.size());

        // Inference timing
        auto inference_start = std::chrono::high_resolution_clock::now();
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names_c;
        for (const auto& name : output_names_) {
            output_names_c.push_back(name.c_str());
        }

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                   input_names.data(), &input_tensor, 1,
                                   output_names_c.data(), output_names_c.size());
        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.0;

        auto total_end = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000.0;

        // Print detailed timing
        std::cout << "  Preprocessing: " << prep_time_ms << " ms" << std::endl;
        std::cout << "  Inference: " << inference_time_ms << " ms" << std::endl;
        std::cout << "  Total: " << result.inference_time_ms << " ms" << std::endl;

        // Process outputs
        processOutputs(outputs, result, config.anomaly_threshold);

    } catch (const std::exception& e) {
        std::cerr << "Detection failed: " << e.what() << std::endl;
    }

    return result;
}
    private:
    void processOutputs(std::vector<Ort::Value>& outputs, DetectionResult& result, float threshold) {
        if (outputs.size() >= 2) {
            // Multiple outputs: score + heatmap
            auto& score_tensor = outputs[0];
            float* score_data = score_tensor.GetTensorMutableData<float>();
            result.anomaly_score = score_data[0];

            auto& heatmap_tensor = outputs[1];
            auto heatmap_shape = heatmap_tensor.GetTensorTypeAndShapeInfo().GetShape();
            float* heatmap_data = heatmap_tensor.GetTensorMutableData<float>();

            if (heatmap_shape.size() >= 2) {
                int h = (int)heatmap_shape[heatmap_shape.size()-2];
                int w = (int)heatmap_shape[heatmap_shape.size()-1];
                result.heatmap = cv::Mat(h, w, CV_32FC1, heatmap_data).clone();

                // Apply Gaussian blur and create mask
                cv::GaussianBlur(result.heatmap, result.heatmap, cv::Size(33, 33), 4.0f);
                cv::threshold(result.heatmap, result.anomaly_mask, threshold, 255.0, cv::THRESH_BINARY);
                result.anomaly_mask.convertTo(result.anomaly_mask, CV_8UC1);

                result.anomalous_pixels = cv::countNonZero(result.anomaly_mask);
                result.anomaly_ratio = (float)result.anomalous_pixels / (h * w);
            }
        } else if (outputs.size() == 1) {
            // Single output - determine type by shape
            auto& tensor = outputs[0];
            auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
            float* data = tensor.GetTensorMutableData<float>();

            int64_t total_elements = 1;
            for (auto dim : shape) total_elements *= dim;

            if (total_elements == 1) {
                result.anomaly_score = data[0];
            } else if (shape.size() >= 2) {
                // Treat as heatmap
                int h = (int)shape[shape.size()-2];
                int w = (int)shape[shape.size()-1];
                result.heatmap = cv::Mat(h, w, CV_32FC1, data).clone();

                cv::GaussianBlur(result.heatmap, result.heatmap, cv::Size(33, 33), 4.0f);
                cv::threshold(result.heatmap, result.anomaly_mask, threshold, 255.0, cv::THRESH_BINARY);
                result.anomaly_mask.convertTo(result.anomaly_mask, CV_8UC1);

                cv::Scalar mean_score = cv::mean(result.heatmap);
                result.anomaly_score = (float)mean_score[0];

                result.anomalous_pixels = cv::countNonZero(result.anomaly_mask);
                result.anomaly_ratio = (float)result.anomalous_pixels / (h * w);
            }
        }

        result.is_anomalous = result.anomaly_score > threshold;
    }
};

// Visualization utility class
class Visualizer {
public:
    static cv::Mat createOverlay(const cv::Mat& original, const cv::Mat& heatmap, double alpha) {
        if (heatmap.empty()) return original.clone();

        cv::Mat hm_norm, hm_u8, hm_color;
        cv::normalize(heatmap, hm_norm, 0.0, 255.0, cv::NORM_MINMAX);
        hm_norm.convertTo(hm_u8, CV_8UC1);
        cv::applyColorMap(hm_u8, hm_color, cv::COLORMAP_JET);

        cv::Mat resized_hm;
        cv::resize(hm_color, resized_hm, original.size());

        cv::Mat overlay;
        cv::addWeighted(original, 1.0 - alpha, resized_hm, alpha, 0.0, overlay);
        return overlay;
    }

    static void addTextInfo(cv::Mat& image, const DetectionResult& result, float threshold) {
        cv::Scalar text_color = result.is_anomalous ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::Scalar bg_color = cv::Scalar(0, 0, 0);

        // Background rectangle
        cv::rectangle(image, cv::Point(5, 5), cv::Point(400, 120), bg_color, -1);
        cv::rectangle(image, cv::Point(5, 5), cv::Point(400, 120), cv::Scalar(255, 255, 255), 2);

        // Text information
        std::string score_text = "Score: " + std::to_string(result.anomaly_score);
        std::string status_text = result.is_anomalous ? "ANOMALOUS" : "NORMAL";
        std::string time_text = "Time: " + std::to_string(result.inference_time_ms) + " ms";

        cv::putText(image, score_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        cv::putText(image, status_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2);
        cv::putText(image, time_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }

    static void drawAnomalyBoundaries(cv::Mat& image, const cv::Mat& mask, const cv::Size& target_size) {
        if (mask.empty()) return;

        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, target_size);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(resized_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > 50) {  // Filter small noise
                cv::drawContours(image, contours, (int)i, cv::Scalar(0, 255, 0), 3);
                cv::Rect bbox = cv::boundingRect(contours[i]);
                cv::rectangle(image, bbox, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
};

// File utility class
class FileUtils {
public:
    static std::vector<fs::path> listImages(const fs::path& dir) {
        std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
        std::vector<fs::path> images;

        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            return images;
        }

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!fs::is_regular_file(entry.path())) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                images.push_back(entry.path());
            }
        }

        std::sort(images.begin(), images.end());
        return images;
    }
};

// Statistics tracking class
class Statistics {
public:
    int normal_count = 0;
    int anomalous_count = 0;
    std::vector<float> all_scores;

    void addResult(const DetectionResult& result) {
        if (result.is_anomalous) {
            anomalous_count++;
        } else {
            normal_count++;
        }
        all_scores.push_back(result.anomaly_score);
    }

    void printSummary() const {
        int total = normal_count + anomalous_count;
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "FINAL STATISTICS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total images: " << total << std::endl;
        std::cout << "Normal: " << normal_count << std::endl;
        std::cout << "Anomalous: " << anomalous_count << std::endl;
        std::cout << "Anomaly rate: " << (total > 0 ? (float)anomalous_count / total * 100.0f : 0.0f) << "%" << std::endl;

        if (!all_scores.empty()) {
            float min_score = *std::min_element(all_scores.begin(), all_scores.end());
            float max_score = *std::max_element(all_scores.begin(), all_scores.end());
            float avg_score = std::accumulate(all_scores.begin(), all_scores.end(), 0.0f) / all_scores.size();
            std::cout << "Score range: [" << min_score << ", " << max_score << "]" << std::endl;
            std::cout << "Average score: " << avg_score << std::endl;
        }
        std::cout << std::string(60, '=') << std::endl;
    }
};

// Main application class
class AnomalyDetectionApp {
private:
    Config config_;
    ONNXAnomalyDetector detector_;
    Statistics stats_;

public:
    bool initialize(int argc, char** argv) {
        if (!config_.parseArgs(argc, argv)) {
            return false;
        }

        if (!detector_.initialize(config_.model_path)) {
            return false;
        }

        std::cout << "Anomaly Detection System initialized successfully!" << std::endl;
        return true;
    }

    void run() {
        auto images = FileUtils::listImages(config_.images_dir);
        if (images.empty()) {
            std::cerr << "No images found in directory!" << std::endl;
            return;
        }

        std::cout << "Found " << images.size() << " images to process" << std::endl;

        for (size_t i = 0; i < images.size(); ++i) {
            processImage(images[i], i + 1, images.size());
        }

        stats_.printSummary();
    }

private:
    void processImage(const fs::path& image_path, int current, int total) {
        std::cout << "\nProcessing (" << current << "/" << total << "): " << image_path.filename() << std::endl;

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load: " << image_path << std::endl;
            return;
        }

        auto result = detector_.detect(image, config_);
        stats_.addResult(result);

        // Create visualization
        cv::Mat display = Visualizer::createOverlay(image, result.heatmap, config_.overlay_alpha);
        Visualizer::addTextInfo(display, result, config_.anomaly_threshold);
        Visualizer::drawAnomalyBoundaries(display, result.anomaly_mask, image.size());

        // Display
        std::string window_name = "Anomaly Detection: " + image_path.filename().string();
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 800, 600);
        cv::imshow(window_name, display);

        std::cout << "Result: " << (result.is_anomalous ? "ANOMALOUS" : "NORMAL")
                  << " (score: " << result.anomaly_score << ")" << std::endl;

        // Save if requested
        if (config_.save_visualizations) {
            fs::path output_path = config_.output_dir / (image_path.stem().string() + "_result.png");
            cv::imwrite(output_path.string(), display);
        }

        int key = cv::waitKey(1);
        cv::destroyWindow(window_name);

        if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "User requested quit" << std::endl;
            return;
        }
    }
};

// Main function
int main(int argc, char** argv) {
    std::cout << "=== OOP ANOMALY DETECTION SYSTEM ===" << std::endl;

    AnomalyDetectionApp app;

    if (!app.initialize(argc, argv)) {
        return 1;
    }

    app.run();

    std::cout << "Processing completed!" << std::endl;
    return 0;
}
