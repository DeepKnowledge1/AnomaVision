#include "oop_anomaly_detector.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>

// -------------------- Config Implementation --------------------
bool Config::parse(int argc, char** argv) {
    if(argc < 3) return false;
    model = argv[1];
    images_dir = argv[2];
    for(int i=3;i<argc;i++){
        std::string a = argv[i];
        if(a=="--save_viz" && i+1<argc){ save_viz=true; out_dir=argv[++i]; }
        else if(a=="--alpha" && i+1<argc) alpha = std::stod(argv[++i]);
        else if(a=="--thresh" && i+1<argc) thresh = std::stof(argv[++i]);
    }
    if(!fs::exists(model) || !fs::exists(images_dir)) return false;
    if(save_viz && !fs::exists(out_dir)) fs::create_directories(out_dir);
    return true;
}

// -------------------- ImageList Implementation --------------------
std::vector<fs::path> ImageList::list(const fs::path& dir) {
    std::vector<fs::path> imgs;
    std::vector<std::string> exts{".jpg",".jpeg",".png",".bmp",".tiff"};
    if(!fs::exists(dir) || !fs::is_directory(dir)) return imgs;
    for(auto &e: fs::directory_iterator(dir)){
        if(!fs::is_regular_file(e.path())) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if(std::find(exts.begin(),exts.end(),ext)!=exts.end()) imgs.push_back(e.path());
    }
    std::sort(imgs.begin(), imgs.end());
    return imgs;
}

// -------------------- Preprocessor Implementation --------------------
Preprocessor::Preprocessor(int H, int W, const std::array<float,3>& mean, const std::array<float,3>& stdev)
    : H_(H), W_(W), mean_(mean), stdev_(stdev) {}

std::vector<float> Preprocessor::preprocess(const cv::Mat& bgr) const {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat r; cv::resize(rgb, r, cv::Size(W_,H_));
    r.convertTo(r, CV_32FC3, 1.0f/255.0f);
    std::vector<cv::Mat> ch(3); cv::split(r, ch);
    std::vector<float> out; out.reserve(3*H_*W_);
    for(int c=0;c<3;++c){
        const cv::Mat& m = ch[c];
        for(int y=0;y<H_;++y){
            const float* ptr = m.ptr<float>(y);
            for(int x=0;x<W_;++x) out.push_back((ptr[x]-mean_[c])/stdev_[c]);
        }
    }
    return out;
}

// -------------------- ONNXModel Implementation --------------------
ONNXModel::ONNXModel(const std::string& model_path, Ort::Env& env, Ort::SessionOptions& opts)
    : env_(env), session_(nullptr), alloc_() {
#ifdef _WIN32
    std::wstring wmodel(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wmodel.c_str(), opts);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
#endif
    fetch_io();
}

const std::string& ONNXModel::input_name() const { return input_name_; }
const std::vector<int64_t>& ONNXModel::input_shape() const { return input_shape_; }
const std::vector<std::string>& ONNXModel::output_names() const { return output_names_; }

std::vector<Ort::Value> ONNXModel::run(float* input_data, size_t input_count) {
    std::vector<int64_t> dims = input_shape_;
    if(dims.size()>0 && dims[0] <= 0) dims[0] = 1;

    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(meminfo, input_data, input_count, dims.data(), dims.size());

    std::vector<const char*> in_names = { input_name_.c_str() };
    std::vector<const char*> out_names_c;
    out_names_c.reserve(output_names_.size());
    for(const auto &n: output_names_) out_names_c.push_back(n.c_str());

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), &input_tensor, 1, out_names_c.data(), out_names_c.size());
    return outputs;
}

void ONNXModel::fetch_io() {
    char* in = session_->GetInputName(0, alloc_);
    input_name_ = in;
    alloc_.Free(in);

    auto info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_shape_.assign(info.begin(), info.end());

    size_t nout = session_->GetOutputCount();
    for(size_t i=0;i<nout;i++){
        char* n = session_->GetOutputName(i, alloc_);
        output_names_.emplace_back(n);
        alloc_.Free(n);
    }
}

// -------------------- Postprocessor Implementation --------------------
Postprocessor::Postprocessor(float threshold) : threshold_(threshold) {}

void Postprocessor::process(std::vector<Ort::Value>& outputs, Result& r) {
    if(outputs.size() >= 2){
        float* sdata = outputs[0].GetTensorMutableData<float>();
        r.score = sdata[0];
        auto shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        float* hdata = outputs[1].GetTensorMutableData<float>();
        int H = (int)shape[shape.size()-2], W = (int)shape[shape.size()-1];
        r.heatmap = cv::Mat(H,W,CV_32FC1,hdata).clone();
    } else if(outputs.size() == 1){
        auto &t = outputs[0];
        auto shape = t.GetTensorTypeAndShapeInfo().GetShape();
        int64_t total=1; for(auto d:shape) if(d>0) total*=d;
        float* data = t.GetTensorMutableData<float>();
        if(total==1) r.score = data[0];
        else {
            int H = (int)shape[shape.size()-2], W = (int)shape[shape.size()-1];
            r.heatmap = cv::Mat(H,W,CV_32FC1,data).clone();
            cv::Scalar m = cv::mean(r.heatmap); r.score = (float)m[0];
        }
    }

    if(!r.heatmap.empty()){
        cv::GaussianBlur(r.heatmap, r.heatmap, cv::Size(33,33), 4.0);
        cv::threshold(r.heatmap, r.mask, threshold_, 255.0, cv::THRESH_BINARY);
        r.mask.convertTo(r.mask, CV_8UC1);
        r.pixels = cv::countNonZero(r.mask);
        r.ratio = (float)r.pixels / (r.heatmap.rows * r.heatmap.cols);
    }
    r.anomalous = r.score > threshold_;
}

// -------------------- Visualizer Implementation --------------------
Visualizer::Visualizer(double alpha) : alpha_(alpha) {}

cv::Mat Visualizer::overlay(const cv::Mat& orig, const cv::Mat& heat) const {
    // Just return original image (no heatmap overlay)
    return orig.clone();
}


void Visualizer::annotate(cv::Mat& img, const Result& r, double t_ms) const {
    cv::Scalar txt = r.anomalous ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
    cv::rectangle(img, {5,5}, {420,110}, {0,0,0}, -1);
    cv::putText(img, "Score: "+to_fixed(r.score,3), {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, txt, 2);
    cv::putText(img, r.anomalous?"ANOMALOUS":"NORMAL", {10,60}, cv::FONT_HERSHEY_SIMPLEX, 0.9, txt, 2);
    cv::putText(img, "Time: "+to_fixed(t_ms,2)+" ms", {10,90}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);
}

void Visualizer::draw_boundaries(cv::Mat& img, const cv::Mat& mask) const {
    if(mask.empty()) return;
    cv::Mat rm; cv::resize(mask, rm, img.size());
    std::vector<std::vector<cv::Point>> ctrs;
    cv::findContours(rm, ctrs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for(size_t i=0;i<ctrs.size();++i){
        if(cv::contourArea(ctrs[i])>50){
            cv::drawContours(img, ctrs, (int)i, {0,255,0}, 3);
            cv::Rect b = cv::boundingRect(ctrs[i]); cv::rectangle(img, b, {0,0,255}, 2);
        }
    }
}

std::string Visualizer::to_fixed(double v, int prec) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss.precision(prec);
    oss<<v;
    return oss.str();
}

// -------------------- App Implementation --------------------
App::App(const Config& cfg) : cfg_(cfg), env_(ORT_LOGGING_LEVEL_WARNING, "det"), opts_(), alloc_() {
    opts_.SetIntraOpNumThreads(1);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    model_ = std::make_unique<ONNXModel>(cfg_.model, env_, opts_);

    auto ishape = model_->input_shape();
    if(ishape.size()<4) throw std::runtime_error("Unexpected input shape from model");
    H_ = (int)ishape[2]; W_ = (int)ishape[3];

    preprocessor_ = std::make_unique<Preprocessor>(H_, W_, cfg_.mean, cfg_.std);
    postprocessor_ = std::make_unique<Postprocessor>(cfg_.thresh);
    visualizer_ = std::make_unique<Visualizer>(cfg_.alpha);
}

int App::run() {
    auto images = ImageList::list(cfg_.images_dir);
    if(images.empty()){ std::cerr<<"No images found\n"; return 1; }
    std::cout<<"Found "<<images.size()<<" images\n";

    std::vector<int64_t> dims = model_->input_shape();
    if(dims.size()>0 && dims[0] <= 0) dims[0] = 1;
    size_t input_count = 1; for(auto d: dims) input_count *= (d>0? d: 1);

    for(size_t i=0;i<images.size();++i){
        auto p = images[i];
        std::cout<<"\nProcessing "<<p.filename().string()<<" ("<<i+1<<"/"<<images.size()<<")\n";
        cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
        if(img.empty()){ std::cerr<<"Failed to open "<<p<<"\n"; continue; }

        auto input_vec = preprocessor_->preprocess(img);
        if(input_vec.size() != input_count){
            if(input_vec.size() > input_count){
                std::cerr<<"Warning: produced more input elements than model expects. Using first N elements.\n";
            } else {
                std::cerr<<"Warning: input size mismatch: produced "<<input_vec.size()<<" expected "<<input_count<<"\n";
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto outputs = model_->run(input_vec.data(), std::min(input_vec.size(), input_count));
        auto t2 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;

        Result res; res.time_ms = elapsed_ms;
        postprocessor_->process(outputs, res);

        cv::Mat disp = visualizer_->overlay(img, res.heatmap);
        visualizer_->annotate(disp, res, elapsed_ms);
        visualizer_->draw_boundaries(disp, res.mask);

        std::string wname = "Result: " + p.filename().string();
        cv::namedWindow(wname, cv::WINDOW_NORMAL);
        cv::resizeWindow(wname, 800, 600);
        cv::imshow(wname, disp);
        if(cfg_.save_viz){
            fs::path outp = cfg_.out_dir / (p.stem().string() + "_res.png");
            cv::imwrite(outp.string(), disp);
        }
        int key = cv::waitKey(1);
        cv::destroyWindow(wname);
        if(key==27 || key=='q' || key=='Q'){ std::cout<<"User quit\n"; break; }

        std::cout<<"Score: "<<res.score<<"  Anomalous: "<<(res.anomalous?"YES":"NO")<<"  Time: "<<res.time_ms<<" ms\n";
        std::cout<<"Number of anomalous pixels: "<<res.pixels<<"  Ratio to total pixels: "<<res.ratio<<" ms\n";
    }

    std::cout<<"\nDone\n";
    return 0;
}
