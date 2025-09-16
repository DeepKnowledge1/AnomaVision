
#include "oop_anomaly_detector.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>


int main(int argc, char** argv) {
    Config cfg;
    if(!cfg.parse(argc, argv)){
        std::cout<<"Usage: "<<argv[0]<<" <model.onnx> <images_dir> [--save_viz out_dir] [--alpha 0.5] [--thresh 0.5]\n";
        return 1;
    }

    try {
        App app(cfg);
        return app.run();
    } catch(const std::exception& ex){
        std::cerr<<"Fatal error: "<<ex.what()<<"\n";
        return 1;
    }
}

// How to run
// .\build\Release\onnx_inference.exe  model.onnx D:/01-DATA/test --save_viz D:/output --alpha 0.5 --thresh 13.0
