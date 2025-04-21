#pragma once
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "engine.h"
#include "utils/config.h"

namespace modnet {

const int kInputH = 288;
const int kInputW = 512;
const int kInputC = 3;
const int kOutputH = 288;
const int kOutputW = 512;
const char* kInputTensorName = "input";
const char* kOutputTensorName = "output";

class MODNet {
public:
    MODNet(const std::string &model_path);
    ~MODNet() = default;
    cv::Mat run(const cv::Mat& img);
    void doInference();
    void preprocess(const cv::Mat& img);
    cv::Mat postprocess();
private:
    std::shared_ptr<TrtEngine> engine_;
};

}