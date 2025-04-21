#pragma once
#include <opencv2/opencv.hpp>

void MODNetPreprocess(const cv::Mat& img, int inputW, int inputH, void* buffer);