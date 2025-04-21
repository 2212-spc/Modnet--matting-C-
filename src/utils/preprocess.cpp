#include "preprocess.h"

void MODNetPreprocess(const cv::Mat &img, int inputW, int inputH, void *buffer)
{

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3);
    cv::subtract(normalized, cv::Scalar(127.5, 127.5, 127.5), normalized);
    cv::divide(normalized, cv::Scalar(127.5, 127.5, 127.5), normalized);
    // split it into three channels
    std::vector<cv::Mat> nchw_channels;
    cv::split(normalized, nchw_channels);

    for (auto &img : nchw_channels)
    {
        img = img.reshape(1, 1);
    }

    cv::Mat nchw;
    cv::hconcat(nchw_channels, nchw);

    memcpy(buffer, nchw.data, 3 * inputH * inputW * sizeof(float));
}
