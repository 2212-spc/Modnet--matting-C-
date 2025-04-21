#include "modnet.h"

#include "utils/preprocess.h"
#include "utils/config.h"

#include <opencv2/dnn/all_layers.hpp>

namespace modnet
{

    MODNet::MODNet(const std::string &model_path)
    {
        engine_.reset(new TrtEngine(model_path));
    }

    cv::Mat MODNet::run(const cv::Mat &img)
    {
        preprocess(img);
        doInference();
        return postprocess();
    }

    void MODNet::doInference()
    {
        engine_->doInference();
    }

    void MODNet::preprocess(const cv::Mat &img)
    {
        MODNetPreprocess(img, kInputW, kInputH, engine_->getHostBuffer(kInputTensorName));
    }

    cv::Mat MODNet::postprocess()
    {
        int output_size = kOutputH * kOutputW;
        static std::vector<float> matte(output_size);

        memcpy(matte.data(), engine_->getHostBuffer(kOutputTensorName), output_size * sizeof(float)); // 将engine_中的outputTensorName的数据拷贝到matte中
        cv::Mat matte_mat(kOutputH, kOutputW, CV_32FC1, matte.data());                                // 将matte的数据拷贝到matte_mat中

        // cv::imwrite("matte.png", matte_mat * 255);                                                    // matte_mat 的值在0-1之间，乘以255之后，就是0-255之间的值了
        return matte_mat;
    }

} // namespace
