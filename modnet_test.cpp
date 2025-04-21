#include <iostream>

#include "gflags/gflags.h"

#include "modnet.h"
#include "utils/draw.h"

#include <glob.h>
#include <vector>
using std::vector;

DEFINE_string(modnet, "modnet-sim.engine", "facenet model path");
DEFINE_string(vid_dir, "", "video files directory");
DEFINE_string(format, "", "export format: foreground, matte, background");
DEFINE_string(bg, "", "background image path");

vector<std::string> globVector(const std::string &pattern)
{
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<std::string> files;
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

void processMatting(std::string filename, modnet::MODNet &modnet, std::string format, std::string bg)
{
    cv::VideoCapture cap;

    cap.open(filename);

    // get height with from cap
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << width << ", height: " << height << ", fps: " << fps << std::endl;

    // h264 save mp4
    cv::VideoWriter writer;
    // ./videos/test1.mp4 -> test
    std::string out_filename = filename.substr(filename.find_last_of("/") + 1);
    out_filename = out_filename.substr(0, out_filename.find_last_of("."));
    out_filename = "./output/" + format + "_" + out_filename + "_out.mp4";

    writer.open(out_filename, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(width, height));

    cv::Mat frame;
    cv::Mat matte;

    cv::Mat bg_img;
    if (format == "background")
    {
        bg_img = cv::imread(bg);
    }
    cv::Mat matte_temp;
    while (cap.read(frame))
    {
        // FPS开始时间
        auto start = std::chrono::high_resolution_clock::now();

        matte = modnet.run(frame);
        // cv2以numpy数组形式打印matte
        // std::cout << cv::format(matte, cv::Formatter::FMT_NUMPY) << std::endl;

        // // COLORMAP_JET 显示为彩色
        // cv::convertScaleAbs(matte, matte_temp, 255, 0);
        // cv::applyColorMap(matte_temp, matte_temp, cv::COLORMAP_JET);
        // cv::resize(matte_temp, matte_temp, cv::Size(width, height));
        // cv::imwrite("./output/matte.jpg", matte_temp);

        draw_matte(frame, matte, format, bg_img);

        // FPS结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;

        // std::cout << "elapsed: " << elapsed << "ms" << std::endl;
        std::cout << "filename: " << filename << ", fps: " << 1000.f / elapsed << std::endl;
        std::string fps_str = "FPS: " + std::to_string(1000.f / elapsed);
        cv::putText(frame, fps_str, cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

        writer.write(frame);
    }
    // release writer
    writer.release();
    // release capture
    cap.release();

    std::cout << "processMatting: " << filename << " done, output: " << out_filename << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // gflags print help message
        std::cout << "Usage: " << argv[0] << " --modnet=modnet-sim.engine --vid_dir=videos" << std::endl;
        return -1;
    }

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::string modnet_path = FLAGS_modnet;
    std::string vid_dir = FLAGS_vid_dir;
    std::string format = FLAGS_format; // foreground, matte, background
    std::string bg = FLAGS_bg;

    modnet::MODNet modnet(modnet_path);

    // list all files in directory
    vector<std::string> files = globVector(vid_dir + "/*.mp4");
    for (auto &file : files)
    {
        // std::cout << file << std::endl;
        processMatting(file, modnet, format, bg);
    }

    return 0;
}