// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"
#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <glog/logging.h>
#include <include/video_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include "auto_log/autolog.h"

DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");
DEFINE_bool(benchmark, true, "Whether use benchmark.");
DEFINE_string(save_log_path, "./log_output/", "Save benchmark log path.");

// detection related
// DEFINE_string(image_dir, "", "Dir of input image.");
// DEFINE_string(rec_model_dir, "", "Path of video rec inference model.");
// DEFINE_int32(max_side_len, 960, "max_side_len of input image.");
// DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
// DEFINE_double(det_db_box_thresh, 0.5, "Threshold of det_db_box_thresh.");
// DEFINE_double(det_db_unclip_ratio, 1.6, "Threshold of det_db_unclip_ratio.");
// DEFINE_bool(use_polygon_score, false, "Whether use polygon score.");
// DEFINE_bool(visualize, true, "Whether show the detection results.");
// classification related
// DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
// DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
// DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");

// recognition related
// DEFINE_int32(max_side_len, 224, "max_side_len of input image.");
DEFINE_string(video_dir, "", "Dir of input video(s).");
DEFINE_string(rec_model_dir, "", "Path of video rec inference model.");
DEFINE_int32(num_seg, 8, "number of frames input to model, which are extracted from a video.");
DEFINE_int32(seg_len, 1, "number of frames from a segment.");
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
DEFINE_string(char_list_file, "../../data/k400/Kinetics-400_label_list.txt", "Path of dictionary.");


using namespace std;
using namespace cv;
using namespace PaddleVideo;


static bool PathExists(const std::string& path)
{
#ifdef _WIN32
    struct _stat buffer;
    return (_stat(path.c_str(), &buffer) == 0);
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}

// #define DEBUG_HSS

int main_rec(std::vector<cv::String> cv_all_video_names)
{
    std::vector<double> time_info = {0, 0, 0}; //声明时间信息
    VideoRecognizer rec(FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_num_seg,
                       FLAGS_gpu_id, FLAGS_gpu_mem,
                       FLAGS_cpu_threads,
                       FLAGS_enable_mkldnn, FLAGS_char_list_file,
                       FLAGS_use_tensorrt, FLAGS_precision); // 实例化一个视频识别类

    // cv::VideoCapture capture; // 创建一个视频对象
    // cv::Mat frame; // 创建一个用于存储视频帧的对象
    for (int i = 0; i < cv_all_video_names.size(); ++i) // 对每个视频做处理
    {
        LOG(INFO) << "The predict video: " << cv_all_video_names[i]; // 处理前打印信息

        std::vector<cv::Mat> frames = Utility::SampleFramesFromVideo(cv_all_video_names[i], FLAGS_num_seg, FLAGS_seg_len);
        #ifdef DEBUG_HSS
            printf("%d\n", frames.size());
            // return 0;
        #endif
        std::vector<double> rec_times;
        rec.Run(frames, &rec_times); // 拿读取到的几帧视频帧送到识别类的run方法里预测

        time_info[0] += rec_times[0];
        time_info[1] += rec_times[1];
        time_info[2] += rec_times[2];
    }
    if (FLAGS_benchmark) {
        AutoLogger autolog("rec",
                           FLAGS_use_gpu,
                           FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn,
                           FLAGS_cpu_threads,
                           1,
                           "dynamic",
                           FLAGS_precision,
                           time_info,
                           cv_all_video_names.size());
        autolog.report();
    }

    return 0;
}


void check_params(char* mode)
{
    if (strcmp(mode, "rec") == 0)
    {
        std::cout << "[" << FLAGS_rec_model_dir << "]" << std::endl;
        std::cout << "[" << FLAGS_video_dir << "]" << std::endl;
        if (FLAGS_rec_model_dir.empty() || FLAGS_video_dir.empty())
        {
            std::cout << "Usage[rec]: ./ppvideo --rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                      << "--video_dir=/PATH/TO/INPUT/VIDEO/" << std::endl;
            exit(1);
        }
    }
    if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" && FLAGS_precision != "int8")
    {
        cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. " << endl;
        exit(1);
    }
}


int main(int argc, char **argv)
{
    if (argc <= 1 || (strcmp(argv[1], "rec") != 0)) // 获取用户输入并检查
    {
        std::cout << "Please choose one mode of [rec] !" << std::endl;
        return -1;
    }
    std::cout << "mode: " << argv[1] << endl; // 输出需要的推理任务类型

    // Parsing command-line
    google::ParseCommandLineFlags(&argc, &argv, true);
    check_params(argv[1]);

    if (!PathExists(FLAGS_video_dir)) // 判断视频所在目录是否存在
    {
        std::cerr << "[ERROR] video path not exist! video_dir: " << FLAGS_video_dir << endl;
        exit(1);
    }

    std::vector<cv::String> cv_all_video_names; // 储存所有视频的名字
    cv::glob(FLAGS_video_dir, cv_all_video_names); // 在FLAGS_video_dir下搜索所有视频的名字，然后储存到cv_all_video_names中
    std::cout << "total videos num: " << cv_all_video_names.size() << endl; // 输出搜索到的视频个数

    if (strcmp(argv[1], "rec") == 0)
    {
        return main_rec(cv_all_video_names); // 执行真正的主程序
    }
    return 0;
}
