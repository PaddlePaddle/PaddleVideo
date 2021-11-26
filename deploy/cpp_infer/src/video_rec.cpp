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

#include <include/video_rec.h>
// #define DEBUG_HSS

namespace PaddleVideo
{

    void VideoRecognizer::Run(std::vector<cv::Mat> &frames, std::vector<double> *times)
    {
        // Copy parameters to the function
        std::vector<cv::Mat> srcframes(this->num_seg, cv::Mat());
        for (int i = 0; i < this->num_seg; ++i)
        {
            frames[i].copyTo(srcframes[i]);
        }
        /* Consistent strategy with *.yaml
        ========================================
        - Scale:
            short_size: 256
        - CenterCrop:
            target_size: 224
        - Image2Array:
        - Normalization:
            mean: [0.485, 0.456, 0.406] (r,g,b)
            std: [0.229, 0.224, 0.225] (r,g,b)
        ========================================
        */
        auto preprocess_start = std::chrono::steady_clock::now();

        /* Preprocess */
        // 1. Scale
        std::vector<cv::Mat> resize_frames(this->num_seg, cv::Mat());
        for (int i = 0; i < this->num_seg; ++i)
        {
            this->scale_op_.Run(srcframes[i], resize_frames[i], this->use_tensorrt_, 256);
        }

        // 2. CenterCrop
        std::vector<cv::Mat> crop_frames(this->num_seg, cv::Mat());
        for (int i = 0; i < this->num_seg; ++i)
        {
            this->centercrop_op_.Run(resize_frames[i], crop_frames[i], this->use_tensorrt_, 224);
        }

        // 3. Normalization
        for (int i = 0; i < this->num_seg; ++i)
        {
            this->normalize_op_.Run(&crop_frames[i], this->mean_, this->scale_, this->is_scale_);
        }

        // 4. Image2Array
        // Declare a tensor to store video frames
        std::vector<float> input(1 * this->num_seg * 3 * crop_frames[0].rows * crop_frames[0].cols, 0.0f);

        int rh = crop_frames[0].rows;
        int rw = crop_frames[0].cols;
        int rc = crop_frames[0].channels();
        for (int i = 0; i < this->num_seg; ++i)
        {
            this->permute_op_.Run(&crop_frames[i], input.data() + i * rh * rw * rc);
        }

        auto preprocess_end = std::chrono::steady_clock::now();

        /* Inference */
        auto input_names = this->predictor_->GetInputNames();
        auto input_t = this->predictor_->GetInputHandle(input_names[0]);
        input_t->Reshape({1 * this->num_seg, 3, crop_frames[0].rows, crop_frames[0].cols});

        auto inference_start = std::chrono::steady_clock::now();
        input_t->CopyFromCpu(input.data());
        this->predictor_->Run(); // Use the inference library to predict

        std::vector<float> predict_batch;
        auto output_names = this->predictor_->GetOutputNames();
        auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
        auto predict_shape = output_t->shape();

        int out_numel = std::accumulate(predict_shape.begin(), predict_shape.end(), 1, std::multiplies<int>());
        predict_batch.resize(out_numel);
        output_t->CopyToCpu(predict_batch.data()); // Copy the model output to predict_batch

        // Convert output (logits) into probabilities
        this->softmax_op_.Inplace_Run(predict_batch);

        auto inference_end = std::chrono::steady_clock::now();

        // output decode
        auto postprocess_start = std::chrono::steady_clock::now();
        std::vector<std::string> str_res;
        int argmax_idx = 0;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;

        argmax_idx = int(Utility::argmax(predict_batch.begin(), predict_batch.end()));
        score += predict_batch[argmax_idx];
        count += 1;
        str_res.push_back(this->label_list_[argmax_idx]);

        auto postprocess_end = std::chrono::steady_clock::now();
        score /= count;
        for (int i = 0; i < str_res.size(); i++)
        {
            std::cout << str_res[i];
        }
        std::cout << "\tscore: " << score << std::endl;

        std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
        times->push_back(double(preprocess_diff.count() * 1000));
        std::chrono::duration<float> inference_diff = inference_end - inference_start;
        times->push_back(double(inference_diff.count() * 1000));
        std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
        times->push_back(double(postprocess_diff.count() * 1000));
    }

    void VideoRecognizer::LoadModel(const std::string &model_dir)
    {
        //   AnalysisConfig config;
        paddle_infer::Config config;
        config.SetModel(model_dir + "/" + this->inference_model_name + ".pdmodel",
                        model_dir + "/" + this->inference_model_name + ".pdiparams");

        if (this->use_gpu_)
        {
            config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
            if (this->use_tensorrt_)
            {
                auto precision = paddle_infer::Config::Precision::kFloat32;
                if (this->precision_ == "fp16")
                {
                    precision = paddle_infer::Config::Precision::kHalf;
                }
                if (this->precision_ == "int8")
                {
                    precision = paddle_infer::Config::Precision::kInt8;
                }
                config.EnableTensorRtEngine(
                    1 << 20, 10, 3,
                    precision,
                    false, false);
//                 std::map<std::string, std::vector<int>> min_input_shape =
//                 {
//                     {"x", {1, 1, 3, 224, 224}}
//                 };
//                 std::map<std::string, std::vector<int>> max_input_shape =
//                 {
//                     {"x", {4, 1 * this->num_seg, 3, 224, 224}}
//                 };
//                 std::map<std::string, std::vector<int>> opt_input_shape =
//                 {
//                     {"x", {1, 1 * this->num_seg, 3, 224, 224}}
//                 };

//                 config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
//                                               opt_input_shape);
            }
        }
        else
        {
            config.DisableGpu();
            if (this->use_mkldnn_)
            {
                config.EnableMKLDNN();
                // cache 10 different shapes for mkldnn to avoid memory leak
                config.SetMkldnnCacheCapacity(10);
            }
            config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
        }

        config.SwitchUseFeedFetchOps(false);
        // true for multiple input
        config.SwitchSpecifyInputNames(true);

        config.SwitchIrOptim(true);

        config.EnableMemoryOptim();
        config.DisableGlogInfo();

        this->predictor_ = CreatePredictor(config);
    }

} // namespace PaddleVideo
