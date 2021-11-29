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

        auto preprocess_start = std::chrono::steady_clock::now();
        /* Preprocess */
        std::vector<cv::Mat> resize_frames;
        std::vector<cv::Mat> crop_frames;
        std::vector<float> input;
        int num_views;
        if (this->inference_model_name == "ppTSM")
        {
            num_views = 1;
            // 1. Scale
            resize_frames = std::vector<cv::Mat>(this->num_seg, cv::Mat());
            for (int i = 0; i < this->num_seg; ++i)
            {
                this->scale_op_.Run(srcframes[i], resize_frames[i], this->use_tensorrt_, 256);
            }

            // 2. CenterCrop
            crop_frames = std::vector<cv::Mat>(num_views * this->num_seg, cv::Mat());
            for (int j = 0; j < this->num_seg; ++j)
            {
                this->centercrop_op_.Run(resize_frames[j], crop_frames[j], this->use_tensorrt_, 224);
            }

            // 3. Normalization(inplace operation)
            for (int i = 0; i < num_views; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->normalize_op_.Run(&crop_frames[i * this->num_seg + j], this->mean_, this->scale_, this->is_scale_);
                }
            }

            // 4. Image2Array
            input = std::vector<float>(1 * num_views * this->num_seg * 3 * crop_frames[0].rows * crop_frames[0].cols, 0.0f);
            int rh = crop_frames[0].rows;
            int rw = crop_frames[0].cols;
            int rc = crop_frames[0].channels();
            for (int i=0; i<num_views; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->permute_op_.Run(&crop_frames[i * this->num_seg + j], input.data() + i * j * rh * rw * rc);
                }
            }
        }
        else if(this->inference_model_name == "ppTSN")
        {
            num_views = 10;
            // 1. Scale
            resize_frames = std::vector<cv::Mat>(this->num_seg, cv::Mat());
            for (int i = 0; i < this->num_seg; ++i)
            {
                this->scale_op_.Run(srcframes[i], resize_frames[i], this->use_tensorrt_, 256);
            }

            // 2. TenCrop
            crop_frames = std::vector<cv::Mat>(num_views * this->num_seg, cv::Mat());
            for (int i = 0; i < this->num_seg; ++i)
            {
                this->tencrop_op_.Run(resize_frames[i], crop_frames, i * num_views, this->use_tensorrt_, 224);
            }

            // 3. Normalization(inplace operation)
            for (int i = 0; i < num_views; ++i)
            {
                for (int j = 0; j < this->num_seg; ++j)
                {
                    this->normalize_op_.Run(&crop_frames[i * this->num_seg + j], this->mean_, this->scale_, this->is_scale_);
                }
            }

            // 4. Image2Array
            input = std::vector<float>(1 * num_views * this->num_seg * 3 * crop_frames[0].rows * crop_frames[0].cols, 0.0f);
            int rh = crop_frames[0].rows;
            int rw = crop_frames[0].cols;
            int rc = crop_frames[0].channels();
            for (int i = 0; i < this->num_seg; ++i)
            {
                for (int j = 0; j < num_views; ++j)
                {
                    this->permute_op_.Run(&crop_frames[i * num_views + j], input.data() + (i * num_views + j) * rh * rw * rc);
                }
            }
        }
        else
        {
            throw "[Error] Not implemented yet";
        }
        auto preprocess_end = std::chrono::steady_clock::now();

        /* Inference */
        auto input_names = this->predictor_->GetInputNames();
        auto input_t = this->predictor_->GetInputHandle(input_names[0]);
        input_t->Reshape({1 * num_views * this->num_seg, 3, crop_frames[0].rows, crop_frames[0].cols});
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
