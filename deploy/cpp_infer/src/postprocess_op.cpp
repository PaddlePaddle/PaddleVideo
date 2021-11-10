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

#include <include/postprocess_op.h>
#include <include/clipper.cpp>

namespace PaddleVideo
{
    void Softmax::Inplace_Run(std::vector<float> &arr)
    {
        const float max_value = *std::max_element(arr.begin(), arr.end());
        float denominator = 0.0f;
        for (size_t i = 0, sz = arr.size(); i < sz; ++i)
        {
            arr[i] = std::exp(arr[i] - max_value);
            denominator += arr[i];
        }
        for (size_t i = 0, sz = arr.size(); i < sz; ++i)
        {
            arr[i] /= denominator;
        }
    }
    std::vector<float> Softmax::Run(const std::vector<float> &arr)
    {
        std::vector<float> prob(arr.begin(), arr.end());
        const float max_value = *std::max_element(arr.begin(), arr.end());
        float denominator = 0.0f;
        for (size_t i = 0, sz = arr.size(); i < sz; ++i)
        {
            prob[i] = std::exp(arr[i] - max_value);
            denominator += prob[i];
        }
        for (size_t i = 0, sz = arr.size(); i < sz; ++i)
        {
            prob[i] /= denominator;
        }
        return prob;
    }

} // namespace PaddleVideo
