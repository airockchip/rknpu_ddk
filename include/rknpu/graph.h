// Copyright (c) 2020 by Fuzhou Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "tensor.h"
#include "operator.h"

namespace rk {
namespace nn {

/** Graph is used to create and save tensor and Operator, and the connection relationship of these Operators.
 * It is mainly used to save model information and is not actually created on the NPU.
*/
class Graph
{
    friend class Exection;
private:
    void *pri_data;

public:
    Graph();
    ~Graph();

    /** Add an Operator.
     * 
     *  @param type [in] Operator Type
     *  @param inputs [in] input tensors
     *  @param outputs [in] outputs tensors
     *  @param attrs [in] attributes of Operator
     *  @param name [in] Operator's name
     *  @return The corresponding Operator is returned on success, otherwise it returns nullptr
    */
    std::shared_ptr<Operator> AddOperator(OperatorType type, std::vector<std::shared_ptr<Tensor>>inputs, std::vector<std::shared_ptr<Tensor>>outputs, void *attrs, std::string name = "");
    /** Create a tensor.
     * 
     *  @param attr [in]  attributes of tensor, cannot be empty.
     *  @param data [in]  tensor data, can be empty.
     *  @return the pointer of tensor
    */
    std::shared_ptr<Tensor> CreateTensor(std::shared_ptr<TensorAttr> attr, void *data);

    /**Set the input and output tensor of the graph.
     * 
     *  @param input_tensors [in] input tensors
     *  @param output_tensors [in] output tensors
     *  @return RK_SUCCESS when success
    */
    int SetInputsOutputs(std::vector<std::shared_ptr<Tensor>> input_tensors, std::vector<std::shared_ptr<Tensor>> output_tensors);

    /** Get the input tensor of graph
     * 
     *  @return input tensors set by SetInputsOutputs()
    */
    std::vector<std::shared_ptr<Tensor>> GetInputs();

    /** Get the output tensor of graph
     * 
     *  @return output tensors set by SetInputsOutputs()
    */
    std::vector<std::shared_ptr<Tensor>> GetOutputs();
};

}
}
