// Copyright (c) 2020 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include <iostream>
#include <vector>
#include <string.h>
#include <map>
#include <fstream>

#include "rknpu/rknpu_pub.h"

using namespace rk::nn;
using namespace std;

#define CHECK_EQ(x, y) do { \
    if ((x) != (y))  { \
    }   \
} while(0)

std::map<PrecisionType,std::string> precesion_map =
{
    {PrecisionType::UNKNOWN,"UNKNOWN"},
    {PrecisionType::INT8,"INT8"},
    {PrecisionType::UINT8,"UINT8"},
    {PrecisionType::INT16,"INT16"},
    {PrecisionType::UINT16,"UINT16"},
    {PrecisionType::INT32,"INT32"},
    {PrecisionType::UINT32,"UINT32"},
    {PrecisionType::FLOAT16,"FLOAT16"},
    {PrecisionType::FLOAT32,"FLOAT32"},
};
std::map<DataLayoutType,std::string> layout_map =
{
    {DataLayoutType::UNKNOWN,"UNKNOWN"},
    {DataLayoutType::NCHW,"NCHW"},
    {DataLayoutType::NHWC,"NHWC"},
};

std::shared_ptr<Operator> createConv(Graph* graph, std::shared_ptr<Tensor> input_tensor, 
        std::shared_ptr<Tensor> output_tensor,
        std::vector<int> filter_dims, 
        std::vector<int> strides,
        std::vector<int> paddings,
        std::vector<int> dilations,
        int groups = 0,
        bool fuse_relu = false)
{
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::vector<std::shared_ptr<Tensor>> outputs;
    auto input_dims = input_tensor->GetDims();
    auto output_dims = output_tensor->GetDims();

    CHECK_EQ(filter_dims.size(), 4);
    CHECK_EQ(input_dims.size(), 4);
    CHECK_EQ(output_dims.size(), 4);

    auto bs = input_dims[0];
    auto ic = input_dims[1];
    auto oc = filter_dims[0];

    CHECK_EQ(output_dims[0], bs);
    CHECK_EQ(output_dims[1], oc);
    CHECK_EQ(strides.size(), 2L);
    CHECK_EQ(dilations.size(), 2L);

    // create dummy weight
    auto weight_attr = std::make_shared<rk::nn::TensorAttr>();

    weight_attr->dims.push_back(filter_dims[0]);
    weight_attr->dims.push_back(filter_dims[1]);
    weight_attr->dims.push_back(filter_dims[2]);
    weight_attr->dims.push_back(filter_dims[3]);

    weight_attr->precision = PrecisionType::INT8;
    weight_attr->layout = DataLayoutType::NCHW;
    weight_attr->role = TensorRole::CONST;
    weight_attr->qntType = QuantizationType::SYMMETRIC;
    weight_attr->qntBits = 8;
    weight_attr->qntParamSymmetric.scale.push_back(1.0);

    int weight_data_size = filter_dims[0] * filter_dims[1] * filter_dims[2] * filter_dims[3];
    int8_t *weight_data = new int8_t[weight_data_size];

    memset(weight_data, 0x01, weight_data_size);
    auto weight =  graph->CreateTensor(weight_attr, (void *)weight_data);

    // create dummy bias
    auto bias_attr = std::make_shared<rk::nn::TensorAttr>();
    bias_attr->dims.push_back(oc);
    bias_attr->precision = PrecisionType::INT32;
    bias_attr->layout = DataLayoutType::NCHW;
    bias_attr->role = TensorRole::CONST;
    bias_attr->qntType = QuantizationType::SYMMETRIC;
    bias_attr->qntBits = 32;
    bias_attr->qntParamSymmetric.scale.push_back(1.0);

    int bias_data_size = oc;
    int32_t* bias_data = new int32_t[bias_data_size];

    memset(bias_data, 0x00, bias_data_size);
    auto bias =  graph->CreateTensor(bias_attr, (void *)bias_data);

    inputs.push_back(input_tensor);
    inputs.push_back(weight);
    inputs.push_back(bias);
    outputs.push_back(output_tensor);

    rk::nn::Conv2DAttr attr;
    attr.ksize[0] = filter_dims[2];
    attr.ksize[1] = filter_dims[3];
    attr.stride[0] = strides[0];
    attr.stride[1] = strides[1];
    attr.pad[0] = paddings[0];
    attr.pad[1] = paddings[1];
    attr.pad[2] = paddings[2];
    attr.pad[3] = paddings[3];
    attr.group = groups;
    attr.weights = oc;
    attr.dilation[0] = dilations[0];
    attr.dilation[1] = dilations[1];
    attr.pad_type = rk::nn::PadType::AUTO;
    attr.has_relu = fuse_relu;
    attr.multiplier = 0;

    auto conv = graph->AddOperator(OperatorType::CONV2D, inputs, outputs, &attr);

    return conv;
}

int CreateGraph(Graph* graph,std::string cache_path,bool save_cache = false) {

    if(save_cache) {
        auto attr_input = std::make_shared<rk::nn::TensorAttr>();
        attr_input->dims.push_back(1);
        attr_input->dims.push_back(1);
        attr_input->dims.push_back(28);
        attr_input->dims.push_back(28);

        attr_input->precision = PrecisionType::INT8;
        attr_input->layout = DataLayoutType::NCHW;
        attr_input->role = TensorRole::DATA;
        attr_input->qntType = QuantizationType::SYMMETRIC;
        attr_input->qntBits = 8;
        attr_input->qntParamSymmetric.scale.push_back(1.0);

        auto input_tensor =  graph->CreateTensor(attr_input, nullptr);

        auto attr_output = std::make_shared<rk::nn::TensorAttr>();
        attr_output->dims.push_back(1);
        attr_output->dims.push_back(1);
        attr_output->dims.push_back(28);
        attr_output->dims.push_back(28);

        attr_output->precision = PrecisionType::INT8;
        attr_output->layout = DataLayoutType::NCHW;
        attr_output->role = TensorRole::DATA;
        attr_output->qntType = QuantizationType::SYMMETRIC;
        attr_output->qntBits = 8;
        attr_output->qntParamSymmetric.scale.push_back(1.0);
        auto output_tensor =  graph->CreateTensor(attr_output, nullptr);

        // conv1 attr
        std::vector<int> filter_dims;
        std::vector<int> strides;
        std::vector<int> paddings;
        std::vector<int> dilations;

        filter_dims.resize(4);
        filter_dims[0] = 3;  // output channel
        filter_dims[1] = 1;  // input channel
        filter_dims[2] = 1; // kernel h
        filter_dims[3] = 1; // kernel w
        strides.resize(2);
        strides[0] = 1;
        strides[1] = 1;
        paddings.resize(4);
        paddings[0] = 0;
        paddings[1] = 0;
        paddings[2] = 0;
        paddings[3] = 0;
        dilations.resize(2);
        dilations[0] = 1;
        dilations[1] = 1;

        std::vector<std::shared_ptr<Tensor>> inputs;
        std::vector<std::shared_ptr<Tensor>> outputs;

        inputs.push_back(input_tensor);
        outputs.push_back(output_tensor);

        graph->EnableCreateCache(cache_path);
        auto conv1 = createConv(graph, input_tensor, output_tensor, filter_dims, strides, paddings, dilations);
        graph->SetInputsOutputs(inputs, outputs);
    } 
    else  
    {
        auto status = graph->LoadCache(cache_path);
        auto input_attrs = graph->GetInputTensorsAttr();
        auto output_attrs = graph->GetOutputTensorsAttr();
        int count=0;
        for(auto attr:input_attrs){
            printf("inputs[%d]: shape={%d,%d,%d,%d},type=%s,fmt=%s\n",count,attr->dims[0],attr->dims[1],attr->dims[2],attr->dims[3],
                precesion_map[attr->precision].c_str(),layout_map[attr->layout].c_str());
            count++;
        }
        count=0;
        for(auto attr:output_attrs){
            printf("outputs[%d]: shape={%d,%d,%d,%d},type=%s,fmt=%s\n",count,attr->dims[0],attr->dims[1],attr->dims[2],attr->dims[3],
                precesion_map[attr->precision].c_str(),layout_map[attr->layout].c_str());
            count++;
        }

        std::vector<std::shared_ptr<rk::nn::Tensor>> exam_inputs;
        std::vector<std::shared_ptr<rk::nn::Tensor>> exam_outputs;
        for(int i = 0;i<input_attrs.size();++i)
        {
            auto in_tensor =  graph->CreateTensor(input_attrs[i], nullptr);
            exam_inputs.push_back(in_tensor);
        }
        for(int i = 0;i<output_attrs.size();++i)
        {
            auto out_tensor =  graph->CreateTensor(output_attrs[i], nullptr);
            exam_outputs.push_back(out_tensor);
        }
        graph->SetInputsOutputs(exam_inputs,exam_outputs);
    }    
    return 0;
}

/**
 * Inference example for save/load by cache
 */ 
int main(int argc, char *argv[]) {
    int status = 0;
    std::vector<InputInfo> inputs;
    std::vector<OutputInfo> outputs;
    std::string cache_path = "";
    if (argc != 3) {
        std::cout << "Usage:\n" << argv[0] <<" <cache_path> <cache_flag>"<<std::endl;
        std::cout <<"\t<cache_flag>:\t\t\"0\":save, \"1\":load\n";
        return -1;
    }
    cache_path = argv[1];
    bool isSave = atoi(argv[2])==0 ? true:false;

    Graph* graph = new Graph();
    CreateGraph(graph,cache_path, isSave);

    Exection* exector = new Exection(graph);

    status = exector->Build();

    if (status != RK_SUCCESS) {
        printf("exector->Build() fail, ret=%d\n", status);
        exit(1);
    }

    InputInfo inputInfo;
    OutputInfo outputInfo;

    inputInfo.index = 0;
    inputInfo.size = 1*1*28*28;
    inputInfo.buf = new int8_t[inputInfo.size];
    inputInfo.pass_through = 0;
    inputInfo.type = PrecisionType::INT8;
    inputInfo.layout = DataLayoutType::NCHW;

    memset(inputInfo.buf, 0x00, inputInfo.size);

    inputs.push_back(inputInfo);

    int8_t * input_data = (int8_t *)inputInfo.buf;
    for (int i=0; i<255; i++) {
        input_data[i] = i;
    }

    outputInfo.index = 0;
    outputInfo.size = 1*1*28*28;
    outputInfo.buf = new int8_t[outputInfo.size];
    outputInfo.want_float = false;
    outputInfo.type = PrecisionType::INT8;
    outputInfo.layout = DataLayoutType::NCHW;

    outputs.push_back(outputInfo);

    status = exector->SetInputs(inputs);
    if (status != RK_SUCCESS) {
        printf("exector->SetInputs() fail, ret=%d\n", status);
        exit(1);        
    }

    status = exector->Run();
    if (status != RK_SUCCESS) {
        printf("exector->Run() fail, ret=%d\n", status);
        exit(1);        
    }

    status = exector->GetOutputs(outputs);

    if (status != RK_SUCCESS) {
        printf("exector->GetOutputs() fail, ret=%d\n", status);
        exit(1);        
    }

    int8_t * output_data = (int8_t *)outputInfo.buf;
    for (int i=0; i<255; i++) {
        printf("input_data[%d]=%d, output[%d]=%d\n", i, input_data[i], i, output_data[i]);
    }

    return 0;
}
