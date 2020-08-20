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

#include <string>
#include "tensor.h"

namespace rk {
namespace nn {

/** Supported operator types
 * <pre>
 * Operator is created by Graph::AddOperator(type, inputs, outputs, attrs, ...)
 * and each Operator has different inputs and outputs, also has different additional attrs.
 * e.g.
 *    CONV2D     inputs: [in, weight, bias]      outputs: [out]      attrs: Conv2DAttr
 *    It means that CONV2D needs to set 3 inputs and 1 outputs, and fill the Conv2DAttr structure. 
 *    simple code as follow:
 *        std::vector<std::shared_ptr<rk::nn::Tensor>> inputs, outputs;
 *        inputs.push_back(in);
 *        inputs.push_back(weight);
 *        inputs.push_back(bias);
 *        outputs.push_back(out);
 *        rk::nn::Conv2DAttr attr;
 *        ...   // fill attr
 *        graph->AddOperator(rk::nn::OperatorType::CONV2D, inputs, outputs, (void*)&attr);
 * </pre>
 */
enum OperatorType {
    ADD  = 1,               ///< inputs: [in1, in2]                      outputs: [out]      attrs: nullptr
    MULTIPLY,               ///< inputs: [in1, in2]                      outputs: [out]      attrs: nullptr
    CONV2D,                 ///< inputs: [in, weight, bias]              outputs: [out]      attrs: Conv2DAttr
    FULLCONNECT,            ///< inputs: [in, weight, bias]              outputs: [out]      attrs: FCAttr
    SOFTMAX,                ///< inputs: [in]                            outputs: [out]      attrs: SoftmaxAttr
    POOL,                   ///< inputs: [in]                            outputs: [out]      attrs: PoolAttr
    LEAKY_RELU,             ///< inputs: [in]                            outputs: [out]      attrs: LeakyReluAttr
    LRN,                    ///< unimplement
    CONCAT,                 ///< inputs: [in1, in2, ...]                 outputs: [out]      attrs: ConcatAttr
    SPLIT,                  ///< unimplement
    ROI_POOL,               ///< unimplement
    BATCH_NORM,             ///< inputs: [in, mean, var, scale, bias]    outputs: [out]      attrs: BatchNormAttr
    PROPOSAL,               ///< unimplement
    DECONVOLUTION,          ///< unimplement
    RESHAPE,                ///< inputs: [in]                            outputs: [out]      attrs: ReshapeAttr
    PERMUTE,                ///< inputs: [in]                            outputs: [out]      attrs: PermuteAttr
    PRELU,                  ///< inputs: [in]                            outputs: [out]      attrs: PReluAttr
    UPSAMPLE,               ///< unimplement
    RELU,                   ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    RELUN,                  ///< unimplement
    LSTM,                   ///< unimplement
    REORG,                  ///< unimplement
    L2_NORMALIZE,           ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    POOLWITHARGMAX,         ///< unimplement
    ARGMAX,                 ///< unimplement
    MAXIMUM,                ///< inputs: [in1, in2]                      outputs: [out]      attrs: nullptr
    L2NORMALIZESCALE,       ///< unimplement
    CROP,                   ///< unimplement
    SUBTRACT,               ///< inputs: [in1, in2]                      outputs: [out]      attrs: nullptr
    RELU6,                  ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    SIGMOID,                ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    TANH,                   ///< unimplement
    SQRT,                   ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    RSQRT,                  ///< unimplement
    DIVIDE,                 ///< unimplement
    DROPOUT,                ///< unimplement
    SHUFFLECHANNEL,         ///< unimplement
    RESIZE,                 ///< unimplement
    REVERSE,                ///< unimplement
    DEPTH2SPACE,            ///< unimplement
    SPACE2DEPTH,            ///< unimplement
    DATACONVERT,            ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    SCALE,                  ///< unimplement
    SLICE,                  ///< inputs: [in]                            outputs: [out]      attrs: SliceAttr
    ELU,                    ///< unimplement
    BATCH2SPACE,            ///< unimplement
    SPACE2BATCH,            ///< unimplement
    PAD,                    ///< inputs: [in]                            outputs: [out]      attrs: PadAttr
    MATRIXMUL,              ///< unimplement
    LSTMUNIT,               ///< unimplement
    LAYER_NORM,             ///< inputs: [in, mean, var]                 outputs: [out]      attrs: LayerNormAttr
    REDUCE,                 ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    INSTANCE_NORM,          ///< unimplement
    TENSORSTACKCONCAT,      ///< unimplement
    STRIDED_SLICE,          ///< inputs: [in]                            outputs: [out]      attrs: StridedSliceAttr
    SIGNAL_FRAME,           ///< unimplement
    A_TIMES_B_PLUS_C,       ///< unimplement
    SVDF,                   ///< unimplement
    ABS,                    ///< unimplement
    CONV1D,                 ///< unimplement
    LRN2,                   ///< unimplement
    POW,                    ///< unimplement
    FLOORDIV,               ///< unimplement
    MINIMUM,                ///< unimplement
    RELU1,                  ///< unimplement
    STACK,                  ///< unimplement
    FLOOR,                  ///< unimplement
    SQUARE,                 ///< unimplement
    NEG,                    ///< inputs: [in]                            outputs: [out]      attrs: nullptr
    EXP,                    ///< unimplement
    HASHTABLE_LOOKUP,       ///< unimplement
    EMBEDDING_LOOKUP,       ///< unimplement
    LSH_PROJECTION,         ///< unimplement
    RNN,                    ///< unimplement
    CLIP,                   ///< inputs: [in]                            outputs: [out]      attrs: ClipAttr
    UNSTACK,                ///< unimplement
    ADDN,                   ///< unimplement
    GATHER,                 ///< inputs: [in]                            outputs: [out]      attrs: GatherAttr
    TOPK,                   ///< unimplement
    SIN,                    ///< unimplement
    LOG,                    ///< unimplement
    ARGMIN,                 ///< unimplement
    ROI_ALIGN,              ///< unimplement   
    LOG_SOFTMAX,            ///< unimplement
};

/** attrbutes of BatchNormalization
 */
struct BatchNormAttr {
    float eps;                      ///< epsilon, use to avoid division by zero.
};

/** attrbutes of Clip
 */
struct ClipAttr {
    float min;                      ///< minimum value
    float max;                      ///< maximum value
};

/** attrbutes of Concat
 */
struct ConcatAttr {
    int axis;                       ///< which axis to concat on
};

/** attrbutes of Conv2D
 */
struct Conv2DAttr {
    uint32_t     ksize[2];          ///< the shape of the convolution kernel
    uint32_t     stride[2];         ///< stride along each spatial axis
    uint32_t     pad[4];            ///< pad left, right, top, bottom
    PadType      pad_type;          ///< pad type, default value shall be AUTO
    uint32_t     weights;           ///< number of weight batch
    uint32_t     group;             ///< number of groups input channels and output channels are divided into.
    uint32_t     dilation[2];       ///< dilation value along each spatial axis of the filter
    int32_t      multiplier;
    bool         has_relu;          ///< fuse relu
};

/** attrbutes of FullConnection
 */
struct FCAttr {
    uint32_t    weights;            ///< number of weight channel
    bool        has_relu;           ///< fuse relu
};

/** attrbutes of Gather
 */
struct GatherAttr {
    int32_t axis;                   ///< which axis to gather on
};

/** attrbutes of LeakyRelu
 */
struct LeakyReluAttr {
    float alpha;                    ///< coefficient of leakage.
};

/** attrbutes of Permute
 */
struct PermuteAttr {
    std::vector<uint32_t> perm;     ///< permute the axes according to the values given
};

/** attrbutes of Pool
 */
struct PoolAttr {
    uint32_t    ksize[2];           ///< the size of the kernel along each axis
    uint32_t    stride[2];          ///< stride along each spatial axis
    uint32_t    pad[4];             ///< pad left, right, top, bottom
    PadType     pad_type;           ///< pad type, default value shall be AUTO
    PoolType    pool_type;          ///< pool type
    RoundType   round_type;         ///< whether to use ceil or floor to compute the output shape
    bool global_pooling;
};

/** attrbutes of Reshape
 */
struct ReshapeAttr {
    std::vector<uint32_t> shapes;   ///< specified shape for output
};

/** attrbutes of Slice
 */
struct SliceAttr {
    std::vector<uint32_t> start;    ///< starting indices of corresponding axis
    std::vector<uint32_t> length;   ///< the slice length of corresponding axis
};

/** attrbutes of Softmax
 */
struct SoftmaxAttr {
    float beta;                     ///< A FLOAT32 value, specifying the positive scaling factor for the exponent, beta.
    uint32_t axis;                  ///< describes the axis of the inputs
};

/** attrbutes of Reduce
 */
struct ReduceAttr {
    ReduceType type;                ///< reduce type
    std::vector<uint32_t> axis;     ///< axis list
    uint32_t axis_num;              ///< the number of axis list
    bool keep_dim;                  ///< keep dim flag
};

/** attrbutes of StridedSlice
 */
struct StridedSliceAttr {
    std::vector<int32_t> begin;     ///< begin indices of corresponding axis
    std::vector<int32_t> end;       ///< end indices of corresponding axis
    std::vector<int32_t> stride;    ///< stride indices of corresponding axis
};

/** attrbutes of Pad
 */
struct PadAttr {
    PadMode mode;                   ///< pad mode
    std::vector<uint32_t> begin;    ///< begin pading number of corresponding axis
    std::vector<uint32_t> end;      ///< end pading number of corresponding axis
    int32_t const_val;              ///< constant value
};

/** attrbutes of LayerNormalization
 */
struct LayerNormAttr {
    float eps;                      ///< epsilon, use to avoid division by zero.
};

/** Operator is used to get the basic imformation.
 *  and the Operator is returned by Graph::AddOperator, don't create it directly.
*/
class Operator
{
public:
    // The default constructor, do not call it directly.
    Operator(/* args */) = default;
    ~Operator() = default;

    /** Get the input tensor of Operator.
     * 
     *  @return the input tensor of Operator
     */  
    virtual std::vector<std::shared_ptr<Tensor>> GetInputs() = 0;

    /** Get the output tensor of Operator.
     * 
     *  @return the output tensor of Operator
     */  
    virtual std::vector<std::shared_ptr<Tensor>> GetOutputs() = 0;

    /** Get the attribute of Operator.
     * 
     *  @return the pointer of attribute
     */  
    virtual void* GetAttrs() = 0;

    /** Get the name of Operator.
     * 
     *  @return the string of name
     */  
    virtual std::string GetName() = 0;
};

}    
}
