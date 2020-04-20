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
#include "rknpu/types.h"

namespace rk {
namespace nn {

/** the role of Tensor
 */
enum class TensorRole {
    VAR = 0,                    ///< middle tensor
    CONST,                      ///< const tensor
    DATA,                       ///< input & output tensor
};

/** the precision of Tensor
 */
enum class PrecisionType : int {
    UNKNOWN = 0,                ///< unknown precision
    INT8 = 1,
    INT16,
    INT32,
    INT64,
    UINT8 = 5,
    UINT16,
    UINT32,
    UINT64,
    FLOAT16 = 9,
    FLOAT32,
    FLOAT64,
    BOOL8 = 12,
    NUM = 13,                   ///< number of fields.
};

/** the data layout of Tensor
 */
enum class DataLayoutType : int {
    UNKNOWN = 0,                ///< unknown layout
    NCHW = 1,
    NHWC = 2,
    ANY = 3,                    ///< any data layout
    NUM = 4,                    ///< number of fields.
};

/** the quantization type of Tensor
 */
enum class QuantizationType : int {
    NONE = 0,                       ///< none quantized
    DFP,                            ///< dynamic fixed point
    AFFINE_ASYMMETRIC,              ///< affine asymmetric
    AFFINE_PERCHANNEL_SYMMETRIC,    ///< affine perchannel symmetric
    SYMMETRIC,                      ///< symmetric
    NA = 0xff,
};

/** the structure of quantization parameter of Dynamic Fixed Point
 */
struct QuantizationParamDFP {
    std::vector<int8_t>  fl;                ///< fractional length
};

/** the structure of quantization parameter of Affine Asymmetric
 */
struct QuantizationParamAffineAsymmetric {
    std::vector<uint32_t>  zero_point;      ///< zero point
    std::vector<float>     scale;           ///< scale
};

/** the structure of quantization parameter of Symmetric
 */
struct QuantizationParamSymmetric {
    std::vector<float>     scale;           ///< scale 
};

/** the structure of Tensor Attribute
 */
struct TensorAttr {
    std::string name;                                               ///< name of tensor
    std::vector<uint32_t> dims;                                     ///< shape of tensor
    PrecisionType precision;                                        ///< precision of tensor
    DataLayoutType layout;                                          ///< data layout of tensor
    TensorRole role;                                                ///< role of tensor

    QuantizationType qntType;                                       ///< quantization type of tensor
    uint8_t qntBits;                                                ///< quantization bits of tensor
    QuantizationParamDFP qntParamDFP;                               ///< Meanful in dynamic fixed point
    QuantizationParamAffineAsymmetric qntParamAffineAsymmetric;     ///< Meanful in affine asymmetric
    QuantizationParamSymmetric qntParamSymmetric;                   ///< Meanful in symmetric
};

/** Tensor is used to get and set imformation.
 *  and the Tensor is returned by Graph::CreateTensor, don't create it directly.
*/
class Tensor
{
public:
    // The default constructor, do not call it directly.
    Tensor(/* args */) = default;
    ~Tensor() = default;

    /** Get the dimensions of Tensor.
     * 
     *  @return the dimensions of Tensor 
     */ 
    virtual std::vector<uint32_t> GetDims() = 0;

    /** Get the precision of Tensor.
     * 
     *  @return the precision of Tensor 
     */ 
    virtual PrecisionType GetPrecision() = 0;

    /** Get the name of Tensor.
     * 
     *  @return the name of Tensor 
     */ 
    virtual std::string& GetName() = 0;

    /** Get the role of Tensor.
     * 
     *  @return the role of Tensor 
     */ 
    virtual TensorRole GetRole() = 0;

    /** Get the attributes of Tensor.
     * 
     *  @return the attributes of Tensor 
     */ 
    virtual const std::shared_ptr<const TensorAttr> GetAttrs()  = 0;

    /** Get the data of Tensor.
     * 
     *  @return the pointer of data
     */  
    virtual const void* GetData() = 0;

    /** Set the precision of Tensor.
     * 
     *  @param precision [in] tensor precision type
     *  @return error code
     */
    virtual int SetPrecision(PrecisionType precision) = 0;

    /** Set the name of Tensor.
     * 
     *  @param name [in] tensor name
     *  @return error code
     */
    virtual int SetName(std::string &name) = 0;

    /** Set the role of Tensor.
     * 
     *  @param role [in] tensor role
     *  @return error code
     */  
    virtual int SetRole(TensorRole role) = 0;

    /** Set the quantization type and the DFP parameter of Tensor.
     * 
     *  @param type [in] quantization type
     *  @param bits [in] quantization bits
     *  @param dfp [in] quantization parameter of DFP
     *  @return error code
     */  
    virtual int SetQntParam(QuantizationType type, uint8_t bits, QuantizationParamDFP &dfp) = 0;

    /** Set the quantization type and the affine asymmetric parameter of Tensor.
     * 
     *  @param type [in] quantization type
     *  @param bits [in] quantization bits
     *  @param affine_asymmetric [in] quantization parameter of affine asymmetric
     *  @return error code
     */  
    virtual int SetQntParam(QuantizationType type, uint8_t bits, QuantizationParamAffineAsymmetric &affine_asymmetric) = 0;

    /** Set the quantization type and the symmetric parameter of Tensor.
     * 
     *  @param type [in] quantization type
     *  @param bits [in] quantization bits
     *  @param symmetric [in] quantization parameter of symmetric
     *  @return error code
     */  
    virtual int SetQntParam(QuantizationType type, uint8_t bits, QuantizationParamSymmetric &symmetric) = 0;
};

}
}
