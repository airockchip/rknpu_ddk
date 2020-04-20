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

#include <vector>
#include <stdint.h>

namespace rk {
namespace nn {

/// error code
enum  {
    RK_SUCCESS = 0,     ///< execute succeed
    RK_FAILURE = -1,    ///< execute failed
    RK_INVALID_INPUTS = -2, ///< Invalid inputs, such as input number mismatch with model
    RK_INVALID_OUTPUTS = -3, ///< Invalid outputs, such as output number mismatch with model
    RK_INVALID_MODEL = -4,   ///< Invalid model, Exection::Build() fail
    RK_INVALID_PARAM = -5,   ///< Invalid parameter
    RK_NO_MEMORY = -6,  ///< memory malloc fail
    RK_DEVICE_UNAVAILABLE = -7, ///< device is unavailable.
    RK_INVALID_TENSOR = -8, ///< Invalid tensor
    RK_INVALID_OP = -9, ///< Operater does not support or is not implemented 
};

/** Pad type enum */
enum class PadType : int
{
    AUTO = 0, ///< decide by driver
    /**
     * VALID Padding: it means no padding and it assumes that all the dimensions are valid
     * so that the input image gets fully covered by a filter and the stride specified by you.
    */
    VALID,
    /**
     * SAME Padding: it applies padding to the input image so that the input image gets 
     * fully covered by the filter and specified stride.It is called SAME because, for stride 1 , 
     * the output will be the same as the input.
    */
    SAME
} ;

/** Pool type enum */
enum class PoolType : int
{
    POOLING_MAX = 0, ///< Calculate the maximum value for each patch of the feature map.    
    POOLING_AVG,     ///< Calculate the average value for each patch on the feature map.
    POOLING_UNKNOWN
} ;

/** Round type enum 
 * <pre>
 *  How to round input to integer
 *  
 * |  x   | Floor | Ceil |
 * |  2   |   2   |   2  |
 * |  2.4 |   2   |   3  |
 * |  2.9 |   2   |   3  |
 * | −2.7 |  −3   |  −2  |
 * |  −2  |  −2   |  −2  |
 * </pre>
*/
enum class RoundType : int
{
    ROUND_CEIL = 0, ///< round input upwards to the nearest integer
    ROUND_FLOOR,    ///< round input downwards to the nearest integer
    ROUND_UNKNOWN
};

}
}