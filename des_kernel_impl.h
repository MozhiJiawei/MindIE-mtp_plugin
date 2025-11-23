/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DES_KERNEL_IMPL_H
#define DES_KERNEL_IMPL_H
#include "kernel_operator.h"
#include "kernel_utils.h"

using namespace AscendC;

static constexpr uint32_t BUFFER_NUM = 1;       // split the UB to 2 equal part to enable ping-pong techniques.
static constexpr uint32_t DOUBLE_BUFFER = 1;     // split the UB to 2 equal part to enable ping-pong techniques.
static constexpr uint32_t BUF_FACTOR = 4;       // 1(g) + 1(workspace) = 2
static constexpr uint32_t OFFSET_WORKSPACE = 1; // the offset of workspace is 1
static constexpr uint32_t ALLIGMENT = 16;
constexpr uint32_t TENSOR_MAX_DIM_NUM = 2;


struct DesNormCommonTilingData
{
    uint32_t numRow;
    uint32_t collNum;
    uint32_t oneCoreRowNum;
    uint32_t numCore{1};
    uint32_t topK{8};
    float treshold;

    uint32_t xStrides[TENSOR_MAX_DIM_NUM];
};


class DesImpl
{
public:
    __aicore__ inline DesImpl() {}
    __aicore__ inline void Init(GM_ADDR globalInputWeigth, GM_ADDR globalExpertIds, GM_ADDR globalWeigthsOut, 
                                                GM_ADDR globalExpertIdsOut, GM_ADDR globalExpertsTotalNum,
                                                DesCustomKernelTilingData &tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();

    __aicore__ inline void Compute();       

private:
    AscendC::TPipe pipe;
    // Input global Tensors
    /////////////////////////////////////////////////////////////////////////////////
    AscendC::GlobalTensor<half> routingWeigthIn;
    AscendC::GlobalTensor<int32_t> expertIds;
    /////////////////////////////////////////////////////////////////////////////////

    // Output global Tensors
    /////////////////////////////////////////////////////////////////////////////////
    AscendC::GlobalTensor<half> routingWeigthOut;
    AscendC::GlobalTensor<int32_t> expertIdsOut;
    AscendC::GlobalTensor<int32_t> expertNumOut;
    /////////////////////////////////////////////////////////////////////////////////

    // Ques
    /////////////////////////////////////////////////////////////////////////////////
    AscendC::TQue<AscendC::QuePosition::VECIN, DOUBLE_BUFFER> inputQueWeigth_;
    AscendC::TQue<AscendC::QuePosition::VECIN, DOUBLE_BUFFER> inputQueExpertId_;

    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> expertOutIdsQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> weigthOutQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> expertNumOutQue_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    /////////////////////////////////////////////////////////////////////////////////
    // tiling params
    uint32_t numRow;
    uint32_t collNum;
    uint32_t oneCoreRowNum;
    uint32_t numCore;
    uint32_t topK;
    float treshold;
};


__aicore__ inline void DesImpl::Init(GM_ADDR globalInputWeigth, GM_ADDR globalExpertIds, GM_ADDR globalWeigthsOut, 
    GM_ADDR globalExpertIdsOut, GM_ADDR globalExpertsTotalNum, DesCustomKernelTilingData &tilingData)
{   
    // Copy tiling
    numRow = tilingData.numRow;
    collNum = tilingData.collNum;
    oneCoreRowNum = tilingData.oneCoreRowNum;
    numCore = tilingData.numCore;
    topK = tilingData.topK;
    treshold = tilingData.treshold;

    // Calculate each core offset
    uint32_t gmOffset = AscendC::GetBlockIdx() * oneCoreRowNum * collNum;

    // set pipes
    routingWeigthIn.SetGlobalBuffer((__gm__ half*)globalInputWeigth + gmOffset);
    expertIds.SetGlobalBuffer((__gm__ int32_t*)globalExpertIds + gmOffset);

    routingWeigthOut.SetGlobalBuffer((__gm__ half*)globalWeigthsOut + gmOffset);
    expertIdsOut.SetGlobalBuffer((__gm__ int32_t*)globalExpertIdsOut + gmOffset);
    expertNumOut.SetGlobalBuffer((__gm__ int32_t*)globalExpertsTotalNum + AscendC::GetBlockIdx() * oneCoreRowNum);

    // input ques
    pipe.InitBuffer(inputQueWeigth_, DOUBLE_BUFFER, oneCoreRowNum * collNum * sizeof(half));
    pipe.InitBuffer(inputQueExpertId_, DOUBLE_BUFFER, oneCoreRowNum * collNum * sizeof(int32_t));
    // calculating ques
    pipe.InitBuffer(calcBuf_, BUF_FACTOR * oneCoreRowNum * collNum * sizeof(half));
    // output ques
    pipe.InitBuffer(expertOutIdsQue_, BUFFER_NUM, oneCoreRowNum * collNum * sizeof(int32_t));
    pipe.InitBuffer(weigthOutQue_, BUFFER_NUM, oneCoreRowNum * collNum * sizeof(half));
    pipe.InitBuffer(expertNumOutQue_, BUFFER_NUM, oneCoreRowNum * ALLIGMENT * sizeof(int32_t));

}

__aicore__ inline void DesImpl::CopyIn()
{
    // Copy expert ids into device
    AscendC::LocalTensor<int32_t> localExperts = inputQueExpertId_.AllocTensor<int32_t>();
    AscendC::DataCopy(localExperts, expertIds, oneCoreRowNum * collNum);
    inputQueExpertId_.EnQue<int32_t>(localExperts);

    // Copy into device export's weigth
    AscendC::LocalTensor<half> localWeigth = inputQueWeigth_.AllocTensor<half>();
    AscendC::DataCopy(localWeigth, routingWeigthIn, oneCoreRowNum * collNum);
    inputQueWeigth_.EnQue<half>(localWeigth);
}

__aicore__ inline void DesImpl::CopyOut()
{
    // copy calculated relevant weigth
    AscendC::LocalTensor<half> localOutWeigth = weigthOutQue_.DeQue<half>();
    AscendC::DataCopy(routingWeigthOut, localOutWeigth, oneCoreRowNum * collNum);
    AscendC::PipeBarrier<PIPE_MTE3>();
    // copy calculated relevant expertIds
    AscendC::LocalTensor<int32_t> localExpertIdsOut = expertOutIdsQue_.DeQue<int32_t>();
    AscendC::DataCopy(expertIdsOut, localExpertIdsOut, oneCoreRowNum * collNum);
    AscendC::PipeBarrier<PIPE_MTE3>();
    // copy relevant experts num
    AscendC::LocalTensor<int32_t> localExpertNumOut = expertNumOutQue_.DeQue<int32_t>();
    AscendC::DataCopy(expertNumOut, localExpertNumOut, oneCoreRowNum * ALLIGMENT);
    AscendC::PipeBarrier<PIPE_MTE3>();
}


__aicore__ inline void DesImpl::Compute()
{   
    // Deque weigth 
    AscendC::LocalTensor<half> localWeigths = inputQueWeigth_.DeQue<half>();
    float normValue;

    // Deque experts Ids
    AscendC::LocalTensor<int32_t> inputExpertsId = inputQueExpertId_.DeQue<int32_t>();

    // temp tensors for norm calculating 
    AscendC::LocalTensor<half> buf = calcBuf_.Get<half>();
    AscendC::LocalTensor<half> work = buf[OFFSET_WORKSPACE * collNum];

    // normalization each row
    for (int i = 0; i < oneCoreRowNum; i++)
    {   
        normValue = 1.f;
        // compute norm for normalization
        normValue /= (float)ComputeSum(localWeigths[i * collNum], work, work, collNum);
        AscendC::PipeBarrier<PIPE_S>();
        // normalization
        AscendC::Muls(localWeigths[i * collNum], localWeigths[i * collNum], (half)normValue, collNum);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // calculate cumSums
    AscendC::LocalTensor<half> cumSumOut = buf[OFFSET_WORKSPACE * 2 * collNum];
    static constexpr AscendC::CumSumConfig cumSumConfig{true, false, false};
    const AscendC::CumSumInfo cumSumInfo{oneCoreRowNum, collNum};
    AscendC::CumSum<half, cumSumConfig>(cumSumOut, cumSumOut, localWeigths, cumSumInfo);
    AscendC::PipeBarrier<PIPE_V>();

    // Allocate output tensors
    AscendC::LocalTensor<half> outWeigthLocal = weigthOutQue_.AllocTensor<half>();
    AscendC::LocalTensor<int32_t> outExpertsIdsLocal = expertOutIdsQue_.AllocTensor<int32_t>();
    AscendC::LocalTensor<int32_t> validExpertsNumLocal = expertNumOutQue_.AllocTensor<int32_t>();
    // set default value
    AscendC::Duplicate(outWeigthLocal, (half)0, oneCoreRowNum * collNum);
    AscendC::Duplicate(outExpertsIdsLocal, (int32_t)777, oneCoreRowNum * collNum);
    // experts iterators
    uint32_t validIter;
    // find a valid experts
    for (int rowIter = 0; rowIter < oneCoreRowNum; rowIter++)
    {   
        validIter = 0;

        for (int i = 0; i < collNum; i++)
        {
            if ((float)cumSumOut.GetValue(rowIter * collNum + i) <= treshold)
            {
                outWeigthLocal.SetValue(rowIter * collNum + validIter, localWeigths.GetValue(rowIter * collNum + i));
                outExpertsIdsLocal.SetValue(rowIter * collNum + validIter, inputExpertsId.GetValue(rowIter * collNum + i));
                validIter++;
                pipe_barrier(PIPE_S);
            }
        }

        outWeigthLocal.SetValue(rowIter * collNum + validIter, localWeigths.GetValue(rowIter * collNum + validIter));
        outExpertsIdsLocal.SetValue(rowIter * collNum + validIter, inputExpertsId.GetValue(rowIter * collNum + validIter));
        validIter++;
        validExpertsNumLocal.SetValue(rowIter, validIter);
        pipe_barrier(PIPE_S);

    }


    weigthOutQue_.EnQue<half>(outWeigthLocal);
    expertOutIdsQue_.EnQue<int32_t>(outExpertsIdsLocal);
    expertNumOutQue_.EnQue<int32_t>(validExpertsNumLocal);
}

__aicore__ inline void DesImpl::Process()
{
    CopyIn();
    Compute();
    CopyOut();
}

#endif // DES_KERNEL_IMPL_H
