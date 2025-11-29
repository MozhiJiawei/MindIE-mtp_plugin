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
static constexpr uint32_t CUMSUM_WORKSPACE = 2; // the offset of workspace is 1
static constexpr uint32_t ALIGNMENT = 32;       // alignment in bytes
constexpr uint32_t TENSOR_MAX_DIM_NUM = 2;


class DesImpl
{
public:
    __aicore__ inline DesImpl() {}
    __aicore__ inline void Init(GM_ADDR globalInputWeigth, GM_ADDR globalExpertIds, GM_ADDR globalWeigthsOut, 
                                                GM_ADDR globalExpertIdsOut, GM_ADDR globalExpertsTotalNum,
                                                DesCustomKernelTilingData &tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CleanUp();
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();

    __aicore__ inline void Compute();

private:
    AscendC::TPipe pipe;
    // Input global Tensors
    /////////////////////////////////////////////////////////////////////////////////
    AscendC::GlobalTensor<int32_t> expertIds;
    AscendC::GlobalTensor<half> routingWeigthIn;

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

    /////////////////////////////////////////////////////////////////////////////////
    // tiling params
    uint32_t numRow;
    uint32_t collNum;
    uint32_t oneCoreRowNum;
    uint32_t numCore;
    uint32_t topK;
    
    uint32_t aligned_half_size;
    uint32_t aligned_half;
    
    uint32_t aligned_exp_num_size;
    uint32_t aligned_exp_num;
    

    AscendC::LocalTensor<int32_t> localExperts;
    AscendC::LocalTensor<half> localWeigth;
    
    AscendC::LocalTensor<half> outWeigthLocal;
    AscendC::LocalTensor<int32_t> outExpertsIdsLocal;
    AscendC::LocalTensor<int32_t> validExpertsNumLocal;

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
    uint32_t expOffset = AscendC::GetBlockIdx() * oneCoreRowNum;

    // set pipes
    routingWeigthIn.SetGlobalBuffer((__gm__ half*)globalInputWeigth + gmOffset);
    expertIds.SetGlobalBuffer((__gm__ int32_t*)globalExpertIds + gmOffset);

    aligned_half_size = oneCoreRowNum * collNum * sizeof(half) > ALIGNMENT ? oneCoreRowNum * collNum * sizeof(half) : ALIGNMENT;
    aligned_half = aligned_half_size / sizeof(half);
    
    aligned_exp_num_size = oneCoreRowNum * sizeof(uint32_t) > ALIGNMENT ? oneCoreRowNum * sizeof(uint32_t) : ALIGNMENT;
    aligned_exp_num = aligned_exp_num_size / sizeof(uint32_t);
    
    routingWeigthOut.SetGlobalBuffer((__gm__ half*)globalWeigthsOut + gmOffset);
    expertIdsOut.SetGlobalBuffer((__gm__ int32_t*)globalExpertIdsOut + gmOffset);
    expertNumOut.SetGlobalBuffer((__gm__ int32_t*)globalExpertsTotalNum + expOffset);

    // input ques
    pipe.InitBuffer(inputQueWeigth_,   DOUBLE_BUFFER, aligned_half_size  );
    pipe.InitBuffer(inputQueExpertId_, DOUBLE_BUFFER, oneCoreRowNum * collNum * sizeof(uint32_t));

    // output ques
    pipe.InitBuffer(expertOutIdsQue_, BUFFER_NUM, oneCoreRowNum * collNum * sizeof(uint32_t));
    pipe.InitBuffer(weigthOutQue_,    BUFFER_NUM, aligned_half_size);
    pipe.InitBuffer(expertNumOutQue_, BUFFER_NUM, aligned_exp_num_size);

}

__aicore__ inline void DesImpl::CopyIn()
{
    // Copy expert ids into device
    localExperts = inputQueExpertId_.AllocTensor<int32_t>();
    AscendC::DataCopy(localExperts, expertIds, oneCoreRowNum * collNum);
    inputQueExpertId_.EnQue<int32_t>(localExperts);

    // Copy into device export's weigth
    localWeigth = inputQueWeigth_.AllocTensor<half>();
    AscendC::DataCopy(localWeigth, routingWeigthIn, aligned_half );
    inputQueWeigth_.EnQue<half>(localWeigth);
}


__aicore__ inline void DesImpl::Compute()
{
    AscendC::printf("/////////////////////////////////////////////////////////////////////////////////\n");
    AscendC::printf("Step1!!!!!!!!\n");
    AscendC::printf("/////////////////////////////////////////////////////////////////////////////////\n");

    // Deque weigth
    AscendC::LocalTensor<half>    localWeigths   = inputQueWeigth_.DeQue<half>();
    AscendC::LocalTensor<int32_t> inputExpertsId = inputQueExpertId_.DeQue<int32_t>();

    AscendC::PipeBarrier<PIPE_ALL>();
    // Allocate output tensors
    outWeigthLocal = weigthOutQue_.AllocTensor<half>();
    outExpertsIdsLocal = expertOutIdsQue_.AllocTensor<int32_t>();
    validExpertsNumLocal = expertNumOutQue_.AllocTensor<int32_t>();

    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Duplicate(outExpertsIdsLocal, (int32_t)129, oneCoreRowNum * collNum); // Here we can fix fake expert numer

    AscendC::PipeBarrier<PIPE_ALL>();
    // set default value
    AscendC::Duplicate(outWeigthLocal, (half)0, aligned_half );
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::printf("/////////////////////////////////////////////////////////////////////////////////\n");
    AscendC::printf("Step2!!!!!!!!\n");
    AscendC::printf("/////////////////////////////////////////////////////////////////////////////////\n");


    // experts iterators
    uint32_t validIter;
    // find a valid experts
    for (int rowIter = 0; rowIter < oneCoreRowNum; rowIter++)
    {
        validIter = 0;
        float cumsum = 0.0;
        for (int i = 0; i < collNum; i++)
        {
            if (cumsum <= treshold) {
		//AscendC::PipeBarrier<PIPE_ALL>();
		//AscendC::printf("Current trash %f \n", threshold);
		//AscendC::PipeBarrier<PIPE_ALL>();
		        float ff = localWeigths.GetValue(rowIter * collNum + i);
//                pipe_barrier(PIPE_S);
//                outWeigthLocal.SetValue(rowIter * collNum + i, (half)ff);
//                AscendC::PipeBarrier<PIPE_ALL>();
//                uint32_t idx = inputExpertsId.GetValue(rowIter * collNum + i);
//                pipe_barrier(PIPE_S);
//                outExpertsIdsLocal.SetValue(rowIter * collNum + i, idx);
//                AscendC::PipeBarrier<PIPE_ALL>();
//                validIter++;
//                pipe_barrier(PIPE_S);
//                AscendC::PipeBarrier<PIPE_ALL>();
//                cumsum += ff;
//                pipe_barrier(PIPE_S);
            }
//
        }

//        outWeigthLocal.SetValue(rowIter * collNum + validIter, localWeigths.GetValue(rowIter * collNum + validIter));
//        AscendC::PipeBarrier<PIPE_ALL>();
//        outExpertsIdsLocal.SetValue(rowIter * collNum + validIter, inputExpertsId.GetValue(rowIter * collNum + validIter));
//        AscendC::PipeBarrier<PIPE_ALL>();
//        validExpertsNumLocal.SetValue(rowIter, validIter);
//        AscendC::PipeBarrier<PIPE_ALL>();
    }

    weigthOutQue_.EnQue<half>(outWeigthLocal);
    expertOutIdsQue_.EnQue<int32_t>(outExpertsIdsLocal);
    expertNumOutQue_.EnQue<int32_t>(validExpertsNumLocal);

}

__aicore__ inline void DesImpl::CopyOut()
{
    // copy calculated relevant weigth
    AscendC::LocalTensor<half> localOutWeigth = weigthOutQue_.DeQue<half>();
    AscendC::DataCopy(routingWeigthOut, localOutWeigth, aligned_half);
    AscendC::PipeBarrier<PIPE_MTE3>();
    // copy calculated relevant expertIds
    AscendC::LocalTensor<int32_t> localExpertIdsOut = expertOutIdsQue_.DeQue<int32_t>();
    AscendC::DataCopy(expertIdsOut, localExpertIdsOut, oneCoreRowNum * collNum);
    AscendC::PipeBarrier<PIPE_MTE3>();
    // copy relevant experts num
    AscendC::LocalTensor<int32_t> localExpertNumOut = expertNumOutQue_.DeQue<int32_t>();
    AscendC::DataCopy(expertNumOut, localExpertNumOut, aligned_exp_num);
    AscendC::PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void DesImpl::Process()
{
    CopyIn();
    Compute();
//    CopyOut();
}

#endif // DES_KERNEL_IMPL_H
