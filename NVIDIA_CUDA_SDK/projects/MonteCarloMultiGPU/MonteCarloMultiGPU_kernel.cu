/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */



#ifndef MONTECARLOMULTIGPU_KERNEL_CUH
#define MONTECARLOMULTIGPU_KERNEL_CUH



//Common types
#include "MonteCarlo_types.h"



////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////
//Store input data in constant memory
#define MAX_OPT_COUNT 256
__device__ __constant__ TEuropeanOption d_OptionData[MAX_OPT_COUNT];



////////////////////////////////////////////////////////////////////////////////
// This function calculates total sum for each of the two input arrays.
// SUM_N must be power of two
////////////////////////////////////////////////////////////////////////////////
template<int SUM_N> __device__ void sumReduce(float *sum, float *sum2){
    for(int stride = SUM_N / 2; stride > 0; stride >>= 1){
        __syncthreads();
        for(int pos = threadIdx.x; pos < stride; pos += blockDim.x){
            sum[pos] += sum[pos + stride];
            sum2[pos] += sum2[pos + stride];
        }
    }
}



////////////////////////////////////////////////////////////////////////////////
// This kernel computes partial integrals over all paths using a multiple thread
// blocks per option. It is used when a single thread block per option would not
// be enough to keep the GPU busy. Execution of this kernel is followed by
// a MonteCarloReduce() to get the complete integral for each option.
////////////////////////////////////////////////////////////////////////////////
__global__ void MonteCarloKernel1(
    TOptionValue *d_SumCall,
    float *d_Random,
    int pathN
){
    const int optionIndex = blockIdx.y;

    const float        S = d_OptionData[optionIndex].S;
    const float        X = d_OptionData[optionIndex].X;
    const float        T = d_OptionData[optionIndex].T;
    const float        R = d_OptionData[optionIndex].R;
    const float        V = d_OptionData[optionIndex].V;
    const float VBySqrtT = V * sqrtf(T);
    const float MuByT    = (R - 0.5f * V * V) * T;

    //One thread per partial integral
    const int   iSum = blockIdx.x * blockDim.x + threadIdx.x;
    const int accumN = blockDim.x * gridDim.x;

    //Cycle through the entire random paths array:
    //derive end stock price for each path
    //accumulate into intermediate global memory array
    TOptionValue sumCall = {0, 0};
    for(int pos = iSum; pos < pathN; pos += accumN){
        float             r = d_Random[pos];
        float endStockPrice = S * __expf(MuByT + VBySqrtT * r);
        float    callProfit = fmaxf(endStockPrice - X, 0);
        sumCall.Expected   += callProfit;
        sumCall.Confidence += callProfit * callProfit;
    }
    d_SumCall[optionIndex * accumN + iSum] = sumCall;
}



////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block 
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy. When this is not the case, using
// more blocks per option is faster, so we use MonteCarloKernel1 plus
// MonteCarloReduce instead.
////////////////////////////////////////////////////////////////////////////////
template<int SUM_N> __global__ void MonteCarloKernel2(
    TOptionValue *d_ResultCall, //Partial sums (+sum of squares) destination
    float *d_Random,            //N(0, 1) random samples array
    int pathN                   //Sample count
){
    __shared__ float s_SumCall[SUM_N];
    __shared__ float s_Sum2Call[SUM_N];

    const int optionIndex = blockIdx.x;
    const float         S = d_OptionData[optionIndex].S;
    const float         X = d_OptionData[optionIndex].X;
    const float         T = d_OptionData[optionIndex].T;
    const float         R = d_OptionData[optionIndex].R;
    const float         V = d_OptionData[optionIndex].V;
    const float  VBySqrtT = V * sqrtf(T);
    const float     MuByT = (R - 0.5f * V * V) * T;

    //Cycle through the entire random paths array:
    //derive end stock price for each path 
    //accumulate partial integrals into intermediate shared memory buffer
    for(int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x){
        TOptionValue sumCall = {0, 0};
        for(int pos = iSum; pos < pathN; pos += SUM_N){
            float              r = d_Random[pos];
            float  endStockPrice = S * __expf(MuByT + VBySqrtT * r);
            float     callProfit = fmaxf(endStockPrice - X, 0);
            sumCall.Expected   += callProfit;
            sumCall.Confidence += callProfit * callProfit;
        }
        s_SumCall[iSum]  = sumCall.Expected;
        s_Sum2Call[iSum] = sumCall.Confidence;
    }

    //Reduce shared memory accumulators
    //and write final result to global memory
    sumReduce<SUM_N>(s_SumCall, s_Sum2Call);
    if(threadIdx.x == 0){
        TOptionValue sumCall = {s_SumCall[0], s_Sum2Call[0]};
        d_ResultCall[optionIndex] = sumCall;
    }
}



////////////////////////////////////////////////////////////////////////////////
//Finalizing reduction for MonteCarloKernel1()
//Final reduction for each per-option accumulator output
////////////////////////////////////////////////////////////////////////////////
template<int SUM_N> __global__ void MonteCarloReduce(
    TOptionValue *d_ResultCall,
    TOptionValue *d_SumCall,
    int accumN
){
    __shared__ float s_SumCall[SUM_N];
    __shared__ float s_Sum2Call[SUM_N];
    TOptionValue *d_SumBase = &d_SumCall[blockIdx.x * accumN];

    //Reduce global memory accumulators array for current option
    //to a set fitting into shared memory
    for(int iSum = threadIdx.x; iSum < SUM_N; iSum += blockDim.x){
        TOptionValue sumCall = {0, 0};
        for(int pos = iSum; pos < accumN; pos += SUM_N){
            TOptionValue sum = d_SumBase[pos];
            sumCall.Expected   += sum.Expected;
            sumCall.Confidence += sum.Confidence;
        }
        s_SumCall[iSum]  = sumCall.Expected;
        s_Sum2Call[iSum] = sumCall.Confidence;
    }

    //Reduce shared memory accumulators
    //and write final result to global memory
    sumReduce<SUM_N>(s_SumCall, s_Sum2Call);
    if(threadIdx.x == 0){
        TOptionValue sumCall = {s_SumCall[0], s_Sum2Call[0]};
        d_ResultCall[blockIdx.x] = sumCall;
    }
}



#endif
