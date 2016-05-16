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

/*
 * This sample evaluates fair call price for a
 * given set of European options using Monte-Carlo approach.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <cutil.h>
#include <multithreading.h>
#include "MonteCarlo_types.h"



////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}

float randFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// CPU reference functions
///////////////////////////////////////////////////////////////////////////////
extern "C" void MonteCarloCPU(
    TOptionValue&   callValue,
    TEuropeanOption optionData,
    float *h_Random,
    int pathN
);

//Black-Scholes formula for call options
extern "C" void BlackScholesCall(
    float& CallResult,
    TEuropeanOption optionData
);



////////////////////////////////////////////////////////////////////////////////
// GPU kernel code
////////////////////////////////////////////////////////////////////////////////
#include "MersenneTwister_kernel.cu"
#include "MonteCarloMultiGPU_kernel.cu"



////////////////////////////////////////////////////////////////////////////////
// GPU-driving host thread
////////////////////////////////////////////////////////////////////////////////
float *h_Random;

typedef struct{
    int device;

    int optionCount;
    TEuropeanOption *h_OptionData;
    TOptionValue    *h_CallValue;

    unsigned int seed;
    int pathN;

    float time;
} TOptionSolver;


static CUT_THREADPROC solverThread(TOptionSolver *solver){
    const int            optN = solver->optionCount;
    const int           pathN = solver->pathN;
    const int  optionDataSize = optN * sizeof(TEuropeanOption);
    const int      resultSize = optN * sizeof(TOptionValue);

    const int    doMultiBlock = (pathN / optN) >= 8192;
    const int blocksPerOption = (optN < 16) ? 64 : 16;
    const int        THREAD_N = 256;
    const int          accumN = THREAD_N * blocksPerOption;

    const int           randN = iAlignUp(pathN, 2 * MT_RNG_COUNT);
    const int         nPerRNG = randN / MT_RNG_COUNT;


    TOptionValue
        *d_Sum,
        *d_Result;

    float *d_Random;
    int i;
    unsigned int hTimer;

    //Init GPU and allocate memory
    CUDA_SAFE_CALL( cudaSetDevice(solver->device) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Random, randN * sizeof(float)) );

    if(doMultiBlock)
        CUDA_SAFE_CALL( cudaMalloc((void **)&d_Sum, resultSize * accumN) );

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Result, resultSize) );

    //Generate random data on this GPU
    seedMTGPU(solver->seed);
    RandomGPU<<<32, 128>>>(d_Random, nPerRNG);
    CUT_CHECK_ERROR("RandomGPU() execution failed\n");
    BoxMullerGPU<<<32, 128>>>(d_Random, nPerRNG);
    CUT_CHECK_ERROR("BoxMullerGPU() execution failed\n");

    //Upload GPU random samples to CPU for reference CPU Monte-Carlo simulation
    if(solver->device == 0)
       CUDA_SAFE_CALL( cudaMemcpy(h_Random, d_Random, pathN * sizeof(float), cudaMemcpyDeviceToHost) );

    //Main computations
    CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
        CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_OptionData, solver->h_OptionData, optionDataSize) );
        if(doMultiBlock){
            dim3 gridMain(blocksPerOption, optN);
            MonteCarloKernel1<<<gridMain, THREAD_N>>>(
                d_Sum,
                d_Random,
                solver->pathN
            );
            CUT_CHECK_ERROR("MonteCarloKernel1() execution failed\n");
            MonteCarloReduce<128><<<optN, 128>>>(
                d_Result,
                d_Sum,
                accumN
            );
            CUT_CHECK_ERROR("MonteCarloReduce() execution failed\n");
        }else{
            MonteCarloKernel2<128><<<optN, 128>>>(
                d_Result,
                d_Random,
                solver->pathN
            );
            CUT_CHECK_ERROR("MonteCarloKernel2() execution failed\n");
        }
        CUDA_SAFE_CALL( cudaMemcpy(solver->h_CallValue, d_Result, resultSize, cudaMemcpyDeviceToHost) );

        for(i = 0; i < optN; i++){
            const double   RT = solver->h_OptionData[i].R * solver->h_OptionData[i].T;
            const double  sum = solver->h_CallValue[i].Expected;
            const double sum2 = solver->h_CallValue[i].Confidence;
            //Derive average from the total sum and discount by riskfree rate 
            solver->h_CallValue[i].Expected = (float)(exp(-RT) * sum / (double)pathN);
            //Standart deviation
            double stdDev = sqrt(((double)pathN * sum2 - sum * sum)/ ((double)pathN * (double)(pathN - 1)));
            //Confidence width; in 95% of all cases theoretical value lies within these borders
            solver->h_CallValue[i].Confidence = (float)(exp(-RT) * 1.96 * stdDev / sqrt((double)pathN));
        }
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL( cutStopTimer(hTimer) );
    solver->time = cutGetTimerValue(hTimer);

    //Shut down this GPU
    CUDA_SAFE_CALL( cudaFree(d_Result) );
    if(doMultiBlock) CUDA_SAFE_CALL( cudaFree(d_Sum) );
    CUDA_SAFE_CALL( cudaFree(d_Random) );

    CUT_THREADEND;
}



#define DO_CPU
#undef DO_CPU

#define PRINT_RESULTS
#undef PRINT_RESULTS



int main(int argc, char **argv){
#ifdef __DEVICE_EMULATION__
    const int         OPT_N = 4;
#else 
    const int         OPT_N = 128;
#endif
    const int MAX_GPU_COUNT = 8;
    const unsigned int SEED = 777;
    const int PATH_N  = 24000000;

    //Input data array
    TEuropeanOption optionData[OPT_N];
    //Final GPU MC results
    TOptionValue callValueGPU[OPT_N];
    //"Theoretical" call values by Black-Scholes formula
    float callValueBS[OPT_N];
    //Solver config
    TOptionSolver optionSolver[MAX_GPU_COUNT];
    //OS thread ID
    CUTThread threadID[MAX_GPU_COUNT];


    //GPU number present in the system
    int GPU_N;
    int gpuBase, gpuIndex;
    int i;

    double
        delta, ref, sumDelta, sumRef, sumReserve;

    CUT_DEVICE_INIT();
    CUDA_SAFE_CALL( cudaGetDeviceCount(&GPU_N) );

    h_Random = (float *)malloc(PATH_N * sizeof(float));
    loadMTGPU(cutFindFilePath("MersenneTwister.dat", argv[0]));

    //Initialize input data
    printf("main(): generating input data...\n");
    srand(123);
    for(i = 0; i < OPT_N; i++){
        optionData[i].S = randFloat(5.0f, 50.f);
        optionData[i].X = randFloat(10.0f, 25.0f);
        optionData[i].T = randFloat(1.0f, 5.0f);
        optionData[i].R = 0.06f;
        optionData[i].V = 0.10f;
        callValueGPU[i].Expected   = -1.0f;
        callValueGPU[i].Confidence = -1.0f;
    }

    //Get option count for each GPU
    for(i = 0; i < GPU_N; i++){
        optionSolver[i].optionCount = OPT_N / GPU_N;
        optionSolver[i].seed  = SEED;
        optionSolver[i].pathN = PATH_N;
    }
    //Take into account cases with "odd" option counts
    for(i = 0; i < (OPT_N % GPU_N); i++)
        optionSolver[i].optionCount++;
    //Assign GPU option ranges
    gpuBase = 0;
    for(i = 0; i < GPU_N; i++){
        optionSolver[i].device = i;
        optionSolver[i].h_OptionData = optionData   + gpuBase;
        optionSolver[i].h_CallValue  = callValueGPU + gpuBase;
        gpuBase += optionSolver[i].optionCount;
    }

    //Start CPU thread for each GPU
    printf("main(): starting multiple host threads...\n");
    for(gpuIndex = 0; gpuIndex < GPU_N; gpuIndex++)
        threadID[gpuIndex] = cutStartThread((CUT_THREADROUTINE)solverThread, &optionSolver[gpuIndex]);

    printf("main(): waiting for GPU results...\n");
    cutWaitForThreads(threadID, GPU_N);

    printf("main(): GPU statistics\n");
    for(i = 0; i < GPU_N; i++){
        printf("GPU #%i\n", optionSolver[i].device);
        printf("Options         : %i\n", optionSolver[i].optionCount);
        printf("Simulation paths: %i\n", optionSolver[i].pathN);
        printf("Time (ms.)      : %f\n", optionSolver[i].time);
        printf("Options per sec.: %f\n", optionSolver[i].optionCount / (optionSolver[i].time * 0.001));
    }


#ifdef DO_CPU
    printf("main(): running CPU MonteCarlo...\n");
        TOptionValue callValueCPU;
        sumDelta = 0;
        sumRef   = 0;
        for(i = 0; i < OPT_N; i++){
            MonteCarloCPU(
                callValueCPU,
                optionData[i],
                h_Random,
                PATH_N
            );
            delta     = fabs(callValueCPU.Expected - callValueGPU[i].Expected);
            ref       = callValueCPU.Expected;
            sumDelta += delta;
            sumRef   += ref;
            printf("Exp : %f | %f\t", callValueCPU.Expected,   callValueGPU[i].Expected);
            printf("Conf: %f | %f\n", callValueCPU.Confidence, callValueGPU[i].Confidence);
        }
    printf("L1 norm: %E\n", sumDelta / sumRef);
#endif

    printf("main(): comparing Monte-Carlo and Black-Scholes results...\n");
        sumDelta   = 0;
        sumRef     = 0;
        sumReserve = 0;
        for(i = 0; i < OPT_N; i++){
            BlackScholesCall(
                callValueBS[i],
                optionData[i]
            );
            delta       = fabs(callValueBS[i] - callValueGPU[i].Expected);
            ref         = callValueBS[i];
            sumDelta   += delta;
            sumRef     += ref;
            if(delta > 1e-6) sumReserve += callValueGPU[i].Confidence / delta;
#ifdef PRINT_RESULTS
            printf("BS: %f; delta: %E\n", callValueBS[i], delta);
#endif
        }
    sumReserve /= OPT_N;
    printf("L1 norm        : %E\n", sumDelta / sumRef);
    printf("Average reserve: %f\n", sumReserve);
    printf((sumReserve > 1.0f) ? "TEST PASSED\n" : "TEST FAILED.\n");

    printf("Shutting down...\n");
    free(h_Random);
    CUT_EXIT(argc, argv);
}
