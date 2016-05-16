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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>



typedef struct{
    float S;
    float X;
    float T;
    float R;
    float V;
}TEuropeanOption;

typedef struct{
    float Expected;
    float Confidence;
}TOptionValue;



////////////////////////////////////////////////////////////////////////////////
// CPU Monte-Carlo for GPU results validation
////////////////////////////////////////////////////////////////////////////////
static float Max(float a, float b){
    return (a > b) ? a : b;
}

extern "C" void MonteCarloCPU(
    TOptionValue&    callValue,
    TEuropeanOption optionData,
    float *h_Random,
    int pathN
){
    const float        S = optionData.S;
    const float        X = optionData.X;
    const float        T = optionData.T;
    const float        R = optionData.R;
    const float        V = optionData.V;
    const float VBySqrtT = V * sqrtf(T);
    const float MuByT    = (R - 0.5f * V * V) * T;

    double sum = 0, sum2 = 0;
    for(int pos = 0; pos < pathN; pos++){
        float    r = h_Random[pos];
        float path = Max(S * expf(MuByT + VBySqrtT * r) - X, 0);
        sum  += path;
        sum2 += path * path;
    }

    //Derive average from the total sum and discount by riskfree rate 
    callValue.Expected = (float)(exp(-R * T) * sum / (double)pathN);
    //Standart deviation
    double stdDev = sqrt(((double)pathN * sum2 - sum * sum)/ ((double)pathN * (double)(pathN - 1)));
    //Confidence width; in 95% of all cases theoretical value lies within these borders
    callValue.Confidence = (float)(exp(-R * T) * 1.96 * stdDev / sqrt((double)pathN));
}



////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for CPU/GPU Monte-Carlo results validation
////////////////////////////////////////////////////////////////////////////////
#define A1 0.31938153f
#define A2 -0.356563782f
#define A3 1.781477937f
#define A4 -1.821255978f
#define A5 1.330274429f
#define RSQRT2PI 0.39894228040143267793994605993438f

//Polynomial approxiamtion of
//cumulative normal distribution function
float CND(float d){
    float
        K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
        cnd = RSQRT2PI * expf(- 0.5f * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

//Black-Scholes formula for call value
extern "C" void BlackScholesCall(
    float& callValue,
    TEuropeanOption optionData
){
    const float S = optionData.S;
    const float X = optionData.X;
    const float T = optionData.T;
    const float R = optionData.R;
    const float V = optionData.V;
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = (logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = CND(d1);
    CNDD2 = CND(d2);

    expRT = expf(- R * T);
    callValue = S * CNDD1 - X * expRT * CNDD2;
}
