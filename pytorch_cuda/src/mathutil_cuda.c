// mathutil_cuda.c
// THC是pytorch底层GPU库
#include <THC/THC.h>
#include "mathutil_cuda_kernel.h"
#include <stdio.h>
extern THCState *state;

int broadcast_sum(THCudaTensor *a_tensor, THCudaTensor *b_tensor, int x, int y)
{
    float *a = THCudaTensor_data(state, a_tensor);
    float *b = THCudaTensor_data(state, b_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    // 这里调用之前在cuda中编写的接口函数
    broadcast_sum_cuda(a, b, x, y, stream);
    printf("nnn");

    return 1;
}