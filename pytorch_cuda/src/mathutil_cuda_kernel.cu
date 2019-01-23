// mathutil_cuda_kernel.cu
// 头文件，最后一个是cuda特有的
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mathutil_cuda_kernel.h"

// 获取GPU线程通道信息
dim3 cuda_gridsize(int n)
{
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    return d;
}
// 这个函数是cuda执行函数，可以看到细化到了每一个元素
__global__ void broadcast_sum_kernel(float *a, float *b, int x, int y, int size)
{
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= size) return;
    int j = i % x; i = i / x;
    int k = i % y;
    a[IDX2D(j, k, y)] += b[k];
}


// 这个函数是与c语言函数链接的接口函数
void broadcast_sum_cuda(float *a, float *b, int x, int y, cudaStream_t stream)
{
    int size = x * y;
    cudaError_t err;

    // 上面定义的函数
    broadcast_sum_kernel<<<cuda_gridsize(size), BLOCK, 0, stream>>>(a, b, x, y, size);

    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
//上面的代码和C语言很像，但是多了一个__global__,这个是cuda中特有类型，这个函数实现向量a和b的element-wise的相加。