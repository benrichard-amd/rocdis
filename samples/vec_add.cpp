#include<hip/hip_runtime.h>

__global__ void vec_add(float* x, float* y, float* z, int n)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < n) {
        z[gid] = x[gid] + y[gid];
    }
}
