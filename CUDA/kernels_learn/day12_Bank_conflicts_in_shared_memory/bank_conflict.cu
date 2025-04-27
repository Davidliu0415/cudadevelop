#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // 线程数，需为32的倍数

__global__ void ConflictKernel(float* out) {
    __shared__ float shared[N][32];  // 32列可能引发 bank 冲突

    int tid = threadIdx.x;

    // 所有线程访问同一列：0
    for (int i = 0; i < 1000; ++i)
        shared[tid][0] = tid * 1.0f;

    __syncthreads();

    out[tid] = shared[tid][0];
}

__global__ void NoConflictKernel(float* out) {
    __shared__ float shared[N][32];

    int tid = threadIdx.x;

    // 每个线程访问不同列，避免 bank 冲突
    for (int i = 0; i < 1000; ++i)
        shared[tid][tid % 32] = tid * 1.0f;

    __syncthreads();

    out[tid] = shared[tid][tid % 32];
}

void runAndMeasure(void (*kernel)(float*), float* d_out, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<1, N>>>(d_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << name << " time: " << time << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    float* d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    std::cout << "Testing bank conflicts...\n";

    runAndMeasure(ConflictKernel, d_out, "With Bank Conflict");
    runAndMeasure(NoConflictKernel, d_out, "Without Bank Conflict");

    cudaFree(d_out);
    return 0;
}
