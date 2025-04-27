#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // 数组大小

// CUDA Kernel
__global__ void sumArray(float* input, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // 计算全局线程 ID

    if (tid < N) {
        // 原子加到 result[0]
        atomicAdd(result, input[tid]);
    }
}

int main() {
    float h_input[N], h_result = 0;
    float *d_input, *d_result;

    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // 简单起见，全部赋值为1
    }

    // 分配设备内存
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // 每个 block 256 个线程，计算需要多少个 block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动核函数
    sumArray<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_result);

    // 同步，确保 kernel 执行结束
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Sum of array elements = " << h_result << std::endl;

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
