//我们现在将原始的 CUDA 矩阵加法代码扩展为支持多个线程块（block）并使用 共享内存（shared memory） 和 线程同步（__syncthreads()） 来实现协同计算。
/*✅ 目标：
使用多个线程块处理任意大小的矩阵。

使用共享内存临时存储块数据，加速访问。

使用 __syncthreads() 进行线程同步，确保线程协作正确。

对线程块的大小做边界判断，保证不会越界。
*/


#include <iostream>
#include <cuda_runtime.h>

#define N 64 // 假设是一个 N × N 的矩阵
#define BLOCK_SIZE 16 // 每个 block 是 16x16 个线程

// CUDA 核函数：执行矩阵加法，使用共享内存加速访问
__global__ void MatAdd(float* A, float* B, float* C, int width)
{
    // 计算全局线程在矩阵中的坐标
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 使用共享内存加速块内数据访问（blockDim.x = BLOCK_SIZE）
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 边界检查，防止访问越界
    if (i < width && j < width)
    {
        // 将全局内存中的数据读入共享内存
        As[threadIdx.y][threadIdx.x] = A[j * width + i]; // 行主序
        Bs[threadIdx.y][threadIdx.x] = B[j * width + i];

        // 同步线程，确保共享内存加载完成
        __syncthreads();

        // 在共享内存中进行加法（这里没必要用共享内存加法，但演示使用）
        float sum = As[threadIdx.y][threadIdx.x] + Bs[threadIdx.y][threadIdx.x];

        // 同步线程，确保加法操作完成（此处可选）
        __syncthreads();

        // 写回结果到全局内存
        C[j * width + i] = sum;
    }
}

int main()
{
    const int size = N * N * sizeof(float);
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // 初始化矩阵 A 和 B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = i * 1.5f;
    }

    // 设备端指针
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置线程块维度和网格维度
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动 CUDA 核函数
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印部分结果
    std::cout << "Matrix C = A + B, 前 5 个元素:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
