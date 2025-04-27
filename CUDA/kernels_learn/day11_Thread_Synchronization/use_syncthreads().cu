#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define TILE_SIZE 32  // 每个 block 计算 TILE_SIZE × TILE_SIZE 的区域

// CUDA kernel：使用共享内存 + 同步机制进行矩阵乘法
__global__ void MatMulShared(float *A, float *B, float *C, int n) {
    // 当前线程的全局索引
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 声明共享内存 tile
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 分阶段计算（每次加载一个 tile 计算部分结果）
    for (int m = 0; m < n / TILE_SIZE; ++m) {
        // 每个线程负责将 A 和 B 的一小块加载进共享内存
        tile_A[threadIdx.y][threadIdx.x] = A[row * n + m * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * n + col];

        // 同步，确保整个 tile 已被加载
        __syncthreads();

        // 当前 tile 内做矩阵乘法
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        // 同步，确保所有线程完成本轮计算后再进行下一轮加载
        __syncthreads();
    }

    // 写结果
    C[row * n + col] = sum;
}

int main() {
    int size = N * N * sizeof(float);

    // 分配主机内存
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // 初始化输入数据
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;  // 可随机填充
        h_B[i] = 1.0f;
    }

    // 分配 GPU 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义 block 与 grid 尺寸
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, N / TILE_SIZE);

    // CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);  // 记录开始时间

    // 启动内核
    MatMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // 等待计算完成

    cudaEventRecord(stop);  // 记录结束时间
    cudaEventSynchronize(stop);

    // 计算用时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU matrix multiplication time: " << milliseconds << " ms" << std::endl;

    // 拷回结果
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result sample C[0] = " << h_C[0] << std::endl;

    // 释放资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
