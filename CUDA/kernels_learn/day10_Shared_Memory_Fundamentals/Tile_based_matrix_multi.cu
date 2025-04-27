/*编译语句nvcc Tile_based_matrix_multi.cu -o Tile_based_matrix_multi
  运行语句.\Tile_based_matrix_multi.exe
*/

#include <cuda_runtime.h>
#include <iostream>

#define N 1024           // 假设矩阵大小为 N × N
#define TILE_SIZE 32     // tile 大小：每个 block 处理 TILE_SIZE × TILE_SIZE 个元素

// CUDA 矩阵乘法内核，使用共享内存加速
__global__ void MatMulShared(float *A, float *B, float *C, int n) {
    // 每个线程的全局行列索引
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 声明 tile 缓存，放在 shared memory 中
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 遍历 tile 步长
    for (int m = 0; m < n / TILE_SIZE; ++m) {
        // 将 A 和 B 中的 tile 装入 shared memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * n + m * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads(); // 所有线程加载完后再执行计算

        // 计算当前 tile 中一行一列的点积
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads(); // 确保当前 tile 计算完再加载下一个 tile
    }

    // 写入结果矩阵 C
    C[row * n + col] = sum;
}

int main() {
    int size = N * N * sizeof(float);

    // 分配主机内存
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // 初始化输入矩阵 A 和 B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;  // 或者 rand() % 100
        h_B[i] = 2.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置线程块和网格维度
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, N / TILE_SIZE);

    // 启动 kernel
    MatMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 拷回结果
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 简单验证结果
    std::cout << "C[0] = " << h_C[0] << std::endl; // 理论上应为 N × 2.0f

    // 清理资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
