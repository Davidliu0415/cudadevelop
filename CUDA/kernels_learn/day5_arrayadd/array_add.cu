#include <iostream>
#include <cuda_runtime.h>
using namespace std;


/*代码片段 | 中文解释
__global__ | 表示这是一个在 GPU 上运行的 kernel 函数，由 CPU 调用
threadIdx.x | 当前线程在线程块中的编号（从 0 开始）
cudaMalloc() | 在 GPU 上分配显存
cudaMemcpy() | 在 CPU 和 GPU 之间复制数据
VecAdd<<<1, N>>>() | 启动 kernel，<<<blocks, threads_per_block>>> 是 CUDA 启动语法
cudaDeviceSynchronize() | 等待 GPU 所有操作完成
cudaFree() | 释放 GPU 上的显存资源*/


/*CPU 初始化数组
     ↓
将数组 A/B 拷贝到 GPU
     ↓
GPU 中 N 个线程并行执行 A[i] + B[i]
     ↓
结果 C 拷贝回 CPU
     ↓
打印结果并释放资源
*/

// CUDA 核函数（kernel）：进行逐元素加法
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    // 每个线程处理数组中的一个元素
    int i = threadIdx.x;

    if (i < N) {  // 防止线程数多于数组长度时越界
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 5;
    size_t size = N * sizeof(float);  // 每个数组大小（字节数）

    // ======================== //
    // 1. 在 CPU 上声明并初始化数据
    // ======================== //
    float h_A[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_B[N] = {10.0, 20.0, 30.0, 40.0, 50.0};
    float h_C[N];  // 用于接收结果

    // ======================== //
    // 2. 在 GPU 上分配内存
    // ======================== //
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);  // 分配 A 数组的显存
    cudaMalloc((void**)&d_B, size);  // 分配 B 数组的显存
    cudaMalloc((void**)&d_C, size);  // 分配 C 数组的显存（输出）

    // ======================== //
    // 3. 将数据从 CPU 拷贝到 GPU
    // ======================== //
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ======================== //
    // 4. 启动 CUDA 内核执行加法
    // ======================== //
    VecAdd<<<1, N>>>(d_A, d_B, d_C, N);  // 启动 1 个 block，N 个线程
    cudaDeviceSynchronize();  // 等待 GPU 执行完成

    // ======================== //
    // 5. 把结果从 GPU 拷贝回 CPU
    // ======================== //
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // ======================== //
    // 6. 打印输出结果
    // ======================== //
    cout << "结果 C = A + B : ";
    for (int i = 0; i < N; ++i) {
        cout << h_C[i] << " ";
    }
    cout << endl;

    // ======================== //
    // 7. 释放显存
    // ======================== //
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
