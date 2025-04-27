// Kernel definition这是一个并行加法
/*编译语句nvcc Kernels.cu -o Kernels
  运行语句.\Kernels.exe
*/

#include <iostream>
using namespace std;

// GPU 内核函数
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
        printf("GPU thread %d: C[%d] = %f\n", i, i, C[i]);
    }
}
/*你在 kernel 中使用了 A[i], B[i], C[i]，但它们是 CPU 主机端的数组（float A[10]）。

而 CUDA 中的规则是：

GPU 代码（__global__ 函数）只能访问设备（GPU）内存，不能直接访问主机内存。

你需要：

把 A、B、C 分配到 GPU（设备）上

把主机数据拷贝到设备

执行 kernel

把计算结果从设备拷贝回主机

*/
int main()
{
    int N = 10;
    size_t size = N * sizeof(float);//主机端的10位的内存大小

    // 主机端数组
    float h_A[10] = {1.2 ,2 ,34 ,4.6 ,45 ,75.5 ,456 ,43.3,99,100.1};
    float h_B[10] = {1.2 ,2 ,34 ,4.6 ,45 ,75.5 ,456 ,43.3,99,100.1};
    float h_C[10];

    // 设备端指针
    float *d_A, *d_B, *d_C;

    // 1. 在设备上分配内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 2. 将主机数据拷贝到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. 启动 kernel，1 个 block，N 个线程
    VecAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // 4. 等待 GPU 执行完成
    cudaDeviceSynchronize();

    // 5. 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6. 主机端输出结果
    cout << "\n主机输出加法结果:\n";
    for (int i = 0; i < N; ++i)
        cout << "C[" << i << "] = " << h_C[i] << endl;

    // 7. 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
