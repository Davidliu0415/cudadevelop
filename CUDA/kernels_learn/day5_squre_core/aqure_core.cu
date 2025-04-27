#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA 核函数：将每个元素平方
__global__ void SquareKernel(float* d_in, float* d_out, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // 计算全局线程索引
    if (i < N) {
        d_out[i] = d_in[i] * d_in[i];  // 简单的计算：平方
    }
}

int main() {
    const int N = 10;
    size_t size = N * sizeof(float);  // 总内存大小 = 元素数量 × float大小

    // -------------------------
    // 1. 主机端数据初始化
    // -------------------------
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i + 1);  // 初始化为 1 到 10
    }

    // -------------------------
    // 2. 设备端内存分配
    // -------------------------
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);  // 分配输入数组显存
    cudaMalloc((void**)&d_out, size); // 分配输出数组显存

    // -------------------------
    // 3. 从主机拷贝数据到设备
    // -------------------------
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // -------------------------
    // 4. 启动内核函数
    /*选择一个block=256是经验之举
    实际上可以选择32的倍数如128.256.512
    这样代码中公式是用来计算需要多少个block以提高效率
    但这种情况在global函数编写时最后一块线程可能会越界
    */
    // -------------------------
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    SquareKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);

    // 同步等待 GPU 完成
    cudaDeviceSynchronize();

    // -------------------------
    // 5. 将结果从设备拷贝回主机
    // -------------------------
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // -------------------------
    // 6. 打印输出结果
    // -------------------------
    cout << "输入值\t平方结果" << endl;
    for (int i = 0; i < N; ++i) {
        cout << h_in[i] << "\t" << h_out[i] << endl;
    }

    // -------------------------
    // 7. 释放 GPU 显存
    // -------------------------
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
