#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

const int N = 1 << 20; // 1M elements实际上是将1向左移动20位这样二进制表示十进制的2^20=1048576

// ✅ 合并内存访问
__global__ void coalesced_access(float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        output[i] = input[i] * 2.0f; // 连续访问 input[i]
}

// ❌ 非合并内存访问
__global__ void non_coalesced_access(float *input, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        output[i] = input[i * 32] * 2.0f; // 线程访问不连续，间隔跳跃
}

void benchmark_kernel(void (*kernel)(float*, float*), const char* name) {
    float *d_in, *d_out;
    float *h_in = new float[N];
    float *h_out = new float[N];

    for (int i = 0; i < N; i++)
        h_in[i] = static_cast<float>(i);

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 启动内核
    kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out);
    cudaDeviceSynchronize();

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("%s: %.3f ms\n", name, elapsed.count());

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 可选：验证输出正确性
    // for (int i = 0; i < 10; i++) printf("%.1f ", h_out[i]);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
}

int main() {
    benchmark_kernel(coalesced_access, "✅ Coalesced Access");
    benchmark_kernel(non_coalesced_access, "❌ Non-Coalesced Access");
    return 0;
}
