/*编译语句nvcc Day1.cu -o Day1
  运行语句.\Day1.exe
*/




#include <stdio.h>

// 设备端的 kernel 函数
__global__ void helloFromGPU() {
    // 只从一个线程打印

        printf("Hello from GPU thread!\n");

}

int main() {
    // 在 GPU 上启动 3 个 block，2 个线程
    helloFromGPU<<<3, 2>>>();

    // 等待 GPU 执行完成
    cudaDeviceSynchronize();

    return 0;
}

