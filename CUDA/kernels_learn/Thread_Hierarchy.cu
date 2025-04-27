// CUDA 矩阵加法（使用二维线程）
/*编译语句nvcc Thread_Hierarchy.cu -o Thread_Hierarchy
  运行语句.\Thread_Hierarchy.exe
*/

//一个线程快最多包括1024个线程
#include <iostream>
using namespace std;

// 定义矩阵大小 N × N
#define N 4  // 可以改成其他数值，如 16、32、1024

// GPU 上运行的 kernel 函数，负责执行矩阵加法
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;  // 当前线程在 block 中的 x 坐标（对应行）
    int j = threadIdx.y;  // 当前线程在 block 中的 y 坐标（对应列）

    // 每个线程负责计算一个元素
    C[i][j] = A[i][j] + B[i][j];
    printf("GPU thread (%d,%d): C[%d][%d] = %f\n", i, j, i, j, C[i][j]);
}

int main()
{
    // 主机端定义三个矩阵（h_A, h_B, h_C），N × N 大小
    float h_A[N][N], h_B[N][N], h_C[N][N];

    // 初始化输入矩阵 A 和 B（主机端）
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_A[i][j] = i + j;           // 举例：对角线为 0,2,4,...
            h_B[i][j] = i * j + 1;       // 举例：含乘法变化
        }
    }

    // 定义设备端指针
    float (*d_A)[N], (*d_B)[N], (*d_C)[N];

    size_t bytes = N * N * sizeof(float);

    // 在设备端申请内存（注意是二维数组，需要特殊处理）
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // 将主机端数据拷贝到设备端
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 设置线程组织方式：一个 block，block 中 N × N 个线程
    dim3 threadsPerBlock(N, N); // 每个 block 是 N × N 的二维线程网格
    /*dim3 是 CUDA 提供的一个用于表示 一维、二维或三维维度信息的结构体，它可以用来定义：
    一个线程块（block）中线程的布局（threads per block）
    网格中线程块的布局（blocks per grid）
    dim3 threadsPerBlock(4, 3); // 表示一个 block 内部是 4 行 3 列的线程，总共 4×3=12 个线程
    */
    int numBlocks = 1;

    // 启动 kernel 函数：执行矩阵加法
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // 等待 GPU 执行完毕
    cudaDeviceSynchronize();

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 在主机端输出加法结果矩阵
    cout <<"\n主机输出矩阵加法结果 C = A + B:"<< endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            cout << h_C[i][j] << "\t";
        }
        cout << endl;
    }

    // 释放设备端内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
