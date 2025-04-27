# CUDA 开发文档
***
## 下载与环境配置
> 操作流程（显卡不是N卡的可以划走了）  
> 1.先安装显卡驱动  
> 2.使用*nvidia-smi*命令查看显卡驱动的版本 
> ![photo1](.\photo\photo1.png)左上角为cuda最高支持版本  
> 3.安装cuda与cuDNN
> > CUDA  
> > 1.[[点我](https://developer.nvidia.com/cuda-toolkit-archive)]前往安装页面  
> >2.下载安装（记得提前下好Visual Studio把c++部分勾选，因为下载cuda时需要借助VS的）   
> >3.环境变量应该是自动添加的所以无需操心  
> >4.判断是否安装：win+R键运行cmd，输入nvcc --version 即可查看版本号   
> 
> cuDNN
> > 1.[[点我](https://developer.nvidia.com/cudnn-downloads)]前往安装页面  
> > 2.需要先行注册，之后依据自己cuda的版本进行安装  
> >3.下载后发现其实cudnn不是一个exe文件，而是一个压缩包，解压后，有三个文件夹，把三个文件夹拷贝到cuda的安装目录下。
> >配置这四个系统变量到Path（具体根据自己的安装路径配置）
> > >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin  
> > >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include  
> > >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib  
> > >C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp  
> >
> >4.首先win+R启动cmd，cd到安装目录下的 …\extras\demo_suite,然后分别执行bandwidthTest.exe和deviceQuery.exe（进到目录后需要直接输“bandwidthTest.exe”和“deviceQuery.exe”）

## VSCODE设置

## 必要操作
### 编译与启动 '.cu' 文件
```
/*先cd到cu文件所处的文件夹
  编译语句nvcc 名字.cu -o 名字
  运行语句.\名字.exe
*/
```
GPT询问样例'（Bank Conflicts in Shared Memory请向我讲解这部分知识并给我提供一个具有详细注释的关于Test an access pattern that causes bank conflicts; measure performance impact.的完整代码以及关于Overlooking bank conflict in shared memory accesses.的错误解析）'
***
|概念 | 类比/解释|
|---|---|
|SM（Streaming Multiprocessor） | 一个“多线程计算工厂”|
|Thread Block | 一批“工人”|
|Warp | 32 个工人组成的执行小组（一起干活）|
|SIMT | 所有工人做的是同一件事，但每个人有自己的工作台（寄存器、地址）|
|GPU 执行模型 | 调度器把线程块安排给工厂，有空就安排新的任务|
|与 CPU 区别 | 没有分支预测，没有推测执行，执行顺序严格一致，但能处理海量线程并行|




