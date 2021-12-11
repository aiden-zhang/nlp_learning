CUDA cuDNN TensorRt区别

CUDA是英伟达针对自家GPU推出的计算框架，通过该框架提供的API，用户可以很方便的使用GPU进行大规模的并行计算。

cuDNN是NVIDIA打造的针对深度神经网络的GPU加速库，通过它可以将深度网络模型进行一定的优化后再通过CUDA送入GPU进行运算，这比自己调用CUDA运算效率更高。

TensorRt是NVIDIA针对自己平台推出的模型加速包，一般只针对模型推理过程，因此一般是部署时用它来加速模型的运行速度，TensorRT主要做了两件事来优化模型推理速度

可参考：

https://www.cnblogs.com/zhibei/p/12331292.html

①降低精度 float32 int32 推理时可以转换成float16 int8

②对网络结构进行重构，根据GPU的特性做了优化，主要对模型进行垂直合并和水平合并

​		垂直合并：把主流深度神经网络的Cov BN Relu三层融合为一层

​		水平合并：将垂直合并后的有相同维度的张量的层再进行融合

