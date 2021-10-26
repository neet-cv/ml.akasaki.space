轻量级Trick的优化组合。

这篇笔记的写作者是[AsTheStarsFall](https://github.com/asthestarsfalll)。

> 论文名称：[PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf)
>
> 作者：Cheng Cui, Tingquan Gao, Shengyu Wei,Yuning Du...
>
> Code：https://github.com/PaddlePaddle/PaddleClas

## 摘要

1. 总结了一些在延迟（latency）几乎不变的情况下精度提高的技术；
2. 提出了一种基于MKLDNN加速策略的轻量级CPU网络，即PP-LCNet。

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007133525281.png" alt="image-20211007133525281" style="zoom:50%;" />

## 介绍

目前的轻量级网络在启用[MKLDNN](https://github.com/oneapi-src/oneDNN)的Intel CPU上速度并不理想，考虑了一下三个基本问题：

1. 如何促使网络学习到更强的特征，但不增加延迟？
2. 在CPU上提高轻量级模型精度的要素是什么？
3. 如何有效地结合不同的策略来设计CPU上的轻量级模型？

## Method

PP-LCNet使用深度可分离卷积作为基础结构，构建了一个类似MobileNetV1的BaseNet，并在其基础上结合现有的技术，从而得到了PP-LCNet

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007191050134.png" alt="image-20211007191050134" style="zoom:67%;" />

参数配置：

<img src="C:\Users\11864\AppData\Roaming\Typora\typora-user-images\image-20211008131450331.png" alt="image-20211008131450331" style="zoom: 67%;" />

### Better activation function

激活函数是神经网络中非线性的来源，因此其质量一定程度上决定着网络的表达能力。

当激活函数由Sigmoid变为ReLU时，网络的性能得到了很大的提升，近来出现了很多超越ReLU的激活函数，如EfficientNet的Swish，MobileNetV3中将其升级为HSwish，避免了大量的指数运算；因此本网络中使用HSwish激活函数代替ReLU。

首先让我们看一下ReLU函数的近似推导：
$$
\begin{align}
f(x)&=\sum_{i=1}^{\infin}\sigma(x-i+0.5) &(stepped\ sigmoid)\\
&\approx log(1+e^x)  &(softplus\ function)\\
&\approx max(0,N(0,1)) &(ReLU\ function)
\end{align}
$$
<img src="https://gitee.com/Thedeadleaf/images/raw/master/606386-20180502160705206-923153087.png" alt="softplus" style="zoom: 80%;" />

出于计算量的考虑和实验验证选择了ReLU

ReLU6：

增加了上界

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007184950617.png" alt="image-20211007184950617" style="zoom: 50%;" />

[Swish](https://www.cnblogs.com/makefile/p/activation-function.html)：
$$
f(x)=x\cdot sigmoid(\beta x)
$$
<img src="https://gitee.com/Thedeadleaf/images/raw/master/606386-20171102101521763-698600913.png" alt="swish" style="zoom:67%;" />

$\beta$是个常数或可训练的参数。Swish 具备无上界有下界、平滑、非单调的特性。
Swish 在深层模型上的效果略优于 ReLU。仅仅使用 Swish 单元替换 ReLU 就能把 Mobile NASNetA 在 ImageNet 上的 top-1 分类准确率提高 0.9%，Inception-ResNet-v 的分类准确率提高 0.6%。

导数：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/606386-20171102101538013-1397340773.png" alt="swish-derivation" style="zoom: 67%;" />



当$β = 0$时,Swish变为线性函数$f(x)=\frac x 2$
当$β → ∞$, $\sigma(x)=\frac{1}{1+e^{−x}}$为0或1，这时Swish变为$ReLU(x)=max(0,x)$
因此Swish函数可以看做是介于线性函数与ReLU函数之间的平滑函数.

HSwish：

Swish函数的计算量是很大的，因此提出了HSwish，H表示Hard，意味着超过某个范围，激活值为常数

对ReLU6除以6再向左平移三个单位可以得到HSigmoid：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007185106870.png" alt="image-20211007185106870" style="zoom: 50%;" />

HSwish的近似公式为$x\cdot h\sigma(x)=\frac{relu6(x+3)}{6}$，图像如下：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007185403938.png" style="zoom:50%;" />

### SE modules at appropriate positions

注意力模块无疑是轻量级网络完美的选择，本文探究了SE模块放置的位置，发现在网络深层的效果较好。

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007190945690.png" alt="image-20211007190945690" style="zoom: 67%;" />

###  Larger convolution kernels

使用更大的卷积核尺寸，发现在网络深层效果较好

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007191200660.png" alt="image-20211007191200660" style="zoom: 67%;" />

### Larger dimensional1×1conv layer after GAP

在网络最后的GAP之后使用Pointwise卷积进行升维，以此提高网络的性能

### Drop out

实验发现drop out可以提高性能

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007191715602.png" alt="image-20211007191715602" style="zoom: 67%;" />

## 实验结果

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007192557037.png" alt="image-20211007192557037" style="zoom: 50%;" />

与其他网络进行对比

<img src="https://gitee.com/Thedeadleaf/images/raw/master/image-20211007192635596.png" alt="image-20211007192635596" style="zoom: 50%;" />