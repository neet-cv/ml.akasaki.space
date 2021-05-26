# Convolutional Block Attention Module

### 这篇笔记的写作者是[AsTheStarsFall](https://github.com/asthestarsfalll)。 

> 论文名称：[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
>
> 作者：Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon，Korea Advanced Institute of Science and Technology, Daejeon, Korea

## 摘要

- CBAM（Convolutional Block Attention Moudule)是一种简单有效的[前馈](https://www.cnblogs.com/samshare/p/11801806.html)卷积神经网络注意力模块。 
- 该模块为混合域注意力机制（）从通道和空间两个方面依次推断attention map。
- CBAM是一个轻量级的通用模块，可以无缝集成到任何CNN中。

**关键词:物体识别，注意机制，门控卷积**

## 介绍

- 卷积神经网络(CNNs)基于其丰富的表达能力显著提高了视觉任务的性能，目前的主要关注网络的三个重要因素：**深度，宽度和基数**（Cardinality）。
- 从LeNet到残差网络，网络变的更加深入，表达形式更加丰富；GoogLeNet表明宽度是提高模型性能的另一个重要因素；Xception和ResNext则通过增加网络的**基数**，在节省参数的同时，来获得比深度、宽度更强的表达能力（引用于ResNext论文）。
- 除了这些因素之外，本文考察了与网络结构设计不同的方面——注意力。

## 注意力机制

- 注意（attention）在人类感知中起着重要的作用。人类视觉系统的一个重要特性是，它不会试图一次性处理整个场景，而是利用一系列的局部一瞥（a sequence of partial glimpses）来获得对显著部分的关注。
- 近年来有一些尝试将注意力机制加入CNN中，如使用Endcoder-Decoder结构注意力模块的Residual Attention Network，使用“Squeeze-Excitation“模块的SEnet。
- 具体可见[注意力机制](https://asthestarsfalll.icu/2021/05/12/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/)

## CBAM

### 整体结构

![image-20210512124618161](https://gitee.com/Thedeadleaf/images/raw/master/image-20210512124618161.png)

- CBAM在混合域（通道域、空间域）上引入注意力机制，拥有更强的表达能力；

- 整个过程可以被概括为：
  $$
  F’=M_c(F)\otimes F\\
  F''=M_s(F')\otimes F'
  $$
  其中$F$为模块的输入，$M_c、M_s$表示通道注意力图和空间注意力图，$\otimes$表示element-wise multiply，在具体的实现过程中会相应的进行广播。

### Channel attention module

![image-20210512130136918](https://gitee.com/Thedeadleaf/images/raw/master/image-20210512130136918.png)

利用通道之间的关系生成通道注意图（channel attention map），具体可见[注意力机制](https://asthestarsfalll.icu/2021/05/12/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/)

通道注意力主要关注的是图像的“什么”更有意义。

实现过程：

1. 对input进行**全局平均池化**和**全局最大池化**来聚集空间内的信息;
2. 通过一个**共享的**MLP(多层感知机)(具体实现可用1X1卷积层),为了减少参数隐藏层的通道数被设置为$\frac{C}{R}$，在第一层之后设置了ReLU函数来引入非线性（类似于SENet，这种结构出现在各种网络之中，作用之一是为了减少参数和计算量，作用之二是为了获得更多的非线性）；
3. 对应求和之后经过一个ReLU层得到最终的Channel attention map
4. 将其与input相乘（会自动进行广播）。

代码复现：

```python
class Channel_module(nn.Module):
    def __init__(self,  in_ch, ratio=16):
        super(Channel_module, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch//ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_ch//ratio, in_ch, 1, bias=False)

    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.global_avg(x))))
        m = self.fc2(self.relu(self.fc1(self.global_avg(x))))
        attention = F.sigmoid(a+m)
        return x*attention
```

### Spatial attention module

![image-20210512133601683](https://gitee.com/Thedeadleaf/images/raw/master/image-20210512133601683.png)

利用空间之间的关系来生成空间注意力图（spatial attention map）

空间注意力主要关注“哪里”有重要信息，与通道注意力相辅相成。

实现过程：

1. 在通道维度上分别进行平均池化和最大池化，然后进行concat；
2. 经过一个7X7的卷积层，将通道数降为1；
3. Sigmoid函数；
4. 与inputs也就是上一层的Channel-refined feature对应相乘。

代码复现：

```python
class Spatial_module(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_module, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3,
                              bias=False)  # 使用padding保持大小不变

    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)  # 沿着channel维度计算均值和最大值
        m, _ = torch.max(x, dim=1, keepdim=True)
        cx = torch.cat([a, m], dim=1)
        cx = F.sigmiod(self.conv(cx))
        return cx*x
```

### Arrangement of attention modules.

以上两个注意力模块计算互补的注意力，考虑到这一点，这两个模块可以并行或是顺序排列。实验表明，顺序排列比并行排列效果更好，其中通道优先顺序略好于空间优先顺序。

## Ablation studies（消融研究）

作者团队首先寻找计算通道注意的有效方法，然后是空间注意。最后，我们考虑如何结合通道关注模块和空间关注模块。

### Channel attention

作者团队比较了3种不同的通道注意力:平均池化、最大池化和两种池化的联合使用。

![image-20210512160839407](https://i.loli.net/2021/05/12/kFItTg6yD1u5joJ.png)

可以看到，最大池化与平均池化同样重要，而SENet忽略了最大池化的重要性。

**对显著部分进行编码的最大池化特征可以补偿对全局信息软编码的平均池化特征**。

在空间注意力的研究当中，将直接使用最大池化特征和平均池化特征，并将R设置为16。

### Spatial attention

作者团队考虑了两种空间注意力的方案：**一是使用通道维度上的平均池化和最大池化，二是使用1X1卷积进行降维**。此外还研究了3X3和7X7卷积核的影响。在实验当中，将空间注意力模块置于通道注意力模块之后。

![image-20210512162421957](https://gitee.com/Thedeadleaf/images/raw/master/image-20210512162421957.png)

可以看到，通道池化的效果更好，同时，使用较大的核会产生更好的精度，这意味着需要一个更大的感受野来决定空间上的重要区域。

### Arrangement of the channel and spatial attention.

作者团队考虑了三种不同的模块安排方案：通道优先，空间优先和并行。

![image-20210512163932604](https://gitee.com/Thedeadleaf/images/raw/master/image-20210512163932604.png)

可以看到，通道优先的效果更好。

### 最终效果

![image-20210512164218849](https://gitee.com/Thedeadleaf/images/raw/master/image-20210512164218849.png)

