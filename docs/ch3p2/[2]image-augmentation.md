# 图像增广

## 概念引入

近年来，深度学习技术深刻推动了计算机视觉系统性能的提升。可以说，当下最先进的计算机视觉应用几乎离不开深度学习。在“卷积神经网络”中已经介绍了计算机视觉领域常使用的深度学习模型，并实践了简单的图像分类任务。近几年深度学习的大规模成功应用主要的就是得益于数据的累积和算例的提升，虽然近几年很多研究者竭力的攻克半监督和无监督学习，减少对大量数据的依赖，但是目前数据在计算机视觉、自然语言处理等人工智能技术领域依然占据着非常重要的地位。甚至可以说，**大规模的数据是计算机视觉成功应用的前提条件**。但是由于种种原因导致数据的采集变的十分困难，因此图像增广技术就在数据的准备过程中占据着举足轻重的角色。**图像增广的目的就是在数据准备阶段进行处理，提高模型的泛化能力**。

---

图像增广（image augmentation）技术通过**对训练图像做一系列随机改变**，来产生相似但又不同的训练样本，从而扩大训练数据集的规模。图像增广的另一种解释是，随机改变训练样本可以降低模型对某些属性的依赖，从而提高模型的泛化能力。常见的图像增广技术有：

- 简单几何变换：镜像变换、旋转、缩放、剪裁、平移
- 表征特征变换：亮度修改、添加噪声、剪切、变换颜色

在**图像增广过程中可以使用其中一种手段进行扩充，也可以使用其中的几种方法进行组合使用**。

---

## 准备

使用python实现图像增广技术较为简单。我们将会使用以下包：

```python
import matplotlib
import cv2
import numpy as np
```

在这里首先对一些常用的函数进行说明：

如果你对matplotlib和cv2的一些基本函数不是很熟悉，可以先参阅[工地matplotlib](../appendix-1/[5]introducing-matplotlib)。

接下来我们准备一段可以用来显示一些图片的函数（这个函数在[总是用到的代码块](../appendix-1/[4]similar-codeblocks)有记录。）：

```python
import matplotlib.pyplot as plt
import math
def check_image_in_dataset_via_plot(dataset, how_many: float = 36., gray_scale=False):
    plt_width = math.ceil(how_many ** .5)
    plt.figure(figsize=(plt_width, plt_width))
    for i in range(round(how_many)):
        plt.subplot(plt_width, plt_width, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if not gray_scale:
            plt.imshow(dataset[i])
        else:
            plt.imshow(dataset[i], cmap=plt.cm.binary)
    plt.show()
```

## 开始

我们将会试着实现一些简单的图像增广。

### 1. 翻转和剪裁



### 2. 改变颜色