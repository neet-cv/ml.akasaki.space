# 	卷积和代码实现

```python
import torch
import numpy as np
```

## 二维互相关运算的实现

回顾之前所讲的互相关运算的大致过程：

在二维互相关运算中，卷积窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动。当卷积窗口滑动到某一位置时，窗口中的输入子数组与核数组按元素相乘并求和，得到输出数组中相应位置的元素。图5.1中的输出数组高和宽分别为2，其中的4个元素由二维互相关运算得出：

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.\\
$$

下面我们将上述过程用函数实现，它接受输入数组`X`与核数组`K`，并输出数组`Y`：

````python
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 计算X的每一项卷积运行结果
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 从左往右，从上往下计算结果
            temp = torch.sum(X[i:i + h, j:j + w] * K)
            temp.type = torch.float32
            Y[i, j] = temp
    return Y

````

我们可以构造图中的输入数组`X`、核数组`K`来验证二维互相关运算的输出：

![img](./src/try-conv-with-code/5.1_correlation.svg)


```python
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])  # 运算矩阵
K = torch.tensor([[0, 1], [2, 3]])  # 卷积核

print(corr2d(X, K))
```

输出内容：


    tensor([[19., 25.],
            [37., 43.]])

## 二维卷积运算

二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。

在Pytorch里你可以使用torch.nn中的Conv2d来便捷的完成，关于Conv2d的padding和stride等参数在下一篇中会有更加详细的讲解，这里先抛出一块砖头。当然你也可以通过阅读官方文档来进行更加深入的理解。


```python
# 构造一个输入通道数为1，输出通道数为1，核数组形状是(1, 2)的二维卷积层
conv2d = torch.nn.Conv2d(1, 1, kernel_size=(1, 2))
```

卷积窗口形状为$p \times q$的卷积层称为$p \times q$卷积层。同样，$p \times q$卷积或$p \times q$卷积核说明卷积核的高和宽分别为$p$和$q$。

## 卷积的用途

卷积运算在计算机视觉中可以将图片中的特征提取出来，比如，物体的边缘。接下来我们尝试使用卷积提取物体的边缘，即找到像素变化的位置：

首先我们构造一张$6\times 8$的图像（即高和宽分别为6像素和8像素的图像）。它中间4列为黑（0），其余为白（1）。


```python
X = torch.tensor(torch.ones((6, 8)))
X[:, 2:6] = torch.zeros(X[:, 2:6].shape)
print(X)
```

输出：


    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])

然后我们构造一个高和宽分别为1和2的卷积核`K`。当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0。


```python
K = torch.tensor([[1, -1]], dtype=torch.float32)
```

下面将输入`X`和我们设计的卷积核`K`做互相关运算。可以看出，我们将从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0。


```python
Y = corr2d(X, K)
print(Y)
```

输出：


    tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])

由此，我们可以看出，卷积层可通过重复使用卷积核有效地表征局部空间。

## 通过数据学习核数组

最后我们来看一个例子，它使用物体边缘检测中的输入数据`X`和输出数据`Y`来学习我们构造的核数组`K`。我们首先构造一个卷积层，将其卷积核初始化成随机数组。接下来在每一次迭代中，我们使用平方误差来比较`Y`和卷积层的输出，然后计算梯度来更新权重。简单起见，这里的卷积层忽略了偏差。

虽然我们之前构造了`Conv2D`类，但由于`corr2d`使用了对单个元素赋值（`[i, j]=`）的操作因而无法自动求梯度。下面我们使用`torch.nn`提供的`Conv2D`类来实现这个例子。


```python
# (samples, channels, rows, cols)
# 这里需要注意Pytroch和tf的区别
# tf2一般是把channels放在最后面
X = torch.reshape(X, (1, 1, 6, 8))
Y = torch.reshape(Y, (1, 1, 6, 7))
print(Y)


# 构造一个输入通道数为1，输出通道数为1（将在“多输入通道和多输出通道”一节介绍通道），核数组形状是(1, 2)的二维卷积层
conv2d = torch.nn.Conv2d(1, 1, kernel_size=(1, 2))

Y_hat = conv2d(X)
optimer = torch.optim.Adam(conv2d.parameters(), lr=3e-1)
for i in range(30):
    Y_hat = conv2d(X)
    loss = (abs(Y_hat - Y)) ** 2
    loss.sum().backward()
    optimer.step()
    optimer.zero_grad()

    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, torch.sum(loss)))
```

输出：

    batch 2, loss 12.860
    batch 4, loss 3.923
    batch 6, loss 2.595
    batch 8, loss 0.327
    batch 10, loss 1.478
    batch 12, loss 1.980
    batch 14, loss 1.928
    batch 16, loss 1.929
    batch 18, loss 0.726
    batch 20, loss 0.582
    batch 22, loss 0.073
    batch 24, loss 0.253
    batch 26, loss 0.421
    batch 28, loss 0.428
    batch 30, loss 0.477


可以看到，10次迭代后误差已经降到了一个比较小的值。现在来看一下学习到的核数组。


```python
print(conv2d._parameters)
```

输出：


    OrderedDict([('weight', Parameter containing:
    tensor([[[[ 1.0212, -1.0267]]]], requires_grad=True)), ('bias', Parameter containing:
    tensor([0.0048], requires_grad=True))])

可以看到，学到的核数组与我们之前定义的核数组`K`较接近。

