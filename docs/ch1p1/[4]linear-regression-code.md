# 线性回归代码实现

在了解了线性回归的背景知识之后，现在我们可以动手实现它了。尽管强大的深度学习框架可以减少大量重复性工作，但若过于依赖它提供的便利，会导致我们很难深入理解深度学习是如何工作的。因此，本节将介绍如何只利用`Tensor`和`Optim`来实现一个线性回归的训练。

首先，导入本节中实验所需的包或模块，其中的matplotlib包可用于作图，且设置成嵌入显示。

``` python 
%matplotlib inline
import torch
print(torch.__version__)
from matplotlib import pyplot as plt
import random
```

    1.7.0+cu110

## 手动实现 

### 生成数据集

我们构造一个简单的人工训练数据集，它可以使我们能够直观比较学到的参数和真实的模型参数的区别。设训练数据集样本数为1000，输入个数（特征数）为2。给定随机生成的批量样本特征 $\boldsymbol{X} \in \mathbb{R}^{1000 \times 2}$，我们使用线性回归模型真实权重 $\boldsymbol{w} = [2, -3.4]^\top$ 和偏差 $b = 4.2$，以及一个随机噪声项 $\epsilon$ 来生成标签
$$
\boldsymbol{y} = \boldsymbol{X}\boldsymbol{w} + b + \epsilon
$$

其中噪声项 $\epsilon$ 服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰。下面，让我们生成数据集。

``` python
num_inputs = 2
num_examples = 1000
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features = torch.randn(num_examples, num_inputs)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.normal(0, 0.01, size=labels.size())
```

注意，`features`的每一行是一个长度为2的向量，而`labels`的每一行是一个长度为1的向量（标量）。

``` python
print(features[0], labels[0])
```

输出：

```
tensor([0.7686, 1.5004]) tensor(0.6350)
```

通过生成第二个特征`features[:, 1]`和标签 `labels` 的散点图，可以更直观地观察两者间的线性关系。

``` python
def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1], labels, 1)
plt.show()
```

![img](./src/linear-regression-code/3.2_output1.png)


### 读取数据

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回`batch_size`（批量大小）个随机样本的特征和标签。

``` python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i: min(i+batch_size, num_examples)]
        yield tf.gather(features, axis=0, indices=j), tf.gather(labels, axis=0, indices=j)
```

让我们读取第一个小批量数据样本并打印。每个批量的特征形状为(10, 2)，分别对应批量大小和输入个数；标签形状为批量大小。

``` python
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```

输出：

    tensor([[ 0.6154,  0.6066],
            [-0.7462, -2.0108],
            [ 1.3828, -1.7290],
            [-0.7054,  1.1876],
            [-0.7499, -0.4315],
            [ 2.2159, -1.2673],
            [-1.4601,  0.4726],
            [-0.4133,  0.0934],
            [-0.1078,  1.0444],
            [ 0.8082,  0.4850]]) tensor([ 3.3624,  9.5424, 12.8516, -1.2418,  4.1639, 12.9290, -0.3342,  3.0615,
             0.4314,  4.1688])


###  初始化模型参数

我们将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

``` python
w = tf.Variable(tf.random.normal((num_inputs, 1), stddev=0.01))
b = tf.Variable(tf.zeros((1,)))
```

### 定义模型

下面是线性回归的矢量计算表达式的实现。我们使用`matmul`函数做矩阵乘法。

``` python
def linreg(X, w, b):
    return tf.matmul(X, w) + b
```

### 定义损失函数

我们使用上一节描述的平方损失来定义线性回归的损失函数。在实现中，我们需要把真实值`y`变形成预测值`y_hat`的形状。以下函数返回的结果也将和`y_hat`的形状相同。

``` python
def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 /2
```

### 定义优化算法

以下的`sgd`函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。

``` python
lr = 0.0001
sgd = torch.optim.SGD([w, b], lr=lr)
```

### 训练模型

在训练中，我们将多次迭代模型参数。在每次迭代中，我们根据当前读取的小批量数据样本（特征`X`和标签`y`），通过调用反向函数`t.gradients`计算小批量随机梯度，并调用优化算法`sgd`迭代模型参数。由于我们之前设批量大小`batch_size`为10，每个小批量的损失`l`的形状为(10, 1)。回忆一下自动求梯度一节。由于变量`l`并不是一个标量，所以我们可以调用`reduce_sum()`将其求和得到一个标量，再运行`t.gradients`得到该变量有关模型参数的梯度。注意在每次更新完参数后不要忘了将参数的梯度清零。

在一个迭代周期（epoch）中，我们将完整遍历一遍`data_iter`函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。虽然迭代周期数设得越大模型可能越有效，但是训练时间可能过长。而有关学习率对模型的影响，我们会在后面“优化算法”一章中详细介绍。

``` python
num_epochs = 15
net = linreg
loss = squared_loss

w.requires_grad = True
b.requires_grad = True

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = torch.sum(loss(net(X, w, b), y))
        l.backward()
        sgd.step()
        sgd.zero_grad()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, torch.mean(train_l)))
```

输出：

```
epoch 1, loss 14.375875
epoch 2, loss 11.591846
epoch 3, loss 9.347400
epoch 4, loss 7.537946
epoch 5, loss 6.079062
epoch 6, loss 4.902782
epoch 7, loss 3.954329
epoch 8, loss 3.189521
epoch 9, loss 2.572775
epoch 10, loss 2.075413
epoch 11, loss 1.674267
epoch 12, loss 1.350752
epoch 13, loss 1.089808
epoch 14, loss 0.879323
epoch 15, loss 0.709535
```

训练完成后，我们可以比较学到的参数和用来生成训练集的真实参数。它们应该很接近。

``` python
print(true_w, w)
print(true_b, b)
```

输出：

```
tensor([ 2.0000, -3.4000]) tensor([ 1.4737, -2.5808], requires_grad=True)
4.2 tensor([3.1709], requires_grad=True)
```



## 使用框架实现

随着深度学习框架的发展，开发深度学习应用变得越来越便利。实践中，我们通常可以用比上一节更简洁的代码来实现同样的模型。在本节中，我们将介绍如何使用Pytorch更方便地实现线性回归的训练。

### 生成数据集

我们生成与上一节中相同的数据集。其中`features`是训练数据特征，`labels`是标签。

```python
import tensorflow as tf

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal(shape=(num_examples, num_inputs), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)
```

### 读取数据

虽然Pytorch对于线性回归可以直接拟合，不用再划分数据集，但我们仍学习一下读取数据的方法


```python
# 继承Dataset类
class MyData(Dataset):
    def __init__(self, feathers, labels):
        super(MyData, self).__init__()
        self.feathers = feathers
        self.labels = labels

    def __getitem__(self, item):
        # 返回数据和标签
        return self.feathers[item], self.labels[item]

    def __len__(self):
        return len(labels)


data = MyData(features, labels)
data_loader = DataLoader(
    dataset=data,
    batch_size=batch_size,
    shuffle=True
)
```

上述方法是一种通用的自定义数据的方法。通过继承Pytorch的Dataset类，你可以灵活的在`__getitem__`方法中编写需要的方法。

```python
for X, y in data_loader:
    print(X, y)
    break
```

    tensor([[ 0.9187, -1.0823],
            [ 0.7477,  0.3213],
            [-1.1145, -0.8220],
            [ 0.3883, -0.8678],
            [ 0.2935,  0.0016],
            [-0.4807,  0.0705],
            [-0.8014, -1.1751],
            [-1.0711, -0.7012],
            [ 0.3632, -1.0502],
            [ 0.2089, -0.8883]]) tensor([9.7233, 4.6026, 4.7678, 7.9202, 4.7875, 3.0168, 6.5926, 4.4453, 8.5026,
            7.6511])

### 定义模型和初始化参数

`Pytorch`中有非常多种的模型定义方法，如果只是定义个非常小型的网络我推荐使用`torch.nn.Sequential()`。我们先定义一个模型变量`model`，它继承于`nn.Module`。在`Pytorch`中，`Sequential`实例可以看作是一个串联各个层的容器。在构造模型时，我们可以在该容器添加层。当给定输入数据时，容器中的每一层将依次推断下一层的输入尺寸。

```python
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        x = self.base(x)
        return x
```

### 定义损失函数

`Pytorch`在`torch.nn`模块中提供了各种损失函数和自定义损失函数的基类，并直接使用它的均方误差损失作为模型的损失函数。

```python
loss = torch.nn.MSELoss()
```

###  定义优化算法

同样，我们也无须自己实现小批量随机梯度下降算法。`torch.optim` 模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。下面我们创建一个用于优化model 所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法。

```python
trainer = torch.optim.SGD(model.parameters(), lr=0.001)
```

### 训练模型

在使用`Pytorch`训练模型时，每一次计算Pytorch都会自动的创建动态计算图，我们通过 `trainer.step()` 来更新权重，并用 `trainer.zero_grad()` 清空当前计算图，反正在迭代训练的过程中权重累加。

```python
num_epochs = 15
for epoch in range(1, num_epochs + 1):
    for (batch, (X, y)) in enumerate(data_loader):
        l = loss(model(X), y.unsqueeze(-1))
        l.backward()
        trainer.step()
        trainer.zero_grad()
    with torch.no_grad():
        l = loss(model(features), labels.unsqueeze(-1))
        print('epoch %d, loss: %f' % (epoch, l))
```

    epoch 1, loss: 28.551250
    epoch 2, loss: 19.325567
    epoch 3, loss: 13.082422
    epoch 4, loss: 8.857162
    epoch 5, loss: 5.997283
    epoch 6, loss: 4.061272
    epoch 7, loss: 2.750603
    epoch 8, loss: 1.863120
    epoch 9, loss: 1.262159
    epoch 10, loss: 0.855155
    epoch 11, loss: 0.579481
    epoch 12, loss: 0.392732
    epoch 13, loss: 0.266208
    epoch 14, loss: 0.180479
    epoch 15, loss: 0.122382



下面我们分别比较学到的模型参数和真实的模型参数。我们可以通过model的`parameters()`来获得其权重（`weight`）和偏差（`bias`）。学到的参数和真实的参数很接近。


```python
# model.parameters() 返回的是一个迭代器
print(true_w, next(model.parameters()))
```

    [2, -3.4] Parameter containing:
    tensor([[ 1.9263, -3.1823]], requires_grad=True)


