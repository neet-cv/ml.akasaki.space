# softmax回归的代码实现

这一节我们来动手实现softmax回归。首先导入本节实现所需的包或模块。

``` python
import torch
import numpy as np
print(torch.__version__)
```

输出：

```
1.7.0+cu110
```

## 代码实现

这部分版本的代码和tf版本的不太一样，我将手动实现和代码实现放在了一块写。下面的代码大部分还是基于框架进行实现的。

### 获取和读取数据

我们将使用Fashion-MNIST数据集，并设置批量大小为256。

``` python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 256
# 训练集加载器
train_loader = DataLoader(
    datasets.FashionMNIST('./fashionmnist_data/',
                          train=True,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),  # 将图片转化为tensor类型
                              transforms.Normalize((0.1307,), (0.3081,))  # 将图片像素值标准化
                          ])),
    batch_size=batch_size,
    shuffle=True)

# 测试集加载器
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True, )

```

### 初始化模型参数

跟线性回归中的例子一样，我们将使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是 $28 \times 28 = 784$：该向量的每个元素对应图像中每个像素。由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为$784 \times 10$和$1 \times 10$的矩阵。`Variable`来标注需要记录梯度的向量。

``` python
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=torch.zeros((num_inputs, num_outputs)).size(), dtype=torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)
```

### 实现 softmax 运算

在介绍如何定义 softmax 回归之前，我们先描述一下对如何对多维`Tensor`按维度操作。在下面的例子中，给定一个`Tensor`矩阵`X`。我们可以只对其中同一列（`axis=0`）或同一行（`axis=1`）的元素求和，并在结果中保留行和列这两个维度（`keepdims=True`）。

``` python
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(torch.sum(X, dim=0, keepdim=True), torch.sum(X, dim=1, keepdim=True))
```

输出：

```
tensor([[5, 7, 9]]) tensor([[ 6],
        [15]])
```

下面我们就可以定义前面小节里介绍的softmax运算了。在下面的函数中，矩阵`logits`的行数是样本数，列数是输出个数。为了表达样本预测各个输出的概率，softmax运算会先通过`exp`函数对每个元素做指数运算，再对`exp`矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。这样一来，最终得到的矩阵每行元素和为1且非负。因此，该矩阵每行都是合法的概率分布。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。

``` python
def softmax(logits, dim=-1):
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=dim, keepdim=True)
```

可以看到，对于随机输入，我们将每个元素变成了非负数，且每一行和为1。

``` python
X = torch.randn(2, 5)
X_prob = softmax(X)
print(X_prob, torch.sum(X_prob, dim=1))
```

输出：

```
tensor([[0.1631, 0.3456, 0.2960, 0.1330, 0.0623],
        [0.0092, 0.4641, 0.0163, 0.0520, 0.4583]]) tensor([1., 1.])
```

###  定义模型

有了softmax运算，我们可以定义上节描述的softmax回归模型了。这里通过`reshpe`函数将每张原始图像改成长度为`num_inputs`的向量。

``` python
def net(X):
    logits = torch.matmul(torch.reshape(X, shape=(-1, W.shape[0])), W) + b
    return softmax(logits)
```

### 定义损失函数

上一节中，我们介绍了softmax回归使用的交叉熵损失函数。为了得到标签的预测概率，我们可以使用`masked_select`函数和`one_hot`函数。在下面的例子中，变量`y_hat`是2个样本在3个类别的预测概率，变量`y`是这2个样本的标签类别。与3.4节（softmax回归）数学表述中标签类别离散值从1开始逐一递增不同，在代码中，标签类别的离散值是从0开始逐一递增的。

``` python
import torch.nn.functional as F

y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.tensor([0, 2], dtype=torch.int64) # torch.one_hot方法要求y是int64类型
mask = F.one_hot(torch.tensor(y), num_classes=3)
mask = mask.type(torch.bool) # masked_select 要求掩码必须是bool类型的
print(torch.masked_select(y_hat, mask))
```

输出：

```
tensor([0.1000, 0.5000])
```

下面实现了3.4节（softmax回归）中介绍的交叉熵损失函数。Pytorch有内置的交叉熵损失函数，下面直接使用了内置的损失函数。

``` python
def cross_entropy(y_hat, y):
    return F.cross_entropy(y_hat, y)
```

### 计算分类准确率

给定一个类别的预测概率分布`y_hat`，我们把预测概率最大的类别作为输出类别。如果它与真实类别`y`一致，说明这次预测是正确的。分类准确率即正确预测数量与总预测数量之比。

为了演示准确率的计算，下面定义准确率`accuracy`函数。其中`torch.argmax(y_hat, dim=1)`返回矩阵`y_hat`每行中最大元素的索引，且返回结果与变量`y`形状相同。相等条件判断式`(torch.argmax(y_hat, dim=1) == y)`是一个数据类型为`bool`的`Tensor`，实际取值为：0（相等为假）或 1（相等为真）。

``` python
def accuracy(y_hat, y):
    class_ = torch.argmax(y_hat, dim=1)
    class_ = class_ == y
    class_ = class_.type(torch.float)
    return torch.mean(class_)
```

让我们继续使用在演示`masked_select`函数时定义的变量`y_hat`和`y`，并将它们分别作为预测概率分布和标签。可以看到，第一个样本预测类别为2（该行最大元素0.6在本行的索引为2），与真实标签0不一致；第二个样本预测类别为2（该行最大元素0.5在本行的索引为2），与真实标签2一致。因此，这两个样本上的分类准确率为0.5。

``` python
print(accuracy(y_hat, y))
```

输出：

```
tensor(0.5000)
```

类似地，我们可以评价模型`net`在数据集`data_iter`上的准确率。注：这里的net和线性回归那节的代码基本相同，我们只需要修改一下输入的大小即可。`net`的定理如下：

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = nn.Sequential(nn.Linear(28 * 28, 10))

    def forward(self, x):
        x = self.base(x)
        return x
```

测试准确度的代码如下：

``` python
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        X = torch.reshape(X, shape=[X.size(0), 28 * 28])  # 将图片打平后输入到网络当中
        y = y.type(torch.int64)
        pre_class = torch.argmax(net(X), dim=1)  # 计算网络的预测类别
        pre_class = pre_class.type(torch.int64) == y
        acc_sum += torch.sum(pre_class.type(torch.float))
        n += y.size(0)
    return acc_sum / n
```

因为我们随机初始化了模型`net`，所以这个随机模型的准确率应该接近于类别个数 10 的倒数即 0.1。

``` python
fashion_mnist_net = Net()
print(evaluate_accuracy(test_loader, fashion_mnist_net))
```

输出：

```
tensor(0.1424)
```

###  训练模型

训练softmax回归的实现跟 3.2（线性回归的从零开始实现）一节介绍的线性回归中的实现非常相似。我们同样使用小批量随机梯度下降来优化模型的损失函数。在训练模型时，迭代周期数`num_epochs`和学习率`lr`都是可以调的超参数。改变它们的值可能会得到分类更准确的模型。

``` python
num_epochs, lr = 10, 0.1

def train_ch3(fusion_mnist_net, train_iter, test_iter, loss, num_epochs, trainer=None):
    trainer = torch.optim.SGD(fashion_mnist_net.parameters(), lr)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = torch.reshape(X, shape=[X.size(0), 28 * 28])  # 将图片打平后输入到网络当中
            y_hat = fusion_mnist_net(X)
            # 计算交叉熵loss
            l = torch.sum(loss(y_hat, y))
            l.backward()
            trainer.step()
            trainer.zero_grad()
            # 计算training loss
            y = y.type(torch.float)
            train_l_sum += l
            # 计算training acc
            y = y.type(torch.int64)
            pred_classes = torch.argmax(y_hat, dim=1)
            train_acc = y == pred_classes
            train_acc = train_acc.type(torch.float)
            train_acc_sum += torch.sum(train_acc)
            n += y.size(0)
        test_acc = evaluate_accuracy(test_iter, fusion_mnist_net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


trainer = torch.optim.SGD(fashion_mnist_net.parameters(), lr)
train_ch3(fashion_mnist_net, train_loader, test_loader, cross_entropy, num_epochs, trainer)
```

输出：

```
epoch 1, loss 0.0048, train acc 0.593, test acc 0.634
epoch 2, loss 0.0032, train acc 0.729, test acc 0.697
epoch 3, loss 0.0028, train acc 0.765, test acc 0.752
epoch 4, loss 0.0026, train acc 0.781, test acc 0.776
epoch 5, loss 0.0025, train acc 0.790, test acc 0.761
epoch 6, loss 0.0025, train acc 0.798, test acc 0.674
epoch 7, loss 0.0024, train acc 0.800, test acc 0.779
epoch 8, loss 0.0023, train acc 0.807, test acc 0.806
epoch 9, loss 0.0022, train acc 0.808, test acc 0.762
epoch 10, loss 0.0022, train acc 0.812, test acc 0.802
```

###  预测

训练完成后，现在就可以演示如何对图像进行分类了。给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）。

``` python
import matplotlib.pyplot as plt

X, y = next(iter(test_loader))

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 这里注意subplot 和subplots 的区别
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(torch.reshape(img, shape=(28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(torch.argmax(net(X), dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
```

![img](./src/softmax-regression-code/3.6_output1.png)

## 小结

* 可以使用softmax回归做多类别分类。与训练线性回归相比，你会发现训练softmax回归的步骤和它非常相似：获取并读取数据、定义模型和损失函数并使用优化算法训练模型。事实上，绝大多数深度学习模型的训练都有着类似的步骤。
