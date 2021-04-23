
# 自动求梯度

在深度学习中，我们经常需要对函数求梯度（gradient）。本节将介绍如何使用MXNet提供的`autograd`模块来自动求梯度。如果对本节中的数学概念（如梯度）不是很熟悉，请首先看一下数学基础内容。

首先从mxnet引入自动梯度求导的模块：

```python
from mxnet import autograd, nd
```

## 1 简单的例子

我们先看一个简单例子：对函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 求关于列向量 $\boldsymbol{x}$ 的梯度。我们先创建变量`x`，并赋初值。

```python
x = nd.arange(4).reshape((4, 1))
x
```

你会得到这样的输出：

```
[[0.]
 [1.]
 [2.]
 [3.]]
<NDArray 4x1 @cpu(0)>
```

为了求有关变量`x`的梯度，我们需要先调用`attach_grad`函数来申请存储梯度所需要的内存。

```python
x.attach_grad()
```

下面定义有关变量`x`的函数。为了减少计算和内存开销，默认条件下MXNet不会记录用于求梯度的计算。我们需要调用`record`函数来要求MXNet记录与求梯度有关的计算。

```python
with autograd.record():
    y = 2 * nd.dot(x.T, x)
```

由于`x`的形状为（4, 1），`y`是一个标量。接下来我们可以通过调用`backward`函数自动求梯度。需要注意的是，如果`y`不是一个标量，MXNet将默认先对`y`中元素求和得到新的变量，再求该变量有关`x`的梯度。

```python
y.backward()
```

函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 关于$\boldsymbol{x}$ 的梯度应为$4\boldsymbol{x}$。现在我们来验证一下求出来的梯度是正确的。

```python
assert (x.grad - 4 * x).norm().asscalar() == 0
x.grad
```

你可以获得这样的输出：

```
[[ 0.]
 [ 4.]
 [ 8.]
 [12.]]
<NDArray 4x1 @cpu(0)>
```

---

## 2 训练模式和预测模式

从上面可以看出，在调用`record`函数后，MXNet会记录并计算梯度。此外，默认情况下`autograd`还会将运行模式从预测模式转为训练模式。这可以通过调用`is_training`函数来查看。

```python
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

你可以看到这样的输出：

```
False
True
```

在有些情况下，同一个模型在训练模式和预测模式下的行为并不相同。我们会在后面的章节（如[“丢弃法”](//todo)一节）详细介绍这些区别。

---

## 3 对python控制流求梯度

使用MXNet的一个便利之处是，即使函数的计算图包含了Python的控制流（如条件和循环控制），我们也有可能对变量求梯度。

这里，我们尝试实现对下面这个函数求梯度：
$$
f(x) = \begin{cases}
\vec{x}, \quad \sum_{i=0}^n x_i \leq 10
\\\\
2\vec{x}, \quad other
\end{cases}
$$
从上面的式子我们不难看出，当`x`中所有元素加和小于等于10即$\sum_{i=0}^n x_i \leq 10$时，该函数对x的梯度为`1`。其他情况下这个函数对`x`的梯度为`2`。

首先，我们在python中定义上面这个函数：

```python
def f(x):
    if x.sum().asscalar() <= 10:
        return x
    return 2 * x
```

接写来，我们随便产生一个x向量进行测试：

```python
x = nd.array([1, 2, 3, 4]) #随便写下一个x向量
x.attach_grad()
with autograd.record():
    y = f(x)
y.backward()
print(x.grad)
```

你会看到这样的结果：

```
[1. 1. 1. 1.]
<NDArray 4 @cpu(0)>
```

`1+2+3+4`确实满足`<=10`的条件，所以结果是`1`。如果你将`x`改为`x = nd.array([1, 2, 3, 4, 5])`的话，由于此时对`x`内全部元素求和的结果是`>10`的，所以你会得到另外一种结果：

```
[2. 2. 2. 2. 2.]
<NDArray 5 @cpu(0)>
```

其他的情况可以自己尝试。

除了含有分支(`if`)的流程可以被自动梯度处理外，循环也可以。考虑下面程序，其中包含Python的条件和循环控制。需要强调的是，这里循环（`while`循环）迭代的次数和条件判断（`if`语句）的执行都取决于输入`a`的值。

```python
def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

我们像之前一样使用`record`函数记录计算，并调用`backward`函数求梯度。

```python
a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
```

我们来分析一下上面定义的`f`函数。事实上，给定任意输入`a`，其输出必然是 `f(a) = x * a`的形式，其中标量系数`x`的值取决于输入`a`。由于`c = f(a)`有关`a`的梯度为`x`，且值为`c / a`，我们可以像下面这样验证对本例中控制流求梯度的结果的正确性。

```python
a.grad == c / a
```

你会得到这样的输出：

```
[1.]
<NDArray 1 @cpu(0)>
```

