# 自动求梯度

```python
import torch
print(torch.__version__)

```

    1.7.0+cu110


在深度学习中，我们经常需要对函数求梯度（gradient）。本节将介绍如何使用tensorflow2.0提供的GradientTape来自动求梯度。

## 简单示例

我们先看一个简单例子：对函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 求关于列向量 $\boldsymbol{x}$ 的梯度。我们先创建变量`x`，并赋初值。


```python
x = torch.reshape(torch.tensor(range(4), dtype=torch.float32), (4, 1))
print(x)
```

    tensor([[0.],
            [1.],
            [2.],
            [3.]])

函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 关于$\boldsymbol{x}$ 的梯度应为$4\boldsymbol{x}$。现在我们来验证一下求出来的梯度是正确的。

```python
x.requires_grad = True
y = 2 * torch.matmul(x.T, x)
y.backward()
print(x.grad.data)
```

    tensor([[ 0.],
            [ 4.],
            [ 8.],
            [12.]])

## 对Python控制流求梯度

即使函数的计算图包含了Python的控制流（如条件和循环控制），我们也有可能对变量求梯度。

考虑下面程序，其中包含Python的条件和循环控制。需要强调的是，这里循环（while循环）迭代的次数和条件判断（if语句）的执行都取决于输入a的值。

```python
def f(a):
    b = a * 2
    while torch.norm(b) < 1000:
        b = b * 2
    if torch.sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

我们来分析一下上面定义的`f`函数。事实上，给定任意输入`a`，其输出必然是 `f(a) = x * a`的形式，其中标量系数`x`的值取决于输入`a`。由于`c = f(a)`有关`a`的梯度为`x`，且值为`c / a`，我们可以像下面这样验证对本例中控制流求梯度的结果的正确性。

```python
a = torch.randn((1, 1), dtype=torch.float32)
a.requires_grad = True
c = f(a)
c.backward()
print(a.grad.data == (c / a).detach())
```

    tensor([[True]])

