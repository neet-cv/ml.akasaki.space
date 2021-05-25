# 张量的基本数学运算

## Outline

在Pytorch（以及很多相似的框架中），Tensor有以下基本运算方法：

- `+`,`-`,`*`,`/`
- `**`,`pow`,`square`
- `sqrt`
- `//`,`%`
- `exp`,`log`
- `@`,`matmul`

我们对它们进行简单地分类：

- element-wise （逐位的运算）: `+`,`-`,`*`,`/`
- matrix-wise （对整个矩阵的运算）: `@`,`matmul`
- dim-wise（维度运算） : `mean`,`max`,`min`,`sum`

---

首先我们先import一下需要的包

```python
# importing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import torch
import pandas as pd

a =torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
print(a @ a)

out: 
tensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])
```

## operators of `+`,`-`,`*`,`/`,`%`,`//`

```python
a = torch.ones([2, 2])
b = torch.full([2, 2], 2.)
print("The original a: ", a.shape)
print("The original b: ", b.shape)
print("a+b = ", a + b)
print("a-b = ", a - b)
print("a*b = ", a * b)
print("a/b = ", a / b)

print("b%a = ", b % a)
print("b//a = ", b // a)
```

上述代码得到的输出：

```
The original a:  torch.Size([2, 2])
The original b:  torch.Size([2, 2])
a+b =  tensor([[3., 3.],
        [3., 3.]])
a-b =  tensor([[-1., -1.],
        [-1., -1.]])
a*b =  tensor([[2., 2.],
        [2., 2.]])
a/b =  tensor([[0.5000, 0.5000],
        [0.5000, 0.5000]])
b%a =  tensor([[0., 0.],
        [0., 0.]])
b//a =  tensor([[2., 2.],
        [2., 2.]])
```

## operators of `math.log`,`exp`

```python
a = torch.ones([2, 2])
print("the original a: ", a)
# 注意，这里的log其实是ln。即以e为底数的log。
print("Log a = ", torch.log(a))
# exp(a) = e^a, 即exp()实现e的次方运算
print("exp a = ", torch.exp(a))
```

上述代码得到以下输出：

```
the original a:  tensor([[1., 1.],
        [1., 1.]])
Log a =  tensor([[0., 0.],
        [0., 0.]])
exp a =  tensor([[2.7183, 2.7183],
        [2.7183, 2.7183]])
```

torch.log 只能实现以e为底数的log。如果想实现其它的log，比如：$log_2 3$这样的东西，可以这样写:

```python
print("log_2^3 = ", torch.log2(torch.tensor(3.)))
# 再试一下log_10^100:  
print("log_10^100 = ", torch.log10(torch.tensor(10.)))
```

以上代码产生以下输出：

```
log_2^3 =  tensor(1.5850)
log_10^100 =  tensor(1.)
```

## operators of `pow`,`sqrt`

```python
b = torch.full([2, 2], 2.)
print("The original b: ", b)
# using pow calculating b^3 :
print("b^3 = ", torch.pow(b, 3))
# the same as :
print("b**3 = ", b ** 3)
# using sqrt calculating √b
print("√b = ", torch.sqrt(b))
```

上述代码将会输出：

```
The original b:  tensor([[2., 2.],
        [2., 2.]])
b^3 =  tensor([[8., 8.],
        [8., 8.]])
b**3 =  tensor([[8., 8.],
        [8., 8.]])
√b =  tensor([[1.4142, 1.4142],
        [1.4142, 1.4142]])
```

## operators of `@`,`matmul`

```python
a = torch.ones([2, 2])
b = torch.full([2, 2], 2.)
# using a@b (标准矩阵乘法)
print(a @ b)
# using matrix multiply (matmul) doing the same thing: 
print(torch.matmul(a, b))
```

上述代码将会输出：

```
tensor([[4., 4.],
        [4., 4.]])
tensor([[4., 4.],
        [4., 4.]])
```

当然，不是非要正方形的二维矩阵才能相乘。这里的矩阵相乘符合线性代数基本操作。同时，三个维度及以上的矩阵相乘将会对最小的两个维度中所有的元素进行相乘。例如：

```python
a = torch.randn([2, 2, 3])
print("a = ", a)
b = torch.randn([2, 3, 4])
print("b = ", b)
print("a@b = ", a @ b)
```

以上代码会输出：

```
a =  tensor([[[-1.7821, -1.7878,  0.4552],
         [-1.1150, -1.0681, -0.7144]],

        [[ 0.1903, -1.8039, -0.5285],
         [-0.5586, -0.9626, -2.2884]]])
b =  tensor([[[ 0.2129,  0.9654,  0.2214,  0.5795],
         [ 0.9861,  0.0555, -1.6041,  0.0314],
         [ 0.8242,  1.8293,  0.2248, -0.9419]],

        [[ 0.3706, -0.9674, -0.2546, -0.7979],
         [ 0.9365,  1.4654, -0.5503, -0.0462],
         [ 0.9061,  0.3743, -0.4906, -0.8350]]])
a@b =  tensor([[[-1.7673, -0.9872,  2.5756, -1.5175],
         [-1.8795, -2.4425,  1.3059, -0.0068]],

        [[-2.0977, -3.0253,  1.2036,  0.3729],
         [-3.1819, -1.7266,  1.7946,  2.4010]]])
```

## summary

回想简单的一维线性回归中，有简单的线性代数式 $y = \omega \cdot x + b$。 在矩阵中，我们可以写出原理相同的矩阵运算版本: $Y = X@W +B$，其中Y、W、X、B都是矩阵。

```python
relu = torch.nn.ReLU()
X = torch.full([2, 2], 2.)
print("X = ", X)
W = torch.randn([2, 2])
print("W = ", W)
b = torch.tensor([0.001])
print("b = ", b)
Y = W @ X + b
print("Y = W@X+b = ", Y)
print("Y after relu: ", relu(Y))
```

这是输出：

```
X =  tensor([[2., 2.],
        [2., 2.]])
W =  tensor([[ 0.7573, -0.2999],
        [ 1.3441,  2.4994]])
b =  tensor([0.0010])
Y = W@X+b =  tensor([[0.9159, 0.9159],
        [7.6881, 7.6881]])
Y after relu:  tensor([[0.9159, 0.9159],
        [7.6881, 7.6881]])
```