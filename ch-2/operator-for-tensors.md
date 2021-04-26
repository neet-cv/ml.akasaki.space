# 张量的基本数学运算

## Outline

在Tensorflow（以及很多相似的框架中），Tensor有以下基本运算方法：

- `+`,`-`,`*`,`/`
- `**`,`pow`,`square`
- `sqrt`
- `//`,`%`
- `exp`,`log`
- `@`,`matmul`

我们对它们进行简单地分类：

- element-wise （逐位的运算）: `+`,`-`,`*`,`/`
- matrix-wise （对整个矩阵的运算）: `@`,`matmul`
- dim-wise（维度运算） : `reduce_mean`,`max`,`min`,`sum`

---

首先我们先import一下需要的包

```python
# importing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

a = tf.constant([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
print(a@a)
tf.Tensor(
[[0 0 0 0]
 [0 0 0 0]
 [0 0 0 0]
 [0 0 0 0]], shape=(4, 4), dtype=int32)
```

## operators of `+`,`-`,`*`,`/`,`%`,`//`

```python
a = tf.ones([2,2])
b = tf.fill([2,2],2.)
print("The original a: ",a.shape)
print("The original b: ",b.shape)
print("a+b = ",a+b)
print("a-b = ",a-b)
print("a*b = ",a*b)
print("a/b = ",a/b)

print("b%a = ",b%a)
print("b//a = ",b//a)
```

上述代码得到的输出：

```
The original a:  (2, 2)
The original b:  (2, 2)
a+b =  tf.Tensor(
[[3. 3.]
 [3. 3.]], shape=(2, 2), dtype=float32)
a-b =  tf.Tensor(
[[-1. -1.]
 [-1. -1.]], shape=(2, 2), dtype=float32)
a*b =  tf.Tensor(
[[2. 2.]
 [2. 2.]], shape=(2, 2), dtype=float32)
a/b =  tf.Tensor(
[[0.5 0.5]
 [0.5 0.5]], shape=(2, 2), dtype=float32)
b%a =  tf.Tensor(
[[0. 0.]
 [0. 0.]], shape=(2, 2), dtype=float32)
b//a =  tf.Tensor(
[[2. 2.]
 [2. 2.]], shape=(2, 2), dtype=float32)
```

## operators of `math.log`,`exp`

```python
a = tf.ones([2,2])
print("the original a: ",a)
# 注意，这里的log其实是ln。即以e为底数的log。
print("Log a = ",tf.math.log(a))
# exp(a) = e^a, 即exp()实现e的次方运算
print("exp a = ",tf.exp(a))
```

上述代码得到以下输出：

```
the original a:  tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32)
Log a =  tf.Tensor(
[[0. 0.]
 [0. 0.]], shape=(2, 2), dtype=float32)
```

tf.math.log 只能实现以e为底数的log。如果想实现其它的log，比如：$log_2 3$这样的东西，可以这样写:

```python
print("log_2^3 = ", tf.math.log(3.)/tf.math.log(2.))
# 再试一下log_10^100:  
print("log_10^100 = ",tf.math.log(100.)/tf.math.log(10.))
```

以上代码产生以下输出：

```
log_2^3 =  tf.Tensor(1.5849625, shape=(), dtype=float32)
log_10^100 =  tf.Tensor(2.0, shape=(), dtype=float32)
```

## operators of `pow`,`sqrt`

```python
b = tf.fill([2,2],2.)
print("The original b: ",b)
# using pow calculating b^3 :
print("b^3 = ",tf.pow(b,3))
# the same as :
print("b**3 = ",b**3)
# using sqrt calculating √b
print("√b = ",tf.sqrt(b))
```

上述代码将会输出：

```
The original b:  tf.Tensor(
[[2. 2.]
 [2. 2.]], shape=(2, 2), dtype=float32)
b^3 =  tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)
b**3 =  tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)
√b =  tf.Tensor(
[[1.4142135 1.4142135]
 [1.4142135 1.4142135]], shape=(2, 2), dtype=float32)
```

## operators of `@`,`matmul`

```python
a = tf.ones([2,2])
b = tf.fill([2,2],2.)
# using a@b (标准矩阵乘法)
print(a@b)
# using matrix multiply (matmul) doing the same thing: 
print(tf.matmul(a,b))
```

上述代码将会输出：

```
tf.Tensor(
[[4. 4.]
 [4. 4.]], shape=(2, 2), dtype=float32)
tf.Tensor(
[[4. 4.]
 [4. 4.]], shape=(2, 2), dtype=float32)
```

当然，不是非要正方形的二维矩阵才能相乘。这里的矩阵相乘符合线性代数基本操作。同时，三个维度及以上的矩阵相乘将会对最小的两个维度中所有的元素进行相乘。例如：

```python
a = tf.random.normal([2,2,3])
print("a = ",a)
b = tf.random.normal([2,3,4])
print("b = ",b)
print("a@b = ",a@b)
```

以上代码会输出：

```
a =  tf.Tensor(
[[[-0.9515763  -1.5930161  -0.80232066]
  [-0.11224517  1.0577781  -2.087813  ]]

 [[-0.6721073   0.06475437  1.1214167 ]
  [ 0.21041955  0.9131027   1.8936794 ]]], shape=(2, 2, 3), dtype=float32)
b =  tf.Tensor(
[[[ 0.6188545  -0.39597458 -0.57622075  0.12333456]
  [-1.5091116  -0.5508188  -0.04195018  1.6269956 ]
  [-0.32907686 -0.35808468 -0.6265103   1.154586  ]]

 [[-1.4198172  -0.7301577   2.3513768   1.1920613 ]
  [ 0.83363074 -1.1223577   0.5119745  -0.5090065 ]
  [ 0.35063308 -0.29540017  0.18837257  0.88924986]]], shape=(2, 3, 4), dtype=float32)
a@b =  tf.Tensor(
[[[ 2.0791771   1.541562    1.1178075  -3.6355407 ]
  [-0.97871786  0.209416    1.3283403  -0.7034028 ]]

 [[ 1.4014565   0.08680002 -1.3359808   0.16306616]
  [ 1.1264198  -1.7378604   1.3189782   1.470012  ]]], shape=(2, 2, 4), dtype=float32)
```

## summary

回想简单的一维线性回归中，有简单的线性代数式 $y = \omega \cdot x + b$。 在矩阵中，我们可以写出原理相同的矩阵运算版本: $Y = X@W +B$，其中Y、W、X、B都是矩阵。

```python
X = tf.fill([2,2],2.)
print("X = ",X)
W = tf.random.normal([2,2])
print("W = ",W)
b = tf.constant([0.001])
print("b = ",b)
Y =W@X+b
print("Y = W@X+b = ",Y)
print("Y after relu: ",tf.nn.relu(Y))
```

这是输出：

```
X =  tf.Tensor(
[[2. 2.]
 [2. 2.]], shape=(2, 2), dtype=float32)
W =  tf.Tensor(
[[-0.8689423  1.2698495]
 [-0.4126347 -1.6587701]], shape=(2, 2), dtype=float32)
b =  tf.Tensor([0.001], shape=(1,), dtype=float32)
Y = W@X+b =  tf.Tensor(
[[ 0.8028144  0.8028144]
 [-4.1418095 -4.1418095]], shape=(2, 2), dtype=float32)
Y after relu:  tf.Tensor(
[[0.8028144 0.8028144]
 [0.        0.       ]], shape=(2, 2), dtype=float32)
```