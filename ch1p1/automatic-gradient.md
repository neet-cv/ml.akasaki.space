```python
import tensorflow as tf
print(tf.__version__)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )
```

    2.0.0

# 2.3 自动求梯度

在深度学习中，我们经常需要对函数求梯度（gradient）。本节将介绍如何使用tensorflow2.0提供的GradientTape来自动求梯度。

## 2.3.1 简单示例

我们先看一个简单例子：对函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 求关于列向量 $\boldsymbol{x}$ 的梯度。我们先创建变量`x`，并赋初值。


```python
x = tf.reshape(tf.Variable(range(4), dtype=tf.float32),(4,1))
x
```

    <tf.Tensor: id=10, shape=(4, 1), dtype=float32, numpy=
    array([[0.],
           [1.],
           [2.],
           [3.]], dtype=float32)>

函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 关于$\boldsymbol{x}$ 的梯度应为$4\boldsymbol{x}$。现在我们来验证一下求出来的梯度是正确的。

```python
with tf.GradientTape() as t:
    t.watch(x)
    y = 2 * tf.matmul(tf.transpose(x), x)
    
dy_dx = t.gradient(y, x)
dy_dx
```

    <tf.Tensor: id=30, shape=(4, 1), dtype=float32, numpy=
    array([[ 0.],
           [ 4.],
           [ 8.],
           [12.]], dtype=float32)>

## 2.3.2 训练模式和预测模式

```python
with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
    dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
    dy_dx = g.gradient(y, x)  # 6.0
dz_dx,dy_dx
```

    WARNING:tensorflow:Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage). Only call GradientTape.gradient inside the context if you actually want to trace the gradient in order to compute higher order derivatives.
    WARNING:tensorflow:Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage). Only call GradientTape.gradient inside the context if you actually want to trace the gradient in order to compute higher order derivatives.
    
    (<tf.Tensor: id=41, shape=(4, 1), dtype=float32, numpy=
     array([[  0.],
            [  4.],
            [ 32.],
            [108.]], dtype=float32)>,
     <tf.Tensor: id=47, shape=(4, 1), dtype=float32, numpy=
     array([[0.],
            [2.],
            [4.],
            [6.]], dtype=float32)>)

## 2.3.3 对Python控制流求梯度

即使函数的计算图包含了Python的控制流（如条件和循环控制），我们也有可能对变量求梯度。

考虑下面程序，其中包含Python的条件和循环控制。需要强调的是，这里循环（while循环）迭代的次数和条件判断（if语句）的执行都取决于输入a的值。

```python
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

我们来分析一下上面定义的`f`函数。事实上，给定任意输入`a`，其输出必然是 `f(a) = x * a`的形式，其中标量系数`x`的值取决于输入`a`。由于`c = f(a)`有关`a`的梯度为`x`，且值为`c / a`，我们可以像下面这样验证对本例中控制流求梯度的结果的正确性。

```python
a = tf.random.normal((1,1),dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(a)
    c = f(a)
t.gradient(c,a) == c/a
```

    <tf.Tensor: id=201, shape=(1, 1), dtype=bool, numpy=array([[ True]])>
