# Tensor的基本操作：合并、分割以及统计

## 合并与分割(Merge and split)

- tf.concat
- tf.split
- tf.stack
- tf.unstack

```python
# importing
import os
# changing env log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
```

### concat

concat用于张量的拼接操作。例如：一共有六个班级需要统计成绩。其中第一个人统计前四个班级的成绩，另一个人统计后两个班级的成绩。假设每个班有35人，每个人有八门科目的成绩，那么两个人获得的成绩单的shape应该分别是[4,35,8]和[2,35,8]，拼接后的成绩单的shape应该是[6,35,8]。

```python
a = tf.ones([4,35,8])
b = tf.ones([2,35,8])
# The operation of concat is seemingly two steps. 1. broadcast. 2. combine 
c = tf.concat([a,b],axis = 0)
print(c.shape)
```

输出：

```
(6, 35, 8)
```

另外一个类似的场景是两个人统计一个班级的成绩信息，该班级一共有35名学生，第一个人统计前32名学生的成绩，第二个人统计后3名学生的成绩，拼接后得到全班的总成绩单。

```python
a = tf.ones([1,32,8])
b = tf.ones([1,3,8])
# The operation of concat is seemingly two steps. 1. broadcast. 2. combine 
c = tf.concat([a,b],axis = 1)
print(c.shape)
```

输出：

```
(1, 35, 8)
```

当然也有这样的场景：现在有四个班级的人考了总共16门考试，其成绩分别记录在了两张表中，每张表记录了8门成绩。现在要将这些成绩放入同一张表：

```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.concat([a,b],axis = -1)
print(c.shape)
```

```
(4, 35, 16)
```


请注意这两个场景在运算时的区别是基于哪个维度进行拼接。第一种场景下对第0维度进行拼接，第二种场景下对第一维度进行拼接。

- concat的使用限制条件为：出了要拼接的维度的大小可以不等之外其它维度需要相等。

### stack

stack用于张量的堆叠操作。例如：现在有两个班级的成绩信息，张量结构为[class,student,scoer]。这两个班级分别属于两个学校，现在要将它们放入一张成绩表中，但是要能区分他们的学校。

```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
# add a new dim and combine them
c = tf.stack([a,b],axis = 0)
print(c.shape)
```

输出：

```
(2, 4, 35, 8)
```

stack可选要扩展维度的位置，例如，我们希望将学校一列放在最后：

```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
# add a new dim and combine them
c = tf.stack([a,b],axis = 3)
print(c.shape)
```

输出：

```
(4, 35, 8, 2)
```

不过一般习惯上把更大的维度（学校）放在前面。

- stack的使用限制条件为shape相等。

### unstack

对应stack，也有unstack。unstack可以在指定的axis上将tensor打散为该axis的size份

```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
# add a new dim and combine them
c = tf.stack([a,b],axis = 0)
print("shape of the origin : ",c.shape)
# unstack
a_2,b_2 = tf.unstack(c,axis = 0)
print("after unstack : a2:",a_2.shape,",b2:",b_2.shape)
```

输出：

```
shape of the origin :  (2, 4, 35, 8)
after unstack : a2: (4, 35, 8) ,b2: (4, 35, 8)
```

### split

unstack的使用场景有限。split的功能更加强大。split大体上有两种用法：

- 第一种：num_or_size_splits是数字，例如"num_or_size_splits=2"的情况，split会将tensor再指定的axis上分成两半。

```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
# add a new dim and combine them
c = tf.stack([a,b],axis = 0)
print("shape of the origin : ",c.shape)
# split into two part on axis 0
res = tf.split(c,axis = 0,num_or_size_splits=2)
print("after split into two part, len = ",len(res),", shape = ",res[0].shape," and ",res[1].shape)
```

输出：

```
shape of the origin :  (2, 4, 35, 8)
after split into two part, len =  2 , shape =  (1, 4, 35, 8)  and  (1, 4, 35, 8)
```

- 第二种：num_or_size_splits是一个list，例如"num_or_size_splits=[1,2,3]"的情况，split会将tensor在指定的axis上分为这个list的size份，在这里是3份，每份的相对大小分别是1、2、3。

```python
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
# add a new dim and combine them
c = tf.stack([a,b],axis = 0)
print("shape of the origin : ",c.shape)
# split into three part on axis 3, relative size = 2 ,2 ,4
res = tf.split(c,axis = 3,num_or_size_splits=[2,2,4])
```

输出：

```
shape of the origin :  (2, 4, 35, 8)
```

## 数据统计

- tf.norm：张量范数（一范数、二范数、...、无穷范数）
- tf.reduce_min：最小值
- tf,reduce_max：最大值
- tf.argmin：最小值位置
- tf.argmax：最大值位置
- tf.equal：张量比较
- tf.unique：独特值

### tf.norm

为了好理解，暂时只讨论向量的范数。向量的二范数的公式为：
$$
^2\sqrt{sum_{i=1}^{size}x_i^2}
$$
向量的n范数的公式为：
$$
^n\sqrt{sum_{i=1}^{size}x_i^n}
$$
可以理解为：范数是一个函数，是一个向量到数值的映射。向量之间无法比较大小，进行范数运算之后就能直接比较大小了。再换句话理解，这是一种特殊的"欧氏距离(x)"，可以比较向量到远点的距离(x)（我瞎理解的）。

先来一个小一点的tensor：

```python
origin = tf.ones([2,2])
# 二范数
print("origin = ",origin,"\nafter norm: ",tf.norm(origin))
# 验证一下二范数的运算方式和我们上面说的是否一致
print("origin = ",origin,"\nafter square-square-aqrt : ",tf.sqrt(tf.reduce_sum(tf.square(origin))))
print("They are the same")
```

```
origin =  tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32) 
after norm:  tf.Tensor(2.0, shape=(), dtype=float32)
origin =  tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32) 
after square-square-aqrt :  tf.Tensor(2.0, shape=(), dtype=float32)
They are the same
```

大一点的tensor：

```python
origin = tf.ones([4,28,28,3])
# more complex example
print("origin = ",origin.shape,"\nafter norm: ",tf.norm(origin)) 
# 验证一下二范数的运算方式和我们上面说的是否一致
print("origin = ",origin.shape,"\nafter square-square-aqrt : ",tf.sqrt(tf.reduce_sum(tf.square(origin))))
print("They are the same")
```

输出：

```
origin =  (4, 28, 28, 3) 
after norm:  tf.Tensor(96.99484, shape=(), dtype=float32)
origin =  (4, 28, 28, 3) 
after square-square-aqrt :  tf.Tensor(96.99484, shape=(), dtype=float32)
They are the same
```

norm除了可以作用在整个张量上，也可以作用在某一个维度上。大概可以理解为对这个维度进行一次unstack然后再对unstack出来的每一个向量求norm。

```python
origin = tf.ones([4,28,28,3])
# norm working on specific axis
print("origin = ",origin.shape,"\nafter norm on axis = 3 : ",tf.norm(origin,axis = 3))
```

输出：

```
origin =  (4, 28, 28, 3) 
after norm on axis = 3 :  tf.Tensor(
[[[1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  ...
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]]

 [[1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  ...
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]]

 [[1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  ...
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]]

 [[1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  ...
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]
  [1.7320508 1.7320508 1.7320508 ... 1.7320508 1.7320508 1.7320508]]], shape=(4, 28, 28), dtype=float32)
```

除了默认的二范数外，norm也可以求n范数。方法是指定ord参数。例如：

```python
origin = tf.ones([2,2])
# 一范数
print("ord = 1 : ",tf.norm(origin,ord=1))
# 二范数
print("ord = 2 : ",tf.norm(origin,ord=2))
# 三范数
print("ord = 3 : ",tf.norm(origin,ord=3))
# 四范数
print("ord = 4 : ",tf.norm(origin,ord=4))
# 五范数
print("ord = 5 : ",tf.norm(origin,ord=5))
```

输出：

```
ord = 1 :  tf.Tensor(4.0, shape=(), dtype=float32)
ord = 2 :  tf.Tensor(2.0, shape=(), dtype=float32)
ord = 3 :  tf.Tensor(1.587401, shape=(), dtype=float32)
ord = 4 :  tf.Tensor(1.4142135, shape=(), dtype=float32)
ord = 5 :  tf.Tensor(1.319508, shape=(), dtype=float32)
```

### tf.reduce_min / max / mean / sum

- tf.reduce_min
- tf.reduce_max
- tf.reduce_mean
- tf.reduce_sum

其实就是求最小值最大值平均值。名字里带着reduce表明，这个操作会有一个类似"打平"的过程。例如，当不指定axis参数时，一个[10,4]的tensor会被"打平"成一个[40]的"list"并求最大值、最小值....；再如，带有axis=2参数时，一个[10,4,10]的tensor会被"降维"变成一个元素为[10,4]的tensor的list，大小是10，然后对着十个元素进行最大、最小....运算。

在整个tensor上操作：

```python
origin = tf.random.normal([4,10])
print("origin = ",origin,"\nreduce_min = ",tf.reduce_min(origin),"\nreduce_max = ",tf.reduce_max(origin),"\nreduce_mean = ",tf.reduce_mean(origin))
```

输出：

```
origin =  tf.Tensor(
[[-1.4178391e+00  1.0799263e+00  1.8141992e+00 -2.9427743e-01
   3.5776252e-01 -6.9446379e-01 -7.1207196e-01  9.6388352e-01
  -2.1230397e+00  4.8318788e-01]
 [-4.1854006e-01 -2.2664030e-01 -9.8776561e-01  3.3819950e-01
   2.4363371e-02 -3.2178679e+00 -2.8521428e-01 -5.3039378e-01
  -1.0285269e+00 -1.2320877e+00]
 [ 6.0093373e-01  1.3320454e-02  9.5860285e-01  1.4495020e+00
   5.1962131e-01  1.1331964e+00 -1.0149366e+00 -5.1126540e-02
  -5.0443190e-01  3.9746460e-01]
 [-4.1444901e-01 -1.2171540e+00 -8.4814447e-01  1.4405949e+00
   7.2787516e-04  1.2379333e+00  1.0925928e+00 -9.9176753e-01
   3.8999468e-02  1.0164096e+00]], shape=(4, 10), dtype=float32) 
reduce_min =  tf.Tensor(-3.2178679, shape=(), dtype=float32) 
reduce_max =  tf.Tensor(1.8141992, shape=(), dtype=float32) 
reduce_mean =  tf.Tensor(-0.08123293, shape=(), dtype=float32)
```

在某个轴上操作：

```python
origin = tf.random.normal([4,10])
print("origin = ",origin)
print("\nreduce_min on axis 1 = ",tf.reduce_min(origin,axis = 1))
print("\nreduce_max on axis 1 = ",tf.reduce_max(origin,axis = 1))
print("\nreduce_mean on axis 1 = ",tf.reduce_mean(origin,axis = 1))
```

输出：

```
origin =  tf.Tensor(
[[-3.1204236   0.67563623 -0.9232384   1.1589053   0.8515049  -0.47955766
  -1.723766    0.12821583 -0.6078169  -0.07115268]
 [-0.03351626  0.5452725   0.4999855  -0.13481826  0.6798329   0.23792107
  -0.6113948   1.3868407   0.24892737 -0.41333905]
 [-0.9676226  -0.3656622  -0.688232    1.721823    0.6695465  -0.44504106
   0.90125936  0.5428907   1.4090685  -0.9626962 ]
 [-0.87203074  0.9285623   0.56897074 -1.4624474   1.8943952  -0.5554827
  -0.8351434  -0.3565093  -1.5708245  -1.1640625 ]], shape=(4, 10), dtype=float32)

reduce_min on axis 1 =  tf.Tensor([-3.1204236 -0.6113948 -0.9676226 -1.5708245], shape=(4,), dtype=float32)

reduce_max on axis 1 =  tf.Tensor([1.1589053 1.3868407 1.721823  1.8943952], shape=(4,), dtype=float32)

reduce_mean on axis 1 =  tf.Tensor([-0.4111693   0.24057117  0.18153341 -0.34245723], shape=(4,), dtype=float32)
```

### tf.argmax/argmin

- tf.argmax
- tf.argmin

用于求最小值和最大值的位置。当不指定axis参数时，默认再维度0上求每个维度下标下的最大、最小值的位置。

argmin

```python
origin = tf.nn.relu(tf.random.normal([4,3])*100)
print("origin = ",origin)
# 维度0下求最大值位置
print("argmax : ",tf.argmax(origin))
```

输出：

```
origin =  tf.Tensor(
[[ 0.        0.        0.      ]
 [ 0.        0.       54.485176]
 [86.468796 74.000046  0.      ]
 [ 0.       29.033602 69.07481 ]], shape=(4, 3), dtype=float32)
argmax :  tf.Tensor([2 2 3], shape=(3,), dtype=int64)
```

argmax

```python
origin = tf.nn.relu(tf.random.normal([4,3,2])*100)
print("origin = ",origin)
# 维度0下求最大值位置，这里对维度0展开会得到二位的tensor，所以得到的最值得位置也会是二维坐标
print("argmax : ",tf.argmax(origin))
```

输出：

```
origin =  tf.Tensor(
[[[  0.        44.180485]
  [ 60.554047  54.124874]
  [ 11.455048  54.916447]]

 [[  0.        70.35009 ]
  [147.33435  110.680046]
  [  0.        59.37093 ]]

 [[  0.        20.160051]
  [  0.        24.07408 ]
  [  0.         0.      ]]

 [[ 84.53291    0.      ]
  [131.28426  103.82523 ]
  [192.91162   31.05115 ]]], shape=(4, 3, 2), dtype=float32)
argmax :  tf.Tensor(
[[3 1]
 [1 1]
 [3 1]], shape=(3, 2), dtype=int64)
```

### tf.equal

用于比较

```python
a = tf.constant([1,2,3,4,5])
b = tf.constant(range(5))
print("a = ",a,", b = ",b)
result = tf.equal(a,b)
print("Equal : ",result)
cast_to_int = tf.reduce_sum(tf.cast(result, dtype=tf.int32))
print("To int32 : ",cast_to_int)
```

输出：

```
a =  tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32) , b =  tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)
Equal :  tf.Tensor([False False False False False], shape=(5,), dtype=bool)
To int32 :  tf.Tensor(0, shape=(), dtype=int32)
```

tf.equal在精确度计算过上似乎有点用。例如，当有一个测试数据集，你的模型跑出来的预测值和测试数据的y做一次equal，然后cast成一个数字，根据大小可以判断accuracy。（也就是相同的部分是准确预测的）

### tf.unique

tf.unique能得到一个包含tensor中所有元素的“set”，并且得到另一个idx的tensor用户标注每一个元素在得到的set里的下标。例如：

```python
origin = tf.constant([4,2,2,4,3])
result, idx = tf.unique(origin) 
print("origin = ",origin,"\nunique : ",result,", \nidx : ",idx)
```

输出：

```
origin =  tf.Tensor([4 2 2 4 3], shape=(5,), dtype=int32) 
unique :  tf.Tensor([4 2 3], shape=(3,), dtype=int32) , 
idx :  tf.Tensor([0 1 1 0 2], shape=(5,), dtype=int32)
```

回忆一下tf2基本操作中的gather，我们可以通过得到的结果的得到的idx将它复原

```python
origin = tf.constant([4,2,2,4,3])
result, idx = tf.unique(origin) 
print("origin = ",origin,"\nunique : ",result,", \nidx : ",idx)
print("using tf.gather : ",tf.gather(result,idx))
```

输出：

```
origin =  tf.Tensor([4 2 2 4 3], shape=(5,), dtype=int32) 
unique :  tf.Tensor([4 2 3], shape=(3,), dtype=int32) , 
idx :  tf.Tensor([0 1 1 0 2], shape=(5,), dtype=int32)
using tf.gather :  tf.Tensor([4 2 2 4 3], shape=(5,), dtype=int32)
```