# Tensor的基本操作：合并、分割以及统计

## 合并与分割(Merge and split)

- torch.cat
- torch.split
- torch.stack
- torch.unbind

```python
# importing
import os
# changing env log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import torch
```

### cat

cat用于张量的拼接操作。例如：一共有六个班级需要统计成绩。其中第一个人统计前四个班级的成绩，另一个人统计后两个班级的成绩。假设每个班有35人，每个人有八门科目的成绩，那么两个人获得的成绩单的shape应该分别是[4,35,8]和[2,35,8]，拼接后的成绩单的shape应该是[6,35,8]。

```python
a = torch.ones([4, 35, 8])
b = torch.ones([2, 35, 8])
# The operation of concat is seemingly two steps. 1. broadcast. 2. combine
c = torch.cat([a, b], 0)
print(c.shape)
```

输出：

```
torch.Size([6, 35, 8])
```

另外一个类似的场景是两个人统计一个班级的成绩信息，该班级一共有35名学生，第一个人统计前32名学生的成绩，第二个人统计后3名学生的成绩，拼接后得到全班的总成绩单。

```python
a = torch.ones([1, 32, 8])
b = torch.ones([1, 3, 8])
# The operation of concat is seemingly two steps. 1. broadcast. 2. combine
c = torch.cat([a, b], 1)
print(c.shape)
```

输出：

```
torch.Size([1, 35, 8])
```

当然也有这样的场景：现在有四个班级的人考了总共16门考试，其成绩分别记录在了两张表中，每张表记录了8门成绩。现在要将这些成绩放入同一张表：

```python
a = torch.ones([4, 35, 8])
b = torch.ones([4, 35, 8])
c = torch.cat([a, b], -1)
print(c.shape)
```

```
torch.Size([4, 35, 16])
```


请注意这两个场景在运算时的区别是基于哪个维度进行拼接。第一种场景下对第0维度进行拼接，第二种场景下对第一维度进行拼接。

- cat的使用限制条件为：出了要拼接的维度的大小可以不等之外其它维度需要相等。

### stack

stack用于张量的堆叠操作。例如：现在有两个班级的成绩信息，张量结构为[class,student,scoer]。这两个班级分别属于两个学校，现在要将它们放入一张成绩表中，但是要能区分他们的学校。

```python
a = torch.ones([4, 35, 8])
b = torch.ones([4, 35, 8])
# add a new dim and combine them
c = torch.stack([a, b], 0)
print(c.shape)
```

输出：

```
torch.Size([2, 4, 35, 8])
```

stack可选要扩展维度的位置，例如，我们希望将学校一列放在最后：

```python
a = torch.ones([4, 35, 8])
b = torch.ones([4, 35, 8])
# add a new dim and combine them
c = torch.stack([a, b], 3)
print(c.shape)
```

输出：

```
torch.Size([4, 35, 8, 2])
```

不过一般习惯上把更大的维度（学校）放在前面。

- stack的使用限制条件为shape相等。

### unbind

对应stack，对应的有unbind。unbind可以在指定的dim上将tensor打散为该dim的size份

```python
a = torch.ones([4, 35, 8])
b = torch.ones([4, 35, 8])
# add a new dim and combine them
c = torch.stack([a, b], 0)
print("shape of the origin : ", c.shape)
# unbind
a_2, b_2 = torch.unbind(c, dim=0)
print("after unbind : a2:", a_2.shape, ",b2:", b_2.shape)
```

输出：

```
shape of the origin :  torch.Size([2, 4, 35, 8])
after unbind : a2: torch.Size([4, 35, 8]) ,b2: torch.Size([4, 35, 8])
```

### split

unbind的使用场景有限。split的功能更加强大。

`torch.split(tensor,split_size_or_sections,dim=0)`  第一个参数是待分割张量、第二个参数有两种形式第一种是分割份数,这就和 torch.chunk()一样了。第二种则是分割方案,split_size_or_sections是一个list,待分割张量将会分割为len(list)份,每一份的大小取決于list中的元素。第三个参数为分割维度。

下面介绍一下split_size_or_sections为数字和为list时的两种情况：

- 第一种：split_size_or_sections是数字，例如"split_size_or_sections=2"的情况，split会将tensor在指定的dim上分成两半。

```python
a = torch.ones([4, 35, 8])
b = torch.ones([4, 35, 8])
# add a new dim and combine them
c = torch.stack([a, b], dim=0)
print("shape of the origin : ", c.shape)
# split into two part on axis 0
res = torch.split(c, dim=1, split_size_or_sections=2)
print("after split into two part, len = ", len(res), ", shape = ", res[0].shape, " and ", res[1].shape)
```

输出：

```
shape of the origin :  torch.Size([2, 4, 35, 8])
after split into two part, len =  2 , shape =  torch.Size([2, 2, 35, 8])  and  torch.Size([2, 2, 35, 8])
```

- 第二种：split_size_or_sections是一个list，例如"split_size_or_sections=[1,2,3]"的情况，split会将tensor在指定的dim上分为这个list的size份，在这里是3份，每份的相对大小分别是1、2、3。

```python
a = torch.ones([4, 35, 8])
b = torch.ones([4, 35, 8])
# add a new dim and combine them
c = torch.stack([a, b], dim=0)
print("shape of the origin : ", c.shape)
# split into three part on axis 3, relative size = 2 ,2 ,4
res = torch.split(c, dim=3, split_size_or_sections=[2, 2, 4])
print("after split into two part, len = ", len(res), ", shape = ", res[0].shape, " and ", res[1].shape, " and ",
      res[2].shape)
```

输出：

```
shape of the origin :  torch.Size([2, 4, 35, 8])
after split into two part, len =  3 , shape =  torch.Size([2, 4, 35, 2])  and  torch.Size([2, 4, 35, 2])  and  torch.Size([2, 4, 35, 4])
```

## 数据统计

- torch.norm：张量范数（一范数、二范数、...、无穷范数）
- torch.min：最小值
- torch.max：最大值
- torch.argmin：最小值位置
- torch.argmax：最大值位置
- torch.equal：张量比较
- torch.unique：独特值

### torch.norm

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
origin = torch.ones([2, 2])
# 二范数
print("origin = ", origin, "\nafter norm: ", torch.norm(origin))
# 验证一下二范数的运算方式和我们上面说的是否一致
print("origin = ", origin, "\nafter square-square-aqrt : ", torch.sqrt(torch.sum(torch.square(origin))))
print("They are the same")
```

```
origin =  tensor([[1., 1.],
                  [1., 1.]]) 
after norm:  tensor(2.)
origin =  tensor([[1., 1.],
                  [1., 1.]]) 
after square-square-aqrt :  tensor(2.)
They are the same
```

大一点的tensor：

```python
origin = torch.ones([4, 28, 28, 3])
# more complex example
print("origin = ", origin.shape, "\nafter norm: ", torch.norm(origin))
# 验证一下二范数的运算方式和我们上面说的是否一致
print("origin = ", origin.shape, "\nafter square-square-aqrt : ", torch.sqrt(torch.sum(torch.square(origin))))
print("They are the same")
```

输出：

```
origin =  torch.Size([4, 28, 28, 3]) 
after norm:  tensor(96.9948)
origin =  torch.Size([4, 28, 28, 3]) 
after square-square-aqrt :  tensor(96.9948)
They are the same
```

norm除了可以作用在整个张量上，也可以作用在某一个维度上。大概可以理解为对这个维度进行一次unstack然后再对unstack出来的每一个向量求norm。

```python
origin = torch.ones([4, 28, 28, 3])
# norm working on specific axis
print("origin = ", origin.shape, "\nafter norm on axis = 3 : ", torch.norm(origin, dim=3))
```

输出：

```
origin =  torch.Size([4, 28, 28, 3]) 
after norm on axis = 3 :  tensor([[[1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         ...,
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321]],

        [[1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         ...,
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321]],

        [[1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         ...,
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321]],

        [[1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         ...,
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321],
         [1.7321, 1.7321, 1.7321,  ..., 1.7321, 1.7321, 1.7321]]])
```

除了默认的二范数外，norm也可以求n范数。方法是指定ord参数。例如：

```python
origin = torch.ones([2, 2])
# fro范数
print("ord = fro : ", torch.norm(origin, p='fro'))
# nuc范数
print("ord = nuc : ", torch.norm(origin, p='nuc'))
```

输出：

```
ord = fro :  tensor(2.)
ord = nuc :  tensor(2.0000)
```

### torch.min / max / mean / sum

- torch.min
- torch.max
- torch.mean
- torch.sum

其实就是求最小值最大值平均值。名字里带着reduce表明，这个操作会有一个类似"打平"的过程。例如，当不指定axis参数时，一个[10,4]的tensor会被"打平"成一个[40]的"list"并求最大值、最小值....；再如，带有axis=2参数时，一个[10,4,10]的tensor会被"降维"变成一个元素为[10,4]的tensor的list，大小是10，然后对着十个元素进行最大、最小....运算。

在整个tensor上操作：

```python
origin = torch.randn([4, 10])
print("origin = ", origin, "\nreduce_min = ", torch.min(origin), "\nreduce_max = ", torch.max(origin),
      "\nreduce_mean = ", torch.mean(origin))
```

输出：

```
origin =  tensor([[-2.4433, -1.1511, -0.2030, -1.3455,  0.6525, -0.6783,  0.7531,  0.6605,
         -1.9095, -0.9665],
        [ 0.2756,  2.2938, -1.0264,  1.1871, -1.9671,  0.3018, -1.6952,  0.8526,
          0.5669, -1.8738],
        [ 0.9387, -1.2025, -1.7915, -0.5040, -0.1888,  0.3096, -1.3362, -2.3146,
         -1.3945,  1.1870],
        [ 0.0379,  0.4525, -1.1179, -1.0721,  1.3222,  0.7806, -1.2831, -1.1209,
         -1.2530,  0.1896]]) 
reduce_min =  tensor(-2.4433) 
reduce_max =  tensor(2.2938) 
reduce_mean =  tensor(-0.4269)
```

在某个轴上操作：

```python
origin = torch.randn([4, 10])
print("origin = ", origin)
print("\nreduce_min on dim 1 = ", torch.min(origin, dim=1))
print("\nreduce_max on dim 1 = ", torch.max(origin, dim=1))
print("\nreduce_mean on dim 1 = ", torch.mean(origin, dim=1))
```

输出：

```
origin =  tensor([[-0.2782,  1.3932, -1.7019, -1.0771,  0.3067, -0.1856,  0.0128, -0.6817,
          0.0550,  1.3197],
        [-0.2669, -0.2080, -0.4126,  1.1578, -0.1088, -1.2211, -1.0403,  1.1367,
         -0.2732, -1.1699],
        [ 1.6385, -0.5989,  0.1612,  1.2463,  0.5802, -1.3011,  0.5324, -1.5297,
          0.1034, -1.2750],
        [ 0.4026,  0.2355, -0.1389,  1.3751, -1.1055, -1.6338, -1.0345,  1.1596,
          0.6468, -1.0662]])

reduce_min on dim 1 =  torch.return_types.min(
values=tensor([-1.7019, -1.2211, -1.5297, -1.6338]),
indices=tensor([2, 5, 7, 5]))

reduce_max on dim 1 =  torch.return_types.max(
values=tensor([1.3932, 1.1578, 1.6385, 1.3751]),
indices=tensor([1, 3, 0, 3]))

reduce_mean on dim 1 =  tensor([-0.0837, -0.2406, -0.0443, -0.1159])
```

### torch.argmax/argmin

- torch.argmax
- torch.argmin

用于求最小值和最大值的位置。当不指定axis参数时，默认再维度0上求每个维度下标下的最大、最小值的位置。

argmin

```python
relu = torch.nn.ReLU()
origin = relu(torch.randn([4, 3]) * 100)
print("origin = ", origin)
# 维度0下求最小值位置
print("argmin : ", torch.argmin(origin))
```

输出：

```
origin =  tensor([[  0.0000,   0.0000,   0.0000],
        [  8.6362, 111.6943,  45.3711],
        [  0.0000, 144.8896,  85.3544],
        [  0.0000,   0.0000,   0.0000]])
argmin :  tensor(0)
```

argmax

```python
relu = torch.nn.ReLU()
origin = relu(torch.randn([4, 3]) * 100)
print("origin = ", origin)
# 维度0下求最大值位置
print("argmax : ", torch.argmax(origin))
```

输出：

```
origin =  tensor([[  0.0000,  12.0928,   0.0000],
        [ 34.4331,  96.3071,  22.3417],
        [ 64.8798,   0.0000,  67.5305],
        [  0.0000,   0.0000, 119.3537]])
argmax :  tensor(11)
```

### torch.equal

用于比较

```python
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor(range(5))
print("a = ", a, ", b = ", b)
result = torch.equal(a, b)
print("Equal : ", result)
```

输出：

```
a =  tensor([1, 2, 3, 4, 5]) , b =  tensor([0, 1, 2, 3, 4])
Equal :  False
```

Torch.equal在精确度计算过上似乎有点用。例如，当有一个测试数据集，你的模型跑出来的预测值和测试数据的y做一次equal，然后cast成一个数字，根据大小可以判断accuracy。（也就是相同的部分是准确预测的）

### torch.unique

torch.unique能得到一个包含tensor中所有元素的“set”. 例如：

```python
origin = torch.tensor([4, 2, 2, 4, 3])
print(torch.unique(origin))
```
