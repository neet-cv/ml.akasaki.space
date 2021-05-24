# Tensor的基本操作：索引、切片和广播



索引和切片能够很方便的帮你拿到你想要的部分数据。

```python
# importing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import torch
import pandas as pd
a = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
print(a@a)
```

```
tensor([[1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]])
```

## List

List是python的builtin，是一种非常灵活的数据载体， 可以包含任何类型的数据

```python
a = ['a', 'b', 'c']
b = [1, 2, 3]
c = [1, 2, 'c']
d = [1, 'b', [3]]
e = [1, b, [c, '4']]
print("a=",a,"b=",b,"c=",c,"d=",d,"e=",e)
```

上述代码产生以下输出：

```
a= ['a', 'b', 'c'] b= [1, 2, 3] c= [1, 2, 'c'] d= [1, 'b', [3]] e= [1, [1, 2, 3], [[1, 2, 'c'], '4']]
```

## Tensor

Tensor意味着张量，是一种自由的多维度数据载体，也是tensorflow使用的数据载体。

```python
tensor_0 = torch.randn(4, 28, 28, 3) # 假设这是一个含有四张28x28像素彩色图片的数据集
print("数据集的shape: ",tensor_0.shape) 
# 省略号...代表任意长度的冒号，只要能在逻辑上推断的都可以使用省略号
print("第一张照片的shape: ",tensor_0[0,:,:,:].shape)
print("和上一行等效，时第一张图片的shape: ",tensor_0[0,...].shape)
# 省略号也可以用来省略前面的冒号
print("所有照片的所有像素的Red通道的shape: ",tensor_0[:,:,:,0].shape)
print("和上一行等效，时所有像素的Red通道的shape: ",tensor_0[...,0].shape)
```

输出：

```
数据集的shape:  torch.Size([4, 28, 28, 3])
第一张照片的shape:  torch.Size([28, 28, 3])
和上一行等效，时第一张图片的shape:  torch.Size([28, 28, 3])
所有照片的所有像素的Red通道的shape:  torch.Size([4, 28, 28])
和上一行等效，时所有像素的Red通道的shape:  torch.Size([4, 28, 28])
```

## 维度变换

维度变换是学习人工神经网络需要理解的基本概念之一。
对于一个多维的数据，可以有不同的理解方式。例如，我们有一份[4,28,28,3]的数据集，代表一个含有四张28x28像素的彩色图片的数据集。接下来我们提供两种理解方式：

1. 这是四张图片。
2. 这是4x28行像素，每行有28个。

其实理论上理解方式可以有很多种。但是无论采用何种方式理解，数据的content本身是不会变化的。我们称不同的理解方式为不同的View。

### torch.reshape

Reshape就是一个用不同的View理解Tensor的过程。Reshape是全连接层常用操作之一。

```python
tensor_1 = torch.randn([4, 28, 28, 3])
print("原来的tensor_1: ", tensor_1.shape)
# tensorflow.reshape需要传入一个原始数据和一个你希望得到的view，就能在允许的范围内进行reshape
print("在reshape为view[4, 28*28, 3]之后: ", torch.reshape(tensor_1, [4, 784, 3]).shape)
# 上述代码中的view可以等价写为以下代码中的形式
print("在reshape为view[4,-1,3]之后: ", torch.reshape(tensor_1, [4, -1, 3]).shape)
# 注意，一个view中只能写下一个-1，reshape方法会根据你写下的-1自动推算维度。如果同时出现多个-1，就会导致无法推算的问题
print("在reshape为view[4,28*28*3]之后: ", torch.reshape(tensor_1, [4, 28 * 28 * 3]).shape)
# 同样，上述代码可以使用-1
print("在reshape为view[4,-1]之后: ", torch.reshape(tensor_1, [4, -1]).shape)
# 这种类型情况下的reshape通常是可逆的。你可以把它reshape回来
print("试图reshape回veiw[4,28,28,3]: ", torch.reshape(torch.reshape(tensor_1, [4, -1]), [4, 28, 28, 3]).shape)
```

以上代码的输出：

```
原来的tensor_1:  torch.Size([4, 28, 28, 3])
在reshape为view[4, 28*28, 3]之后:  torch.Size([4, 784, 3])
在reshape为view[4,-1,3]之后:  torch.Size([4, 784, 3])
在reshape为view[4,28*28*3]之后:  torch.Size([4, 2352])
在reshape为view[4,-1]之后:  torch.Size([4, 2352])
试图reshape回veiw[4,28,28,3]:  torch.Size([4, 28, 28, 3])
```

###  transpose and permute

`torch.reshape`可以帮助你获得不同view下的content，但是并不能改变content。而`转置(transpose,permute)`可以帮助你更换content中各个维度的顺序。 转置操作可以是线性代数意义上的(默认情况)，也可以根据你传入的参数按需改变content。

```python
matrix = torch.randn([4, 3, 2, 1])
print("原矩阵形状: ", matrix.shape)
# 不带参数的默认的转置是线性代数意义上的矩阵转置
print("转置后的矩阵形状: ", matrix.T.shape)
# 可以传入参数规定矩阵中各个维度出现的顺序
print("将维度顺序换为0,1,3,2后的形状: ", matrix.permute(0, 1, 3, 2).shape)
```

以上代码输出：

```
原矩阵形状:  torch.Size([4, 3, 2, 1])
转置后的矩阵形状:  torch.Size([1, 2, 3, 4])
将维度顺序换为0,1,3,2后的形状:  torch.Size([4, 3, 1, 2])
```

### torch.unsqueeze

还是用学生成绩的例子，现在有分别来自两个学校的四个班级，每个班级35名学生，每名学生有8门课有成绩。

```python
# students' score of the two schools
course_score = torch.randn([4, 35, 8])
print("原来的成绩的shape: ", course_score.shape)
# 追加维度时axis给出0，则在首位追加一个维度
print("在下标0前追加一个维度后的shape: ", course_score.unsqueeze(0).shape)
# 给出的axis等于现存维度总数，会在最后追加一个维度
print("在下标3处追加一个维度后的shape: ", course_score.unsqueeze(3).shape)
# 给出的axis是任意小于维度总数的整数x，会在现有的x维度之前追加一个维度。
print("在下标2之前追加一个维度后的shape: ", course_score.unsqueeze(2).shape)
# 给出的axis是负数-x，则会从后往前index到x并追加一个维度
print("给出axis=-1追加一个维度后的shape: ", course_score.unsqueeze(-1).shape)
print("给出axis=-4追加一个维度后的shape: ", course_score.unsqueeze(-4).shape)
```

### torch.squeeze

`squeeze`本身是压榨、挤压的意思。其功能如起义，能够帮你将为1的维度挤压掉。

```python
# only squeeze for shape=1 dim
shape_0 = torch.zeros([1, 2, 1, 1, 3])
print("原始的shape_0的shape: ", shape_0.shape)
print("挤压掉为1的维度后的shape: ", shape_0.squeeze(1).shape)
print("挤压掉下标为0的维度后的shape: ", shape_0.squeeze(0).shape)
print("挤压掉下标为2的维度后的shape: ", shape_0.squeeze(2).shape)
print("挤压掉下标为-2的维度后的shape: ", shape_0.squeeze(-2).shape)
```

## torch.broadcast_tensors

以班级和学生成绩为例，现在有四个班级，每个班级有35名学生，每个学生有八门课的成绩。现规定[4,35,8]为全体学生的成绩。例如，第二门科目是物理，而本次物理考试较难，老师打算为所有学生的物理成绩加上5分。也就是，对于每个学生的成绩[4,x](一个八维数据)加上[0,5,0,0,0,0,0,0]。我们可以这样操作：

```python
origin = torch.randn([4, 35, 8])
another = torch.tensor([0, 5, 0, 0, 0, 0, 0, 0])
result, _ = torch.broadcast_tensors(another, origin)
# 这样another就会被扩展到origin的维度
print(result.shape)
```

[4, 16, 16, 32]习惯上将前面的维度称为大维度，后面的维度称为小维度。Broadcasting会先将小维度对齐，然后对于之前的每一个相对较大的维度，如果缺失，就补1。在这之后，这些为1的维度会被扩展到与目标相同的size。例如，将[32]进行broadcasting，目标是[4,16,16,32]:

```
(1) [32]->[1,1,1,32]
(2) [1,1,1,32]->[4,16,16,32]
```

原API描述为：

```
(1) Insert 1 dim ahead if needed
(2) Expand dims with size 1 to the same size
```

以下是一个更为现实的例子
![example](https://camo.githubusercontent.com/bee1dc67dd6f7870c9833871ee986832002be43c/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230313030393130343332363637332e706e673f782d6f73732d70726f636573733d696d6167652f77617465726d61726b2c747970655f5a6d46755a33706f5a57356e6147567064476b2c736861646f775f31302c746578745f6148523063484d364c7939696247396e4c6d4e7a5a473475626d56304c315a7063335668624552316333513d2c73697a655f31362c636f6c6f725f4646464646462c745f3730237069635f63656e746572)

需要注意的是比并不是所有的情况都能broadcasting。例如[4]与[1,3]，4并不能通过broadcasting变为3。

- 关于优化

  broadcasting进行了特别的优化。在进行broadcasting的过程中不会产生大量的数据复制。这会帮你减少复制带来的内存占用。

  > 请注意，你并不能随便Broadcasting。

例如，[2,32,32,14]并不能通过Broadingcast变为[4,32,32,14]。允许broadcasting的条件要求被扩展的维度缺失或为1.

## Summary of dim expanding

也就是说，我们现在有了三种扩展维度的方式。但是在Pytorch的使用过程中，我推荐使用expand搭配unsqueeze方法进行维度扩展。这种方式最为直观和简单。

`使用unsqueeze和expand中需要注意内存共享问题`

使用unsqueeze和expand时，如果你希望扩展之后每个对象的(比如[4,3,312,312]可以看做4张图像)的内容在运算后是一样的那你就不需要做额外的处理。此时扩展前的数据和扩展后的数据是共享内存的。比如我有一个1x3x312x312的张量。它经过expand处理后变成了4x3x312x312。表面上看起来是扩大了4倍，但是其实4个维度共享内存空间的。

如果你希望在维度处理后，每一个维度的数据是独立的，那你就需要做一个数据克隆。使用`.clone()`来使得输入和输出使用不同的内存。

















