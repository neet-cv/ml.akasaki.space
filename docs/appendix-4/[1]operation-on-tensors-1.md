# Tensor的基本操作：索引、切片和广播



索引和切片能够很方便的帮你拿到你想要的部分数据。

```python
# importing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
a = tf.constant([[0,1,0,0],[1,0,1,0],[0,0,0,1],[0,1,0,0]])
print(a@a)
```

```
tf.Tensor(
[[1 0 1 0]
 [0 1 0 1]
 [0 1 0 0]
 [1 0 1 0]], shape=(4, 4), dtype=int32)
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
tensor_0 = tf.random.normal([4,28,28,3]) # 假设这是一个含有四张28x28像素彩色图片的数据集
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
数据集的shape:  (4, 28, 28, 3)
第一张照片的shape:  (28, 28, 3)
和上一行等效，时第一张图片的shape:  (28, 28, 3)
所有照片的所有像素的Red通道的shape:  (4, 28, 28)
和上一行等效，时所有像素的Red通道的shape:  (4, 28, 28)
```

## Selective index

使用冒号等能获得连续的一段数据，而SelectiveIndex能获取可以定义具体的采样方式。

### tensorflow.gather

```python
# indices意味指标。在使用gather时需要一个选择的指标作为参数。
# 举个例子，有4个班级，每个班级5名学生，每个学生有5门课程。数据集表示所有学生的所有课程的成绩。
course_score = tf.random.normal([4,35,8])
print("二班和三班成绩的shape: ",tf.gather(course_score,axis=0,indices=[2,3]).shape) # 在维度0中选择下标2和3。
print("通过简单索引获得相同效果: ",course_score[2:4].shape) # 与上述语句等效的传统的选择方式
# 使用gather能让你在选择顺序上下手脚
print("三班和二班的成绩的shape: ",tf.gather(course_score,axis=0,indices=[3,2]).shape) # 改变了选择的顺序
print("抽取每个班第 2 3 4 个学生的成绩",tf.gather(course_score,axis=1,indices=[1,2,3]).shape) # gather可以取任意维度
print("抽取所有学生的第 2 3 门课程的成绩",tf.gather(course_score,axis=2,indices=[1,2]).shape) # 还是演示改变选择的维度
```

以上代码的输出：

```
二班和三班成绩的shape:  (2, 35, 8)
通过简单索引获得相同效果:  (2, 35, 8)
三班和二班的成绩的shape:  (2, 35, 8)
抽取每个班第 2 3 4 个学生的成绩 (4, 3, 8)
抽取所有学生的第 2 3 门课程的成绩 (4, 35, 2)
```

### tensorflow.gather_nd

```python
# gater_nd 能够使用更复杂的筛选条件
print("取第1个班级的第2个同学的第3门课的成绩",tf.gather_nd(course_score,[0,1,2]).shape)
print("分别取第1和第2个班级的第3个和第4个学生的成绩",tf.gather_nd(course_score,[[0,1],[2,3]]).shape)
```

输出：

```
取第1个班级的第2个同学的第3门课的成绩 ()
分别取第1和第2个班级的第3个和第4个学生的成绩 (2, 8)
```

### tensorflow.boolean_mask

```python
# boolean_mask 能通过一个布尔序列进行筛选。boolean_mask可以在某种程度上实现reshape
print("对第一个维度进行筛选，只选择前两个",tf.boolean_mask(tensor_0,mask=[True,True,False,False]).shape)
print("筛选出每张图片所有像素的R和G通道",tf.boolean_mask(tensor_0,axis=3,mask=[True,True,False]))

a=tf.ones([2,3,4])
# 在这里我们选用一个两行三列的mask，涉及到原样本a的前两个维度的筛选
print("筛选出[0,0]<4>[1,1]<4>[1,2]<4>",tf.boolean_mask(a,[[True,False,False],[False,True,True]])); # 不懂，等着再看看
```

## 维度变换

维度变换是学习人工神经网络需要理解的基本概念之一。
对于一个多维的数据，可以有不同的理解方式。例如，我们有一份[4,28,28,3]的数据集，代表一个含有四张28x28像素的彩色图片的数据集。接下来我们提供两种理解方式：
这是四张图片。
这是4x28行像素，每行有28个。
其实理论上理解方式可以有很多种。但是无论采用何种方式理解，数据的content本身是不会变化的。我们称不同的理解方式为不同的View。

### tensorflow.reshape

Reshape就是一个用不同的View理解Tensor的过程。Reshape是全连接层常用操作之一。

```python
tensor_1 = tf.random.normal([4,28,28,3])
print("原来的tensor_1: ",tensor_1.shape)
# tensorflow.reshape需要传入一个原始数据和一个你希望得到的view，就能在允许的范围内进行reshape
print("在reshape为view[4, 28*28, 3]之后: ",tf.reshape(tensor_1,[4,784,3]).shape)
# 上述代码中的view可以等价写为以下代码中的形式
print("在reshape为view[4,-1,3]之后: ",tf.reshape(tensor_1,[4,-1,3]).shape)
# 注意，一个view中只能写下一个-1，reshape方法会根据你写下的-1自动推算维度。如果同时出现多个-1，就会导致无法推算的问题
print("在reshape为view[4,28*28*3]之后: ",tf.reshape(tensor_1,[4,28*28*3]).shape)
# 同样，上述代码可以使用-1
print("在reshape为view[4,-1]之后: ",tf.reshape(tensor_1,[4,-1]).shape)
# 这种类型情况下的reshape通常是可逆的。你可以把它reshape回来
print("试图reshape回veiw[4,28,28,3]: ",tf.reshape(tf.reshape(tensor_1,[4,-1]),[4,28,28,3]).shape)
```

以上代码的输出：

```
原来的tensor_1:  (4, 28, 28, 3)
在reshape为view[4, 28*28, 3]之后:  (4, 784, 3)
在reshape为view[4,-1,3]之后:  (4, 784, 3)
在reshape为view[4,28*28*3]之后:  (4, 2352)
在reshape为view[4,-1]之后:  (4, 2352)
试图reshape回veiw[4,28,28,3]:  (4, 28, 28, 3)
```

### tensorflow.transpose

`tensorflow.reshape`可以帮助你获得不同view下的content，但是并不能改变content。而`转置(transpose)`可以帮助你更换content中各个维度的顺序。 转置操作可以是线性代数意义上的(默认情况)，也可以根据你传入的参数按需改变content。

```python
matrix = tf.random.normal([4,3,2,1])
print("原矩阵形状: ",matrix.shape)
# 不带参数的默认的转置是线性代数意义上的矩阵转置
print("转置后的矩阵形状: ",tf.transpose(matrix).shape)
# 可以传入参数规定矩阵中各个维度出现的顺序
print("将维度顺序换为0,1,3,2后的形状: ",tf.transpose(matrix,perm=[0,1,3,2]).shape)
```

以上代码输出：

```
原矩阵形状:  (4, 3, 2, 1)
转置后的矩阵形状:  (1, 2, 3, 4)
将维度顺序换为0,1,3,2后的形状:  (4, 3, 1, 2)
```

### tensorflow.expand_dims

还是用学生成绩的例子，现在有分别来自两个学校的四个班级，每个班级35名学生，每名学生有8门课有成绩。

```python
# students' score of the two schools  
course_score = tf.random.normal([4,35,8])
print("原来的成绩的shape: ",course_score.shape)
# 追加维度时axis给出0，则在首位追加一个维度
print("在下标0前追加一个维度后的shape: ",tf.expand_dims(course_score,axis=0).shape)
# 给出的axis等于现存维度总数，会在最后追加一个维度
print("在下标3处追加一个维度后的shape: ",tf.expand_dims(course_score,axis=3).shape)
# 给出的axis是任意小于维度总数的整数x，会在现有的x维度之前追加一个维度。
print("在下标2之前追加一个维度后的shape: ",tf.expand_dims(course_score,axis=2).shape)
# 给出的axis是负数-x，则会从后往前index到x并追加一个维度  
print("给出axis=-1追加一个维度后的shape: ",tf.expand_dims(course_score,axis=-1).shape)
print("给出axis=-4追加一个维度后的shape: ",tf.expand_dims(course_socre,axis=-4).shape)
```

### tensorflow.queeze

`squeeze`本身是压榨、挤压的意思。其功能如起义，能够帮你将为1的维度挤压掉。

```python
# only squeeze for shape=1 dim
shape_0 = tf.zeros([1,2,1,1,3])
print("原始的shape_0的shape: ",shape_0.shape)
print("挤压掉为1的维度后的shape: ",tf.squeeze(shape_0).shape)
print("挤压掉下标为0的维度后的shape: ",tf.squeeze(shape_0,axis=0).shape)
print("挤压掉下标为2的维度后的shape: ",tf.squeeze(shape_0,axis=2).shape)
print("挤压掉下标为-2的维度后的shape: ",tf.squeeze(shape_0,axis=-2).shape)
```



## tensorflow.broadcast

以班级和学生成绩为例，现在有四个班级，每个班级有35名学生，每个学生有八门课的成绩。现规定 `[4,35,8]` 为全体学生的成绩。例如，第二门科目是物理，而本次物理考试较难，老师打算为所有学生的物理成绩加上5分。也就是，对于每个学生的成绩 `[4,x]` (一个八维数据)加上 `[0,5,0,0,0,0,0,0]`。我们可以这样操作：

```python
origin = tensorflow.random.normal([4,35,8])
another = tensorflow.constant([0,5,0,0,0,0,0,0])
result = tensorflow.broadcast_to(another, origin)
# 这样another就会被扩展到origin的维度
```

`[4, 16, 16, 32]` 习惯上将前面的维度称为大维度，后面的维度称为小维度。Broadcasting会先将小维度对齐，然后对于之前的每一个相对较大的维度，如果缺失，就补1。在这之后，这些为1的维度会被扩展到与目标相同的size。例如，将[32]进行broadcasting，目标是 `[4,16,16,32]`:

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

  > 请注意，你并不额能随便Broadcasting。

例如，`[2,32,32,14]` 并不能通过Broadingcast变为 `[4,32,32,14]`。允许broadcasting的条件要求被扩展的维度缺失或为1.

## Summary of dim expanding

也就是说，我们现在有了三种扩展维度的方式。以[3,4]扩展到[2,3,4]为例：
第一种是broadcasting:

```python
origin = tf.ones([3,4])
result = tf.broadcast_to(origin, [2,3,4])
print(result)
```

输出：

```
tf.Tensor(
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]], shape=(2, 3, 4), dtype=float32)
```

第二种是expand_dims:

```python
origin = tf.ones([3,4])
result = tf.expand_dims(origin, axis=0)
result = tf.tile(result, [2,1,1])
print(result)
```

输出：

```
tf.Tensor(
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]], shape=(2, 3, 4), dtype=float32)
```

可以看到，两种方法得到的结果是一样的。不过区别是broadcasting存在优化，而expand和tile并没有优化。