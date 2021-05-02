# 眼熟的代码块

这些代码块经常被使用。建议熟悉它们，并经常看看它们。

## 1.数据集相关

### 1.1 使用tensorflow自带的工具加载常见数据集

tensorflow包含了[mnist](https://en.wikipedia.org/wiki/MNIST_database)、fashion-mnist、cifar-10、cifar-100以及boston_housing等数据集的下载、读取功能。读取这些数据，你需要import相关包：

```python
from tensorflow.keras import datasets
```

`tensorflow.keras.dataset`包含的数据集读取方法返回训练集和测试集。

#### 1.1.1 载入数据集中的数据

例子：载入mnist手写识别数据集中的数据：

```python
(training_x, training_y), (testing_x, testing_y) = datasets.mnist.load_data()
```

例子：载入fashion-mnist服装数据集中的数据：

```python
(training_x, training_y), (testing_x, testing_y) = datasets.fashion_mnist.load_data()
```

你应该大致掌握了相应方法，它有一个通用的写法：

```python
(training_x, training_y), (testing_x, testing_y) = datasets.数据集名称.load_data()
```

这时你会得到四个数据部分，也就是训练集的图像`training_x`、训练图像对应的标签`training_y`、测试集的图像`testing_x`以及测试集的图像对应的标签`testing_y`。

此时你可以显示和检查一部分图片。详见本页的**1.2 查看部分数据集中图像**中相关内容。

#### 1.1.2 构建数据集

此时应该让这些数据成为两个小数据集，第一个是训练数据集，第二个是测试数据集。你可以使用`tensorflow.data.Dataset.from_tensor_slices()`方法完成这个过程：

```python
training_dataset = tf.data.Dataset.from_tensor_slices((training_x, training_y))
testing_dataset = tf.data.Dataset.from_tensor_slices((testing_x, testing_y))
```

当然，你也可以不这样做。具体取决于你希望怎么迭代你的数据集。上述方法也不仅限于从tensorflow工具包里导入的数据集，在本地的数据集也可以被这样构建为运行时数据集。

#### 1.1.3 标准化和前处理

如果你不知道什么是标准化，请参考[附录-常见名词](../appendix/similar-vocabularies.md)中有关标准化、去量纲的词条。``

在训练图像数据时，特征图一般是float32的，并且要标准化。而对应的标签一般是整数（索引）。所以，我们一般对x和y做以下前处理：

```python
training_x = (training_x.astype('float32') / 255.)
testing_x = (testing_x.astype('float32') / 255.)
```

#### 1.1.4 指定batch大小以及将数据集打乱

在你经常看到各种SGD（参阅[附录-常见名词](../appendix/similar-vocabularies.md)中关于随机梯度下降的词条）方法中，需要批量取用数据集，也就是我们经常指定的batch。tensorflow也提供了指定`Dataset`对象batch size的方法：

```python
training_dataset = training_dataset.batch(128)
```

在这之后，往往我们将数据集打乱：

```python
training_dataset = training_dataset.shuffle(100)
```

testing_x和testing_y是否有必要打乱请自行根据需求决定。

#### 1.1.5 总结

把上面的整个过程写在一起，就是：

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, initializers

(training_x, training_y), (testing_x, testing_y) = datasets.fashion_mnist.load_data()
training_x = (training_x.astype('float32') / 255.)
testing_x = (testing_x.astype('float32') / 255.)
batch_size = 128

training_dataset = tf.data.Dataset.from_tensor_slices((training_x, training_y))
training_dataset = training_dataset.batch(batch_size)
testing_dataset = tf.data.Dataset.from_tensor_slices((testing_x, testing_y))
testing_dataset = testing_dataset.batch(batch_size)
```

于是你的到了一份可以直接投入训练和测试的数据集。

### 1.2 查看部分数据集中图像

当你导入了一些图片作为数据集时，可以显示一部分图片，检查导入的是否正确。这不是必要的，但是很好玩。我们定义一个这样的函数方便经常使用：

```python
import matplotlib.pyplot as plt
import math
def check_image_in_dataset_via_plot(dataset, how_many: float = 36., gray_scale=False):
    plt_width = math.ceil(how_many ** .5)
    plt.figure(figsize=(plt_width, plt_width))
    for i in range(round(how_many)):
        plt.subplot(plt_width, plt_width, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if not gray_scale:
            plt.imshow(dataset[i])
        else:
            plt.imshow(dataset[i], cmap=plt.cm.binary)
        # marking index under the picture
        # plt.xlabel(class_names[labels[i][0]])
    plt.show()
```

其中`dataset`是一个list，其中包含若干图片。`how_many`决定你要显示几张图片，`gray_scale`决定是否显示为灰度图（可以指定灰度是因为有的数据集确实是灰度图，比如`fashion-mnist`）。

举个使用的例子，我们导入了一份`fashion-mnist`数据集中的图片，显示其中的36灰度图：

```python
check_image_in_dataset_via_plot(imsges, how_many=36, gray_scale=True)
```

其中images是图像的list。你会看到这样的显示：

![image-20210413100216656](src/similar-codeblocks/image-20210413100216656.png)



## 2.计算机视觉相关