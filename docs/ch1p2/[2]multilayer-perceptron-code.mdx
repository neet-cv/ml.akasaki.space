# 多层感知机的代码实现

我们已经从上一节里了解了多层感知机的原理。下面，我们一起来动手实现一个多层感知机。首先导入实现所需的包或模块。

``` python
import tensorflow as tf
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_tensorflow
import d2lzh_tensorflow2 as d2l
print(tf.__version__)
```

## 手动实现

### 获取和读取数据

这里继续使用Fashion-MNIST数据集。我们将使用多层感知机对图像进行分类。

``` python
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train/255.0
x_test = x_test/255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
```

### 定义模型参数

我们在3.6节（softmax回归的从零开始实现）里已经介绍了，Fashion-MNIST数据集中图像形状为 $28 \times 28$，类别数为10。本节中我们依然使用长度为 $28 \times 28 = 784$ 的向量表示每一张图像。因此，输入个数为784，输出个数为10。实验中，我们设超参数隐藏单元个数为256。

``` python
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens),mean=0, stddev=0.01, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs),mean=0, stddev=0.01, dtype=tf.float32))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.1))
```

###  定义激活函数

这里我们使用基础的`max`函数来实现ReLU，而非直接调用`relu`函数。

``` python
def relu(x):
    return tf.math.maximum(x,0)
```

### 定义模型

同softmax回归一样，我们通过`reshape`函数将每张原始图像改成长度为`num_inputs`的向量。然后我们实现上一节中多层感知机的计算表达式。

``` python
def net(X):
    X = tf.reshape(X, shape=[-1, num_inputs])
    h = relu(tf.matmul(X, W1) + b1)
    return tf.math.softmax(tf.matmul(h, W2) + b2)
```

### 定义损失函数

为了得到更好的数值稳定性，我们直接使用Tensorflow提供的包括softmax运算和交叉熵损失计算的函数。

``` python
def loss(y_hat,y_true):
    return tf.losses.sparse_categorical_crossentropy(y_true,y_hat)
```

### 训练模型

训练多层感知机的步骤和3.6节中训练softmax回归的步骤没什么区别。我们直接调用`d2l`包中的`train_ch3`函数，它的实现已经在3.6节里介绍过。我们在这里设超参数迭代周期数为5，学习率为0.5。

``` python
num_epochs, lr = 5, 0.5
params = [W1, b1, W2, b2]
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
```

输出：

```
epoch 1, loss 0.8208, train acc 0.693, test acc 0.804
epoch 2, loss 0.4784, train acc 0.822, test acc 0.832
epoch 3, loss 0.4192, train acc 0.843, test acc 0.850
epoch 4, loss 0.3874, train acc 0.857, test acc 0.858
epoch 5, loss 0.3651, train acc 0.864, test acc 0.860
```

---

## 使用框架实现

下面我们使用Tensorflow来实现上一节中的多层感知机。首先导入所需的包或模块。

``` python
import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
```

### 定义模型

和softmax回归唯一的不同在于，我们多加了一个全连接层作为隐藏层。它的隐藏单元个数为256，并使用ReLU函数作为激活函数。

``` python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu',),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 读取数据并训练模型

我们使用之前训练softmax回归几乎相同的步骤来读取数据并训练模型。

``` python
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.5),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5,
              batch_size=256,
              validation_data=(x_test, y_test),
              validation_freq=1)
```

输出：

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 2s 33us/sample - loss: 0.7428 - accuracy: 0.7333 - val_loss: 0.5489 - val_accuracy: 0.8049
Epoch 2/5
60000/60000 [==============================] - 1s 22us/sample - loss: 0.4774 - accuracy: 0.8247 - val_loss: 0.4823 - val_accuracy: 0.8288
Epoch 3/5
60000/60000 [==============================] - 1s 21us/sample - loss: 0.4111 - accuracy: 0.8497 - val_loss: 0.4448 - val_accuracy: 0.8401
Epoch 4/5
60000/60000 [==============================] - 1s 21us/sample - loss: 0.3806 - accuracy: 0.8600 - val_loss: 0.5326 - val_accuracy: 0.8132
Epoch 5/5
60000/60000 [==============================] - 1s 21us/sample - loss: 0.3603 - accuracy: 0.8681 - val_loss: 0.4217 - val_accuracy: 0.8448
<tensorflow.python.keras.callbacks.History at 0x7f9868e12310>
```
