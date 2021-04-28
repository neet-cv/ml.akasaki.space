#  丢弃法

除了前一节介绍的权重衰减以外，深度学习模型常常使用丢弃法（dropout）来应对过拟合问题。丢弃法有一些不同的变体。本节中提到的丢弃法特指倒置丢弃法（inverted dropout）。

## 方法

回忆一下，多层感知机中曾经描述一个单隐藏层的多层感知机。其中输入个数为4，隐藏单元个数为5，且隐藏单元$h_i$（$i=1, \ldots, 5$）的计算表达式为

$$
h_i = \phi\left(x_1 w_{1i} + x_2 w_{2i} + x_3 w_{3i} + x_4 w_{4i} + b_i\right)
$$

这里$\phi$是激活函数，$x_1, \ldots, x_4$是输入，隐藏单元$i$的权重参数为$w_{1i}, \ldots, w_{4i}$，偏差参数为$b_i$。当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为$p$，那么有$p$的概率$h_i$会被清零，有$1-p$的概率$h_i$会除以$1-p$做拉伸。丢弃概率是丢弃法的超参数。具体来说，设随机变量$\xi_i$为0和1的概率分别为$p$和$1-p$。使用丢弃法时我们计算新的隐藏单元$h_i'$

$$
h_i' = \frac{\xi_i}{1-p} h_i
$$

由于$E(\xi_i) = 1-p$，因此

$$
E(h_i') = \frac{E(\xi_i)}{1-p}h_i = h_i
$$

即**丢弃法不改变其输入的期望值**。让我们对图3.3中的隐藏层使用丢弃法，一种可能的结果如下图所示，其中$h_2$和$h_5$被清零。这时输出值的计算不再依赖$h_2$和$h_5$，在反向传播时，与这两个隐藏单元相关的权重的梯度均为0。由于在训练中隐藏层神经元的丢弃是随机的，即$h_1, \ldots, h_5$都有可能被清零，输出层的计算无法过度依赖$h_1, \ldots, h_5$中的任一个，从而在训练模型时起到正则化的作用，并可以用来应对过拟合。在测试模型时，我们为了拿到更加确定性的结果，一般不使用丢弃法。

![img](src/dropout/3.13_dropout.svg)

## 从零开始实现

根据丢弃法的定义，我们可以很容易地实现它。下面的`dropout`函数将以`drop_prob`的概率丢弃`X`中的元素。

``` python
import tensorflow as tf
import numpy as np
from tensorflow import keras, nn， losses
from tensorflow.keras.layers import Dropout, Flatten, Dense

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return tf.zeros_like(X)
    #初始mask为一个bool型数组，故需要强制类型转换
    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) < keep_prob
    return tf.cast(mask, dtype=tf.float32) * tf.cast(X, dtype=tf.float32) / keep_prob
```

我们运行几个例子来测试一下`dropout`函数。其中丢弃概率分别为0、0.5和1。

``` python
X = tf.reshape(tf.range(0, 16), shape=(2, 8))
dropout(X, 0)
```

``` python
dropout(X, 0.5)
```

``` python
dropout(X, 1.0)
```

### 定义模型参数

实验中，我们依然使用softmax回归的从零开始实现中介绍的Fashion-MNIST数据集。我们将定义一个包含两个隐藏层的多层感知机，其中两个隐藏层的输出个数都是256。

``` python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = tf.Variable(tf.random.normal(stddev=0.01, shape=(num_inputs, num_hiddens1)))
b1 = tf.Variable(tf.zeros(num_hiddens1))
W2 = tf.Variable(tf.random.normal(stddev=0.1, shape=(num_hiddens1, num_hiddens2)))
b2 = tf.Variable(tf.zeros(num_hiddens2))
W3 = tf.Variable(tf.random.truncated_normal(stddev=0.01, shape=(num_hiddens2, num_outputs)))
b3 = tf.Variable(tf.zeros(num_outputs))

params = [W1, b1, W2, b2, W3, b3]
```

### 定义模型

下面定义的模型将全连接层和激活函数ReLU串起来，并对每个激活函数的输出使用丢弃法。我们可以分别设置各个层的丢弃概率。通常的建议是把靠近输入层的丢弃概率设得小一点。在这个实验中，我们把第一个隐藏层的丢弃概率设为0.2，把第二个隐藏层的丢弃概率设为0.5。我们可以通过参数`is_training`函数来判断运行模式为训练还是测试，并只需在训练模式下使用丢弃法。

``` python
drop_prob1, drop_prob2 = 0.2, 0.5

def net(X, is_training=False):
    X = tf.reshape(X, shape=(-1,num_inputs))
    H1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    if is_training:# 只在训练模型时使用丢弃法
      H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = nn.relu(tf.matmul(H1, W2) + b2)
    if is_training:
      H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return tf.math.softmax(tf.matmul(H2, W3) + b3)
```



### 训练和测试模型

这部分与之前多层感知机的训练和测试类似。

``` python
from tensorflow.keras.datasets import fashion_mnist

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    global sample_grads
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X, is_training=True)
                l = loss(y_hat, tf.one_hot(y, depth=10, axis=-1, dtype=tf.float32))
            
            grads = tape.gradient(l, params)
            if trainer is None:
                
                sample_grads = grads
                params[0].assign_sub(grads[0] * lr)
                params[1].assign_sub(grads[1] * lr)
            else:
                trainer.apply_gradients(zip(grads, params))  # “softmax回归的简洁实现”一节将用到

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

        
loss = tf.losses.CategoricalCrossentropy()
num_epochs, lr, batch_size = 5, 0.5, 256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test,tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

输出：

```
epoch 1, loss 0.0369, train acc 0.530, test acc 0.663
epoch 2, loss 0.0261, train acc 0.658, test acc 0.704
epoch 3, loss 0.0232, train acc 0.694, test acc 0.727
epoch 4, loss 0.0215, train acc 0.714, test acc 0.741
epoch 5, loss 0.0204, train acc 0.727, test acc 0.748
```



## 简洁实现

在Tensorflow2.0中，我们只需要在全连接层后添加`Dropout`层并指定丢弃概率。在训练模型时，`Dropout`层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时（即`model.eval()`后），`Dropout`层并不发挥作用。

``` python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256,activation='relu'),
    Dropout(0.2),
    keras.layers.Dense(256,activation='relu'),
    Dropout(0.5),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])


```

下面训练并测试模型。

``` python
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=256,validation_data=(x_test, y_test),
                    validation_freq=1)
```

输出：

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 [==============================] - 2s 28us/sample - loss: 0.6704 - accuracy: 0.7647 - val_loss: 0.4495 - val_accuracy: 0.8352
Epoch 2/5
60000/60000 [==============================] - 1s 15us/sample - loss: 0.4306 - accuracy: 0.8447 - val_loss: 0.4112 - val_accuracy: 0.8525
Epoch 3/5
60000/60000 [==============================] - 1s 14us/sample - loss: 0.3902 - accuracy: 0.8590 - val_loss: 0.3898 - val_accuracy: 0.8588
Epoch 4/5
60000/60000 [==============================] - 1s 15us/sample - loss: 0.3618 - accuracy: 0.8687 - val_loss: 0.3590 - val_accuracy: 0.8713
Epoch 5/5
60000/60000 [==============================] - 1s 14us/sample - loss: 0.3423 - accuracy: 0.8756 - val_loss: 0.3617 - val_accuracy: 0.8718
```



## 小结

* 我们可以通过使用丢弃法应对过拟合。
* 丢弃法只在训练模型时使用。

