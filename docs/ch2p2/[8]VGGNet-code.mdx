---
sidebar_position: 8
---

# VGG的代码实现

这里不会包含任何讲解的内容，讲解的内容请参考[VGG：可复用的网络块](./[7]VGGNet)。

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    # 卷积层 01
    Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    # 卷积层 02
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', ),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
    Dropout(0.2),  # 丢弃层
    # 卷积层 03
    Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    # 卷积层 04
    Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
    Dropout(0.2),  # 丢弃层
    # 卷积层 05
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层1
    Activation('relu'),  # 激活层1
    # 卷积层 06
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层1
    Activation('relu'),  # 激活层1
    Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层1
    Activation('relu'),  # 激活层1
    # 卷积层 07
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
    Dropout(0.2),  # 丢弃层
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    # 卷积层 08
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    # 卷积层 09
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),
    # 卷积层 10
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    # 卷积层 11
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    # 卷积层 12
    Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
    BatchNormalization(),  # BN层
    Activation('relu'),  # 激活层
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),  # 池化层
    Dropout(0.2),  # 丢弃层
    # 打平进入全连接
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),  # 丢弃层
    Dense(512, activation='relu'),
    Dropout(0.2),  # 丢弃层
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
```

