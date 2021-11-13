---
sidebar_position: 5
---

# AlexNet代码实现

这里不会包含任何讲解的内容，讲解的内容请参考[AlexNet：更深的卷积神经网络](./[4]AlexNet)。

```python
分类import tensorflow as tf
from tensorflow.keras import layers, datasets, Sequential

(training_x, training_y), (testing_x, testing_y) = datasets.cifar10.load_data()
training_x = (training_x.astype('float32') / 255.)
testing_x = (testing_x.astype('float32') / 255.)

model = Sequential([
    # 第一层卷积
    layers.Conv2D(filters=96, kernel_size=(3, 3)),
    # 第一层卷积具有BN层
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    # 第二层卷积
    layers.Conv2D(filters=256, kernel_size=(3, 3)),
    # 第二层卷积具有BN层
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    # 第三层卷积，并没有BN层，也没有池化
    layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                  activation='relu'),
    # 第四层卷积，并没有BN层，也没有池化
    layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                  activation='relu'),
    # 第五层卷积，并没有BN层
    layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    # 打平进入全连接进行分类
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(training_x, training_y, batch_size=32, epochs=20, validation_data=(testing_x, testing_y))
```

