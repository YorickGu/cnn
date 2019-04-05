import math

import tensorflow as tf

# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

class VDCNN():
    def __init__(self, num_classes, sequence_max_length=1024, num_quantized_chars=69, embedding_size=16,
                 use_he_uniform=True):
        # input tensors
        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training = tf.placeholder(tf.bool)
        num_filters = 128  # 卷积核个数   number of convolution kernel
        kernel_sizes = [3, 4, 5]  # 定义卷积层的大小
        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if use_he_uniform:
                self.embedding_W = tf.get_variable(name='lookup_W', shape=[num_quantized_chars, embedding_size],
                                                   initializer=tf.keras.initializers.he_uniform())
            else:
                self.embedding_W = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0),
                                               name="embedding_W")
            self.embedded_characters = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
            print("-" * 20)
            print("Embedded Lookup:", self.embedded_characters.get_shape())
            print("-" * 20)

        with tf.name_scope('cnn'):  # 词向量 卷积核个数，卷积核尺寸，卷积核训练，然后池化最大的,没有激活函数
            convs = []
            for fsz in kernel_sizes:
                l_conv = tf.layers.conv1d(self.embedded_characters, num_filters, fsz, activation=tf.nn.relu)
                l_pool = tf.reduce_max(l_conv, reduction_indices=[1])

                convs.append(l_pool)

        # fc1
        with tf.variable_scope('fc1'):
            outputs = tf.concat(convs, 1)
            w = tf.get_variable('w', [outputs.get_shape()[1], num_classes], initializer=he_normal,
                                regularizer=regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(1.0))
            self.fc3 = tf.matmul(outputs, w) + b


        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc3, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3, labels=self.input_y)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(losses) + sum(regularization_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
