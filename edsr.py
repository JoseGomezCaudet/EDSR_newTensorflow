from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, scaling_factor=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform')
        self.relu = tf.keras.layers.ReLU()
        self.scaling_factor = scaling_factor

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x *= self.scaling_factor
        return inputs + x
    


class Edsr(tf.keras.Model):
    def __init__(self, B, F, scale):
        super(Edsr, self).__init__()
        self.B = B
        self.F = F
        self.scale = scale
        self.res_blocks = [ResBlock(F) for _ in range(B)]
        self.conv1 = tf.keras.layers.Conv2D(F, (3, 3), padding='same', kernel_initializer='glorot_uniform')
        self.conv2 = tf.keras.layers.Conv2D(F, (3, 3), padding='same', kernel_initializer='glorot_uniform')
        self.conv3 = tf.keras.layers.Conv2D(3 * (scale ** 2), (3, 3), padding='same', kernel_initializer='glorot_uniform')

    def call(self, inputs):
        x = self.conv1(inputs)
        out1 = x

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.conv2(x)
        x += out1

        x = self.conv3(x)
        x = tf.nn.depth_to_space(x, self.scale)
        return x