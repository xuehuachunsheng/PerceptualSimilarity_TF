#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 09/08/2018 4:00 PM  
# @Author  : yanxuewu  
# @File    : DistModel.py
from util import util
import numpy as np
import tensorflow as tf
class DistModel(object):
    def __init__(self, torch_model_path='/Users/wuyanxue/PycharmProjects/PerceptualSimilarity_TF/alex.pth', trainable=False):
        self.alexnet, self.alexnetinnet = util.torch_to_tf_model(torch_model_path, nin_trainable=trainable)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()

    def distance(self, in0, in1):
        assert in0.shape == in1.shape
        return self.alexnetinnet.distance(in0, in1)

    def init_train(self, **kwargs):
        pass

    def train_batch_data(self, batch_data):
        ref_imgs = []
        p0_imgs = []
        p1_imgs = []
        judges = []
        for ref_img, p0_img, p1_img, judge in batch_data:
            ref_imgs.append(ref_img)
            p0_imgs.append(p0_img)
            p1_imgs.append(p1_img)
            judges.append(judge)

        ref_imgs_tensor = tf.convert_to_tensor(np.array(ref_imgs), dtype=tf.float32)
        p0_imgs_tensor = tf.convert_to_tensor(np.array(p0_imgs), dtype=tf.float32)
        p1_imgs_tensor = tf.convert_to_tensor(np.array(p1_imgs), dtype=tf.float32)
        judge_tensor = tf.convert_to_tensor(np.array(judges), dtype=tf.float32)

        d0 = self.distance(ref_imgs_tensor, p0_imgs_tensor)
        d1 = self.distance(ref_imgs_tensor, p1_imgs_tensor)

    def distance(self, in1_tensor, in2_tensor):

        # The input must be 64x64 scale
        # Both Input1 and Input2 are input images with shape [N, H, W, C] ndarray

        # Batch Normalization
        input1_bn = (in1_tensor - tf.contrib.framework.broadcast_to(self.alexnetinnet.shift, shape=tf.shape(in1_tensor))) / tf.contrib.framework.broadcast_to(self.alexnetinnet.scale, shape=tf.shape(in1_tensor))
        input2_bn = (in2_tensor - tf.contrib.framework.broadcast_to(self.alexnetinnet.shift, shape=tf.shape(in2_tensor))) / tf.contrib.framework.broadcast_to(self.alexnetinnet.scale, shape=tf.shape(in2_tensor))

        # Resize the input to 68x68 to ensure the output of the first layer is the same to that in torch

        # Step 2. Manully padding the input to Nx68x68x3
        # [N, H, W, C]
        input1_value_68 = tf.pad(input1_bn, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), mode='CONSTANT')
        input2_value_68 = tf.pad(input2_bn, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), mode='CONSTANT')

        relu_values1 = self.alexnet.forward(input1_value_68)
        relu_values2 = self.alexnet.forward(input2_value_68)

        # # Assert each output of the layers in alext has the correct shape.
        # assert relu_values1[0].shape[1:] == (15, 15, 64)
        # assert relu_values1[1].shape[1:] == (7, 7, 192)
        # assert relu_values1[2].shape[1:] == (3, 3, 384)
        # assert relu_values1[3].shape[1:] == (3, 3, 256)
        # assert relu_values1[4].shape[1:] == (3, 3, 256)

        diffs = []
        for out1, out2 in zip(relu_values1, relu_values2):
            # Step 3. Normalize
            out1_norm = util.normalize_tensor(out1)
            out2_norm = util.normalize_tensor(out2)
            diffs.append((out1_norm - out2_norm)**2)

        # The difference is the same as that in pytorch
        convs_value = self.alexnetinnet.forward(diffs)
        result = tf.reduce_mean(tf.reduce_mean(convs_value[0], axis=2), axis=1)
        for conv in convs_value[1:]:
            result += tf.reduce_mean(tf.reduce_mean(conv, axis=2), axis=1)
        #result = tf.reshape(result, shape=(result.shape[0], 1, 1, result.shape[1]))
        result = tf.squeeze(result)

        return result
