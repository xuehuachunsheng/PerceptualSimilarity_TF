#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 14/08/2018 5:56 PM  
# @Author  : yanxuewu  
# @File    : AlexNetInNet.py  

from . import AlexNet
import tensorflow as tf
import numpy as np
class AlexNetInNet:
	def __init__(self, alexnet=None, ex_weights=None, trainable=False):
		self._alexnet = alexnet
		self.trainable = trainable
		self.__net_in_net(ex_weights)

		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)


	def __net_in_net(self, ex_weights=None): # Construct the net-in-net structure
		self.chns = [64, 192, 384, 256, 256]
		self.ex_weights = []
		if ex_weights is None:
			for i in range(5):
				self.ex_weights.append(tf.truncated_normal([self.chns[i]],  dtype=tf.float32, stddev=1e-1))
		else:
			for i in range(5):
				c_tensor = tf.convert_to_tensor(ex_weights[i], dtype=tf.float32)
				self.ex_weights.append(c_tensor)
		# Input shape [N, H, W, C]
		self.diff1 = tf.placeholder(dtype=tf.float32, shape=[None, 16, 16, 64])
		self.diff2 = tf.placeholder(dtype=tf.float32, shape=[None, 7, 7, 192])
		self.diff3 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3, 384])
		self.diff4 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3, 256])
		self.diff5 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3, 256])

		# Extra Layer 1
		dropout1 = tf.nn.dropout(self.diff1, keep_prob=0.5, name='ex_dropout1')
		kernel1 = tf.Variable(tf.reshape(self.ex_weights[0], (1, 1, self.chns[0], 1)), dtype=tf.float32, name='ex_kernel1')
		conv1 = tf.nn.conv2d(dropout1, kernel1, strides=[1, 1, 1, 1], padding='SAME', name='ex_conv1')

		dropout2 = tf.nn.dropout(self.diff2, keep_prob=0.5, name='ex_dropout2')
		kernel2 = tf.Variable(tf.reshape(self.ex_weights[1], (1, 1, self.chns[1], 1)), dtype=tf.float32, name='ex_kernel2')
		conv2 = tf.nn.conv2d(dropout2, kernel2, strides=[1, 1, 1, 1], padding='SAME', name='ex_conv2')

		dropout3 = tf.nn.dropout(self.diff3, keep_prob=0.5, name='ex_dropout3')
		kernel3 = tf.Variable(tf.reshape(self.ex_weights[2], (1, 1, self.chns[2], 1)), dtype=tf.float32, name='ex_kernel3')
		conv3 = tf.nn.conv2d(dropout3, kernel3, strides=[1, 1, 1, 1], padding='SAME', name='ex_conv3')

		dropout4 = tf.nn.dropout(self.diff4, keep_prob=0.5, name='ex_dropout4')
		kernel4 = tf.Variable(tf.reshape(self.ex_weights[3], (1, 1, self.chns[3], 1)), dtype=tf.float32, name='ex_kernel4')
		conv4 = tf.nn.conv2d(dropout4, kernel4, strides=[1, 1, 1, 1], padding='SAME', name='ex_conv4')

		dropout5 = tf.nn.dropout(self.diff5, keep_prob=0.5, name='ex_dropout5')
		kernel5 = tf.Variable(tf.reshape(self.ex_weights[4], (1, 1, self.chns[4], 1)), dtype=tf.float32, name='ex_kernel5')
		conv5 = tf.nn.conv2d(dropout5, kernel5, strides=[1, 1, 1, 1], padding='SAME', name='ex_conv5')

		self.convs = [conv1, conv2, conv3, conv4, conv5]


	def distance(self, in1, in2):
		# obtain two feature mapss

		# Step 1. preprocess

		# Step 2. Input1 and Input2
		relu_values1 = self._alexnet.relu_values(in1)
		relu_values2 = self._alexnet.relu_values(in2)

		# Step 3. Normalize

		# Step 4. Solve difference between these two values

	@property
	def net(self):
		return self._alexnet









