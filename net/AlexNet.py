#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 14/08/2018 2:39 PM  
# @Author  : yanxuewu  
# @File    : AlexNet.py  
import tensorflow as tf
import numpy as np
class AlexNet(object):
	def __init__(self, weights=None, biases=None, trainable=False):
		'''
		:param weights: must be a length-5 ndarray and each element is a ndarray with different shape
		Note that each element in weights must be shape as [kernel_size, kernel_size, in_C, out_C]
		:param biases: must be a length-5 ndarray and each element is a ndarray with different shape
		Note that each element in weights must be shape as [out_C]
		:param trainable: if the parameters are trainable
		'''
		# The input images with shape [N, H, W, 3] placeholder
		self.images = tf.placeholder(tf.float32, [None, None, None, 3])
		self.weights = []
		self.biases = []

		if weights is None: # Randomly initialize the parameters
			conv_kernel1 = tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1)
			self.weights.append(conv_kernel1)
			conv_kernel2 = tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1)
			self.weights.append(conv_kernel2)
			conv_kernel3 = tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1)
			self.weights.append(conv_kernel3)
			conv_kernel4 = tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1)
			self.weights.append(conv_kernel4)
			conv_kernel5 = tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1)
			self.weights.append(conv_kernel5)
		else: # Load parameters
			for i, weight in enumerate(weights):
				c_tensor = tf.convert_to_tensor(weight)
				self.weights.append(c_tensor)
		if biases is None: # Randomly initialize the parameters
			bias1 = tf.constant(0.0, shape=[64], dtype=tf.float32, stddev=1e-1)
			bias2 = tf.constant(0.0, shape=[192], dtype=tf.float32, stddev=1e-1)
			bias3 = tf.constant(0.0, shape=[384], dtype=tf.float32, stddev=1e-1)
			bias4 = tf.constant(0.0, shape=[256], dtype=tf.float32, stddev=1e-1)
			bias5 = tf.constant(0.0, shape=[256], dtype=tf.float32, stddev=1e-1)
			self.biases = [bias1, bias2, bias3, bias4, bias5]
		else:
			for bias in biases:
				c_tensor = tf.convert_to_tensor(bias)
				self.biases.append(c_tensor)
		self.trainable = trainable
		self.__network()

		self.sess = tf.Session()

		# Initialize
		init = tf.global_variables_initializer()
		self.sess.run(init)

	def __network(self):
		# Conv1
		kernel1 = tf.Variable(self.weights[0], dtype=tf.float32, trainable=self.trainable, name='kernel1')
		bias1 = tf.Variable(self.biases[0], dtype=tf.float32, trainable=self.trainable, name='bias1')
		# Here, the output is not the same as pytorch implementation
		# [N, H, W, C] = [N, 16, 16, C]
		# However the pytorch is [N, C, 15, 15]
		conv1 = tf.nn.conv2d(self.images, kernel1, strides=[1, 4, 4, 1], padding='SAME', name='conv1')
		bias_conv1 = tf.nn.bias_add(conv1, bias1, name='bias_conv1')
		relu1 = tf.nn.relu(bias_conv1, name='relu1')

		# Conv2
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
		kernel2 = tf.Variable(self.weights[1], dtype=tf.float32, trainable=self.trainable, name='kernel2')
		bias2 = tf.Variable(self.biases[1], dtype=tf.float32, trainable=self.trainable, name='bias2')
		conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
		bias_conv2 = tf.nn.bias_add(conv2, bias2, name='bias_conv2')
		relu2 = tf.nn.relu(bias_conv2, name='relu2')

		# Conv3
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
		kernel3 = tf.Variable(self.weights[2], dtype=tf.float32, trainable=self.trainable, name='kernel3')
		bias3 = tf.Variable(self.biases[2], dtype=tf.float32, trainable=self.trainable, name='bias3')
		conv3 = tf.nn.conv2d(pool2, kernel3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
		bias_conv3 = tf.nn.bias_add(conv3, bias3, name='bias_conv3')
		relu3 = tf.nn.relu(bias_conv3, name='relu3')

		# Conv4
		kernel4 = tf.Variable(self.weights[3], dtype=tf.float32, trainable=self.trainable, name='kernel4')
		bias4 = tf.Variable(self.biases[3], dtype=tf.float32, trainable=self.trainable, name='bias4')
		conv4 = tf.nn.conv2d(relu3, kernel4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
		bias_conv4 = tf.nn.bias_add(conv4, bias4, name='bias_conv4')
		relu4 = tf.nn.relu(bias_conv4, name='relu4')

		# Conv5
		kernel5 = tf.Variable(self.weights[4], dtype=tf.float32, trainable=self.trainable, name='kernel5')
		bias5 = tf.Variable(self.biases[4], dtype=tf.float32, trainable=self.trainable, name='bias5')
		conv5 = tf.nn.conv2d(relu4, kernel5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
		bias_conv5 = tf.nn.bias_add(conv5, bias5, name='bias_conv5')
		relu5 = tf.nn.relu(bias_conv5, name='relu5')

		self.relus = [relu1, relu2, relu3, relu4, relu5]

	def relu_values(self, inX):
		return self.sess.run(self.relus, feed_dict={self.images: inX})





