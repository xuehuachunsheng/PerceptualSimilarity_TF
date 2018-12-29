#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 15/08/2018 3:50 PM  
# @Author  : yanxuewu  
# @File    : AlexNetTest.py  
import tensorflow as tf
import numpy as np
class AlexNetTest(tf.test.TestCase):
	def test_shapes(self):
		from util.util import torch_to_tf_model
		images = np.random.randn(50, 64, 64, 3)
		alexnetinnet = torch_to_tf_model('/Users/wuyanxue/PycharmProjects/PerceptualSimilarity_TF/alex.pth')
		values = alexnetinnet.net.relu_values(images)
		#self.assertEquals(values[0].shape, (50, 15, 15, 64))
		self.assertEquals(values[0].shape, (50, 16, 16, 64))
		self.assertEquals(values[1].shape, (50, 7, 7, 192))
		self.assertEquals(values[2].shape, (50, 3, 3, 384))
		self.assertEquals(values[3].shape, (50, 3, 3, 256))
		self.assertEquals(values[4].shape, (50, 3, 3, 256))

if __name__ == '__main__':
	tf.test.main()
