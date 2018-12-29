#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 09/08/2018 12:50 PM  
# @Author  : yanxuewu  
# @File    : data_load_test.py  

import tensorflow as tf
from data.data_load import TwoAfcDataLoader
import numpy as np
class TwoAfcDataLoaderTest(tf.test.TestCase):
	def test_load_data(self):
		twoafcobject = TwoAfcDataLoader(data_path='/Users/wuyanxue/Documents/GitHub/datasets/percceptual_metric/2afc', data_type='val', distort_types=['cnn', 'deblur'])

		for i, batch_data in enumerate(twoafcobject.load_data(batch_size=50)):
			if i > 5:
				break
			self.assertEqual(np.shape(batch_data), (50, 4))
			self.assertEqual(np.shape(batch_data[0][0])[2], 3)
			self.assertEqual(np.shape(batch_data[0][1])[2], 3)
			self.assertEqual(np.shape(batch_data[0][2])[2], 3)
			self.assertAllInRange(batch_data[0][3], 0, 1)


if __name__ == '__main__':
	tf.test.main()

