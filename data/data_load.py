#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 08/08/2018 2:25 PM  
# @Author  : yanxuewu  
# @File    : data_load.py
import os
import queue
import random
import threading
from PIL import Image
import numpy as np
from data.data_preprocess import batch_process
class DataLoader(object):
	def __init__(self, data_path, data_type, distort_types):
		#super(DataLoader, self).__init__(data_path, data_type, distort_types)
		self.data_path = data_path
		self.data_type = data_type
		self.distort_types = distort_types

	def load_data(self, batch_size, shuffle):
		'''
		This method should be implemented by its subclasses
		:return: a batch of data
		'''

		raise NotImplementedError


class TwoAfcDataLoader(DataLoader):
	def __init__(self, data_path, data_type='train', distort_types=['cnn', 'mix', 'traditional']):
		'''
		:param data_path: str, the data stored path
		:param data_type: str, optional 'train', 'val'
		:param distort_types: ndarray, ['cnn', 'mix', 'traditional', ...]
		'''
		super(TwoAfcDataLoader, self).__init__(data_path, data_type, distort_types)

		# distort type paths
		distort_paths = []
		for distort_type in distort_types:
			distort_paths.append(os.path.join(data_path, data_type, distort_type))

		input_paths = []

		ele_paths = ['ref', 'p0', 'p1', 'judge']
		for path in distort_paths:
			print('Searching current input path: ', path)
			c_file_names = os.listdir(os.path.join(path, ele_paths[0]))
			for img_name in c_file_names:
				print('Searching input images: ', img_name)
				c_ref_path = os.path.join(path, ele_paths[0], img_name)
				c_p0_path = os.path.join(path, ele_paths[1], img_name)
				c_p1_path = os.path.join(path, ele_paths[2], img_name)
				c_judge_path = os.path.join(path, ele_paths[3], img_name[:-3]+'npy')

				input_path_tuple = (c_ref_path, c_p0_path, c_p1_path, c_judge_path)
				input_paths.append(input_path_tuple)

		self.input_paths = input_paths

	def load_data(self, batch_size, shuffle=True):
		'''

		:param batch_size:
		:return: ndarray with size [batch_size, 4] where [:, 0] is the
		'''

		input_queue = queue.Queue(maxsize=5*batch_size)

		def en_queue():
			while True:
				random.shuffle(self.input_paths) # Shuffle data in each epoch
				for ele in self.input_paths:
					input_queue.put(ele)
		self.queue_thread = threading.Thread(target=en_queue)
		# When the major thread is finished, this thread would be killed immediately.
		self.queue_thread.setDaemon(True)
		self.queue_thread.start()

		def convert_paths_to_data(paths):
			c_data = []
			for one_input_path in paths:
				#  ndarray
				ref_img_ = Image.open(one_input_path[0]).convert('RGB')
				p0_img_ = Image.open(one_input_path[1]).convert('RGB')
				p1_img_ = Image.open(one_input_path[2]).convert('RGB')

				# Preprocess the input data
				# The input data must be in [0, 255]
				# process an image object and return a ndarray with shape [H, W, C]
				ref_img = batch_process(ref_img_)[0]
				p0_img = batch_process(p0_img_)[0]
				p1_img = batch_process(p1_img_)[0]
				# float32
				judge = np.load(one_input_path[3])[0]
				c_data.append((ref_img, p0_img, p1_img, judge))
			return c_data

		batch_data_path = []
		while True:
			for i in range(batch_size):
				batch_data_path.append(input_queue.get())
			# Convert path to instance
			batch_data = convert_paths_to_data(batch_data_path)
			yield batch_data
			batch_data_path = []


class JndDataLoader(DataLoader):
	def __init__(self, data_path, data_type='val', distort_types=['cnn', 'traditional']):
		'''
		:param data_path: str, the data stored path
		:param data_type: str, optional 'train', 'val'
		:param distort_types: ndarray, ['cnn', 'mix', 'traditional', ...]
		'''
		super(JndDataLoader, self).__init__(data_path, data_type, distort_types)
		#TODO


	def load_data(self, batch_size, shuffle=True):

		print('This is the jnd data loader implentation')
		# TODO
		pass


class TFRecordConverter(object):
	def __init__(self, data_path, data_type='train'):
		pass

