#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 09/08/2018 10:50 AM  
# @Author  : yanxuewu  
# @File    : data_preprocess.py  
from PIL import Image
import numpy as np
from skimage import img_as_float
def batch_resize(batch_imgs, size=(64, 64)):
	'''
	Convert the batch_imgs with shape [N, H, W, C] to [N, size[0], size[1], C].
	And the value range is [0, 255]
	:param batch_imgs: an nd-array with shape [N, H, W, C] or an object can be converted to nd-array.
	:param size: the size
	:return: a ndarray stored the image after resized.
	'''
	batch_imgs_np = np.array(batch_imgs)
	if len(batch_imgs_np.shape) == 3:
		batch_imgs_np = np.array([batch_imgs_np, ])
	img_resized = []
	for img in batch_imgs_np:
		img_resized.append(img.resize(size, Image.BILINEAR))
	return np.array(img_resized)

def batch_normalizeto1(batch_imgs):
	'''
	Convert the batch_imgs with shape [N, H, W, C] with value range [0, 255] to [-1, 1].
	And the value range is [0, 255]
	:param batch_imgs: an ndarray with shape [N, H, W, C] or an object can be converted to nd-array
	:return:
	'''

	batch_imgs_np = np.array(batch_imgs)
	if len(batch_imgs_np.shape) == 3:
		batch_imgs_np = np.array([batch_imgs_np, ])
	imgs_norm = []
	for img in batch_imgs_np:
		imgs_norm.append(img_as_float(img))
	return np.array(imgs_norm)

def batch_process(batch_imgs):
	batch_imgs = batch_resize(batch_imgs)
	batch_imgs = batch_normalizeto1(batch_imgs)
	return batch_imgs
