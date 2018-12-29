#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 14/08/2018 11:13 AM  
# @Author  : yanxuewu  
# @File    : util.py  

import torch
import tensorflow as tf

from torchvision import models
from net.AlexNet import AlexNet
from net.AlexNetInNet import AlexNetInNet
import numpy as np
def torch_to_tf_model(torch_model_path, model_type='alexnet', use_gpu=False):
	'''
	Convert the torch model to tensorflow model
	:param torch_model_path:
	:param model_type:
	:param use_gpu:
	:return: An alex net implemented by tensorflow
	'''

	kw = {}
	if not use_gpu:
		kw['map_location'] = 'cpu'
	# Load pretrained weights in the torch alexnet
	torch_model_features = models.alexnet(pretrained=True).features
	torch_model_features_conv = [torch_model_features[0], torch_model_features[3], torch_model_features[6],
								 torch_model_features[8], torch_model_features[10]]

	# Create tf model inputs
	alexnet_weights = []
	alexnet_biases = []
	for layer in torch_model_features_conv:
		# Weight converted
		c_array = layer.weight.detach().numpy()
		c_array_ = np.transpose(c_array, (2, 3, 1, 0)) # Convert to tf [H, W, in_C, out_C]
		alexnet_weights.append(c_array_)
		# Bias converted
		c_array = layer.bias.detach().numpy()
		c_array_ = np.squeeze(c_array)
		alexnet_biases.append(c_array_)

	tf_alexnet = AlexNet(weights=alexnet_weights, biases=alexnet_biases)

	# load extra weight w
	ex_weights = []
	weights = torch.load(torch_model_path, **kw)

	for st, t_weight in weights.items():
		c_array = np.squeeze(t_weight.numpy()) # Remove all single element dimensions
		ex_weights.append(c_array)

	tf_alexnetinnet = AlexNetInNet(tf_alexnet, ex_weights)

	return tf_alexnetinnet



if __name__ == '__main__':
	torch_to_tf_model('/Users/wuyanxue/PycharmProjects/PerceptualSimilarity_TF/alex.pth')