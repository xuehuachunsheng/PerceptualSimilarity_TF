#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 14/08/2018 10:18 AM  
# @Author  : yanxuewu  
# @File    : train.py  


opt_datasets = ['traditional', 'cnn', 'mix']
opt_model = 'net-lin'
opt_net = 'alex'
opt_batch_size = 50
opt_nepoch = 5
opt_nepoch_decay = 5
opt_freq = 5000
opt_checkpoint_dir = './checkpoints'
# from pretrained model

from models import DistModel
from data.data_load import TwoAfcDataLoader
model = DistModel
model.initialize()

dataset = TwoAfcDataLoader(data_path='/Users/wuyanxue/Documents/GitHub/datasets/percceptual_metric')
dataset = dataset.load_data()
for epoch in range(1, opt_nepoch + opt_nepoch_decay + 1):
	for i, data in enumerate(dataset):
		model.put_data(data)
		model.train()

		if model.global_step_value % opt_freq == 0:
			model.save(save_dir=opt_checkpoint_dir)


