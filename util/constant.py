#!/usr/bin/env python  
# _*_ coding:utf-8 _*_  
#  
# @Version : 1.0  
# @Time    : 15/08/2018 2:47 PM  
# @Author  : yanxuewu  
# @File    : constant.py  

alex_weights_shapes = [
	(11, 11, 3, 64),
	(5, 5, 64, 192),
	(3, 3, 192, 384),
	(3, 3, 384, 256),
	(3, 3, 256, 256),
]

alex_bias_shapes = [
	(64, ),
	(192, ),
	(384, ),
	(256, ),
	(256, ),
]

alex_ex_weights_shapes = alex_bias_shapes