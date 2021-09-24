# -*-coding:utf-8-*- 

import torch

class HyParams:

	# CMU-MOSI
	ACOUSTIC_DIM = 5
	VISUAL_DIM = 20
	TEXT_DIM = 768
	"""

	# CMU-MOSEI
	ACOUSTIC_DIM = 74
	VISUAL_DIM = 35
	TEXT_DIM = 768
	"""
	DEVICE = torch.device('cuda')