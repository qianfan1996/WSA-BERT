# -*-coding:utf-8-*- 
import numpy as np
import os
import random
import torch

def set_random_seed(seed):
	print("Random Seed: {}".format(seed))

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.deterministic = True

	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_device():
	if torch.cuda.is_available():
		return "cuda"
	else:
		return "cpu"