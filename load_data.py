# -*-coding:utf-8-*- 
import pickle

def load_pkl(file_path, mode='train'):
	with open(file_path, 'rb') as file:
		info = pickle.load(file)
		raw_text = list(info[mode]['raw_text'])
		audio = info[mode]['audio']
		vision = info[mode]['vision']
		label = info[mode]['regression_labels']
	return raw_text, audio, vision, label