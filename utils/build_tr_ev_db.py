import os
import numpy as np
from multiprocessing import Pool

__author__ = "Tong Wang"

train_ratio = 0.8

tags = []
mode = "train"

def file_cp(idx):
	for item in ['wavs', 'mels', 'mags']:
		suffix = 'wav' if item == 'wavs' else 'npy'
		comm = "cp data/all/{0}/{1}.{2} data/{3}/{0}/".format(item, tags[idx], suffix, mode)
		os.system(comm)


def build_train_eval():
	with open("data/all/metadata.csv", 'r') as fin:
		lines = fin.readlines()
		global tags
		tags = [x.split('|')[0] for x in lines]

	idxs = np.arange(len(tags))
	np.random.shuffle(idxs)

	item_idxs = {} 
	item_idxs['train'] = idxs[:int(len(idxs)*train_ratio)]
	item_idxs['train'].sort()
	item_idxs['eval'] = idxs[int(len(idxs)*train_ratio):]
	item_idxs['eval'].sort()
	
	global mode
	for mode in ['train', 'eval']:
		# mkdir 
		if not os.path.exists("data/{}".format(mode)):
			os.mkdir("data/{}".format(mode))
		for item in ['wavs', 'mels', 'mags']:
			if not os.path.exists("data/{}/{}".format(mode,item)):
				os.mkdir("data/{}/{}".format(mode,item))

		# write data info file
		with open("data/{}/metadata.csv".format(mode), 'w') as fout:
			for idx in item_idxs[mode]:
				fout.write("%s\n"%lines[idx].strip())
		
		# cp wav files
		pool = Pool()
		pool.map(file_cp, item_idxs[mode])
		pool.close()
		pool.join()
