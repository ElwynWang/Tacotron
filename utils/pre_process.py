# -*- coding: utf-8 -*-

from __future__ import print_function, division

from multiprocessing import Pool
import numpy as np
import os

from signal_process import load_spectrograms

__author__ = "Tong Wang"


def load_audio(audio_path):
	fname, mel, mag = load_spectrograms(audio_path)
	sub_paths = audio_path.strip().split('/')
	prefix = '/'.join(sub_paths[:-2])
	
	np.save(prefix+"/mels/{}".format(sub_paths[-1].replace('wav', 'npy')), mel)
	np.save(prefix+"/mags/{}".format(sub_paths[-1].replace('wav', 'npy')), mag)

def pre_process():
	for mode in ['all']:
	#for mode in ['train', 'eval']:
		# for train and eval samples
		main_path = "data/{}/wavs".format(mode)
		files = os.listdir(main_path)
		file_paths = [os.path.join(main_path,f) for f in files]

		pool = Pool()
		pool.map(load_audio, file_paths)
		pool.close()
		pool.join()


