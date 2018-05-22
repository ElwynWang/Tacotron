# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import re
import unicodedata
import codecs

import numpy as np
import tensorflow as tf

from Hyperparameters import Hyperparams as Hp

__author__ = "Tong Wang"

################################################################################
# Convenience functions for Text Processing and Hudge Data Loading
################################################################################

def load_vocab():
    char2idx = {char:idx for idx, char in enumerate(Hp.characs)}
    idx2char = {idx:char for idx, char in enumerate(Hp.characs)}
    return char2idx, idx2char

def text_normalize(text):
    # normalize the text and remove illegal char
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')
    text = re.sub(u"[^{}]".format(Hp.characs), " ", text)
    text = re.sub("[ ]+",  " ", text)

    return text


def parser(transcript, ref_path):
    transcript = tf.decode_raw(transcript, tf.int32) # decode string to int
    transcript = tf.pad(transcript, ([0, Hp.num_charac], ))[:Hp.num_charac] # pad charac[0] to fill in the whole max length Tx

    def load_audio(audio_path):
        sub_paths = audio_path.strip().split('/')
        main_path = '/'.join(sub_paths[:-2])
        fname = os.path.basename(audio_path)
        
        mel_path = main_path + "/mels/{}".format(sub_paths[-1].replace('wav', 'npy'))  # mel spectrogrom
        mag_path = main_path + "/mags/{}".format(sub_paths[-1].replace('wav', 'npy'))  # wavform

        mel = np.load(mel_path)
        mag = np.load(mag_path)
        ref_len = mel.shape[0] if mel.shape[1]==Hp.num_mels*Hp.reduction_factor \
                                else mel.shape[0]*mel.shape[1]//Hp.num_mels*Hp.reduction_factor
        ref_len = np.array(ref_len, dtype = np.int32)
        
        return mel, mag, ref_len 

    spectrogrom, wavform, ref_len = tf.py_func(load_audio, [ref_path], [tf.float32, tf.float32, tf.int32])
    
    transcript = tf.reshape(transcript, [Hp.num_charac,])
    
    spectrogrom = tf.reshape(spectrogrom, [-1, Hp.num_mels*Hp.reduction_factor])
    
    #ref_len = tf.cast(spectrogrom.get_shape()[0], tf.int32)
    wavform = tf.reshape(wavform, [-1, Hp.num_fft//2+1])

    inputs = list((transcript, spectrogrom, tf.reshape(ref_len, [1]), tf.reshape(tf.constant(0), [1]), spectrogrom, wavform))
    #inputs = {'transcript':transcript, 'reference':spectrogrom, 'ref_len':tf.reshape(ref_len,[1]),
    #'speaker':tf.reshape(tf.constant(0), [1]), 'decoder':spectrogrom, 'labels':wavform} 
    
    return inputs




def input_fn(mode):
    data_dir = Hp.train_data_dir if mode == "train" else Hp.eval_data_dir
    batch_size = Hp.train_batch_size if mode == "train" else Hp.eval_batch_size

    char2idx, idx2char = load_vocab()

    lines = codecs.open(os.path.join(data_dir, 'metadata.csv'), 'r', 'utf-8').readlines()

    transcripts = []
    ref_paths = []

    for line in lines:
        fname, _, transcript = line.strip().split('|')
        ref_path = os.path.join(data_dir, 'wavs', fname+'.wav')
        ref_paths.append(ref_path)

        transcript = text_normalize(transcript)+u"␃" # EOS
        transcript = [char2idx[char] for char in transcript]
        transcripts.append(np.array(transcript, np.int32).tostring())

    transcripts = tf.convert_to_tensor(transcripts)
    ref_paths = tf.convert_to_tensor(ref_paths)

    dataset = tf.data.Dataset.from_tensor_slices((transcripts,ref_paths))

    if mode == "train":
        dataset = dataset.shuffle(buffer_size = 100000)
        dataset = dataset.repeat()
    
    dataset = dataset.map(parser,  num_parallel_calls = Hp.data_num_parallel_calls)
    #dataset = dataset.batch(batch_size)
    dataset = dataset.padded_batch(batch_size = batch_size, padded_shapes=(
        [Hp.num_charac], [None, Hp.num_mels*Hp.reduction_factor], [1], [1], 
        [None, Hp.num_mels*Hp.reduction_factor], [None, Hp.num_fft//2+1]))

    dataset = dataset.prefetch(1)

    iterator = dataset.make_one_shot_iterator() #one_shot_iterator can automatically initialize

    #return iterator
    batch_inputs = iterator.get_next()
    names = ['transcript', 'reference', 'ref_len', 'speaker', 'decoder', 'labels']
    batch_inputs = {name:inp for name,inp in zip(names, batch_inputs)}
    
    return batch_inputs


'''
def parser_synthesize(transcript,ref_path):
    transcript = tf.decode_raw(transcript, tf.int32) # decode string to int
    transcript = tf.pad(transcript, ([0, len(characs)], ))[:len(characs)] # pad charac[0] to fill in the whole max length Tx

    def load_audio(audio_path):
        sub_paths = audio_path.strip().split('/')
        main_path = '/'.join(sub_paths[:-2])
        fname = os.path.basename(audio_path)
        
        mel = main_path + "mels/{}".format(sub_paths[-1].replace('wav', 'npy'))  # mel spectrogrom
        mag = main_path + "mags/{}".format(sub_paths[-1].replace('wav', 'npy'))  # wavform

        return np.load(mel), np.load(mag) 

    spectrogrom, wavform = tf.py_func(load_audio, [ref_path], [tf.float32, tf.float32])
    
    transcript.set_shape([len(characs),])
    spectrogrom.set_shape([None, num_mels*reduction_factor])
    wavform.set_shape([None, num_fft//2+1])

    inputs = {'transcript':transcript, 'reference':spectrogrom, 'speaker':tf.constant(0), 'decoder':spectrogrom} 
    labels = wavform

    return inputs, labels


def input_fn_synthesize(characs, transcripts_path, ref_audio_path, num_mels, num_fft, reduction_factor,
     batch_size, num_parallel_calls, multi_gpu):

    load_vocab(characs)

    lines = codecs.open(transcripts_path, 'r', 'utf-8').readlines()[1:]
    ref_audios = os.listdir(ref_audio_path)

    transcripts = []
    ref_paths = []

    for line in lines:
        transcript = text_normalize(line.split(" ",1)[-1].strip())+u"" #EOS
        transcript = [char2idx[char] for char in transcript]

        for i in range(len(ref_audios)):
            transcripts.append(np.array(transcript,np.int32).tostring())
            ref_paths.append(ref_audios[i])

    transcripts = tf.convert_to_tensor(transcripts)
    ref_paths = tf.convert_to_tensor(ref_paths)

    dataset = tf.data.Dataset.from_tensor_slices((transcripts,ref_paths))

    if multi_gpu:
        print ("Warning: GPU Num must be a aliquot of Batch Size!")
        
    dataset = dataset.map(parser_synthesize,  num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(1)

    iterator = dataset.make_one_shot_iterator()
    inputs, labels = iterator.get_next()
    return inputs, labels
'''

