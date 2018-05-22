# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os,sys,codecs
import numpy as np
from scipy.io.wavfile import write
import tensorflow as tf

import data
from model import Model
from Hyperparameters import Hyperparams as Hp
from utils import signal_process


__author__ = "Tong Wang"


def synthesize(session_config):
    Hp.numgpus = 1
    # Load data
    # transcripts
    char2idx, idx2char = data.load_vocab()
    
    lines = codecs.open(Hp.synthes_data_text, 'r', 'utf-8').readlines()[1:] # skip the first head line
    # skip the first field(number), text normalization, E: EOS
    lines_normalize = [data.text_normalize(line.split(" ", 1)[-1]).strip() + u"␃" for line in lines]
    
    transcripts = np.zeros((len(lines), Hp.num_charac), np.int32)

    for i,line in enumerate(lines_normalize):
        transcripts[i, :len(line)] =  [char2idx[char] for char in line]

    # tile each transcript to a batch and the batch_size is the number of ref kinds (16)
    transcripts_num = transcripts.shape[0] # num of transcripts
    transcripts = np.tile(transcripts, (1, Hp.synthes_batch_size))
    transcripts = transcripts.reshape(transcripts_num, Hp.synthes_batch_size, Hp.num_charac)

    # ref audios
    mels, maxlen = [], 0
    files = [os.path.join(Hp.synthes_ref_audio_dir, x) for x in os.listdir(Hp.synthes_ref_audio_dir)]
    for f_path in files:
        _, mel, _ = signal_process.load_spectrograms(f_path)
        #mel = np.reshape(mel, (-1, Hp.num_mels))
        maxlen = max(maxlen, mel.shape[0])
        mels.append(mel)

    assert len(mels)==Hp.synthes_batch_size

    ref = np.zeros((len(mels), maxlen, Hp.num_mels*Hp.reduction_factor), np.float32)
    for i,m in enumerate(mels):
        ref[i, :m.shape[0], :] = m

    ref_lens = np.ones((len(mels), 1), np.int32)*maxlen
    speaker = np.ones((len(mels), 1), np.int32)


    # Load Graph
    model = Model(mode = "synthes")
    print ("Synthesize Graph Loaded") 

    saver = tf.train.Saver()

    save_sample_dir = os.path.join(Hp.logdir, "synthesize")
    if not os.path.exists(save_sample_dir):
        os.mkdir(save_sample_dir)

    with tf.Session(config = session_config) as sess:
        latest_model = tf.train.latest_checkpoint(os.path.join(Hp.logdir, "models"))

        if Hp.restore_model is not None and Hp.restore_model != latest_model:
            print ("Restore Model from Specific Model")
            restore_model = Hp.restore_model
        else:
            print ("Restore Model from Last Checkpoint")
            restore_model = latest_model

        saver.restore(sess, restore_model)

        for text_idx in range(transcripts_num):
            mag_hats, aligns =sess.run([
                                    model.mag_hat,
                                    model.alignments],
                                    {model.inputs_transcript[0]: transcripts[text_idx],
                                    model.inputs_reference[0]: ref,
                                    model.inputs_ref_lens[0]: ref_lens,
                                    model.inputs_speaker[0]:speaker})

            save_sample_path = os.path.join(save_sample_dir, "sample_{}".format(text_idx+1))
            if not os.path.exists(save_sample_path):
                os.mkdir(save_sample_path)

            for i in range(len(mag_hats)):
                wav_hat = signal_process.spectrogrom2wav(mag_hats[i])
                write(os.path.join(save_sample_path, 'style_{}.wav'.format(i+1)), Hp.sample_rate, wav_hat)
                signal_process.plot_alignment(aligns[i], gs = i+1, mode = "save_fig", path = save_sample_path)

            print ("Done! Synthesize for sample {}".format(text_idx+1))
        print ("All jobs Done!")













    



        
