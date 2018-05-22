# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import numpy as np
from scipy.io.wavfile import write
import tensorflow as tf


import data
from model import Model
from Hyperparameters import Hyperparams as Hp
from utils import signal_process

__author__ = "Tong Wang"


def eval(session_config):
    with tf.Session(config = session_config) as sess:
        batch_inputs = data.input_fn(mode = "eval")
        
        model = Model(mode = "eval", inputs = batch_inputs)

        print ("Evaluation Graph Loaded") 

        saver = tf.train.Saver(max_to_keep = Hp.max_to_keep)
    
        init = tf.global_variables_initializer()
        sess.run(init)

        summary_writer = tf.summary.FileWriter(Hp.logdir, graph=sess.graph)

        
        latest_model = tf.train.latest_checkpoint(os.path.join(Hp.logdir, "models"))
        global_step = int(latest_model.split('-')[1])
        
        saver.restore(sess, latest_model)
        print ("Restore Model from Last Checkpoint")
	   
        rounds = Hp.eval_sample_num // Hp.eval_batch_size
        loss = []
        mag_hat = []
        mag_gt = []
        mel_hat = []
        mel_gt = []
        align = []

        for i in range(rounds):
            out =sess.run([
                    model.loss,
                    model.mag_hat,
                    model.merged_labels,
                    model.mel_hat,
                    model.inputs_reference,
                    model.alignments])

            loss.append(out[0])
            mag_hat.extend(out[1])
            mag_gt.extend(out[2])
            mel_hat.extend(out[3])
            mel_gt.extend(np.concatenate(out[4], axis=0))
            align.extend(out[5])

        
        print ("saving Sample during Evaluation")
        # store a sample to listen to both on tensorboard and local files
        save_sample_dir = os.path.join(Hp.logdir, "eval")
        if not os.path.exists(save_sample_dir):
            os.mkdir(save_sample_dir)

        with open(os.path.join(save_sample_dir, "loss"), 'a+') as fout:
            fout.write("Step:{}\tLoss:{}\n".format(global_step, np.mean(np.array(loss))))

        wav_hat = signal_process.spectrogrom2wav(mag_hat[0])
        ground_truth = signal_process.spectrogrom2wav(mag_gt[0])
        signal_process.plot_alignment(align[0], gs = global_step, mode = "save_fig", path = save_sample_dir)


        write(os.path.join(save_sample_dir, 'gt_{}.wav'.format(global_step)), Hp.sample_rate, ground_truth)
        write(os.path.join(save_sample_dir, 'hat_{}.wav'.format(global_step)), Hp.sample_rate, wav_hat)


        merged = sess.run(tf.summary.merge(
            [tf.summary.audio("eval/sample_gt"+str(global_step), tf.expand_dims(ground_truth, 0), Hp.sample_rate),
            tf.summary.audio("eval/sample_hat_gs"+str(global_step), tf.expand_dims(wav_hat, 0), Hp.sample_rate),
            tf.summary.image("eval/attention_gs"+str(global_step), signal_process.plot_alignment(out[2][0], gs=global_step, mode="with_return"))]))
        
        summary_writer.add_summary(merged, global_step)
                    









        

