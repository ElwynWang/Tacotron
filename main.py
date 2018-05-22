# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os,sys
import tensorflow as tf

from Hyperparameters import Hyperparams as Hp
import train
import evaluation
import synthesize


__author__ = "Tong Wang"


def main(argv):
    # Set Enviroment and GPU Options
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    tf.logging.set_verbosity(tf.logging.INFO)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads = Hp.inter_op_parallelism_threads,
        intra_op_parallelism_threads = Hp.intra_op_parallelism_threads,
        allow_soft_placement = True,
        log_device_placement = False,
        gpu_options = gpu_options)
    session_config.gpu_options.allow_growth = True

    # Set log dir specifically
    Hp.logdir = os.path.join(Hp.logdir, "test{}".format(sys.argv[1])) 
    
    if sys.argv[2] == 'train':
        # Train branch (Train branch also contains Eval branch, see train.py and Hyperparameter.py for more details)
        print ("Training Mode")
        train.train(session_config)

    elif sys.argv[2] == 'eval':
        #Eval
        print ("Evaluation Mode")
        evaluation.eval(session_config)

    elif sys.argv[2] == 'synthes':
        print ("Synthesize Mode")
        synthesize.synthesize(session_config)

    else:
        print ("Uncognized mode! You need type mode chosen from train/eval/synthes.")


if __name__ == "__main__":
    if len(sys.argv)!=3:
        print ('python main.py num mode\nExample: python main.py 1 train/eval/synthes')
    
    main(sys.argv)
