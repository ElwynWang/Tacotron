# -*- coding: utf-8 -*-
import tensorflow as tf

__author__ = "Tong Wang"

class Hyperparams(object):
    ### training params ###
    num_gpus = 4

    learning_rate = 0.001
    
    train_batch_size = 32*4
    eval_batch_size = 32*4
    synthes_batch_size = 16 # equal kinds of ref_audios
    
    train_step = 200000
    max_to_keep = 200
    train_from_restore = False
    train_step_per_eval = 2000
    save_model_step = 1000
    
    eval_sample_num = 2620 #2620


    inter_op_parallelism_threads = 20
    intra_op_parallelism_threads = 20


    ### IO params ###
    logdir = "runs"
    
    train_data_dir = "data/train"
    eval_data_dir = "data/eval"

    synthes_data_text = "data/synthesize/text.txt"
    synthes_ref_audio_dir = "data/synthesize/wavs/"

    restore_model = None
    data_num_parallel_calls = 20

    ### model params ###
    num_charac = 200 #Tx
    charac_embed_size = 256
    num_speakers = 1 
    speaker_embed_size = 16 
    num_encoder_banks = 16 
    num_post_banks = 8 
    num_enc_highway_layers = 4 
    num_post_highway_layers = 4  
    num_attention_nodes = 256 
    num_decoder_nodes = 256 

    characs = u'''␀␃ !',-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz''' # ␀: Padding \u2400 ␃: End of Sentence \u2403
    
    data_format = "channels_last"
    dtype = tf.float32


    ### audio signal params ###
    reduction_factor = 2 

    num_mels = 80 
    num_fft = 2048 ## fft points (samples)
    sample_rate = 22050
    frame_shift = 0.0125 #seconds
    frame_length = 0.05 #seconds
    hop_length = int(sample_rate*frame_shift)
    win_length = int(sample_rate*frame_length)

    power = 1.2 # Exponent for amplifying the predicted magnitude
    preemphasis = 0.97

    grilimn_iter = 50

    max_db = 100
    ref_db = 20 

    


    


    
