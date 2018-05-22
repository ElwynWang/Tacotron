# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import modules
from Hyperparameters import Hyperparams as Hp


################################################################################
# Prosody Transfer for Expressive Tacotron Model
################################################################################

class Model(object):
	def __init__(self, mode):
		self.models = []
		self.inputs_transcript = []
		self.inputs_reference = []
		self.inputs_reference_lengths =[]
		self.inpus_speaker = []
		self.inputs_decoder = []
		self.labels = []

		self.memory = []
		self.mel_hat = []
		self.alignments = []
		self.mag_hat = []
		self.wavform = []

		for gpu_id in range(Hp.num_gpus):
            with tf.device('gpu:%d'%gpu_id):
                with tf.name_scope('tower_%d'%gpu_id) as scope:
                    with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
						self.inputs_transcript.append(tf.placeholder(tf.int32, shape=[None，Hp.num_charac], name = "inputs_transcript") )#text [batch, Tx]
						
						self.inputs_reference.append(tf.placeholder(tf.float32,  #ref audio melspectrogrom [batch, Ty(?)//r, n_mels*r]
							shape=[None, None, Hp.num_mels * Hp.reduction_factor], name = "inputs_reference"))
						
						self.inputs_reference_lengths.append(tf.placeholder(tf.float32, shape=[None,1]), name = "inputs_reference_lengths")
						
						self.inpus_speaker.append(tf.placeholder(tf.int32, shape=[None,1], name = "intpus_speaker")) #speaker id [batch, 1]
						
						inpus_decoder.append(tf.placeholder(tf.float32,  # decoder melspectrogrom [batch, Ty//r, n_mels*r]
							shape=[None, None, Hp.num_mels * Hp.reduction_factor], name = "inputs_decoder") )
						
						labels.append(tf.placeholder(tf.float32, shape=[None, None, Hp.num_fft//2+1])) #magnitude
					
						training = True if mode="train" else False

						# Encoder
						# transcript encoder
						text = modules.transcript_encoder(
								inputs = self.inputs_transcript[gpu_id], 
								embed_size = Hp.charac_embed_size, 
								K = Hp.num_encoder_banks, 
								highway_layers = Hp.num_enc_highway_layers, 
								training = training)  # outputs: [Batch_size, Text length, 256]

						text =tf.identity(text, name = "text_enc")

						# reference encoder
						if mode == "train":
							batch_size = Hp.train_batch_size
						elif mode == "eval":
							batch_size = Hp.eval_batch_size
						else
							batch_size = Hp.synthes_batch_size 

						inputs_reference = tf.reshape(self.inputs_reference[gpu_id], [batch_size, -1, Hp.num_mels])
						# expand the dims inputs_reference [batch, Ty, n_mels] from 3 to 4 for conv2d [batch,Ty, n_mels, 1]
						inputs_reference = tf.expand_dims(inputs_reference, -1)
						
						prosody = modules.reference_encoder(inputs = inputs_reference, training = training) #[batch, 128]
						prosody = tf.expand_dims(prosody,1)  #[batch, 1 ,128]

						#[batch, Tx, 128] replicate prosody for all Tx steps
						prosody = tf.tile(prosody, [1, Hp.num_charac, 1], name = "prosody_enc") 
						

						# speaker
						speaker = modules.embedding(
							inputs = self.inputs_speaker[gpu_id], 
							charac_size = Hp.num_speakers, 
							embed_size = Hp.speaker_embed_size) # [batch, speaker_embed_size]
						speaker = tf.expand_dims(speaker, 1)
						speaker = tf.tile(speaker, [1, Hp.num_charac, 1], name = "speaker_embed")

						memory = tf.concat([text, prosody, speaker], axis = -1, name = "memory")  # [batch, Tx, Dt+Ds+Dp ]
						self.memory.append(memory)

						# Spectrogrom Decoder
						# we concat f0 frame and remove the last frame of original melspectrogrom since it will not be sent to the deconder 
						intpus_decoder = tf.concat((tf.zeros_like(self.intpus_decoder[gpu_id][:,:1,:]), self.intpus_decoder[gpu_id][:,:-1,:]), 1) #[batch, Ty/r, num_mels*r]

						mel_hat, alignments = attention_gru_decoder(
							inputs = inputs_decoder, 
							inputs_lengths = self.inputs_reference_lengths[gpu_id],
							memory = memory, 
							attention_rnn_nodes = Hp.num_attention_nodes, 
							decoder_rnn_nodes = Hp.num_decoder_nodes, 
							num_mels = Hp.num_mels, 
							reduction_factor = Hp.reduction_factor,
							max_iters = Hp.max_iters 
							training = training)  #[batch, Ty/r, num_mels*r]

						alignments = tf.identity(alignments, name = "alignments")
						mel_hat =tf.identity(mel_hat, name = "melspectrogrom_pred")

						mag_hat = modules.cbhg_postprocessing(
							inputs = mel_hat, 
							num_mels = self.num_mels, 
							num_fft = self.num_fft, 
							K = self.num_post_banks, 
							highway_layers = self.num_post_highway_layers, 
							training = training)  # [batch, Ty, 1+n_fft//2]
						
						mag_hat =tf.identity(mag_hat, name = "magnitude_pred")
		
						wavform = tf.py_func(signal_process.Spectrogrom2Wav, [mag_hat[0]], tf.float32, name = "wavform")

						self.mel_hat.append(mel_hat)
						self.alignments.append(alignments)
						self.mag_hat.append(mag_hat)
						self.wavform.append(wavform)

		

		





