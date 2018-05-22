# -*- coding: utf-8 -*-

from __future__ import division, print_function

import tensorflow as tf

import modules
from Hyperparameters import Hyperparams as Hp


__author__ = "Tong Wang"

################################################################################
# Prosody Transfer for Expressive Tacotron Model
################################################################################

class Model(object):
    def single_model(self, gpu_id):
        if self.mode == "train" or self.mode == "eval":
            inputs_transcript = self.inputs_transcript[gpu_id]
            inputs_reference = self.inputs_reference[gpu_id]
            inputs_ref_lens = self.inputs_ref_lens[gpu_id]
            inputs_speaker = self.inputs_speaker[gpu_id]
            inputs_decoder = self.inputs_decoder[gpu_id] 
        
        training = True if self.mode=="train" else False

        # Encoder
        # transcript encoder
        text = modules.transcript_encoder(
                inputs = inputs_transcript, 
                embed_size = Hp.charac_embed_size, 
                K = Hp.num_encoder_banks, 
                highway_layers = Hp.num_enc_highway_layers, 
                training = training)  # outputs: [Batch_size, Text length, 256]

        text =tf.identity(text, name = "text_enc")

        # reference encoder
        if self.mode == "train":
            batch_size = Hp.train_batch_size//Hp.num_gpus
        elif self.mode == "eval":
            batch_size = Hp.eval_batch_size//Hp.num_gpus
        else:
            batch_size = Hp.synthes_batch_size//HP.num_gpus

        inputs_reference_reshape = tf.reshape(inputs_reference, [batch_size, -1, Hp.num_mels])
        # expand the dims inputs_reference [batch, Ty, n_mels] from 3 to 4 for conv2d [batch,Ty, n_mels, 1]
        inputs_reference_reshape = tf.expand_dims(inputs_reference_reshape, -1)
        
        prosody = modules.reference_encoder(inputs = inputs_reference_reshape, training = training) #[batch, 128]
        prosody = tf.expand_dims(prosody,1)  #[batch, 1 ,128]

        #[batch, Tx, 128] replicate prosody for all Tx steps
        prosody = tf.tile(prosody, [1, Hp.num_charac, 1], name = "prosody_enc") 
        

        # speaker
        speaker = modules.embedding(
            inputs = inputs_speaker, 
            charac_size = Hp.num_speakers, 
            embed_size = Hp.speaker_embed_size,
            scope = "speaker") # [batch, 1, speaker_embed_size] [32,1,16]
        
        speaker = tf.tile(speaker, [1, Hp.num_charac, 1], name = "speaker_embed")

        memory = tf.concat([text, prosody, speaker], axis = -1, name = "memory")  # [batch, Tx, Dt+Ds+Dp ]
        #self.memory.append(memory)

        # Spectrogrom Decoder
        # we concat f0 frame and remove the last frame of original melspectrogrom since it will not be sent to the deconder
        if self.mode == "train": 
            inputs_decoder = tf.concat((tf.zeros_like(inputs_decoder[:,:1,:]), inputs_decoder[:,:-1,:]), 1) #[batch, Ty/r, num_mels*r]

        mel_hat, alignments = modules.attention_gru_decoder(
            inputs = inputs_decoder, 
            inputs_lengths = inputs_ref_lens,
            memory = memory, 
            attention_rnn_nodes = Hp.num_attention_nodes, 
            decoder_rnn_nodes = Hp.num_decoder_nodes, 
            num_mels = Hp.num_mels, 
            reduction_factor = Hp.reduction_factor,
            max_iters = self.max_len_per_batch,
            training = training)  #[batch, Ty/r, num_mels*r]

        alignments = tf.identity(alignments, name = "alignments")
        mel_hat =tf.identity(mel_hat, name = "melspectrogrom_pred")

        mag_hat = modules.cbhg_postprocessing(
            inputs = mel_hat, 
            num_mels = Hp.num_mels, 
            num_fft = Hp.num_fft, 
            K = Hp.num_post_banks, 
            highway_layers = Hp.num_post_highway_layers, 
            training = training)  # [batch, Ty, 1+n_fft//2]
        
        mag_hat =tf.identity(mag_hat, name = "magnitude_pred")

        #wavform = tf.py_func(signal_process.Spectrogrom2Wav, [mag_hat[0]], tf.float32, name = "wavform")  # generate a sample to listen to
        
        return mel_hat, alignments, mag_hat 



    def add_loss_op(self):
        self.tower_loss1, self.tower_loss2, self.tower_loss = [], [], []

        def calc_loss(gpu_id):
            with tf.device('gpu:%d'%gpu_id):
                with tf.name_scope('tower_%d'%gpu_id) as scope:
                    with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                        loss1 = tf.reduce_mean(tf.abs(self.tower_mel_hat[gpu_id] - self.inputs_reference[gpu_id]))
                        loss2 = tf.reduce_mean(tf.abs(self.tower_mag_hat[gpu_id] - self.labels[gpu_id]))
                        return loss1, loss2

        for i in range(Hp.num_gpus):
            loss1, loss2 = calc_loss(i)
            self.tower_loss1.append(loss1)
            self.tower_loss2.append(loss2)
            self.tower_loss.append(loss1+loss2)       

        self.loss1 = tf.reduce_mean(tf.stack(self.tower_loss1), name = "seq2seq_loss")
        self.loss2 = tf.reduce_mean(tf.stack(self.tower_loss2), name = "output_loss")
        self.loss = tf.reduce_mean(tf.stack(self.tower_loss), name = "total_loss")



    def add_train_op(self):
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        
        def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.):
            step = tf.cast(global_step + 1, dtype=tf.float32)
            return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

        self.learning_rate = learning_rate_decay(Hp.learning_rate, self.global_step) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)

        self.tower_grads=[]
        def calc_grad(gpu_id):
            with tf.device('gpu:%d'%gpu_id):
                with tf.name_scope('tower_%d'%gpu_id) as scope:
                    with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                        ## gradient clipping
                        gvs = self.optimizer.compute_gradients(self.tower_loss[gpu_id])
                        return gvs

        for i in range(Hp.num_gpus):
            self.tower_grads.append(calc_grad(i))
  

        def average_gradients(tower_grads):
            average_grads=[]

            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = [g for g, _ in grad_and_vars]
                # Average over the 'tower' dimension.
                grad = tf.stack(grads, 0)
                grad = tf.reduce_mean(grad, 0)
                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v=grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
            return average_grads

        self.grad = average_gradients(self.tower_grads)
        
        def grad_clip(gvs):
            clipped = []
            for grad, var in gvs:
                grad = tf.clip_by_norm(grad, 5.)
                clipped.append((grad, var))
            return clipped
        
        self.grad = grad_clip(self.grad)

        minimize_op = self.optimizer.apply_gradients(self.grad, global_step = self.global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        self.train_op = tf.group(minimize_op, update_ops)


    def add_summary_op(self):
        if self.mode == "train" or self.mode == "eval":
            tf.summary.scalar('{}/seq2seq_loss'.format(self.mode), self.loss1)
            tf.summary.scalar('{}/output_loss'.format(self.mode), self.loss2)
            tf.summary.scalar('{}/total_loss'.format(self.mode), self.loss)

            #tf.summary.image("{}/mel_gt".format(self.mode), tf.expand_dims(tf.stack(self.inputs_reference), -1), max_outputs=1)
            #tf.summary.image("{}/mel_hat".format(self.mode), tf.expand_dims(tf.stack(self.mel_hat), -1), max_outputs=1)
            #tf.summary.image("{}/mag_gt".format(self.mode), tf.expand_dims(tf.stack(self.labels), -1), max_outputs=1)
            #tf.summary.image("{}/mag_hat".format(self.mode), tf.expand_dims(tf.stack(self.mag_hat), -1), max_outputs=1)

            if self.mode == "train":
                tf.summary.scalar('{}/learning_rate'.format(self.mode), self.learning_rate)
            

            #if mode == "eval":
            #   tf.summary.audio("{}/sample".format(mode), tf.expand_dims(tf.stack(self.wavform), 0), Hp.sample_rate)

            
        
    def feed_all_gpu(self, inputs):
        batch_size = Hp.train_batch_size if self.mode == "train" else Hp.eval_batch_size
        batch_per_gpu = batch_size//Hp.num_gpus
        
        for i in range(Hp.num_gpus):
            self.inputs_transcript[i] = tf.slice(inputs['transcript'], [i*batch_per_gpu,0], [batch_per_gpu,-1])
            self.inputs_reference[i] = tf.slice(inputs['reference'], [i*batch_per_gpu,0,0], [batch_per_gpu, -1, -1])
            #self.inputs_ref_lens[i] = tf.slice(inputs['ref_len'], [i*batch_per_gpu,0], [batch_per_gpu,-1])
            self.inputs_ref_lens[i] = tf.tile(tf.reshape(self.max_len_per_batch, [1,1]), [batch_per_gpu,1])
            self.inputs_speaker[i] = tf.slice(inputs['speaker'], [i*batch_per_gpu,0], [batch_per_gpu,-1])
            
            # evaluation also needs slice labels since we want to calculate loss
            self.labels[i] = tf.slice(inputs['labels'], [i*batch_per_gpu,0,0], [batch_per_gpu, -1, -1]) 
            
            if self.mode == "train":
                self.inputs_decoder[i] = tf.slice(inputs['decoder'], [i*batch_per_gpu,0,0], [batch_per_gpu, -1, -1])
                



    def __init__(self, mode, inputs = None):
        with tf.device('/cpu:0'):
            self.mode = mode

            self.inputs_transcript = [None]*Hp.num_gpus
            self.inputs_reference = [None]*Hp.num_gpus
            self.inputs_ref_lens = [None]*Hp.num_gpus
            self.inputs_speaker = [None]*Hp.num_gpus
            self.inputs_decoder = [None]*Hp.num_gpus
            self.labels = [None]*Hp.num_gpus
            
            if inputs is not None:
                self.max_len_per_batch = tf.cast(tf.reduce_max(inputs['ref_len']), dtype = tf.int32)
                self.feed_all_gpu(inputs)
            else:
                self.max_len_per_batch = tf.constant(100)

            
            self.tower_memory = [None]*Hp.num_gpus
            self.tower_mel_hat = [None]*Hp.num_gpus
            self.tower_alignments = [None]*Hp.num_gpus
            self.tower_mag_hat = [None]*Hp.num_gpus
            #self.tower_wavform = [None]*Hp.num_gpus

            for gpu_id in range(Hp.num_gpus):
                with tf.device('gpu:%d'%gpu_id):
                    with tf.name_scope('tower_%d'%gpu_id):
                        with tf.variable_scope('cpu_variables', reuse = gpu_id>0):
                            if mode == "synthes":
                                self.inputs_transcript[gpu_id] = tf.placeholder(tf.int32, 
                                                                shape=[None, Hp.num_charac], name = "inputs_transcript") #text [batch, Tx]
                                self.inputs_reference[gpu_id] = tf.placeholder(tf.float32,  #ref audio melspectrogrom [batch, Ty(?)//r, n_mels*r]
                                                                shape=[None, None, Hp.num_mels * Hp.reduction_factor], name = "inputs_reference")
                                self.inputs_ref_lens[gpu_id] = tf.placeholder(tf.int32, shape=[None,1], name = "inputs_ref_lens")
                                self.inpus_speaker[gpu_id] = tf.placeholder(tf.int32, shape=[None,1], name = "intpus_speaker") #speaker id [batch, 1]
                                self.inpus_decoder[gpu_id] = tf.placeholder(tf.float32,  # decoder melspectrogrom [batch, Ty//r, n_mels*r]
                                                                shape=[None, None, Hp.num_mels * Hp.reduction_factor], name = "inputs_decoder") 
                                self.labels[gpu_id] = tf.placeholder(tf.float32, shape=[None, None, Hp.num_fft//2+1]) #magnitude
                            
                            mel_hat, alignments, mag_hat = self.single_model(gpu_id)
                            
                            self.tower_mel_hat[gpu_id] = mel_hat
                            self.tower_alignments[gpu_id] = alignments
                            self.tower_mag_hat[gpu_id] = mag_hat
                            #self.wavform[gpu_id] = wavform
            
            if self.mode == "train" or self.mode == "eval":
                self.merged_labels = tf.concat(self.labels, 0)
           
            self.mel_hat = tf.concat(self.tower_mel_hat, 0)
            self.alignments = tf.concat(self.tower_alignments, 0)
            self.mag_hat = tf.concat(self.tower_mag_hat, 0)
            
            if self.mode == "train":
                self.add_loss_op()
                self.add_train_op()
            elif self.mode == "eval":
                self.add_loss_op()
            
            self.add_summary_op()
            self.merged = tf.summary.merge_all()

        

        





