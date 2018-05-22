# -*- coding: utf-8 -*-

from __future__ import division, print_function

import tensorflow as tf

from layers import *
import rnn_utils 

__author__ = "Tong Wang"

################################################################################
# Convenience modules and components for building the Expressive Tacotron model.
################################################################################

def cbhg(inputs, K, highway_layers, training, scope = "cbhg"):
    with tf.variable_scope(scope):
        # conv1d banks
        outputs = conv1d_banks(inputs = inputs, K = K, filters = 128, training = training)

        # max pooling
        outputs = tf.layers.max_pooling1d(inputs = outputs, pool_size = 2, strides = 1, padding = "same")

        # conv1d projections
        outputs = conv1d(inputs = outputs, kernel_size = 3, filters=128, scope = "conv1d_1")
        outputs = batch_norm(outputs, training, scope = "bn1")
        outputs = tf.nn.relu(outputs)

        outputs = conv1d(inputs = outputs, kernel_size = 3, filters =128, scope = "conv1d_2")
        outputs = batch_norm(outputs, training, scope = "bn2")

        # residual
        outputs = outputs + inputs

        # highway
        for i in range(highway_layers):
            outputs = highway(inputs = outputs, nodes = 128, scope = "highway_{}".format(i))

        # BiGRU
        outputs = gru(inputs = outputs, nodes = 128, bidirection = True)

    return outputs



def transcript_encoder(inputs, embed_size, K, highway_layers, training, scope = "transcript_encoder"):
    # inputs: 2d tensor text represented by characters, shape = [Batch_size, Text Length]
    # outputs: [Batch_size, Text length, 256]

    with tf.variable_scope(scope):
        charac_size = inputs.get_shape().as_list()[-1]
        embed = embedding(inputs, charac_size, embed_size, scope="transcript_encoder")  #[batch, Tx, embed]
        
        prenet_out = prenet(embed, 256, 128, training) #[batch, Tx, 128]

        text = cbhg(inputs = prenet_out, K=K, highway_layers = highway_layers, training = training)
    return text



def reference_encoder(inputs, training, scope = "reference_encoder"):
    # inputs: Melspectrogram of reference audio. [Batch_size, ?, num_mels, 1]
    # outputs: Prosody vectors. [Batch_size, 128]
    # 6 layers of conv2d to [batch, ?/64, num_mels/64, 128]
    with tf.variable_scope(scope):
        inputs = tf.layers.conv2d(inputs = inputs, filters = 32, kernel_size = 3, strides = 2, padding = "SAME")
        inputs = batch_norm(inputs, training, scope = "bn1")
        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(inputs = inputs, filters = 32, kernel_size = 3, strides = 2, padding = "SAME")
        inputs = batch_norm(inputs, training, scope = "bn2")
        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(inputs = inputs, filters = 64, kernel_size = 3, strides = 2, padding = "SAME")
        inputs = batch_norm(inputs, training, scope = "bn3")
        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(inputs = inputs, filters = 64, kernel_size = 3, strides = 2, padding = "SAME")
        inputs = batch_norm(inputs, training, scope = "bn4")
        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(inputs = inputs, filters = 128, kernel_size = 3, strides = 2, padding = "SAME")
        inputs = batch_norm(inputs, training, scope = "bn5")
        inputs = tf.nn.relu(inputs)

        inputs = tf.layers.conv2d(inputs = inputs, filters = 128, kernel_size = 3, strides = 2, padding = "SAME")
        inputs = batch_norm(inputs, training, scope = "bn6")
        inputs = tf.nn.relu(inputs)

        # reshape [batch,?/64, num_mels/64, 128] to [batch, ?/64, num_mels/64*128]
        N, _ , W, C = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [N, -1, W*C])

        # convert to [batch, ?/64,128]
        inputs = gru(inputs = inputs, nodes = 128)

        # only get the last step, [batch, 128]
        inputs = inputs[:,-1,:]

        #[batch, 128]
        prosody = tf.layers.dense(inputs =inputs, units = 128, activation = tf.nn.tanh)

    return prosody



def attention_gru_decoder(inputs, inputs_lengths, memory, attention_rnn_nodes, decoder_rnn_nodes, 
    num_mels, reduction_factor, max_iters, training, scope = "attention_gru_decoder"):
    # input [batch, Ty/r, num_mels*r]
    # outputs [batch, Ty/r, num_mels*r] inputs and outputs both are shifted log melspectrogram of audio files, while at the 
    # training stage, the input is ground truth and the output is predicted audio file and at the testing stage, both are 
    # predicted audio features(log melspectrogram)
    # memory [batch, Tx, Ds+Dr+Dp]

    # Apply 1-layer GRU attention to decoder
    # nodes: attention size, # of rnn nodes
    # inputs: decoder inputs [batch, T', D']
    # memory: outputs of encoder [batch, T, D]
    # outputs: [batch, T', numunits]
    # state: the last step [batch, numunits]
    # T' = Ty/r, D' = n_mels

    with tf.variable_scope(scope):
        batch_size = memory.get_shape().as_list()[0]

        bahdanau_attention = tf.contrib.seq2seq.BahdanauAttention(
            num_units = attention_rnn_nodes,
            memory = memory)

        decoder_cell = tf.contrib.rnn.GRUCell(attention_rnn_nodes)

        # wrap cell with attention mechanism, we use default value of output_attention (false), 
        #that the output at each time is the output of "cell".
        # alignment_hist=true which stroe alignment history from all time steps (as a time major "TensorArray", so 
        # we must call stack() and transpose)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
            cell = rnn_utils.DecoderPrenetWrapper(decoder_cell, training),
            attention_mechanism = bahdanau_attention,
            alignment_history = True,
            output_attention = False)

        decoder_cell = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.OutputProjectionWrapper(cell_with_attention, attention_rnn_nodes),
            tf.contrib.rnn.ResidualWrapper(cell = tf.contrib.rnn.GRUCell(decoder_rnn_nodes)),
            tf.contrib.rnn.ResidualWrapper(cell = tf.contrib.rnn.GRUCell(decoder_rnn_nodes))
            ], state_is_tuple = True)

        output_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, num_mels*reduction_factor)
        
        decoder_init_state = output_cell.zero_state(batch_size = batch_size, dtype = tf.float32) #[batch, Ty/r, num_mels*r]

        if training:
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = inputs,
                sequence_length = tf.cast(tf.reshape(inputs_lengths, [batch_size]), tf.int32),
                time_major = False,
                name = "training_helper")
        else:
            helper = rnn_utils.InferenceHelper(batch_size = batch_size, out_size = num_mels*reduction_factor)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = output_cell,
            helper = helper,
            initial_state = decoder_init_state)

        (decoder_outputs, _), final_decoder_states, _ = tf.contrib.seq2seq.dynamic_decode(  
            decoder = decoder,   # decoder_outputs:(outputs:[batch, Ty/r, mel*r]),  _: sample_id:[batch, Ty/r]
            impute_finished = True,
            maximum_iterations = max_iters)  #final_decoder_states(many items), _:final_sequence_lengths

        alignments = tf.transpose(final_decoder_states[0].alignment_history.stack(), [1,2,0])
        mel_hats = tf.identity(decoder_outputs, name = "mel_hats") #[batch, Ty/r, num_mels*r]
        
    return mel_hats, alignments # pred_mels, align



def cbhg_postprocessing(inputs, num_mels, num_fft, K, highway_layers, training, scope = "cbhg_postprocessing"):
    # inputs [batch, Ty/r, num_mels*r]
    with tf.variable_scope(scope):
        # reshape [batch, Ty/r, num_mels*r] to [batch, Ty, num_mels]
        inputs = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1, num_mels])

        # conv1d banks
        outputs = conv1d_banks(inputs = inputs, K = K, filters = 128, training = training) #[batch, Ty, 128*K]

        # max pooling
        outputs = tf.layers.max_pooling1d(inputs = outputs, pool_size = 2, strides = 1, padding = "same") #[batch, Ty, 128*K]

        # conv1d projections
        outputs = conv1d(inputs = outputs, kernel_size = 3, filters= 256, scope ="conv1d_1") #[batch, Ty, 256]
        outputs = batch_norm(outputs, training, scope = "bn1")
        outputs = tf.nn.relu(outputs)

        outputs = conv1d(inputs = outputs, kernel_size = 3, filters = 80, scope = "conv1d_2") # 80=num_mels [batch, Ty, 80]
        outputs = batch_norm(outputs, training, scope = "bn2")

        # residual
        outputs = outputs + inputs

        # Extra affine transformation for dimensionality sync
        outputs = tf.layers.dense(outputs, 128) # (N, Ty, E/2)

        # highway
        for i in range(highway_layers):
            outputs = highway(inputs = outputs, nodes = 128, scope = "highway_{}".format(i))

        # BiGRU
        outputs = gru(inputs = outputs, nodes = 128, bidirection = True)

        # convert outputs to [batch, Ty, 1+n_fft//2]
        outputs = tf.layers.dense(outputs, 1+num_fft//2) 

        return outputs



'''
def attention_decoder_gru(inputs, memory, nodes, scope = "attention_dec_gru"):
    # Apply 1-layer GRU attention to decoder
    # nodes: attention size, # of rnn nodes
    # inputs: decoder inputs [batch, T', D']
    # memory: outputs of encoder [batch, T, D]
    # outputs: [batch, T', numunits]
    # state: the last step [batch, numunits]
    # T' = Ty/r, D' = n_mels

    with tf.variable_scope(scope):
        bahdanau_attention = tf.contrib.seq2seq.BahdanauAttention(
            num_units = nodes,
            memory = memory)

        decoder_cell = tf.contrib.rnn.GRUCell(nodes)
        
        # wrap cell with attention mechanism, we use default value of output_attention (false), 
        #that the output at each time is the output of "cell".
        # alignment_hist=true which stroe alignment history from all time steps (as a time major "TensorArray", so 
        # we must call stack() and transpose)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
            cell = decoder_cell,
            attention_machanism = bahdanau_attention,
            attention_layer_size = nodes,
            alignment_history = True)

        outputs, state = tf.nn.dynamic_rnn(
            cell = cell_with_attention,
            inputs = inputs,
            dtype = tf.float32)

        return outputs,state


def decoder_spectrogram(inputs, memory, attention_rnn_nodes, decoder_rnn_nodes, num_mels, 
    reduction_factor, training, scope = "decoder_spectrogram"):
    # input [batch, Ty/r, num_mels*r]
    # outputs [batch, Ty/r, num_mels*r] inputs and outputs both are shifted log melspectrogram of audio files, while at the 
    # training stage, the input is ground truth and the output is predicted audio file and at the testing stage, both are 
    # predicted audio features(log melspectrogram)
    # memory [batch, Tx, 256]

    with tf.variable_scope(scope):
        #prenet
        inputs = prenet(inputs, 256, 128, training) #[batch, Ty/r, num_mels*r] to []

        #Attention 
        outputs, state = attention_decoder_gru(inputs =inputs, memory = memory, nodes=attention_rnn_nodes)

        # tracing alignment results
        # time major stored alignment hist, must stack and transpose
        alignments = tf.transpose(state.alignment_history.stack(), [1,2,0])

        # 2 layer residual GRU;
        # attention: since the outputs [batch, Ty/r, 256] and the memory [batch, Tx, 256+128] have the different shape 
        # more than one dim, we just use outs to do next steps instead of concating them together in the original Tacotron
        outputs = outputs + gru(inputs = outputs, nodes = decoder_rnn_nodes)
        outputs = outputs + gru(inputs = outputs, nodes = decoder_rnn_nodes)

        # outputs of spectrogram 
        mel_hats = tf.layers.dense(outputs, num_mels*reduction_factor) #[batch, Ty/r, num_mels*r]

        return mel_hats, alignments
'''















