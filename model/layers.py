# -*- coding: utf-8 -*-

from __future__ import division, print_function

import tensorflow as tf

__author__ = "Tong Wang"

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


################################################################################
# Convenience functions of simple layers for building the Expressive Tacotron model.
################################################################################

class EmbeddingLayer(tf.layers.Layer):
    ''' calculate input embeddings with shared weights'''
    def __init__(self, vocab_size, hidden_size, scope):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_scope = scope


    def build(self, _):
        #create and initialize weights
        with tf.variable_scope(self.embed_scope, reuse = tf.AUTO_REUSE):
            self.lookup_table = tf.get_variable("table_weights", [self.vocab_size, self.hidden_size],
            initializer = tf.random_normal_initializer(0., self.hidden_size**-0.5))
        
        self.built = True
    

    def call(self,x):
        ''' get token embeddings of x '''
        with tf.name_scope(self.embed_scope):
            embeddings = tf.gather(self.lookup_table, x)

            # scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5
            return embeddings




def embedding(inputs, charac_size, embed_size, scope="embedding"):
    """ Embed the input character """
    embed_layer = EmbeddingLayer(charac_size, embed_size, scope)

    return embed_layer(inputs)

    '''
    with tf.variable_scope(scope):
        lookup_table = tf.Variable(
            initial_value = tf.truncated_normal([charac_size, embed_size], mean=0.0, stddev=0.01, dtype=tf.float32), 
            name='lookup_table')

        return tf.nn.embedding_lookup(lookup_table, inputs)
    '''


def batch_norm(inputs, training, data_format = "channels_last", scope="bn"):
    """Performs a batch normalization using a standard set of parameters."""  
    # We set fused=True for a significant performance boost.
    
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(
            inputs = inputs, axis = 1 if data_format == 'channels_first' else len(inputs.get_shape().as_list())-1, 
            momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON, 
            center = True, scale = True, training = training, fused= True)



def conv1d(inputs, kernel_size=1, filters=128, padding = "SAME", dilation_rate = 1, 
    data_format = "channels_last", scope = "conv1d"):

    with tf.variable_scope(scope):
        return tf.layers.conv1d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            padding = padding,
            data_format = data_format,
            dilation_rate = dilation_rate,
            kernel_initializer = tf.variance_scaling_initializer())



def conv1d_banks(inputs, K, filters, training, scope = "conv1d_banks"):
    ''' CBHG component, apply series of conv1d  seperately '''
    # inputs [batch, T, C]
    # output [batch, T, K*filters]

    with tf.variable_scope(scope):
        outputs = tf.concat(
            [conv1d(inputs = inputs, 
                    filters = filters, 
                    kernel_size = k, 
                    scope = 'num_{}_conv1d'.format(k)) for k in range(1, K+1)],
            axis=-1)

        outputs = batch_norm(outputs, training)
        outputs = tf.nn.relu(outputs)

        return outputs



def gru(inputs, nodes, bidirection = False, scope = "gru"):
    # inputs [N, TimeStep, D]
    # outputs [N, TimeStep, nodes*2 if bidirectional else nodes]

    with tf.variable_scope(scope):
        if bidirection:
            gru_fw_cell = tf.contrib.rnn.GRUCell(nodes)
            gru_bw_cell = tf.contrib.rnn.GRUCell(nodes)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = gru_fw_cell,
                cell_bw = gru_bw_cell,
                inputs = inputs,
                dtype = tf.float32)

            outputs = tf.concat(outputs, axis=2)
        else:
            cell = tf.contrib.rnn.GRUCell(nodes)
            outputs, _ =tf.nn.dynamic_rnn(
                cell = cell, 
                inputs =inputs,
                dtype = tf.float32)

        return outputs



def prenet(inputs, nodes_dense1, nodes_dense2, training, scope = "prenet"):
    with tf.variable_scope(scope):
        outputs = tf.layers.dense(
            inputs = inputs, 
            units = nodes_dense1,
            activation = tf.nn.relu,
            name = "dense1")

        outputs = tf.layers.dropout(
            inputs = outputs, 
            rate = 0.5, 
            training = training,
            name = "dropout1")

        outputs = tf.layers.dense(
            inputs = outputs, 
            units = nodes_dense2,
            activation = tf.nn.relu,
            name = "dense2")

        outputs = tf.layers.dropout(
            inputs = outputs, 
            rate = 0.5, 
            training = training,
            name = "dropout2")

    return outputs



def highway(inputs, nodes, scope = "highway"):
    ''' huber https://arxiv.org/abs/1505.00387'''
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs = inputs,
            units = nodes,
            activation = tf.nn.relu,
            name = "hidden")

        T = tf.layers.dense(
            inputs = inputs,
            units = nodes,
            activation = tf.nn.sigmoid,
            bias_initializer=tf.constant_initializer(-1.0),
            name = "transform_gate")

        outputs = H*T + inputs*(1-T)

    return outputs