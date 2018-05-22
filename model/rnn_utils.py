# -*- coding: utf-8 -*-
from __future__ import division, print_function

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.seq2seq import CustomHelper

from layers import prenet 

__author__ = "Tong Wang"

class DecoderPrenetWrapper(RNNCell):
	"make RNN inputs through a prenet before sending them into the cell"    
	def __init__(self, cell, training):
		super(DecoderPrenetWrapper, self).__init__()
		self._cell = cell
		self._training = training

	@property
	def state_size(self):
	    return self._cell.state_size

	@property
	def output_size(self):
	    return self._cell.output_size
	
	def call(self, inputs, state):
		prenet_out = prenet(inputs, 256, 128, self._training, scope = "decoder_prenet")
		return self._cell(prenet_out, state)

	def zero_state(self, batch_size, dtype):
		return self._cell.zero_state(batch_size, dtype)
	



class InferenceHelper(CustomHelper):
    "Custom helper for evaluating and synthesize stages"



    def _initialize_fn(self):
        # we always reconstruct the whole output
        finished = tf.tile([False], [self._batch_size])
        next_inputs = tf.zeros([self._batch_size, self._out_size], dtype=tf.float32)
        return (finished, next_inputs)

    def _sample_fn(self, time, outputs, state):
        # we're not sampling from a vocab so we don't care about this function
        return tf.zeros(self._batch_size, dtype=tf.int32)

    def _next_inputs_fn(self, time, outputs, state, sample_ids):
        del time, sample_ids
        finished = tf.tile([False], [self._batch_size])
        next_inputs = outputs
        return (finished, next_inputs, state)

    def __init__(self, batch_size, out_size):
    	super(InferenceHelper, self).__init__(self._initialize_fn, self._sample_fn, self._next_inputs_fn)
        self._batch_size = batch_size
        self._out_size = out_size