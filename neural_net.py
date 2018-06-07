import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
import random
import complexity

class Model:
    def __init__(self, config):
        self.n_classes = config['n_classes']
        self.n_filters = config['n_filters']
        self.n_layers = config['n_layers']
        self.kernel_size = config['kernel_size']
        self.fc_units = config['fc_units']
        self.keep_prob = tf.constant(0.75)

    def model(self, features):
        model_complexity = {}
        model = {}
        for l in range(self.n_layers):
            if l == 0:
                layer_input = features
            else:
                layer_input = model[str(l-1)]
            conv = tf.layers.conv2d(inputs=layer_input,
                                      filters=self.n_filters,
                                      kernel_size=self.kernel_size,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      name='conv'+str(l))
            pool = tf.layers.max_pooling2d(inputs=conv,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool'+str(l))

            model[str(l)] = pool
            print('pool shape', pool.shape[3])
        print('model', model)

        '''
        common part to all the models:
            - result as a 1D vector
            - fully connected layer
        '''

        feature_dim = pool.shape[1] * pool.shape[2] * pool.shape[3]
        reshaped = tf.reshape(pool, [-1, feature_dim])
        fc = tf.layers.dense(reshaped, self.fc_units, activation=tf.nn.relu, name='fc')
        logits = tf.layers.dense(fc, self.n_classes, name='logits')

        fc_flops = complexity.fc_flops(reshaped.shape[1], self.fc_units)

        model_complexity['fc_flops'] = pool.shape[3] * self.fc_units
        return logits, model_complexity
