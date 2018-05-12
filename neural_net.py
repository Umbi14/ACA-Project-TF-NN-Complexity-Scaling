import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
import random

class Model:
    def __init__(self, config):
        self.n_classes = config['n_classes']
        self.n_filters = config['n_filters']
        self.keep_prob = tf.constant(0.75)

    def model(self, features):
        '''
        Function to build the neural net
        '''
        conv1 = tf.layers.conv2d(inputs=features,
                                  filters=self.n_filters,
                                  kernel_size=[5, 5],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool1')
        conv2 = tf.layers.conv2d(inputs=pool1,
                                  filters=self.n_filters,
                                  kernel_size=[5, 5],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool2')

        '''
        common part to all the models:
            - result as a 1D vector
            - fully connected layer
        '''

        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.layers.dense(pool2, 512, activation=tf.nn.relu, name='fc')
        logits = tf.layers.dense(fc, self.n_classes, name='logits')
        return logits
