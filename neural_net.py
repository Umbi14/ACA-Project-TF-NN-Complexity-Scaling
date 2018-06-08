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
        model_complexity = {'conv': {}, 'pool' : {}, 'fc' : {}, 'total_mem' : 0, 'tot_parmas' : 0}
        model = {}
        for l in range(self.n_layers):
            if l == 0:
                layer_input = features
            else:
                layer_input = model[str(l-1)]

            '''
            convolutional layer
            '''
            conv = tf.layers.conv2d(inputs=layer_input,
                                      filters=self.n_filters,
                                      kernel_size=self.kernel_size,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      name='conv'+str(l))

            # convolutional layer complexity
            model_complexity['conv'][str(l)] = {}
            model_complexity['conv'][str(l)]['description'] = str(layer_input.get_shape().as_list()[1]) + 'x' + str(layer_input.get_shape().as_list()[2]) + 'x' + str(self.n_filters)
            model_complexity['conv'][str(l)]['flops'] = complexity.conv_flops(self.kernel_size, layer_input.get_shape(), self.n_filters, conv.get_shape())
            # weights = (kernel width * kernel heigth * input channles) * self.n_filters
            model_complexity['conv'][str(l)]['weights'] = complexity.weights(self.kernel_size,layer_input.get_shape(), self.n_filters)
            # memory = input width * input height * num of filters
            model_complexity['conv'][str(l)]['memory'] = complexity.memory(layer_input.get_shape(), self.n_filters)
            model_complexity['total_mem'] += model_complexity['conv'][str(l)]['memory']
            model_complexity['tot_parmas'] += model_complexity['conv'][str(l)]['weights']

            '''
            max pool layer
            '''
            pool = tf.layers.max_pooling2d(inputs=conv,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool'+str(l))
            # pool layer complexity
            model_complexity['pool'][str(l)] = {}
            model_complexity['pool'][str(l)]['description'] = str(pool.get_shape().as_list()[1]) + 'x' + str(pool.get_shape().as_list()[2]) + 'x' + str(self.n_filters)
            model_complexity['pool'][str(l)]['memory'] = complexity.memory(pool.get_shape(), self.n_filters)
            model_complexity['total_mem'] += model_complexity['pool'][str(l)]['memory']

            model[str(l)] = pool
        print('model', model)

        '''
        common part to all the models:
            - result as a 1D vector
            - fully connected layer
        '''

        feature_dim = pool.shape[1] * pool.shape[2] * pool.shape[3]
        reshaped = tf.reshape(pool, [-1, feature_dim])
        fc = tf.layers.dense(reshaped, self.fc_units, activation=tf.nn.relu, name='fc')
        print('**************', fc.get_shape())
        logits = tf.layers.dense(fc, self.n_classes, name='logits')

        model_complexity['fc']['0'] = {}
        model_complexity['fc']['0']['description'] = '1x1x' + str(fc.get_shape().as_list()[1])
        model_complexity['fc']['0']['flops'] = complexity.fc_flops(reshaped.get_shape().as_list()[1], self.fc_units)
        # memory = n of neurons
        model_complexity['fc']['0']['memory'] = fc.get_shape().as_list()[1]
        # weigths = input mem * n of neurons
        model_complexity['fc']['0']['weights'] = fc.get_shape().as_list()[1] * model_complexity['pool'][str(l)]['memory']
        model_complexity['total_mem'] += model_complexity['fc']['0']['memory']
        model_complexity['tot_parmas'] += model_complexity['fc']['0']['weights']

        # tot_memory multiplied by 4 bytes
        model_complexity['total_mem'] = model_complexity['total_mem'] * 4
        return logits, model_complexity
