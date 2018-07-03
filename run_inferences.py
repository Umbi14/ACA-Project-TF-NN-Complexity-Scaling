import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import time
import tensorflow as tf
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
import random
from infer import Infer
import itertools

config_file = json.load(open('./config.json'))
num_inferences = 10 # number of iteration for each configuration

for _ in range(num_inferences):
    pass
    for c in itertools.product(config_file['n_layers'], config_file['n_filters'], config_file['batch_size'],
                                        config_file['input_size'],config_file['n_classes'], config_file['kernel_size'],config_file['fc_units'],
                                        config_file['kernel_stride']):
        config = {'n_layers' : c[0],
                    'n_filters' : c[1],
                    'batch_size' : c[2],
                    'input_size' : c[3],
                    'n_classes' : c[4],
                    'kernel_size' : c[5],
                    'fc_units' : c[6],
                    'kernel_stride' : c[7]}
        print(config)
        sl = Infer(config)
        sl.get_data()
        sl.model()
        sl.loss()
        sl.optimizer()
        sl.infer()
        del sl
