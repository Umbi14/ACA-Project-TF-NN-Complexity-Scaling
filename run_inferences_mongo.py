"""
Run inferences and store the result in mongodb
"""
from pymongo import MongoClient
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

#get mongo collection
collection_name = 'config_56x56_nfilters'
mongo = MongoClient()
db = mongo.complexity_scaling   # db
collection = db[collection_name]    # collection
num_inferences = 1 # number of iteration for each configuration
store = True

count = 0
for c in itertools.product(config_file['n_layers'], config_file['n_filters'], config_file['batch_size'],
                            config_file['input_size'],config_file['n_classes'], config_file['kernel_size'],config_file['fc_units'],
                            config_file['kernel_stride']):
    print('configuration', count)
    count += 1
    for i in range(num_inferences):
        print(i)
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
        model_complexity = sl.infer()
        model_complexity['config'] = config
        if store:
            collection.insert(model_complexity)
        del sl
