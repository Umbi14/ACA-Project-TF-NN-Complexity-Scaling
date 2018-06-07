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
a = ["foo", "melon"]
b = [True, False]
c = [(x,y) for x in a for y in b]
print(c)

config_file = json.load(open('./config.json'))

for n_layers in config_file['n_layers']:
    for n_filters in config_file['n_filters']:
        for batch_size in config_file['batch_size']:
            for input_size in config_file['input_size']:
                for n_classes in config_file['n_classes']:
                    for kernel_size in config_file['kernel_size']:
                        for fc_units in config_file['fc_units']:
                            config = {'n_layers' : n_layers,
                                        'n_filters' : n_filters,
                                        'batch_size' : batch_size,
                                        'input_size' : input_size,
                                        'n_classes' : n_classes,
                                        'kernel_size' : kernel_size,
                                        'fc_units' : fc_units}
                            print(config)
                            sl = Infer(config)
                            sl.get_data()
                            sl.model()
                            sl.loss()
                            sl.optimizer()
                            sl.infer()
                            del sl



'''
if __name__ == '__main__':
    sl = Infer(config)
    sl.get_data()
    sl.model()
    sl.loss()
    sl.optimizer()
    sl.infer()
'''
