from pymongo import MongoClient
from matplotlib import pyplot as plt
import numpy as np
import json
import itertools

config_file = json.load(open('./config.json'))

#get mongo collection
mongo = MongoClient()
db = mongo.complexity_scaling   # db

legend = []
for i in config_file['input_size']:
    print('i', i)
    y = []
    for c in itertools.product(config_file['n_layers'], config_file['n_filters'], config_file['batch_size'],
                                        config_file['n_classes'], config_file['kernel_size'],config_file['fc_units'],config_file['kernel_stride']):
        config = {'n_layers' : c[0],
                    'n_filters' : c[1],
                    'batch_size' : c[2],
                    'input_size' : i,
                    'n_classes' : c[3],
                    'kernel_size' : c[4],
                    'fc_units' : c[5],
                    'kernel_stride' : c[6]}
        print(config)
        collection_name = 'config_56x56_nfilters'#'config_' + str(i[0]) + 'x' + str(i[1]) + '_4'
        print(c)
        collection = db[collection_name]    # collection
        tuples = list(collection.find({'config': config}))
        inference_times = np.asarray([t['tot_parmas'] for t in tuples ]) #inference_time #total_mem tot_flops
        print(np.mean(inference_times))
        y.append(np.mean(inference_times))
    plt.plot(config_file['n_filters'], y)
    legend.append(str(i[0]) + 'x' + str(i[1]))



plt.xlabel('num. filters per conv layer')
plt.ylabel('tot parameters')
plt.legend(legend)
plt.show()
