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
from neural_net import Model
import pprint


class Infer:
    def __init__(self, config):
        self.config = config
        #dataset
        self.batch_size = config['batch_size']

        self.input_dim = config['input_size']
        dataset_name = self.get_dataset_name(self.input_dim) #config['dataset_name']
        self.test_img_path = 'data/' + dataset_name + '/test/inputs'

        self.resized_dim = [self.input_dim[0],self.input_dim[1]]

        self.n_classes = config['n_classes']

    def get_dataset_name(self, input_dim):
        """
        Given the input size, get the name of the folder where the dataset is contained
        """
        name = str(input_dim[0])
        for i in input_dim[1:]:
            name += 'x' + str(i)
        return name


    def get_dataset(self, img_path):
        '''
        Parameters: paths to the dataset folder
        Return: tuple of 2 tensors:
            filenames: list of paths of each image in th edataset
            labels: list of labels
        '''
        inputs_file_paths = glob.glob(os.path.join(img_path, '*'))

        # assaign random class
        labels = []
        for _ in inputs_file_paths:
            c = [0]*self.n_classes
            random_index = int(self.n_classes * random.random())
            c[random_index] = 1
            labels.append(c)

        filenames = tf.constant(inputs_file_paths)
        labels = tf.constant(labels)
        return (filenames, labels)

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)  #image_decoded = tf.image.decode_png(image_string)
        image_reshaped = tf.reshape(image_decoded, self.input_dim)
        image_resized = tf.image.resize_images(image_reshaped, self.resized_dim)
        image_float32 = tf.image.convert_image_dtype(image_resized, dtype = tf.float32)
        return image_float32, label

    def get_data(self):
        # using two numpy arrays
        test_data = self.get_dataset(self.test_img_path)  # tuple of (inputs filenames, labels)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.map(self._parse_function).batch(self.batch_size).repeat()    #test_data[0].shape[0]

        #iterator
        iterator = tf.data.Iterator.from_structure(test_dataset.output_types,test_dataset.output_shapes)
        self.features, self.labels = iterator.get_next()

        #init
        self.test_init = iterator.make_initializer(test_dataset)    # initializer for test_dataset

    def model(self):
        '''
        Function to build the neural net
        '''
        model = Model(self.config)
        self.logits, self.model_complexity = model.model(self.features)

    def loss(self):
        '''
        Loss function
        '''
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
        self.loss = tf.reduce_mean(entropy, name='loss')

    def optimizer(self):
        '''
        Optimizer
        '''
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def infer(self):

        '''for variable in tf.trainable_variables():
            print('*', variable, '*')
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print('shape', shape)
            print('len(shape)', len(shape))'''

        '''run_metadata = tf.RunMetadata()
        # Print trainable variable parameter statistics to stdout.
        ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder

        print('*** memory ***')
        memory = tf.profiler.profile(
            tf.get_default_graph(),
            run_meta = run_metadata,
            options=ProfileOptionBuilder.time_and_memory())
        print('memory:', memory)

        print('*** param_stats ***')
        param_stats = tf.profiler.profile(
            tf.get_default_graph(),
            run_meta = run_metadata,
            options=ProfileOptionBuilder.trainable_variables_parameter())
        print('total params:', param_stats.total_parameters)'''

        with tf.Session() as sess:
            print('in the session')
            test_len = len(glob.glob(os.path.join(self.test_img_path, '*')))
            n_batches = test_len // self.batch_size
            sess.run(tf.global_variables_initializer())
            print('variables initialized')
            self.training = False
            # initialise iterator with test data
            sess.run(self.test_init)
            start = time.time()
            for i in range(n_batches):
                sess.run(self.logits)
            # inferece time (of a single image) is compute as the total inference time of the whole test set,
            # divided by the number of element in the data set
            inference_time = (time.time() - start) / test_len
            self.model_complexity['inference_time'] = inference_time
            pprint.pprint(self.model_complexity)
            for i in range(self.config['n_layers']):
                print('CONV'+str(i), '['+self.model_complexity['conv'][str(i)]['description']+']', 'memory', self.model_complexity['conv'][str(i)]['memory'], 'weights', self.model_complexity['conv'][str(i)]['weights'], 'FLOPS', self.model_complexity['conv'][str(i)]['flops'])
                print('POOL'+str(i), '['+self.model_complexity['pool'][str(i)]['description']+']', 'memory', self.model_complexity['pool'][str(i)]['memory'])
            print('FC', '['+self.model_complexity['fc']['0']['description']+']', 'memory', self.model_complexity['fc']['0']['memory'], 'weights', self.model_complexity['fc']['0']['weights'], 'FLOPS', self.model_complexity['fc']['0']['flops'])

            print('TOTAL PARAMETERS', self.model_complexity['tot_parmas'])
            print('TOTAL MEMORY', self.model_complexity['total_mem'], 'bytes')
            print('TOTAL FLOPS', self.model_complexity['tot_flops'])
            print('inference took', self.model_complexity['inference_time'], 'seconds')

        # delete the graph so that a new one can be build with different configurations
        tf.reset_default_graph()
        return self.model_complexity
