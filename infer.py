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

config = json.load(open('./config.json'))
BATCH_SIZE = config['batch_size']

class StreetLearning:
    def __init__(self):
        #dataaset
        dataset_name = config['dataset_name']#'einstein' #'kitti_data_road'
        self.train_img_path = 'data/' + dataset_name + '/trainingtraining/inputs'
        self.train_target_img_path = 'data/' + dataset_name + '/trainingtraining/targets'
        self.test_img_path = 'data/' + dataset_name + '/testingtesting/inputs'
        self.test_target_img_path = 'data/' + dataset_name + '/testingtesting/targets'

        self.input_dim = config['input_size'] #[1242,375,3] #this must be the right size of the images
        self.resized_dim = [self.input_dim[0],self.input_dim[1]]#[92,28]


    def get_dataset(self, img_path, target_img_path):
        '''
        Parameters: paths to the dataset folder
        Return: tuple of 2 tensors:
            filenames: list of paths of each image in th edataset
            labels: list of labels
        '''
        inputs_file_paths = glob.glob(os.path.join(img_path, '*'))
        labels = []
        for input_file_path in inputs_file_paths:
            #Extract target filepath
            file_header = input_file_path.split("/")[-1].split("_")[0]
            file_number = input_file_path.split("_")[-1]
            target_file_path = target_img_path + "/" + file_header + '_'
            if file_header == 'um':
                target_file_path += 'lane_'
                labels.append([1,0])
            else:
                labels.append([0,1])
                target_file_path += 'road_'
            target_file_path += file_number
            #labels.append([1]*self.n_classes)#(target_file_path)  #np.random.sample(self.n_classes) #[random.randint(0,self.n_classes-1)]*self.n_classes

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
        test_data = self.get_dataset(self.test_img_path, self.test_target_img_path)  # tuple of (inputs filenames, labels)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.map(self._parse_function).batch(BATCH_SIZE).repeat()    #test_data[0].shape[0]

        #iterator
        iterator = tf.data.Iterator.from_structure(test_dataset.output_types,test_dataset.output_shapes)
        self.features, self.labels = iterator.get_next()

        #init
        self.test_init = iterator.make_initializer(test_dataset)    # initializer for test_dataset

    def model(self):
        '''
        Function to build the neural net
        '''
        model = Model(config)
        self.logits = model.model(self.features)

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

    def train(self):
        with tf.Session() as sess:
            print('in the session')
            test_len = len(glob.glob(os.path.join(self.test_img_path, '*')))
            n_batches = test_len // BATCH_SIZE
            sess.run(tf.global_variables_initializer())
            print('variables initialized')
            self.training = False
            # initialise iterator with test data
            sess.run(self.test_init)
            start = time.time()
            for _ in range(n_batches):
                print(sess.run(self.logits))
            print('inference took', time.time() - start, 's')


if __name__ == '__main__':
    sl = StreetLearning()
    sl.get_data()
    sl.model()
    sl.loss()
    sl.optimizer()
    sl.train()
