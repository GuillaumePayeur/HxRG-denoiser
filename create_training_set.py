import numpy as np
from astropy.io import fits
import tensorflow as tf
import os
from tqdm import tqdm

def generate_training_set(ramp_folder):
    '''
    This code takes the ribbon sequences previously saved and packages
    them into a training dataset for the neural network. The data is saved
    as tfrecords files in a folder with the same name as "ramp_folder" but with
    a suffix "_tfrecords".

    ramp_001/*.fits will have tfrecords in ramp_001_tfrecords/
    '''
    # Randomly ordering all available ribbon sequences
    indexes = np.arange(5,4091)
    np.random.shuffle(indexes)

    # Splitting the ribbon sequences into 8 tfrecords files of 500 training
    # examples
    print('Creating training dataset')
    for i in tqdm(range(8)):
        x = np.zeros((500, 98, 128, 40, 3),dtype=np.float16)
        y = np.zeros((500, 32*128, 3),dtype=np.float16)

        # Loading the ribbon sequences
        for j in range(500):
            for k in range(3):
                x[j,:,:,:,k] = np.load('{}_ribbons/r_{}.npy'.format(ramp_folder,indexes[i*j+j]-1+k)).astype('float16')
                y[j,:,k] = np.load('{}_ribbons/r_labels_{}.npy'.format(ramp_folder,indexes[i*j+j]-1+k)).astype('float16')
        # Reshaping the numpy files
        n1, n2, n3, n4, n5 = x.shape[0:5]
        x = np.reshape(x, (n1, n2*n3*n4*n5))

        # writing the data to tfrecords file
        if not os.path.isdir('{}_tfrecords'.format(ramp_folder)):
            os.mkdir('{}_tfrecords'.format(ramp_folder))
        writer = tf.io.TFRecordWriter('{}_tfrecords/data_{}.tfrecords'.format(ramp_folder,i))

        for m in range(x.shape[0]):
            x_temp = x[m]
            y_temp = y[m]

            feature = {'x':  _bytes_feature(tf.compat.as_bytes(x_temp.tostring())),
                       'y':  _bytes_feature(tf.compat.as_bytes(y_temp.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
