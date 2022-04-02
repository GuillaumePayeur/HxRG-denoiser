import numpy as np
import tensorflow as tf
import os
################################################################################
# Program that takes npy files and converts them to tfrecords

# Directory where the npy files are located
npy_dir = 'D:\\dataset_v7'
# Directory where the files should be saved
tfrecords_dir = 'D:\\dataset_v7'
################################################################################

files_to_write = np.arange(0,47,1).tolist()

# Helper function
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for i in range(len(files_to_write)):
    print('starting file', files_to_write[i])

    # Loading the npy files
    os.chdir(npy_dir)
    x = np.load('data_{}.npy'.format(files_to_write[i])).astype('float16')
    y = np.load('labels_{}.npy'.format(files_to_write[i])).astype('float16')
    os.chdir(tfrecords_dir)

    # Reshaping the numpy files
    n1, n2, n3, n4, n5 = x.shape[0:5]

    x = np.reshape(x, (n1, n2*n3*n4*n5))

    # writing the tfrecords file
    writer = tf.io.TFRecordWriter('data_{}.tfrecords'.format(files_to_write[i]))

    for m in range(x.shape[0]):

        x_temp = x[m]
        y_temp = y[m]

        feature = {'x':  _bytes_feature(tf.compat.as_bytes(x_temp.tostring())),
                   'y':  _bytes_feature(tf.compat.as_bytes(y_temp.tostring()))}

        # Serializing to string and writing to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
