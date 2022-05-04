import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU as LeakyReLU
import numpy as np
import tensorflow.keras.backend as K
import os
import glob
import sys

'''
This code trains the neural network using the tfrecords files previously
created. It saves the neural network weights after every epoch in a folder with
the same name as "ramp_folder" but with a suffix "_NN".

ramp_001/*.fits will have NN weights in ramp_001_NN/
'''
# Setting up the decoder for the tfrecords files
def _parse_function(proto):
    keys_to_features = {'x': tf.io.FixedLenFeature([], tf.string),
                        'y': tf.io.FixedLenFeature([], tf.string)}

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    parsed_features['x'] = tf.io.decode_raw(parsed_features['x'], tf.float16)
    parsed_features['y'] = tf.io.decode_raw(parsed_features['y'], tf.float16)

    return parsed_features['x'], parsed_features['y']

# Setting up the generator, which loads training examples from the tfrecords
# files and provides them to the neural network
def create_dataset(file_names, batch_size):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    iterator = iter(dataset)

    while True:
        try:
            x, y = iterator.get_next()

            x = tf.reshape(x/90, [-1, 98, 128, 40, 3])
            x = np.transpose(x, (0, 2, 3, 1, 4)).reshape(x.shape[0], 128, 98*40*3)

            y = tf.reshape(y/60, [-1, 128*32, 3])
            y_mid = y[:,:,1]
            y = np.zeros((y.shape[0],128*32*3))
            y[:,1*32*128:2*32*128] = y_mid

            x = np.array(x)
            y = np.array(y)
        except tf.errors.OutOfRangeError:
            break
        yield x, y

# Defining the L4 loss function.
def L4(y_true, y_pred):
    y_pred = y_pred[:,0:1*32*128]
    y_true = y_true[:,1*32*128:2*32*128]
    return K.mean((y_pred-y_true)**4)

def train_nn(ramp_folder,NN_name):
    print('Training neural network')
    # loop through tfrecords files
    files = glob.glob(ramp_folder+'_tfrecords/*.tfrecords')
    num_files = len(files)
    # All tfrecords files are used as training files except the last
    tfrecords_train = tuple(files[0:-1])
    tfrecords_val = (files[-1])

    # Setting up the batches
    train_batch_size = 16
    validation_batch_size = 16
    steps_per_epoch = ((num_files-1)*500 // train_batch_size)
    validation_steps = 500 // validation_batch_size
    # Setting up the hyperparameters of the network
    epochs = 50
    gru_size = 208
    conv_size = 64
    kernel_size = 5
    leak = 0.1
    dropout = 0.05
    rec_dropout = 0.05
    lr = 2e-3

    # Building the network
    inputs = keras.layers.Input((128, 98*40*3))
    gru_forward = keras.layers.GRU(gru_size, activation='tanh', recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True)(inputs)
    gru_backward = keras.layers.GRU(gru_size, activation='tanh', recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True, go_backwards=True)(inputs)
    gru_backward = keras.layers.Lambda(lambda x: K.reverse(x,1))(gru_backward)
    gru = keras.layers.Concatenate(axis=2)([gru_forward, gru_backward])
    lambdaa = keras.layers.Lambda(lambda x: K.expand_dims(x, 3))(gru)
    conv = keras.layers.Conv2D(conv_size, (kernel_size,kernel_size), activation=LeakyReLU(leak), padding='same')(lambdaa)
    conv = keras.layers.Conv2D(conv_size, (kernel_size,kernel_size), activation=LeakyReLU(leak), padding='same')(conv)
    conv = keras.layers.Conv2D(1, (kernel_size,kernel_size), activation=LeakyReLU(leak), padding='same')(conv)
    gru = keras.layers.GRU(32, activation='tanh', recurrent_activation='sigmoid', dropout=dropout, recurrent_dropout=rec_dropout, return_sequences=True)(conv[:,:,:,0])
    outputs = keras.layers.Flatten()(gru)
    outputs = keras.layers.Concatenate(axis=1)([outputs,outputs,outputs])

    model = keras.Model(inputs=inputs, outputs=outputs, name='model_series')
    model.summary()

    # Compiling the network and fitting it to the data
    model.compile(optimizer='adam', loss=L4)
    for j in range(epochs):
        print(j)
        lr = lr*0.98
        K.set_value(model.optimizer.learning_rate, lr)
        model.fit(create_dataset(tfrecords_train, train_batch_size),
            epochs=1,
            steps_per_epoch=steps_per_epoch,
            validation_data=create_dataset(tfrecords_val, validation_batch_size),
            validation_steps=validation_steps,
            callbacks=[],
            verbose=1,
            max_queue_size=10
            )

        # Saving the model every epoch
        model.save(ramp_folder+'_NN/{}_epoch_{}.h5'.format(NN_name,j+1))
