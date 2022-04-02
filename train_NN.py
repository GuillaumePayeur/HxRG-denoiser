import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU as LeakyReLU
import numpy as np
import tensorflow.keras.backend as K
import os
import sys

# Creating the tuples that contains the names of the tfrecords files
tfrecords_val = ('data_46.tfrecords')
tfrecords_train = []
for i in range(46):
    tfrecords_train.append('data_{}.tfrecords'.format(i))
tfrecords_train = tuple(tfrecords_train)

# Setting up the decoder for the tfrecords files
def _parse_function(proto):
    keys_to_features = {'x': tf.io.FixedLenFeature([], tf.string),
                        'y': tf.io.FixedLenFeature([], tf.string)}

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    parsed_features['x'] = tf.io.decode_raw(parsed_features['x'], tf.float16)
    parsed_features['y'] = tf.io.decode_raw(parsed_features['y'], tf.float16)

    return parsed_features['x'], parsed_features['y']

# Setting up the generator
def create_dataset(file_names, batch_size):

    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    iterator = iter(dataset)

    while True:
        try:
            x, y = iterator.get_next()

            x = tf.reshape(x/90, [-1, 98, 127, 40, 3])
            x = x[:,:,:,0:32,:]
            x = np.transpose(x, (0, 2, 3, 1, 4)).reshape(x.shape[0], 127, 98*32*3)

            y = tf.reshape(y/60, [-1, 127*32, 3])
            y_left = y[:,:,0]
            y_mid = y[:,:,1]
            y_right = y[:,:,2]
            y = np.zeros((y.shape[0],127*32*3))
            y[:,0:1*32*127] = np.clip(np.abs(y_left-y_mid),0,0.083)
            y[:,1*32*127:2*32*127] = y_mid
            y[:,2*32*127:3*32*127] = np.clip(np.abs(y_right-y_mid),0,0.083)

            x = np.array(x)
            y = np.array(y)

        except tf.errors.OutOfRangeError:
            break

        yield x, y

# # Creating a lr scheduler
# def schedule(epoch):
#     if epoch <= 4:
#         return 1e-3 #3e-4
#     elif epoch <= 7:
#         return 8e-4 #1e-4
#     else:
#         return 5e-4 #3e-5
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

# Defining the loss function:
def MQE(y_true, y_pred):
    y_pred = y_pred[:,0:1*32*127]
    delta_left = y_true[:,0:1*32*127]
    y_true_mid = y_true[:,1*32*127:2*32*127]
    delta_right = y_true[:,2*32*127:3*32*127]
    multiplier = 1 + 60*derivative_multiplier*(delta_right+delta_left)
    return K.mean(multiplier*((y_pred-y_true_mid))**4)

# Changing the directory to the directory containing the data
data_dir = (sys.argv)[1]
os.chdir(data_dir)
# Changing the directory to the directory containing the data
# data_dir = 'D:\\dataset_v6'
# os.chdir(data_dir)

# Setting up the batches
train_batch_size = 16
validation_batch_size = 16
steps_per_epoch = (23500 // train_batch_size)
validation_steps = 500 // validation_batch_size
# Setting up the hyperparameters of the network
epochs = 50
gru_size = 208
conv_size = 64
kernel_size = 5
derivative_multiplier = 0
leak = 0.1
dropout = 0.05
rec_dropout = 0.05

# Building the network
inputs = keras.layers.Input((127, 98*32*3))
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

lr = 2e-3
# Compiling the network and fitting it to the data
model.compile(optimizer='adam', loss=MQE)
for j in range(epochs):
    lr = lr*0.98
    K.set_value(model.optimizer.learning_rate, lr)
    os.chdir(data_dir)
    model.fit(create_dataset(tfrecords_train, train_batch_size),
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        validation_data=create_dataset(tfrecords_val, validation_batch_size),
        validation_steps=validation_steps,
        callbacks=[],
        verbose=2,
        max_queue_size=10
        )

    # Saving the model every epoch
    os.chdir('/home/payeur/scratch/NIR/checkpoints')
    model.save('model_v7_5_{}.h5'.format(j+1))
