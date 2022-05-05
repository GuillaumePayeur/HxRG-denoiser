import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU as LeakyReLU
import os
import tensorflow.keras.backend as K
from astropy.io import fits
from tqdm import tqdm
import glob

def generate_NN_predictions(ramp_folder,NN_name):
    '''
    This code takes in a directory containing the sequences of ribbons for a
    data cube, a neural network, and returns the predictions of the neural
    network. The predictions on individual row are saved in a folder with
    the same name as "ramp_folder" but with a suffix "_predictions".

    ramp_001/*.fits will have predictions in ramp_001_predictions/

    Additionally we combine the predictions into a 4096x4096 fits filled, and
    save it in the working directory
    '''
    # Creating folder to contain the predictions
    outfolder = ramp_folder+'_predictions/'
    if not os.path.isdir(outfolder):
        print('We create {} to contain the neural network predictions'.format(outfolder))
        os.mkdir(outfolder)

    # Loading the model (chosing the last epoch available)
    files = glob.glob(ramp_folder+'_NN/{}_*.h5'.format(NN_name))
    files.sort()
    model = keras.models.load_model(files[-1], compile=False, custom_objects={'LeakyReLU': LeakyReLU})

    # Predicting on every row of the array one by one.
    print('generating predictions of the neural network')
    for i in tqdm(range(5, 4091)):
        # Loading the data
        x = np.zeros((98,128,40,3))
        for j in range(3):
            x[:,:,:,j] = np.load(ramp_folder+'_ribbons/r_{}.npy'.format(i-1+j))
        x = np.expand_dims(x,0)
        x = np.transpose(x, [0,2,3,1,4])
        x = np.reshape(x, (x.shape[0], 128, 98*40*3))

        # Making the predictions
        y_pred = model.predict(x/90, batch_size=3)*60
        y_pred = y_pred[0,0:128*32]

        # Saving the predictions
        np.save(outfolder+'/predictions_{}.npy'.format(i), y_pred)
        K.clear_session()

    # Combining the predictions into one image
    print('combining the predictions into one image')
    image_ML = np.zeros((4096,4096))
    # Loading the predictions
    for i in tqdm(range(5, 4091)):
        y_pred = np.load(outfolder+'/predictions_{}.npy'.format(i))
        for j in range(0,128):
            for k in range(1,32,2):
                image_ML[i,j+k*128] = y_pred[j*32+k]
            for k in range(0,32,2):
                image_ML[i,127-j+k*128] = y_pred[j*32+k]

    # Saving the predictions as a fits file
    fits.writeto(ramp_folder+'_{}_predictions.fits'.format(NN_name), np.array([]))
    fits.append(ramp_folder+'_{}_predictions.fits'.format(NN_name), image_ML)
