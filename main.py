from create_fake_ramp import *
from create_ribbon_sequences import *
from create_training_set import *
from train_NN import *
from predict import *
'''
This code trains a demonstration HxRG denoiser. It takes in one 'scene' image,
generates one synthetic ramp, generates a small training set using it, and
trains a neural network using it. We also provide code to test the trained
network
'''

'''
scene should be the name of the high-SNR 'scene' image, which is expected to be
a 4096x4096 fits file. The synthetic training data cube is generated using it.
An example file "TEFF3500.fits" is provided.
'''
scene = 'TEFF3500.fits'
'''
ramp_folder should be a directory containing individual dark readout files. We
assume that the dark data cube has 100 readouts. The 100 files are expected to
be  the only fits files in the folder and their chronological sequence should
match what one would get from a 'sort' of the file names. A naming sequence of
the readouts could be
    readout_001.fits ... readout_099.fits
An example collection of dark readouts is provided as ramp_001.tar.gz To use it,
extract the archive to ramp_001.
'''
ramp_folder = 'ramp_001'

# Creating the training dataset and doing the training
create_ramp(ramp_folder,scene)
extract_ribbon_sequences(ramp_folder)
generate_training_set(ramp_folder)
train_nn(ramp_folder,NN_name='demo')

# Testing the neural network
generate_NN_predictions(ramp_folder,NN_name='demo')
