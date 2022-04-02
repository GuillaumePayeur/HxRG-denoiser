from astropy.io import fits
import os
import numpy as np
################################################################################
# Program that creates fits files containing the data
files_to_write = np.arange(0,47).tolist()
################################################################################

# Creating an array containing the names of the images
# names = ['051','052','053','102','103','104']
names = ['911','913','916','917','919','920']

# Creating numpy arrays to hold the ribbons and labels
data = np.zeros((500, 98, 127, 40, 3))
labels = np.zeros((500, 32*127, 3))

# Filling the array with the ribbons and making sure there is no bad pixel
for k in files_to_write:
    print('starting npy file', k)
    os.chdir('C:\\Users\\Guill\\Documents\\NIRPSML\\three_cols\\splits')
    indexes_1 = np.load('splits_1_{}.npy'.format(k))
    indexes_2 = np.load('splits_2_{}.npy'.format(k))
    for m in range(500):
        image = indexes_1[m]
        index = indexes_2[m]
        os.chdir('C:\\Users\\Guill\\Documents\\NIRPS\\data_{}_fixedv2'.format(names[image]))
        for j in range(3):
            data[m,:,:,:,j] = np.load('data_{}_{}.npy'.format(names[image],index-1+j)).astype('float16')
            labels[m,:,j] = np.load('labels_{}_{}.npy'.format(names[image],index-1+j)).astype('float16')

    # Saving the data as numpy files
    os.chdir('D:\\dataset_v7')
    np.save('data_{}.npy'.format(k), data.astype('float16'))
    np.save('labels_{}.npy'.format(k), labels.astype('float16'))
