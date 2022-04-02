from astropy.io import fits
import os
import numpy as np
################################################################################
# Program that creates fits files containing the data
################################################################################
# Creating an array containing the names of the images
names = ['050']

# Creating numpy arrays to hold the ribbons and labels
data = np.zeros((98, 127, 40))
labels = np.zeros((32*127))

# Filling the array with the ribbons and making sure there is no bad pixel
for name in names:
    print('starting file', name)
    for m in range(4,4091):
        os.chdir('C:\\Users\\Guill\\Downloads\\ribbons_{}'.format(name))
        print('starting series', m)

        for i in range(0,127):
            fits_file = fits.open('ribbon_{}-{}.fits'.format(i,m))

            fits_1 = (fits_file[1].data).astype('float16')
            fits_2 = (fits_file[2].data).astype('float16')
            fits_3 = (fits_file[3].data).astype('float16')
            if np.min(fits_file[2].data) == 0:

                bad_indexes = []
                for index_2, column in enumerate(fits_2):
                    if np.min(column) == 0:
                        bad_indexes.append(index_2)

                for index_2 in bad_indexes:
                    if index_2 < 32:
                        while True:
                            n = np.random.randint(0, 32)
                            if n not in bad_indexes:
                                fits_1[index_2] = fits_1[n]
                                fits_3[index_2] = fits_3[n]
                                break
                    else:
                        while True:
                            n = np.random.randint(32, 40)
                            if n not in bad_indexes:
                                fits_1[index_2] = fits_1[n]
                                break

            data[:,i,:] = np.transpose(fits_1, (1, 0))[0:98,:]
            labels[(i)*32:(i+1)*32] = fits_3

        # Saving the data as numpy files
        os.chdir('C:\\Users\\Guill\\OneDrive\\Documents\\NIRPS\\val_data')
        np.save('data_{}_{}.npy'.format(name,m), data.astype('float16'))
        np.save('labels_{}_{}.npy'.format(name,m), labels.astype('float16'))
