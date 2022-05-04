import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
import os

def extract_ribbon_sequences(ramp_folder, doplot = False):
    """
    This code tranforms a series of readouts from an H4RG IR array and the
    high SNR 'scene' that is seen at very low flux within that series of reads
    into 'ribbon sequences', which are used to train machine-learning algorithm
    as described in Payeur et al. 2022.

    One passes two strings, one with that provides the name of the high-SNR
    'scene' image, which is expected to be a 4096x4096 fits file. The second
    is the folder where individual dark readout files are stored. They are
    expected to be the only fits files in the folder and their chronological
    sequence should match what one would get from a 'sort' of the file names.
    A naming sequence of the readouts could be
        readout_001.fits ... readout_099.fits

    extracted ribbon sequences will be in a folder with the same name as
    "ramp_folder" but with a suffix "_ribbons".

    ramp_001/*.fits will have ribbon sequences in ramp_001_ribbons/
    """

    scene = fits.getdata('scene_'+ramp_folder+'.fits')

    # set the output folder
    outfolder = ramp_folder+'_ribbons/'
    # we check if the output folder exists. If not, we create it
    if not os.path.isdir(outfolder):
        print('We create {} to contain ribbons'.format(outfolder))
        os.mkdir(outfolder)

    # find files in ramp folder
    files = glob.glob('pp_'+ramp_folder+'/*.fits')
    files.sort() # sort to avoid the weird ordering of glob

    # create a cube that contains all readouts
    cube = np.zeros([4096, 4096, len(files)])
    for i in tqdm(range(len(files)),leave = False):
        cube[:,:,i] =  fits.getdata(files[i])

    # We skip reference pixels in the training set
    print('Creating ribbon sequences')
    for y0 in tqdm(np.arange(4,4092),leave=False):

        # Create arrays that holds the data from one row as a sequence of ribbons
        r = np.zeros((98, 128, 40))
        r_labels = np.zeros((32*128))

        # we extract every 128 pixels
        for x0 in range(128):
            x = x0+np.arange(32)*128
            y = y0+np.zeros(32, dtype = int)

            # contains the 32 amplifiers and 8 corresponding reference pixels
            ribbon = np.zeros([40,len(files)])

            ribbon[0:32,:] = cube[y,x,:]
            ribbon[32:36,:] = cube[y0,0:4,:] # left ref pixels
            ribbon[36:,:] = cube[y0,-4:,:] # right ref pixels

            if doplot: # just in case we asked for some plots
                plt.imshow(ribbon, origin = 'lower')
                plt.xlabel('Nth readout')
                plt.ylabel('Nth amplifier')
                plt.savefig('ribbon1.png')
                plt.show()

            # subtracting the mean of each ribbon as we are looking for the
            # slope only.
            for i in range(40):
                ribbon[i,:] -= np.nanmean(ribbon[i,:])

            # we reject pixels for which there is a 10-sigma discrepancy between
            # slope and the running MAD
            amps = np.zeros(32)
            for i in range(32):
                # we assume that we have 100 reads
                fit =  np.polyfit(np.arange(len(files)),ribbon[i,:],1)
                amps[i] = fit[0]

            rms = np.nanmedian(np.abs(np.ravel(ribbon)[1:] -
                                      np.ravel(ribbon)[:-1]))

            mask = (np.abs(ribbon) < 10*rms)*1.0

            if doplot:
                # display one sample ribbon and save to a file
                plt.imshow(ribbon, origin = 'lower')
                plt.xlabel('Nth readout')
                plt.ylabel('Nth amplifier')
                plt.savefig('ribbon.png')
                plt.show()

                plt.plot(amps*99)
                plt.plot(scene[y,x])
                plt.show()

            # Replacing bad pixels by neighboring pixels
            if np.min(mask) == 0:
                bad_indexes = []
                for i, column in enumerate(mask):
                    if np.min(column) == 0:
                        bad_indexes.append(i)

                for i in bad_indexes:
                    if i < 32:
                        for _ in range(100):
                            n = np.random.randint(0, 32)
                            if n not in bad_indexes:
                                ribbon[i] = ribbon[n]
                                break
                    else:
                        for _ in range(100):
                            n = np.random.randint(32, 40)
                            if n not in bad_indexes:
                                ribbon[i] = ribbon[n]
                                break
            # ordering ribbons into a ribbon sequence
            r[:,x0,:] = np.transpose(ribbon, (1, 0))[0:98,:]
            r_labels[(x0)*32:(x0+1)*32] = scene[y,x]

        # Saving the ribbon sequences as numpy files
        np.save('{}/r_{}.npy'.format(outfolder,y0), r.astype('float16'))
        np.save('{}/r_labels_{}.npy'.format(outfolder,y0), r_labels.astype('float16'))
