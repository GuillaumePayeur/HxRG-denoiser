import numpy as np
from astropy.io import fits
import glob
from tqdm import tqdm
import os

def xtract_ribbon(high_snr_scene_file, folder_pp_readouts,  doplot = False,
                  outfolder = None):
    """
    This code tranforms a series of readouts from an H4RG IR array and the
    high SNR 'scene' that is seen at very low flux within that series of reads
    into 'ribbons', which are used to train machine-learning algorithm as
    described in Pailleur et al. 2022.

    One passes two strings, one with that provides the name of the high-SNR
    'scene' image, which is expected to be a 4096x4096 fits file. The second
    is the folder where individual readout files are stored. They are
    expected to be the only fits files in the folder and their chronological
    sequence should match what one would get from a 'sort' of the file names.
    A naming sequence of the readouts could be
        readout_001.fits ... readout_099.fits

    extracted ribbons will be in a folder with the same name as
    "folder_pp_readouts" but with a suffix "_ribbons" unless one sets the
    outfolder to a string.

    ramps_001/*.fits will have ribbons in ramps_001_ribbons/
    """

    scene = fits.getdata(high_snr_scene_file)

    # set the output folder
    if type(outfolder) != str:
        outfolder = folder_pp_readouts+'_ribbons/'
    # we check if the output folder exists. If not, we create it
    if not os.path.isdir(outfolder):
        print('We create {} to contain ribbons'.format(outfolder))
        os.mkdir(outfolder)

    # find files in ramp folder
    files = glob.glob(folder_pp_readouts+'/*.fits')
    files.sort() # sort to avoid the weird ordering of glob

    # create a cube that contains all readouts
    cube = np.zeros([4096, 4096, len(files)])
    for i in tqdm(range(len(files)),leave = False):
        cube[:,:,i] =  fits.getdata(files[i])

    # we extract every 128 pixels
    for x0 in tqdm(np.arange(0,127)):
        # We skip reference pixels in the training set
        for y0 in tqdm(np.arange(4,4091),leave=False):
            outname = outfolder+'/ribbon_'+str(x0)+'-'+str(y0)

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

            # we reject pixls for which there is a 10-sigma discrepancy between
            # slope and the running MAD
            amps = np.zeros(32)
            for i in range(32):
                # we assume that we have 100 reads
                fit =  np.polyfit(np.arange(len(files)),ribbon[i,:],1)
                amps[i] = fit[0]

            rms = np.nanmedian(np.abs(np.ravel(ribbon)[1:] -
                                      np.ravel(ribbon)[:-1]))

            mask = np.abs(ribbon) < 10*rms

            # save in a standardised output.
            # 1st extension is the ribbon
            # 2nd extension is the mask of valid pixels (<10 sigma)
            # 3rd extension is the input scence used for training
            hdu1 = fits.PrimaryHDU()
            h = fits.Header()
            h['SIMPLE'] = True
            hdu1.header = h
            hdu1.header['NEXTEND'] = 3
            hdu2 = fits.ImageHDU(ribbon)
            hdu2.header['EXTNAME'] = ('ribbon')

            hdu3 = fits.ImageHDU(mask*1.0)
            hdu3.header['EXTNAME'] = ('mask of valid pixels')

            hdu4 = fits.ImageHDU( scene[y,x])
            hdu4.header['EXTNAME'] = ('scene')

            new_hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4])
            new_hdul.writeto(outname +'.fits', clobber=True)

            if doplot:
                # display one sample ribbon and save to a file
                plt.imshow(ribbon, origin = 'lower')
                plt.xlabel('Nth readout')
                plt.ylabel('Nth amplifier')
                plt.savefig('ribbon2.png')
                plt.show()

                plt.plot(amps*99)
                plt.plot( scene[y,x])
                plt.show()

