![test](https://github.com/GuillaumePayeur/HxRG-denoiser/blob/main/HxRG_denoiser.PNG?raw=true)
## Abstract
We present a new procedure rooted in deep learning to construct science images from data cubes collected by astronomical instruments using HxRG detectors in low-flux regimes. It improves on the drawbacks of the conventional algorithms to construct 2D images from multiple readouts by using the readout scheme of the detectors to reduce the impact of correlated readout noise. We train a convolutional recurrent neural network on simulated astrophysical scenes added to laboratory darks to estimate the flux on each pixel of science images. This method achieves a reduction of the noise on constructed science images when compared to standard flux-measurement schemes (correlated double sampling, up-the-ramp sampling), which results in a reduction of the error on the spectrum extracted from these science images. Over simulated data cubes created in a low signal-to-noise ratio regime where this method could have the largest impact, we find that the error on our constructed science images falls faster than a 1/sqrt(N) decay, and that the spectrum extracted from the images has, averaged over a test set of three images, a standard error reduced by a factor of 1.85 in comparison to the standard up-the-ramp pixel sampling scheme.
## Article
Checkout our [paper](https://arxiv.org/pdf/2205.01866.pdf) for an in depth discussion of our methods and results
## Getting Started
This repository contains code the code needed to train a denoising neural network as described in our [paper](https://arxiv.org/pdf/tba.pdf). We provide sample data collected by the instrument NIRPS to test the code with. It can be downloaded [here](https://www.astro.umontreal.ca/~artigau/ml/). This data may be replaced by your own, and the code may be modified to suit your needs. 
## Dependencies
[Tensorflow](https://github.com/tensorflow/tensorflow "Tensorflow on GitHub")  
[Keras](https://github.com/keras-team/keras "Keras on GitHub")  
[Astropy](https://github.com/astropy/astropy "Astropy on GitHub")  
[tqdm](https://github.com/tqdm/tqdm "tqdm on GitHub")
## Citing This Work
```
biblitex code tba
```
