"""

    This module contains utility functions for the Geko package.
    Not sure of everything that will go in here but it should be a place to put
    basic functions that are used in multiple places in the package.
	Written by A L Danhaive: ald66@cam.ac.uk
"""

# imports
from jax import image
import jax.numpy as jnp
from jax.scipy.stats import norm


from skimage.filters import threshold_otsu
from skimage.morphology import closing
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import dilation, disk
from skimage.color import label2rgb

from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
from photutils.background import Background2D, MedianBackground

from matplotlib import pyplot as plt

def oversample(image_low_res, factor, wave_factor, method='nearest'):
    image_high_res = image.resize(image_low_res, (int(
        image_low_res.shape[0]*factor), int(image_low_res.shape[1]*wave_factor)), method=method)
    image_high_res /= factor*wave_factor

    return image_high_res


def resample(grism_spectrum, factor, wave_factor):
    # blocks = grism_spectrum[int(factor/4):int(grism_spectrum.shape[0]-factor/4)].reshape((int(grism_spectrum.shape[0]/(factor)), factor, int(grism_spectrum.shape[1]/wave_factor), wave_factor))
    blocks = grism_spectrum.reshape(
        (int(grism_spectrum.shape[0]/(factor)), factor, int(grism_spectrum.shape[1]/wave_factor), wave_factor))
    grism_obs_res = jnp.sum(blocks, axis=(1, 3))
    return grism_obs_res


def oversample_errors(error_map, factor, wave_factor):
    # Repeat each element in the original array along both dimensions
    repeated_errors = jnp.kron(error_map, jnp.ones((factor, wave_factor)))

    # Divide each element by the respective oversampling factors
    oversampled_errors = repeated_errors / jnp.sqrt(factor*wave_factor)

    return oversampled_errors


def resample_errors(error_map, factor, wave_factor):

    blocks = error_map.reshape(
        (int(error_map.shape[0]/(factor)), factor, int(error_map.shape[1]/wave_factor), wave_factor))
    # resampled_errors = jnp.sqrt(jnp.sum(blocks**2, axis=(1,3)))
    resampled_errors = jnp.linalg.norm(blocks, axis=(1, 3))
    return resampled_errors


def scale_distribution(distribution, mu, sigma, max, min):
    """

        Rescale a uniform distribution to its target distribution, chosen based on input parameters

    """
    if mu == None:
        #if mu is None, then the distribution is uniform
        rescaled_distribution = scale_uniform(distribution, max, min)
    elif max == None:
        #if mu is not None but max is None, then the distribution is Gaussian
        rescaled_distribution = scale_gaussian(distribution, mu, sigma)
    else:
        #if mu and max are not None, then the distribution is truncated Gaussian
        rescaled_distribution = scale_truncated_gaussian(
            distribution, mu, sigma, max, min)
        
    return rescaled_distribution


def scale_uniform(distribution, max, min):
    """
        Rescale a distribution to a uniform distribution
    """
    rescaled_distribution = distribution * (max - min) + min
    return rescaled_distribution


def scale_gaussian(distribution, mu, sigma):
    """
        Rescale a distribution to a Gaussian distribution
    """
    rescaled_distribution = norm.ppf(distribution) * sigma + mu
    return rescaled_distribution

def scale_truncated_gaussian(distribution, mu, sigma, high, low):
    """
        Rescale a distribution to a truncated Gaussian distribution
    """
    rescaled_distribution = norm.ppf(norm.cdf(low) + distribution*(norm.cdf(high)-norm.cdf(low)))*sigma + mu
    return rescaled_distribution


def find_best_sample(inference_data, variable, mu, sigma, max, min, MLS):
    """
       Find the best sample from an array of variables and return it
       MLS is either none to take median sample or it contains the indices with the higest
       likelihood sample
    """
    #even if only one variable is passed, make sure it is still a list (with only one entry)
    variable = variable
    best_sample = []

    for i, var in enumerate(variable):

        #inference_data is modified even if it is not returned
        inference_data.posterior[var].data = scale_distribution(inference_data.posterior[var].data, mu[i], sigma[i], max[i], min[i])
        inference_data.prior[var].data = scale_distribution(inference_data.prior[var].data, mu[i], sigma[i], max[i], min[i])
        
        if MLS == None:
            # print('Taking posterior median sample')
            best_sample.append(jnp.array(inference_data.posterior[var].median(dim=["chain", "draw"])))
        else:
            # print('Taking the maximum likelihood sample')
            best_sample.append(jnp.array(inference_data.posterior[var].isel(chain=MLS[0], draw=MLS[1])))

    
    return best_sample


def make_mask(image, n = 1, threshold_sigma = 3):
    """
       Make n masks of the image.
    """

    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (20,20) , filter_size=(5, 5),bkg_estimator = bkg_estimator)
    threshold = threshold_sigma * bkg.background_rms
    segment_map = detect_sources(image, threshold, npixels=10)
    cat = SourceCatalog(image, segment_map)
    #select only the central source
    idx_source = jnp.argmin((cat.xcentroid-image.shape[0]//2)**2 + (cat.ycentroid-image.shape[0]//2)**2)

    label_source = idx_source + 1
    segment_map.keep_labels([label_source])


    plt.imshow(image*segment_map.data, origin='lower')
    plt.show()

    #separate two sources if n == 2
    if n == 2:
        segm_deblend = deblend_sources(image, segment_map, npixels=10, nlevels=32, contrast=0.001)
        plt.imshow(segm_deblend.data, origin='lower')
        plt.show()

        #if more than 2 labels, keep the central one and merge the others
        if segm_deblend.nlabels > 2:
            idx_source = jnp.argmin((cat.xcentroid-image.shape[0]//2)**2 + (cat.ycentroid-image.shape[0]//2)**2)
            label_source = idx_source + 1
            new_label = segm_deblend.labels[segm_deblend.labels != label_source][0]
            for label in segm_deblend.labels:
                if label != label_source:
                    segm_deblend.reassign_label(label, new_label)
        cat = SourceCatalog(image, segm_deblend)
        plt.imshow(segm_deblend.data, origin='lower')
        plt.plot(cat.xcentroid, cat.ycentroid, 'ro')
        plt.show()
    else: 
        segm_deblend = None
    
    return segment_map, segm_deblend


def compute_gal_props(image, n=1, threshold_sigma=3):
	'''
		compute PA of galaxy in the image using skimage measure regionprops

	'''

	seg_1comp, seg_2comp = make_mask(image, n, 1)
	if n == 1:   
		sources = SourceCatalog(image, seg_1comp)
	if n == 2:
		sources = SourceCatalog(image, seg_2comp)
	PA = []
	inc = []
	for i in range(len(sources)):
		PA.append(sources.orientation[i].value) #orientation is in degrees
		inc.append(jnp.arccos(1/sources.elongation[i].value)* (180/jnp.pi))
    
	print('PA : ', PA)
	print('inc : ', inc)
                
	return seg_1comp, seg_2comp, PA, inc


#things to add here
#sampling uniform
#creating x,y meshgrid (which takes 3 lines usually)
#computing Bayes factyor