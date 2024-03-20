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
    variable = [variable]
    rescaled_posterior = []
    rescaled_prior = []
    best_sample = []

    for i, var in enumerate(variable):
        rescaled_posterior.append(scale_distribution(
            inference_data.posterior[var].data, mu[i], sigma[i], max[i], min[i]))
        rescaled_prior.append(scale_distribution(
            inference_data.prior[var].data, mu[i], sigma[i], max[i], min[i]))
        
        if MLS == None:
            print('Taking posterior median sample')
            best_sample.append(jnp.array(inference_data.posterior[var].median(dim=["chain", "draw"])))
        else:
            print('Taking the maximum likelihood sample')
            best_sample.append(jnp.array(inference_data.posterior[var].isel(chain=MLS[0], draw=MLS[1])))
    
    return rescaled_posterior, rescaled_prior, best_sample


