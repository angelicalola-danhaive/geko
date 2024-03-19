# imports

#importing my own modules
import model_jax as model
import preprocess as pre
import plotting

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import random, vmap
# from jax.scipy.special import logsumexp
from jax.scipy.stats import norm,mode, expon
import matplotlib.pyplot as plt
# import pandas as pd

import numpyro
from numpyro import handlers
# from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam, LocScaleReparam
# from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_median,init_to_sample, init_to_value

import scipy
from scipy.constants import c, pi
import statistics as st
import math

# useful for plotting
from numpyro.infer import Predictive
from jax import random
from jax import image
import arviz as az
import corner

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, binary_closing
from skimage.color import label2rgb

from astropy.convolution import Box2DKernel, convolve
from astropy.nddata import block_reduce
from skimage.morphology import binary_dilation, dilation, disk

from astropy.io import fits
from astropy.table import Table
from astropy.modeling.models import Sersic2D

import galsim

import argparse
import yaml



jax.config.update('jax_enable_x64', True)
numpyro.set_host_device_count(2)
# numpyro.set_platform('gpu')

# np.set_printoptions(precision=15, floatmode='maxprec')
# jnp.set_printoptions(precision=15, floatmode='maxprec')
# import figure_setup as setup

XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

# plotting settings

# setup.configure_plots()


class Fit_Numpyro():
	def __init__(self, obs_map=None, obs_error=None, obs_map_low = None, obs_error_low = None, grism_object=None, factor=1, wave_factor=1, fitting_object=None, flux_prior=None, x0 = None, y0 = None, cov_matrix = None, fluxes_errors = None, snr_flux = None, error_scaling_low = None, alpha = None, x0_vel = None, y0_vel = None, wavelength = None):
		""" Class to fit model to data

						Parameters
						----------
						obs_map 
								2D array of the EL map from grism
						sigma
								2D array with the error/noise associated with each pixel
						grism_object
								object of the Grism class, initialized for the galaxy we want to model
						fitting_object
								initialize the class with the results of a fit
								has to be az.from_netcdf 

						flux_prior 
								2D array containing fluxes to be used as mean of prior distribution (from medium band)

						Attributes
						----------
		"""
		self.obs_map = obs_map
		self.obs_error = obs_error

		self.obs_map_low = obs_map_low
		self.obs_error_low = obs_error_low

		self.grism_object = grism_object

		self.factor = factor
		self.wave_factor = wave_factor

		self.data = fitting_object

		self.flux_prior = flux_prior

		self.x0 = x0
		self.y0 = y0

		self.x0_vel = x0_vel
		self.y0_vel = y0_vel

		self.cov_matrix = cov_matrix

		self.fluxes_errors = fluxes_errors
		self.snr_flux = snr_flux
		self.error_scaling_low = error_scaling_low
		self.alpha = alpha

		#this is the wavelength taken from config file that will be the center of our wave prior
		self.wavelength = wavelength

		# define the grid on which functions are
		# x = jnp.linspace(0 - grism_object.jcenter,
		# 				 self.grism_object.direct.shape[1]-1 - grism_object.jcenter, self.grism_object.direct.shape[1])
		# y = jnp.linspace(0 - grism_object.icenter,
		# 				 self.grism_object.direct.shape[0]-1 - grism_object.icenter, self.grism_object.direct.shape[0])
		# x = jnp.linspace(0 - grism_object.jcenter/self.factor,
		# 				 self.grism_object.direct.shape[1]/self.factor-1 - grism_object.jcenter/self.factor, self.grism_object.direct.shape[1])
		# y = jnp.linspace(0 - grism_object.icenter/self.factor,
		# 				 self.grism_object.direct.shape[0]/self.factor-1 - grism_object.icenter/self.factor, self.grism_object.direct.shape[0])
		# self.x, self.y = jnp.meshgrid(x, y)

		if (self.x0_vel != None) and (self.x0_vel is not isinstance(x0,np.ndarray)):
			print('Setting the velocity centroid as defined in the config file')
			x = jnp.linspace(0 - grism_object.jcenter,
							self.grism_object.direct.shape[1]-1 - self.x0_vel, self.grism_object.direct.shape[1]*self.factor)
			y = jnp.linspace(0 - grism_object.icenter,
							self.grism_object.direct.shape[0]-1 - self.y0_vel, self.grism_object.direct.shape[0]*self.factor)
			self.x, self.y = jnp.meshgrid(x, y)

			x = jnp.linspace(0 - grism_object.jcenter,
							self.grism_object.direct.shape[1]-1 - self.x0_vel, self.grism_object.direct.shape[1]*self.factor*10)
			y = jnp.linspace(0 - grism_object.icenter,
							self.grism_object.direct.shape[0]-1 - self.y0_vel, self.grism_object.direct.shape[0]*self.factor*10)
			self.x_high, self.y_high = jnp.meshgrid(x, y)
		else:
			x = jnp.linspace(0 - grism_object.jcenter,
							self.grism_object.direct.shape[1]-1 - grism_object.jcenter, self.grism_object.direct.shape[1]*self.factor)
			y = jnp.linspace(0 - grism_object.icenter,
							self.grism_object.direct.shape[0]-1 - grism_object.icenter, self.grism_object.direct.shape[0]*self.factor)
			self.x, self.y = jnp.meshgrid(x, y)

			x = jnp.linspace(0 - grism_object.jcenter,
							self.grism_object.direct.shape[1]-1 - grism_object.jcenter, self.grism_object.direct.shape[1]*self.factor*10)
			y = jnp.linspace(0 - grism_object.icenter,
							self.grism_object.direct.shape[0]-1 - grism_object.icenter, self.grism_object.direct.shape[0]*self.factor*10)
			self.x_high, self.y_high = jnp.meshgrid(x, y)
		#have to fix this (I can probably delete it?)
		if x0 is not None and isinstance(x0,np.ndarray) and x0.shape == (2,):
			self.x_2 = jnp.linspace(0 - x0[1],self.grism_object.direct.shape[1]-1 - x0[1], self.grism_object.direct.shape[1]*self.factor)
			self.y_2 = jnp.linspace(0 - y0[1], self.grism_object.direct.shape[0]-1 - y0[1], self.grism_object.direct.shape[0]*self.factor)
			self.x_2, self.y_2 = jnp.meshgrid(self.x_2, self.y_2)
		# print(self.x, self.y)
		# print(grism_object.jcenter, grism_object.icenter)
	def set_bounds(self, model,flux_prior, flux_bounds, flux_type, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, obs_map_bounds, y_factor, clump = None, mask = None, delta_V_bounds = [0,0], sigma0_mean = None, sigma0_disp = None, PA_normal = None, clump_v_prior = None, clump_sigma_prior = None, clump_flux_prior = None):
		self.flux_prior = flux_prior
		# these first two are in the form (scale, high)
		self.flux_bounds = flux_bounds
		self.flux_type = flux_type
		self.PA_bounds = PA_bounds
		self.PA_normal = PA_normal
		# these are in the form (low, high)
		self.i_bounds = i_bounds
		self.Va_bounds = Va_bounds
		self.r_t_bounds = r_t_bounds
		self.sigma0_mean = sigma0_mean
		self.sigma0_disp = sigma0_disp
		self.sigma0_bounds = sigma0_bounds
		self.obs_map_bounds = obs_map_bounds
		self.clump = clump
		self.mask = mask
		self.delta_V_bounds = delta_V_bounds

		self.y_factor = y_factor

		self.clump_v_prior = clump_v_prior
		self.clump_sigma_prior = clump_sigma_prior
		self.clump_flux_prior = clump_flux_prior

	    # compute the parameters of the sampling distributionS

		# plt.imshow(self.flux_prior, origin='lower')
		# plt.title('flux prior')
		# plt.show()
		# fluxes
		if model == 'two_disc_model':
			self.mu = jnp.array(self.flux_prior)
			self.sigma = self.flux_bounds[0]*jnp.array(self.flux_prior)
			self.high =(self.flux_bounds[1]* jnp.array(self.flux_prior)+ self.mu - self.mu)/self.sigma
			self.low =(jnp.zeros(self.flux_prior.shape)-self.mu)/self.sigma
			self.mu_1 = jnp.where(self.mask[0] == 0, 0.0, self.mu)
			self.sigma_1 = jnp.where(self.mask[0] == 0, 0.000001, self.sigma)
			self.high_1 = jnp.where(self.mask[0] == 0, 0.000002, self.high)
			self.low_1 = jnp.where(self.mask[0] == 0, -0.000002, self.low)
			self.mu_2 = jnp.where(self.mask[1] == 0, 0.0, self.mu)
			self.sigma_2 = jnp.where(self.mask[1] == 0, 0.000001, self.sigma)
			self.high_2 = jnp.where(self.mask[1] == 0, 0.000002, self.high)
			self.low_2 = jnp.where(self.mask[1] == 0, -0.000002, self.low)
			self.mu_PA_1 = self.PA_bounds[0][0]
			self.sigma_PA_1 = self.PA_bounds[0][1]*90
			self.mu_PA_2 = self.PA_bounds[1][0]
			self.sigma_PA_2 = self.PA_bounds[1][1]*90
		elif model == 'unified_two_discs_model':
			self.mu = jnp.array(self.flux_prior)
			self.sigma = self.flux_bounds[0]*jnp.array(self.flux_prior)
			self.high =(self.flux_bounds[1]* jnp.array(self.flux_prior)+ self.mu - self.mu)/self.sigma
			self.low =(jnp.zeros(self.flux_prior.shape)-self.mu)/self.sigma
			# correct for the mask = 0 pixels
			if self.mask is not None:
				self.mu = jnp.where(self.mask == 0, 0.0, self.mu)
				self.sigma = jnp.where(self.mask == 0, 0.000001, self.sigma)
				self.high = jnp.where(self.mask == 0, 0.000002, self.high)
				self.low = jnp.where(self.mask == 0, -0.000002, self.low)
			if self.mask.shape[0] != 2: #because if 2 distinct masks then mask = [mask1, mask2]
				print('Setting both PAs to the same value')
				self.mu_PA_1 = self.PA_bounds[0]
				self.sigma_PA_1 = self.PA_bounds[1]*90
				self.mu_PA_2 = self.PA_bounds[0]
				self.sigma_PA_2 = self.PA_bounds[1]*90
			else:
				self.mu_PA_1 = self.PA_bounds[0][0]
				self.sigma_PA_1 = self.PA_bounds[0][1]*90
				self.mu_PA_2 = self.PA_bounds[1][0]
				self.sigma_PA_2 = self.PA_bounds[1][1]*90
		else:
			if flux_type == 'lin':
				#if the fluxes are negative, set their prior to zero
				self.mu = jnp.maximum(jnp.zeros(self.flux_prior.shape), jnp.array(self.flux_prior))
				self.sigma = self.flux_bounds[0]*jnp.array(self.flux_prior)
				self.high =(self.flux_bounds[1]* jnp.array(self.flux_prior)+ self.mu - self.mu)/self.sigma
				self.low =(jnp.zeros(self.flux_prior.shape)-self.mu)/self.sigma
				# correct for the mask = 0 pixels
				if self.mask is not None:
					self.mask = jnp.array(self.mask)
					# self.mu = jnp.where(self.mask == 0, 0.0, self.mu)
					# self.sigma = jnp.where(self.mask == 0, 0.000001, self.sigma)
					# self.high = jnp.where(self.mask == 0, 0.000002, self.high)
					# self.low = jnp.where(self.mask == 0, -0.000002, self.low)

					#only fitting for pixels in the mask
					self.mask_shape = len(jnp.where(self.mask == 1)[0])
					self.masked_indices = jnp.where(self.mask == 1)
					self.mu = self.mu[jnp.where(self.mask == 1)]
					self.sigma = self.sigma[jnp.where(self.mask == 1)]
					self.high = self.high[jnp.where(self.mask == 1)]
					self.low = self.low[jnp.where(self.mask == 1)]
			elif flux_type == 'log':
				self.mu = jnp.log10(jnp.array(self.flux_prior))
				self.sigma = self.flux_bounds[1]
				self.high = ((self.mu + self.flux_bounds[0]) - self.mu)/self.sigma
				self.low = ((self.mu - self.flux_bounds[0])-self.mu)/self.sigma

			# Postion Angle
			if self.PA_bounds[1] != 'const':
				self.mu_PA = self.PA_bounds[0]
				self.sigma_PA = self.PA_bounds[1]*90
				# self.high_PA = (self.PA_bounds[0]+45 - self.mu_PA)/self.sigma_PA
			# self.low_PA = (self.PA_bounds[0]-45 - self.mu_PA)/self.sigma_PA

		# Velocity Dispersion
		if self.sigma0_bounds != 'const':
			if self.sigma0_mean is not None:
				print('manually setting mean to: ', self.sigma0_mean)
				self.mu_sigma0 = self.sigma0_mean
			else:
				self.mu_sigma0 = 100
			if self.sigma0_disp is not None:
				print('manually setting dispersion to: ', self.sigma0_disp)
				self.sigma_sigma0 = self.sigma0_disp
			else:
				self.sigma_sigma0 = 10
			self.high_sigma0 = (self.sigma0_bounds[1] - self.mu_sigma0)/self.sigma_sigma0
			self.low_sigma0 = (self.sigma0_bounds[0]-self.mu_sigma0)/self.sigma_sigma0

		# Two component model - clump velocity and fluxes
		if clump is not None:
			self.mu_v = jnp.zeros(self.flux_prior.shape)
			self.sigma_v = self.clump*self.clump_v_prior+ 10
			self.high_v = ((self.delta_V_bounds[0]) - self.mu_v)/self.sigma_v
			self.low_v = ((self.delta_V_bounds[1])-self.mu_v)/self.sigma_v
			# self.clump_exp_width = 0.5*self.clump*self.flux_prior
			self.clump_exp_width = self.clump_flux_prior*self.clump*self.flux_prior
			self.sigma_sigma0_clump = self.clump_sigma_prior[1]
			self.mu_sigma0_clump = self.clump_sigma_prior[0]

		# plt.imshow(self.flux_prior, origin='lower')
		# plt.title('flux prior')
		# plt.show()

	def __str__(self):
		# print all of the attributes of the class
		return 'Fit_Numpyro Class: \n' + ' - factor = ' + str(self.factor)+ '\n - wave_factor = '+ str(self.wave_factor) + '\n grism object = ' + str(grism_object)
	
	def full_parametric_renormalized(self):
		"""

				Galaxy model where the velocities are modelled following an exponential disk (PA, i, Va, r_t) and the dispersion is assumed to be constant
				The variables are renormalized after the sampling to get a better posterior geometry!

		"""
		# fluxes = numpyro.sample('fluxes', dist.Uniform(), sample_shape=self.flux_prior.shape)
		# # manually computing the ppf for a truncated normal distribution
		# fluxes = norm.ppf(norm.cdf(self.low) + fluxes*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu

		#only fit for fluxes in the region of the mask
		#with this method to whole changing the mus for pixels outside the mask is not necessary (in the initializing section)
		fluxes_sample = numpyro.sample('fluxes', dist.Uniform(), sample_shape=(int(self.mask_shape),))

		# manually computing the ppf for a truncated normal distribution
		fluxes_sample = norm.ppf(norm.cdf(self.low) + fluxes_sample*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
		
		fluxes = jnp.zeros_like(self.flux_prior)
		fluxes = fluxes.at[self.masked_indices].set(fluxes_sample)

		# fluxes = self.flux_prior

		# using Numpyro's transform for the normal distribution

		# reparam_config = {"fluxes": TransformReparam()}
		# with numpyro.handlers.reparam(config=reparam_config):
		# 	# in order to use TransformReparam we have to express the prior
		# 	# over betas as a TransformedDistribution
		# 	fluxes = numpyro.sample("fluxes",dist.TransformedDistribution(dist.TruncatedNormal(jnp.zeros(self.flux_prior.shape), jnp.ones_like(self.flux_prior), low = self.low, high = self.high),AffineTransform(self.mu, self.sigma),),)

		if self.flux_type == 'log':
			fluxes = jnp.power(10, fluxes)

		fluxes = oversample(fluxes, self.factor, self.factor)

		if self.PA_bounds[1] == 'const':
			Pa = 30
		else:
			Pa = numpyro.sample('PA', dist.Uniform())
			#sample the mu_PA + 0 or 180 (orientation of velocity field)
			rotation = numpyro.sample('rotation', dist.Uniform())

			#simulate a bernouilli discrete distribution
			PA_morph = self.mu_PA + round(rotation)*180

			Pa = norm.ppf(Pa)*self.sigma_PA + PA_morph
			# Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA
		if self.i_bounds == 'const':
			i = 60
		else:
			i = numpyro.sample('i', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		if self.Va_bounds == 'const':
			Va  = 600
		else:
			Va = numpyro.sample('Va', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		if self.r_t_bounds == 'const':
			r_t = 2
		else:
			r_t = numpyro.sample('r_t', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

		if self.sigma0_bounds == 'const':
			sigma0 = 100
		else:
			sigma0 = numpyro.sample('sigma0', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			# sigma0 = norm.ppf(sigma0)*self.sigma_sigma0 + self.mu_sigma0
		# sigma0 = norm.ppf(  norm.cdf(self.low_sigma0) + fluxes*(norm.cdf(self.high_sigma0)-norm.cdf(self.low_sigma0)) )*self.sigma_sigma0 + self.mu_sigma0
		# sigma0 = 100

		#sampling the velocity centroids
		# x0 = numpyro.sample('x0', dist.Uniform())
		# x0 = norm.ppf(x0)*(2) + self.x0
		# y0 = numpyro.sample('y0', dist.Uniform())
		# y0 = norm.ppf(y0)*(2) + self.y0

		# x = jnp.linspace(0 - x0,self.grism_object.direct.shape[1]-1 - x0, self.grism_object.direct.shape[1]*self.factor*10)
		# y = jnp.linspace(0 - y0,self.grism_object.direct.shape[0]-1 - y0, self.grism_object.direct.shape[0]*self.factor*10)
		# X,Y = jnp.meshgrid(x, y)

		# velocities = jnp.array(v(X,Y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
		velocities = jnp.array(v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
		# print(velocities[15])

		# velocities = Va*jnp.ones_like(self.x)
		# velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')
		
		dispersions = sigma0*jnp.ones_like(velocities)

		#sample a shift in the dispersion wavelength
		corrected_wavelength = numpyro.sample('wavelength', dist.Uniform())
		corrected_wavelength = norm.ppf(corrected_wavelength)*0.001 + self.wavelength

		self.grism_object.set_wavelength(corrected_wavelength)

		self.model_map = self.grism_object.disperse(fluxes, velocities, dispersions)

		self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)
		# self.model_map = resample(self.model_map, self.factor, self.wave_factor)

		self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0,1))*9 + 1
		# self.error_scaling = 1
		numpyro.sample('obs', dist.Normal(self.model_map,self.error_scaling*self.obs_error), obs=self.obs_map)
		# numpyro.sample('obs', dist.Normal(self.model_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:],
		# 		     self.error_scaling*self.obs_error[self.obs_map_bounds[0]:self.obs_map_bounds[1],:]), 
		# 			 obs=self.obs_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:])
		
		# compare the model fluxes with the flux prior as well
		# numpyro.sample('fluxes_prior', dist.Normal(fluxes, 0.1*fluxes), obs=oversample(self.flux_prior, self.factor))
		# numpyro.sample('obs', dist.Normal(self.model_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:],
		# 		     self.obs_error[self.obs_map_bounds[0]:self.obs_map_bounds[1],:]), 
		# 			 obs=self.obs_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:])

	
	def two_disc_model(self):
		"""
				2 discs with a velocity offset between them
		"""

		fluxes_1 = numpyro.sample('fluxes_1', dist.Uniform(), sample_shape=self.flux_prior.shape)
		# manually computing the ppf for a truncated normal distribution
		fluxes_1 = norm.ppf(norm.cdf(self.low_1) + fluxes_1*(norm.cdf(self.high_1)-norm.cdf(self.low_1)))*self.sigma_1 + self.mu_1

		fluxes_2 = numpyro.sample('fluxes_2', dist.Uniform(), sample_shape=self.flux_prior.shape)
		# manually computing the ppf for a truncated normal distribution
		fluxes_2 = norm.ppf(norm.cdf(self.low_2) + fluxes_2*(norm.cdf(self.high_2)-norm.cdf(self.low_2)))*self.sigma_2 + self.mu_2

		fluxes_1 = oversample(fluxes_1, self.factor, self.factor)
		fluxes_2 = oversample(fluxes_2, self.factor, self.factor)

		Pa_1 = numpyro.sample('PA_1', dist.Uniform())
		Pa_1 = norm.ppf(Pa_1)*self.sigma_PA_1 + self.mu_PA_2

		Pa_2 = numpyro.sample('PA_2', dist.Uniform())
		Pa_2 = norm.ppf(Pa_2)*self.sigma_PA_2 + self.mu_PA_2

		i_1 = numpyro.sample('i_1', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		i_2 = numpyro.sample('i_2', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

		Va_1 = numpyro.sample('Va_1', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		Va_2 = numpyro.sample('Va_2', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

		r_t_1 = numpyro.sample('r_t_1', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
		r_t_2 = numpyro.sample('r_t_2', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

		sigma0_1 = numpyro.sample('sigma0_1', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# sigma0_1 = norm.ppf(sigma0_1)*self.sigma_sigma0 + self.mu_sigma0

		sigma0_2 = numpyro.sample('sigma0_2', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# sigma0_2 = norm.ppf(sigma0_2)*self.sigma_sigma0 + self.mu_sigma0

		velocities_1 = jnp.array(v(self.x, self.y, jnp.radians(Pa_1),jnp.radians(i_1), Va_1, r_t_1))
		velocities_2 = jnp.array(v(self.x_2, self.y_2, jnp.radians(Pa_2),jnp.radians(i_2), Va_2, r_t_2))

		v_offset = numpyro.sample('v_offset', dist.Uniform())*1000 - 500

		velocities_2 = velocities_2 + v_offset

		# velocities = velocities_1 + velocities_2
		
		dispersions_1 = sigma0_1*jnp.ones_like(velocities_1)
		dispersions_2 = sigma0_2*jnp.ones_like(velocities_2)

		self.model_map = self.grism_object.disperse(fluxes_1, velocities_1, dispersions_1) + self.grism_object.disperse(fluxes_2, velocities_2, dispersions_2)

		self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)
		# self.model_map = resample(self.model_map, self.factor, self.wave_factor)

		self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0,1))*9 + 1

		numpyro.sample('obs', dist.Normal(self.model_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:],
				     self.error_scaling*self.obs_error[self.obs_map_bounds[0]:self.obs_map_bounds[1],:]), 
					 obs=self.obs_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:])

	def unified_two_discs_model(self):
		"""
				2 component model that unifies the two_component_model and the two_disc_model to be able to 
				fit 2 discs with arbitrary (sampled) kinematic centroids and velocity offset
		"""
		
		fluxes = numpyro.sample('fluxes', dist.Uniform(), sample_shape=self.flux_prior.shape)
		# manually computing the ppf for a truncated normal distribution
		fluxes = norm.ppf(norm.cdf(self.low) + fluxes*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
		# sample the fraction of flux in the first component in each pixel
		f1 = numpyro.sample('f1', dist.Uniform(), sample_shape=fluxes.shape)

		fluxes = oversample(fluxes, self.factor, self.factor)
		f1 = oversample(f1, self.factor, self.factor)*self.factor**2

		# define the two flux components
		fluxes_1 = f1*fluxes
		fluxes_2 = (1-f1)*fluxes

		# sample the centroids of the two components, guassian dist centered on input x0_vel, y0_vel, arbitrary width of 2
		x0_1 = numpyro.sample('x0_1', dist.Uniform())
		x0_1 = norm.ppf(x0_1)*(2) + self.x0_vel[0]
		y0_1 = numpyro.sample('y0_1', dist.Uniform())
		y0_1 = norm.ppf(y0_1)*(2) + self.y0_vel[0]
		
		x0_2 = numpyro.sample('x0_2', dist.Uniform())
		x0_2 = norm.ppf(x0_2)*(2) + self.x0_vel[1]
		y0_2 = numpyro.sample('y0_2', dist.Uniform())
		y0_2 = norm.ppf(y0_2)*(2) + self.y0_vel[1]

		# make the grids for both components
		x_1 = jnp.linspace(0 - x0_1,self.grism_object.direct.shape[1]-1 - x0_1, self.grism_object.direct.shape[1]*self.factor*10)
		y_1 = jnp.linspace(0 - y0_1,self.grism_object.direct.shape[0]-1 - y0_1, self.grism_object.direct.shape[0]*self.factor*10)
		x_1, y_1 = jnp.meshgrid(x_1, y_1)

		x_2 = jnp.linspace(0 - x0_2,self.grism_object.direct.shape[1]-1 - x0_2, self.grism_object.direct.shape[1]*self.factor*10)
		y_2 = jnp.linspace(0 - y0_2,self.grism_object.direct.shape[0]-1 - y0_2, self.grism_object.direct.shape[0]*self.factor*10)
		x_2, y_2 = jnp.meshgrid(x_2, y_2)

		# sample the kinematic parameters of the two components
		Pa_1 = numpyro.sample('PA_1', dist.Uniform())
		Pa_1 = norm.ppf(Pa_1)*self.sigma_PA_1 + self.mu_PA_2

		Pa_2 = numpyro.sample('PA_2', dist.Uniform())
		Pa_2 = norm.ppf(Pa_2)*self.sigma_PA_2 + self.mu_PA_2

		i_1 = numpyro.sample('i_1', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		i_2 = numpyro.sample('i_2', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

		Va_1 = numpyro.sample('Va_1', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		Va_2 = numpyro.sample('Va_2', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

		r_t_1 = numpyro.sample('r_t_1', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
		r_t_2 = numpyro.sample('r_t_2', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

		sigma0_1 = numpyro.sample('sigma0_1', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# sigma0_1 = norm.ppf(sigma0_1)*self.sigma_sigma0 + self.mu_sigma0

		sigma0_2 = numpyro.sample('sigma0_2', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# sigma0_2 = norm.ppf(sigma0_2)*self.sigma_sigma0 + self.mu_sigma0

		# sample the velocity offset between the two components
		v_offset = numpyro.sample('v_offset', dist.Uniform())*1000 - 500

		# compute the velocities and dispersions of the two components
		velocities_1 = jnp.array(v(x_1, y_1, jnp.radians(Pa_1),jnp.radians(i_1), Va_1, r_t_1))
		velocities_2 = jnp.array(v(x_2, y_2, jnp.radians(Pa_2),jnp.radians(i_2), Va_2, r_t_2))

		#resample the velocities by 10
		velocities_1 = image.resize(velocities_1, (int(velocities_1.shape[0]/10), int(velocities_1.shape[1]/10)), method='nearest')
		velocities_2 = image.resize(velocities_2, (int(velocities_2.shape[0]/10), int(velocities_2.shape[1]/10)), method='nearest')

		velocities_2 = velocities_2 + v_offset

		dispersions_1 = sigma0_1*jnp.ones_like(velocities_1)
		dispersions_2 = sigma0_2*jnp.ones_like(velocities_2)

		# compute the model map
		self.model_map = self.grism_object.disperse(fluxes_1, velocities_1, dispersions_1) + self.grism_object.disperse(fluxes_2, velocities_2, dispersions_2)

		# self.model_map = self.grism_object.disperse(fluxes, f1*velocities_1 + (1-f1)*velocities_2, f1*dispersions_1 + (1-f1)*dispersions_2)

		self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)

		# sample the error scaling
		self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0,1))*9 + 1

		# sample the observed map
		numpyro.sample('obs', dist.Normal(self.model_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:],
				     self.error_scaling*self.obs_error[self.obs_map_bounds[0]:self.obs_map_bounds[1],:]),
					 obs=self.obs_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:])
		
# # MODEL FOR REAL DATA => ERICA's
# 	def one_comp_MLE(self):

# 		if self.PA_bounds[1] == 'const':
# 			Pa = 30
# 		else:
# 			Pa = numpyro.sample('PA', dist.Uniform())
# 			Pa = norm.ppf(Pa)*self.sigma_PA + self.mu_PA
# 			# Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA
# 		if self.i_bounds == 'const':
# 			i = 60
# 		else:
# 			i = numpyro.sample('i', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
# 		if self.Va_bounds == 'const':
# 			Va  = 600
# 		else:
# 			Va = numpyro.sample('Va', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
# 		if self.r_t_bounds == 'const':
# 			r_t = 2
# 		else:
# 			r_t = numpyro.sample('r_t', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

# 		if self.sigma0_disp is not None:
# 			sigma0 = numpyro.sample('sigma0', dist.Uniform())
# 			sigma0 = norm.ppf(sigma0)*self.sigma_sigma0 + self.mu_sigma0
# 		else:
# 			sigma0 = numpyro.sample('sigma0', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
# 			# sigma0 = norm.ppf(sigma0)*self.sigma_sigma0 + self.mu_sigma0
# 		# sigma0 = norm.ppf(  norm.cdf(self.low_sigma0) + fluxes*(norm.cdf(self.high_sigma0)-norm.cdf(self.low_sigma0)) )*self.sigma_sigma0 + self.mu_sigma0
# 		# sigma0 = 100

# 		x = jnp.linspace(0 - self.grism_object.jcenter, self.flux_prior.shape[1]-1 -self.grism_object.jcenter, self.flux_prior.shape[1]*10)
# 		y = jnp.linspace(0 - self.grism_object.icenter, self.flux_prior.shape[0]-1 -self.grism_object.icenter, self.flux_prior.shape[0]*10)
# 		X_high, Y_high = jnp.meshgrid(x, y)

# 		velocities = jnp.array(v(X_high, Y_high, jnp.radians(Pa),jnp.radians(i), Va, r_t))
# 		# velocities = Va*jnp.ones_like(self.x)
# 		velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='bicubic')
		
# 		dispersions = sigma0*jnp.ones_like(velocities)

# 		# velocities = jnp.where(self.mask == 0, 0, velocities)
# 		# dispersions = jnp.where(self.mask == 0, 1e-6, dispersions)

# 		alpha = numpyro.sample('alpha', dist.Uniform(0,1))*(1-0.00001) +0.00001
# 		#sample alpha in the log space
# 		# alpha = numpyro.sample('alpha', dist.Uniform())*(jnp.log10(100)-jnp.log10(0.0001)) + jnp.log10(0.0001)
# 		# alpha = 10**alpha

# 		# error_scaling_high = numpyro.sample('error_scaling_high', dist.Uniform(0,1))*1000
# 		# sample error_scaling_high in the log space
# 		# error_scaling_high = numpyro.sample('error_scaling_high', dist.Uniform())*(jnp.log10(1000)-jnp.log10(0.001)) + jnp.log10(0.001)
# 		# error_scaling_high = 10**error_scaling_high
# 		# error_scaling_low = numpyro.sample('error_scaling_low', dist.Uniform(0,1))*10
# 		# error_scaling_fluxes = numpyro.sample('error_scaling_fluxes', dist.Uniform(0,1))

# 		fluxes, fluxes_low = self.compute_MLE(self.obs_map, self.cov_matrix*(2**2), velocities, dispersions, self.factor, alpha,self.wave_factor)
# 		# fluxes, fluxes_low = self.compute_MLE(self.obs_map, self.cov_matrix*error_scaling**2, velocities, dispersions, self.factor, alpha)

# 		flux_prior_high = oversample(self.flux_prior, self.factor, self.factor)

# 		fluxes_errors = jnp.max(jnp.abs(self.flux_prior))/50*jnp.ones_like(self.flux_prior) #+ jnp.sqrt(jnp.abs(self.flux_prior))/5 #error calculated as a S/N
# 		# fluxes_errors = jnp.where(self.mask == 0, self.flux_prior*10,fluxes_errors) #- self.flux_prior/10  #*jnp.ones_like(self.flux_prior)
# 		# fluxes_errors = (self.flux_prior.sum()*jnp.ones_like(self.flux_prior) - self.flux_prior/10)*error_scaling_fluxes  #*jnp.ones_like(self.flux_prior)

# 		model_map = self.grism_object.disperse(fluxes, velocities, dispersions)
# 		model_map_low = resample(model_map, self.y_factor*self.factor, self.wave_factor)

# 		obs_map_low = self.obs_map_low
# 		# error_high = 0.29763962506647706
# 		# error_map_low = (error_high*math.sqrt(self.factor*self.y_factor*self.wave_factor))*jnp.ones_like(obs_map_low)
# 		error_map_low = self.obs_error_low*2

# 		reshaped_fluxes = jnp.reshape(fluxes_low, (1, fluxes_low.shape[0]*fluxes_low.shape[1]))
# 		reshaped_fluxes_errors = jnp.reshape(fluxes_errors, (1, fluxes_errors.shape[0]*fluxes_errors.shape[1]))
# 		reshaped_obs_map_low = jnp.reshape(obs_map_low, (1, obs_map_low.shape[0]*obs_map_low.shape[1]))
# 		reshaped_error_map_low = jnp.reshape(error_map_low, (1, error_map_low.shape[0]*error_map_low.shape[1]))
# 		reshaped_model_map_low = jnp.reshape(model_map_low, (1, model_map_low.shape[0]*model_map_low.shape[1]))
# 		reshaped_flux_prior = jnp.reshape(self.flux_prior, (1, self.flux_prior.shape[0]*self.flux_prior.shape[1]))
# 		# numpyro.sample('obs', dist.Normal(fluxes,(fluxes_errors/2)*jnp.ones_like(flux_prior_high)),obs=flux_prior_high)
# 		# numpyro.sample('obs', dist.Normal(fluxes_low,(fluxes_errors)),obs=self.flux_prior)
# 		numpyro.sample('obs', dist.Normal(jnp.concatenate((reshaped_fluxes, reshaped_model_map_low), axis=1),jnp.concatenate((reshaped_fluxes_errors, reshaped_error_map_low), axis=1)),obs=jnp.concatenate((reshaped_flux_prior, reshaped_obs_map_low), axis=1))


# MODEL FOR SIM DATA TESTING
	def one_comp_MLE(self):

		if self.PA_bounds[1] == 'const':
			Pa = 30
		else:
			Pa = numpyro.sample('PA', dist.Uniform())
			Pa = norm.ppf(Pa)*self.sigma_PA + self.mu_PA
			# Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA
		if self.i_bounds == 'const':
			i = 60
		else:
			i = numpyro.sample('i', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		if self.Va_bounds == 'const':
			Va  = 600
		else:
			Va = numpyro.sample('Va', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		if self.r_t_bounds == 'const':
			r_t = 2
		else:
			r_t = numpyro.sample('r_t', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

		if self.sigma0_bounds == 'const':
			sigma0 = 100
		else:
			sigma0 = numpyro.sample('sigma0', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			# print('lower sigma0 bound:', self.sigma0_bounds[0], 'upper sigma0 bound:', self.sigma0_bounds[1], 'sigma0:', sigma0)
			# sigma0 = norm.ppf(sigma0)*self.sigma_sigma0 + self.mu_sigma0
		# sigma0 = norm.ppf(  norm.cdf(self.low_sigma0) + fluxes*(norm.cdf(self.high_sigma0)-norm.cdf(self.low_sigma0)) )*self.sigma_sigma0 + self.mu_sigma0
		# sigma0 = 100

		# x = jnp.linspace(0 - 14, 28-1 -14, 28*10)
		# y = jnp.linspace(0 - 14, 28-1 -14, 28*10)
		# x = jnp.linspace(0 - 28, 56-1 -28, 56)
		# y = jnp.linspace(0 - 28, 56-1 -28, 56)
		# print(self.grism_object.jcenter,self.grism_object.icenter, self.grism_object.direct.shape)
		#IF YOU HAVE FACTOR != THAN 1 THEN YOU NEED TO CHANGE THE CENTER OF THE GRID
		x = jnp.linspace(0 - self.grism_object.jcenter, self.grism_object.direct.shape[1]-1 -self.grism_object.jcenter, self.grism_object.direct.shape[1]*10)
		y = jnp.linspace(0 - self.grism_object.icenter, self.grism_object.direct.shape[0]-1 -self.grism_object.icenter, self.grism_object.direct.shape[0]*10)
		# print(self.grism_object.jcenter,self.grism_object.icenter, self.grism_object.direct.shape)
		X_high, Y_high = jnp.meshgrid(x, y)

		velocities = jnp.array(v(X_high, Y_high, jnp.radians(Pa),jnp.radians(i), Va, r_t))
		# velocities = Va*jnp.ones_like(self.x)
		velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')
		dispersions = sigma0*jnp.ones_like(velocities)

		fluxes_errors = self.fluxes_errors #jnp.max(jnp.abs(self.flux_prior))/50*jnp.ones_like(self.flux_prior) #+ jnp.sqrt(jnp.abs(self.flux_prior))/5 #error calculated as a S/N

		# velocities = jnp.where(self.flux_prior/fluxes_errors < self.snr_flux, 0.0, velocities)
		# dispersions = jnp.where(self.flux_prior/fluxes_errors < self.snr_flux, 0.0, dispersions)


		# velocities = jnp.where(self.flux_prior < self.flux_prior.max()*0.02, 0, velocities)
		# dispersions = jnp.where(self.flux_prior < self.flux_prior.max()*0.02, 0, dispersions)

		alpha = self.alpha
		#numpyro.sample('alpha', dist.Uniform(0,1))*(0.1-0.00001) +0.00001
		#sample alpha in the log space
		# alpha = numpyro.sample('alpha', dist.Uniform())*(jnp.log10(1)-jnp.log10(0.00001)) + jnp.log10(0.00001)
		# alpha = 10**alpha

		# error_scaling_high = 1#numpyro.sample('error_scaling_high', dist.Uniform(0,1))*9+1
		# sample error_scaling_high from a standard normal distribution
		# error_scaling_high = norm.ppf(error_scaling_high)*10
		# error_scaling_high = numpyro.sample('error_scaling_high', dist.Normal(0,1))*5
		# sample error_scaling_high in the log space
		# error_scaling_high = numpyro.sample('error_scaling_high', dist.Uniform())*(jnp.log10(1000)-jnp.log10(1)) + jnp.log10(1)
		# error_scaling_high = 10**error_scaling_high

		error_scaling_low = self.error_scaling_low
		#numpyro.sample('error_scaling_low', dist.Uniform(0,1))*9+1
		# error_scaling_low = numpyro.sample('error_scaling_low', dist.Normal(0,1))*5
		# error_scaling_low = norm.ppf(error_scaling_low)*5
		# error_scaling_fluxes = numpyro.sample('error_scaling_fluxes', dist.Uniform(0,1))*10

		fluxes, fluxes_low , gaussian, flux_covariance = self.compute_MLE(self.obs_map, self.cov_matrix*error_scaling_low, velocities, dispersions, self.factor, alpha, self.wave_factor)
		# fluxes, fluxes_low = self.compute_MLE(self.obs_map, self.cov_matrix*error_scaling**2, velocities, dispersions, self.factor, alpha)

		#renormalizing the model flux to the flux prior
		# fluxes = fluxes*(self.flux_prior.max()/fluxes.max())

		# flux_prior_high = oversample(self.flux_prior, self.factor, self.factor)

		# fluxes_errors = (self.flux_prior.sum()*error_scaling_fluxes)*jnp.ones_like(self.flux_prior)
		# fluxes_errors = jnp.where(self.mask == 0, self.flux_prior*10,fluxes_errors) #- self.flux_prior/10  #*jnp.ones_like(self.flux_prior)
		# fluxes_errors = (self.flux_prior.sum()*jnp.ones_like(self.flux_prior) - self.flux_prior/50)*error_scaling_fluxes  #*jnp.ones_like(self.flux_prior)

		# fluxes_low= jnp.where(self.flux_prior<self.flux_prior.max()*0.02, 0.0, fluxes_low)
		# model_map = self.grism_object.disperse(fluxes_low, velocities, dispersions)
		model_map = jnp.matmul( fluxes[:,None, :] , gaussian)[:,0,:]
		model_map_low = resample(model_map, self.y_factor*self.factor, 1) #this is always one because the resampling is done before gaussian
		# model_map_low = model_map

		obs_map_low = self.obs_map_low
		# error_high = 0.29763962506647706
		# error_map_low = (error_high*math.sqrt(self.factor*self.y_factor*self.wave_factor))*jnp.ones_like(obs_map_low)
		error_map_low = self.obs_error_low*error_scaling_low  #/2
		# fluxes_errors = fluxes_errors*error_scaling_fluxes
		masked_flux = jnp.where(jnp.abs(self.flux_prior/fluxes_errors) < self.snr_flux, 0.0, self.flux_prior)
		# I NEED TO REDEFINE IT IF I JUST WANT TO SCALE THE CENTRAL PIXEL
		fluxes_errors_masked = jnp.where(jnp.abs(self.flux_prior/fluxes_errors) < self.snr_flux, fluxes_errors*1e6, fluxes_errors*error_scaling_low)

		reshaped_fluxes = jnp.reshape(fluxes, fluxes.shape[0]*fluxes.shape[1])
		# reshaped_fluxes_errors = jnp.reshape(fluxes_errors_masked, (1, fluxes_errors_masked.shape[0]*fluxes_errors_masked.shape[1]))
		reshaped_obs_map_low = jnp.reshape(obs_map_low, obs_map_low.shape[0]*obs_map_low.shape[1])
		reshaped_error_map_low = jnp.reshape(error_map_low, error_map_low.shape[0]*error_map_low.shape[1])
		reshaped_model_map_low = jnp.reshape(model_map_low, model_map_low.shape[0]*model_map_low.shape[1])
		reshaped_masked_flux_prior = jnp.reshape(masked_flux, masked_flux.shape[0]*masked_flux.shape[1])
		reshaped_flux_prior = jnp.reshape(self.flux_prior, self.flux_prior.shape[0]*self.flux_prior.shape[1])
		# flux_covariance = flux_covariance +jax.scipy.linalg.block_diag(*fluxes_errors_masked)
		reshaped_covariance = jax.scipy.linalg.block_diag(*flux_covariance) + jnp.diag(jnp.reshape(fluxes_errors_masked**2, self.fluxes_errors.shape[0]*self.fluxes_errors.shape[1]))
		reshaped_full_covariance = jax.scipy.linalg.block_diag(*flux_covariance,jnp.diag(reshaped_error_map_low))
		# print(reshaped_full_covariance)

		# reshaped_flux_uncertainty = jnp.reshape(flux_uncertainty, (1, flux_uncertainty.shape[0]*flux_uncertainty.shape[1]))
		# numpyro.sample('obs', dist.Normal(fluxes,(fluxes_errors/2)*jnp.ones_like(flux_prior_high)),obs=flux_prior_high)
		# numpyro.sample('obs', dist.Normal(fluxes_low,(fluxes_errors_masked)),obs=masked_flux)
		# numpyro.sample('obs', dist.Normal(jnp.concatenate((reshaped_fluxes, reshaped_model_map_low), axis=1),jnp.concatenate((reshaped_fluxes_errors, reshaped_error_map_low), axis=1)),obs=jnp.concatenate((reshaped_flux_prior, reshaped_obs_map_low), axis=1))

		# numpy_fluxes = np.zeros((fluxes.shape[0],fluxes.shape[1]))
		# for i in range(28):
		# 	for j in range(28):
		# 		numpy_fluxes[i,j] = fluxes[i,j]
		# plt.imshow(numpy_fluxes,origin='lower')
		# plt.savefig('fitting_results/' + 'test' + '.png', dpi=300)
		# numpyro.sample( 'obs', dist.MultivariateNormal(fluxes, jnp.reshape(jnp.sqrt(jnp.diag(reshaped_covariance)), fluxes.shape)), obs=masked_flux)
		#print the sum of the residuals
		numpyro.sample( 'obs', dist.MultivariateNormal(reshaped_fluxes, jnp.diag(jnp.diag(reshaped_covariance))), obs=reshaped_flux_prior)
		# numpyro.sample( 'obs', dist.MultivariateNormal(jnp.concatenate((reshaped_fluxes, reshaped_model_map_low)), reshaped_full_covariance), obs=jnp.concatenate((reshaped_flux_prior, reshaped_obs_map_low)))


	def compute_MLE_model(self,Pa, i , Va, r_t, sigma0):

		#IF YOU HAVE FACTOR != THAN 1 THEN YOU NEED TO CHANGE THE CENTER OF THE GRID
		x = jnp.linspace(0 - self.grism_object.jcenter, self.grism_object.direct.shape[1]-1 -self.grism_object.jcenter, self.grism_object.direct.shape[1]*10)
		y = jnp.linspace(0 - self.grism_object.icenter, self.grism_object.direct.shape[0]-1 -self.grism_object.icenter, self.grism_object.direct.shape[0]*10)
		# print(self.grism_object.jcenter,self.grism_object.icenter, self.grism_object.direct.shape)
		X_high, Y_high = jnp.meshgrid(x, y)

		velocities = jnp.array(v(X_high, Y_high, jnp.radians(Pa),jnp.radians(i), Va, r_t))
		# velocities = Va*jnp.ones_like(self.x)
		velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')
		dispersions = sigma0*jnp.ones_like(velocities)

		fluxes_errors = self.fluxes_errors #jnp.max(jnp.abs(self.flux_prior))/50*jnp.ones_like(self.flux_prior) #+ jnp.sqrt(jnp.abs(self.flux_prior))/5 #error calculated as a S/N

		# velocities = jnp.where(self.flux_prior/fluxes_errors < self.snr_flux, 0.0, velocities)
		# dispersions = jnp.where(self.flux_prior/fluxes_errors < self.snr_flux, 0.0, dispersions)


		# velocities = jnp.where(self.flux_prior < self.flux_prior.max()*0.02, 0, velocities)
		# dispersions = jnp.where(self.flux_prior < self.flux_prior.max()*0.02, 0, dispersions)

		alpha = self.alpha
		#numpyro.sample('alpha', dist.Uniform(0,1))*(0.1-0.00001) +0.00001
		#sample alpha in the log space
		# alpha = numpyro.sample('alpha', dist.Uniform())*(jnp.log10(1)-jnp.log10(0.00001)) + jnp.log10(0.00001)
		# alpha = 10**alpha

		# error_scaling_high = 1#numpyro.sample('error_scaling_high', dist.Uniform(0,1))*9+1
		# sample error_scaling_high from a standard normal distribution
		# error_scaling_high = norm.ppf(error_scaling_high)*10
		# error_scaling_high = numpyro.sample('error_scaling_high', dist.Normal(0,1))*5
		# sample error_scaling_high in the log space
		# error_scaling_high = numpyro.sample('error_scaling_high', dist.Uniform())*(jnp.log10(1000)-jnp.log10(1)) + jnp.log10(1)
		# error_scaling_high = 10**error_scaling_high

		error_scaling_low = self.error_scaling_low
		#numpyro.sample('error_scaling_low', dist.Uniform(0,1))*9+1
		# error_scaling_low = numpyro.sample('error_scaling_low', dist.Normal(0,1))*5
		# error_scaling_low = norm.ppf(error_scaling_low)*5
		# error_scaling_fluxes = numpyro.sample('error_scaling_fluxes', dist.Uniform(0,1))*10

		fluxes, fluxes_low , gaussian, flux_covariance = self.compute_MLE(self.obs_map, self.cov_matrix*error_scaling_low, velocities, dispersions, self.factor, alpha, self.wave_factor)
		# fluxes, fluxes_low = self.compute_MLE(self.obs_map, self.cov_matrix*error_scaling**2, velocities, dispersions, self.factor, alpha)

		#renormalizing the model flux to the flux prior
		# fluxes = fluxes*(self.flux_prior.max()/fluxes.max())

		# flux_prior_high = oversample(self.flux_prior, self.factor, self.factor)

		# fluxes_errors = (self.flux_prior.sum()*error_scaling_fluxes)*jnp.ones_like(self.flux_prior)
		# fluxes_errors = jnp.where(self.mask == 0, self.flux_prior*10,fluxes_errors) #- self.flux_prior/10  #*jnp.ones_like(self.flux_prior)
		# fluxes_errors = (self.flux_prior.sum()*jnp.ones_like(self.flux_prior) - self.flux_prior/50)*error_scaling_fluxes  #*jnp.ones_like(self.flux_prior)

		# fluxes_low= jnp.where(self.flux_prior<self.flux_prior.max()*0.02, 0.0, fluxes_low)
		# model_map = self.grism_object.disperse(fluxes_low, velocities, dispersions)
		model_map = jnp.matmul( fluxes[:,None, :] , gaussian)[:,0,:]
		model_map_low = resample(model_map, self.y_factor*self.factor, 1)
		# model_map_low = model_map

		# obs_map_low = self.obs_map_low
		# error_high = 0.29763962506647706
		# error_map_low = (error_high*math.sqrt(self.factor*self.y_factor*self.wave_factor))*jnp.ones_like(obs_map_low)
		error_map_low = self.obs_error_low*error_scaling_low
		# fluxes_errors = fluxes_errors*error_scaling_fluxes
		masked_flux = jnp.where(jnp.abs(self.flux_prior/fluxes_errors) < self.snr_flux, 0.0, self.flux_prior)
		# I NEED TO REDEFINE IT IF I JUST WANT TO SCALE THE CENTRAL PIXEL
		fluxes_errors_masked = jnp.where(jnp.abs(self.flux_prior/fluxes_errors) < self.snr_flux, fluxes_errors*1e6, fluxes_errors*error_scaling_low)

		reshaped_fluxes = jnp.reshape(fluxes, (fluxes.shape[0]*fluxes.shape[1]))
		# reshaped_fluxes_errors = jnp.reshape(fluxes_errors_masked, (1, fluxes_errors_masked.shape[0]*fluxes_errors_masked.shape[1]))
		# reshaped_obs_map_low = jnp.reshape(obs_map_low, (1, obs_map_low.shape[0]*obs_map_low.shape[1]))
		reshaped_error_map_low = jnp.reshape(error_map_low, (error_map_low.shape[0]*error_map_low.shape[1]))
		# reshaped_model_map_low = jnp.reshape(model_map_low, (1, model_map_low.shape[0]*model_map_low.shape[1]))
		reshaped_flux_prior = jnp.reshape(masked_flux, (masked_flux.shape[0]*masked_flux.shape[1]))
		# flux_covariance = flux_covariance +jax.scipy.linalg.block_diag(*fluxes_errors_masked)
		reshaped_covariance = jax.scipy.linalg.block_diag(*flux_covariance) + jnp.diag(jnp.reshape(self.fluxes_errors**2, self.fluxes_errors.shape[0]*self.fluxes_errors.shape[1]))
		fluxes_errors_total = np.zeros_like(fluxes_errors)
		for i in range(len(flux_covariance)):
			fluxes_errors_total[i] = jnp.sqrt(jnp.diag(flux_covariance[i])+fluxes_errors[i]**2)
		# reshaped_flux_uncertainty = jnp.reshape(flux_uncertainty, (1, flux_uncertainty.shape[0]*flux_uncertainty.shape[1]))
		# numpyro.sample('obs', dist.Normal(fluxes,(fluxes_errors/2)*jnp.ones_like(flux_prior_high)),obs=flux_prior_high)
		# numpyro.sample('obs', dist.Normal(fluxes_low,(fluxes_errors_masked)),obs=masked_flux)
		# numpyro.sample('obs', dist.Normal(jnp.concatenate((reshaped_fluxes, reshaped_model_map_low), axis=1),jnp.concatenate((reshaped_fluxes_errors, reshaped_error_map_low), axis=1)),obs=jnp.concatenate((reshaped_flux_prior, reshaped_obs_map_low), axis=1))

		# numpyro.sample( 'obs', dist.MultivariateNormal(reshaped_fluxes, reshaped_covariance), obs=reshaped_flux_prior)
		return fluxes, fluxes_errors_total, model_map_low

	def two_component_model(self):

		# define the first component which is a simple disk model
		disc_fluxes = numpyro.sample('fluxes', dist.Uniform(), sample_shape=self.flux_prior.shape)
		# manually computing the ppf for a truncated normal distribution
		disc_fluxes = norm.ppf(norm.cdf(self.low) + disc_fluxes*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
		# reparam_config = {"fluxes": TransformReparam()}
		# with numpyro.handlers.reparam(config=reparam_config):
		# 		# in order to use TransformReparam we have to express the prior
		# 		# over betas as a TransformedDistribution
		# 		disc_fluxes = numpyro.sample("fluxes",dist.TransformedDistribution(dist.TruncatedNormal(jnp.zeros(self.flux_prior.shape), jnp.ones_like(self.flux_prior), low = self.low, high = self.high),AffineTransform(self.mu, self.sigma),),)

		disc_fluxes = oversample(disc_fluxes, self.factor, self.factor)

		if self.PA_normal is not None:
				Pa = numpyro.sample('PA', dist.Normal())
				Pa = Pa*self.sigma_PA + self.mu_PA
		else:
			Pa = numpyro.sample('PA', dist.Uniform())
			Pa = norm.ppf(Pa)*self.sigma_PA + self.mu_PA
		# Pa = 30
		# Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA

		i = numpyro.sample('i', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		Va = numpyro.sample('Va', dist.Uniform())*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		r_t = numpyro.sample('r_t', dist.Uniform())*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

		# disc_sigma0 = numpyro.sample('sigma0', dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		disc_sigma0 = numpyro.sample('sigma0', dist.Uniform())
		disc_sigma0 = norm.ppf(disc_sigma0)*self.sigma_sigma0 + self.mu_sigma0
		# sigma0 = norm.ppf(  norm.cdf(self.low_sigma0) + fluxes*(norm.cdf(self.high_sigma0)-norm.cdf(self.low_sigma0)) )*self.sigma_sigma0 + self.mu_sigma0
		# disc_sigma0 = 100

		disc_velocities = v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t)
		# disc_velocities = image.resize(disc_velocities, (int(disc_velocities.shape[0]/10), int(disc_velocities.shape[1]/10)), method='bicubic')
		
		disc_dispersions = disc_sigma0*jnp.ones(disc_velocities.shape)
		self.disc_model_map = self.grism_object.disperse(disc_fluxes, disc_velocities, disc_dispersions)

		# now construct and disperse the second component
		clump_fluxes = numpyro.sample('clump_fluxes', dist.Uniform(), sample_shape=self.flux_prior.shape)
		# width of this also depending on flux? but not on radius, centered on zero but also cutoff at zero
		# an exponeniatal distribution with a scale of flux_prior
		# clump_fluxes = expon.ppf(clump_fluxes, scale=self.flux_prior)
		clump_fluxes = -self.clump_exp_width*jnp.log(1-clump_fluxes)
		clump_fluxes = oversample(clump_fluxes, self.factor, self.factor)
		# velocities = centered on zero with a width depending on flux and radius

		clump_velocities = numpyro.sample('clump_velocities', dist.Uniform(), sample_shape=self.flux_prior.shape)
		clump_velocities = norm.ppf(clump_velocities)*self.sigma_v + self.mu_v
		clump_velocities = image.resize(clump_velocities, (int(clump_velocities.shape[0]*factor), int(clump_velocities.shape[1]*factor)), method='bicubic')

		# this will often be set to the same as disc_dispersions
		clump_sigma0 = numpyro.sample('clump_sigma0', dist.Uniform())
		clump_sigma0 = norm.ppf(clump_sigma0)*self.sigma_sigma0_clump + self.mu_sigma0_clump
		clump_dispersions = clump_sigma0*jnp.ones(clump_velocities.shape)

		self.clump_model_map = self.grism_object.disperse(clump_fluxes, clump_velocities, clump_dispersions)

		self.model_map = self.disc_model_map + self.clump_model_map

		self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)
		numpyro.sample('obs', dist.Normal(self.model_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:], 
				    self.obs_error[self.obs_map_bounds[0]:self.obs_map_bounds[1],:]), 
		 obs=self.obs_map[self.obs_map_bounds[0]:self.obs_map_bounds[1],:])
		

	# def linear_model(self):


	def run_inference(self, model_name, num_samples=2000, num_warmup=2000, high_res=False, median=True, step_size=1, adapt_step_size=True, target_accept_prob=0.8, max_tree_depth=10, num_chains=5):

		# self.nuts_kernel = NUTS(self.full_parametric_renormalized, init_strategy=init_to_median(num_samples=1000), step_size=step_size, adapt_step_size=adapt_step_size,
		# 						target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=max_tree_depth)

		# true_fluxes,true_fluxes_errors_total, true_model_map_low = self.compute_MLE_model(30, 60, 600, 3, 150)
		if model_name == 'two_component_model':
			self.nuts_kernel = NUTS(self.two_component_model, init_strategy=init_to_median(num_samples=1000), step_size=step_size, adapt_step_size=adapt_step_size,
								target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=max_tree_depth, find_heuristic_step_size = True)
		elif model_name == 'one_component_model':
			self.nuts_kernel = NUTS(self.full_parametric_renormalized, init_strategy=init_to_median(num_samples=1000), step_size=step_size, adapt_step_size=adapt_step_size,
								target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=max_tree_depth, find_heuristic_step_size = True)
		elif model_name == 'two_disc_model':
			self.nuts_kernel = NUTS(self.two_disc_model, init_strategy=init_to_median(num_samples=1000), step_size=step_size, adapt_step_size=adapt_step_size,
								target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=max_tree_depth, find_heuristic_step_size = True)	
		elif model_name == 'unified_two_discs_model':
			self.nuts_kernel = NUTS(self.unified_two_discs_model, init_strategy=init_to_median(num_samples=1000), step_size=step_size, adapt_step_size=adapt_step_size,
								target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=max_tree_depth, find_heuristic_step_size = True)
		elif model_name == 'MLE':
			self.nuts_kernel = NUTS(self.one_comp_MLE, init_strategy=init_to_value(values = {"Pa": 0.5, "i": 0.66, "Va": 0.6, "r_t": 0.30, "sigma0": 0.30}), step_size=step_size, adapt_step_size=adapt_step_size,
								target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=max_tree_depth, find_heuristic_step_size = True)
		print('max tree: ', max_tree_depth)
		print('step size: ', step_size)
		print('warmup: ', num_warmup)
		print('samples: ', num_samples)
		self.mcmc = MCMC(self.nuts_kernel, num_samples=num_samples,
						 num_warmup=num_warmup, num_chains=num_chains)
		self.rng_key = random.PRNGKey(4)
		self.mcmc.run(self.rng_key, extra_fields=("potential_energy", "accept_prob"))

		
		# self.mcmc.run(self.rng_key)
		print('done')

		self.mcmc.print_summary()


		# # self.data = az.from_numpyro(self.mcmc)
		# samples = self.mcmc.get_samples()
		# nlnp = self.mcmc.get_extra_fields()["potential_energy"]
		# ind_best = nlnp.argmin()
		# best = {k: samples[k][ind_best] for k in samples.keys()}
		# best_PA = norm.ppf(best['PA'])*self.sigma_PA + self.mu_PA
		# best_i = best['i']*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		# best_Va = best['Va']*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		# best_r_t = best['r_t']*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
		# best_sigma0 = best['sigma0']*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		
		# print('Best parameters: ', best_PA, best_i, best_Va, best_r_t, best_sigma0)

		# first = {k: samples[k][0] for k in samples.keys()}
		# first_PA = norm.ppf(first['PA'])*self.sigma_PA + self.mu_PA
		# first_i = first['i']*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		# first_Va = first['Va']*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		# first_r_t = first['r_t']*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
		# first_sigma0 = first['sigma0']*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# print('First parameters: ', first_PA, first_i, first_Va, first_r_t, first_sigma0)

		# best_model_flux,best_fluxes_errors_total, best_model_map_low = self.compute_MLE_model(best_PA, best_i, best_Va, best_r_t, best_sigma0)
		# # plt.imshow(best_model_flux, origin='lower')
		# # plt.savefig('fitting_results/' + 'best_model_flux' + '.png', dpi=300)
		# plot_image(best_model_flux, 14,14,28,save_to_folder='MLE_degtest', name='best_model_flux')

		# # plt.imshow(fluxes_test, origin='lower')
		# # # plt.savefig('fitting_results/' + 'true_model_flux' + '.png', dpi=300)
		# # plt.imshow(self.flux_prior, origin='lower')
		# # plt.savefig('fitting_results/' + 'flux_prior' + '.png', dpi=300)
		# plot_image(true_fluxes, 14,14,28,save_to_folder='MLE_degtest', name='true_model_flux')
		# plot_image(self.flux_prior, 14,14,28,save_to_folder='MLE_degtest', name='flux_prior')
		# plot_image_residual(best_model_flux, self.flux_prior, best_fluxes_errors_total, 14,14,28,save_to_folder='MLE_degtest', name='best_residual')
		# plot_image_residual(true_fluxes, self.flux_prior, true_fluxes_errors_total, 14,14,28,save_to_folder='MLE_degtest', name='true_residual')
		# print('Best nlp: ', nlnp[ind_best])
		# print('Truth nlp: ', nlnp[0])
		# plot_grism_residual(self.obs_map_low,best_model_map_low, self.obs_error_low, 14,28,self.wave_axis, save_to_folder='MLE_degtest', name='best_grism_residual')
		# plot_grism_residual(self.obs_map_low,true_model_map_low, self.obs_error_low, 14,28,self.wave_axis, save_to_folder='MLE_degtest', name='true_grism_residual')

		# #print the residual of best model and true model with self.flux_prior
		# print('Best model residual: ', jnp.sum((best_model_flux-self.flux_prior)**2/best_fluxes_errors_total**2))
		# print('True model residual: ', jnp.sum((true_fluxes-self.flux_prior)**2/true_fluxes_errors_total**2))

#  --------------MLE --------------------------
	def compute_MLE(self, obs_map_high, cov_matrix, velocities, dispersions,factor,alpha, wave_factor ):


		gaussian = self.grism_object.compute_gaussian(velocities, dispersions)
		blocks = gaussian.reshape((velocities.shape[0],velocities.shape[0], int(gaussian.shape[2]/wave_factor), wave_factor)) #make this into self.wave_factor or smth

		gaussian = jnp.sum(blocks, axis=-1)
		# gaussian = jnp.where(gaussian<1e-10, 0.0, gaussian)
		gaussian_factor = 1 #jnp.sqrt(jnp.linalg.cond(gaussian))
		scaling_factor_gaussian = 1 #jnp.max(gaussian_factor)
		gaussian = gaussian*scaling_factor_gaussian
		gaussian_transpose = jnp.swapaxes(gaussian,1,2)
		# Calculate the scaling factor for the covariance matrix
		scaling_factor_cov = 1 #jnp.max(jnp.abs(cov_matrix[0]))
		# print(scaling_factor_cov)
		# scaling_factor_cov =1 
		#jnp.max(jnp.abs(cov_matrix[0]))
		# print('scaling factor: ', scaling_factor_cov)
		# Apply scaling to the covariance matrix
		cov_matrix_scaled = cov_matrix/scaling_factor_cov
		cov_matrix_inv = jnp.linalg.inv(cov_matrix_scaled)
		# print(cov_matrix_inv.shape)
		first_half = jnp.einsum('ijk,ikl->ijl', gaussian,cov_matrix_inv)
		sigma_f_total = jnp.einsum('ijk,ikl->ijl', first_half, gaussian_transpose)
		# print(sigma_f_total[28])
		# print('sigma_f_total min: ', sigma_f_total.min())
		# sigma_f_inv = jnp.linalg.pinv(sigma_f_total+ alpha * np.eye(obs_map_high.shape[0]))
		# # print('sigma_f_inv min: ', sigma_f_inv.min())
		# second_half = jnp.einsum('ijk,ikl->ijl', sigma_f_inv,first_half)
		# fluxes = jnp.einsum('ijk,ik->ij', second_half, obs_map_high)
		#scaling_factor = 1 #jnp.sqrt(jnp.linalg.cond(obs_map_high))
		scaling_factor =  1 #jnp.sqrt(jnp.linalg.cond(obs_map_high)) #jnp.max(jnp.abs(obs_map_high))
		# print('scaling factor: ', scaling_factor)
		#print('other scaling factor: ', jnp.max(jnp.abs(obs_map_high)))
		second_half = jnp.einsum('ijk,ik->ij', first_half,obs_map_high*scaling_factor)
		# fluxes = np.zeros((28,28))
		 #jnp.sqrt(jnp.linalg.cond(sigma_f_total))
		scaling_factor_second = jnp.max(jnp.sqrt(jnp.linalg.cond(second_half))) #jnp.max(jnp.abs(second_half)) #jnp.sqrt(jnp.linalg.cond(second_half))#1 #jnp.max(jnp.abs(second_half))
		#jnp.sqrt(jnp.linalg.cond(second_half))

		# sigma_f_total = jnp.where(sigma_f_total<1e-5, 0.0, sigma_f_total)
		scaling_factor_sigma = jnp.max(jnp.sqrt(jnp.linalg.cond(sigma_f_total)))#1#jnp.max(jnp.abs(sigma_f_total))
		# print('scaling factor sigma: ', scaling_factor_sigma)
		# plt.imshow(sigma_f_total[14], origin='lower')
		# plt.show()
		# print('DET sigma: ', jnp.linalg.det(sigma_f_total*scaling_factor_sigma))
		# print(sigma_f_total[28], (sigma_f_total*scaling_factor_sigma)[28])
		# print(sigma_f_total.shape)
		# fluxes = jnp.linalg.solve(sigma_f_total*scaling_factor_sigma+ alpha * np.eye(obs_map_high.shape[0]), second_half*scaling_factor_second)

		# reg_matrix = np.zeros((28,28,28))
		# for i in range(28):
		# 	for j in range(28):
		# 		for k in range(28):
		# 			reg_matrix[i,j,k] = np.exp(-(j-k)**2/0.5**2)
		
		# j_index, k_index = jnp.indices((28,28))
		# reg_matrix = jnp.exp(-(j_index-k_index)**2/1) #jnp.eye(28) 
		# # ones = jnp.ones((28,1,1))
		# # reg_matrix = reg_matrix*ones
		# print(reg_matrix)

		j_index, k_index = jnp.indices((obs_map_high.shape[0],obs_map_high.shape[0]))
		# print(sigma_f_total[14,14])
		reg_matrix = jnp.exp(-(j_index-k_index)**2/0.001**2)
		ones = jnp.ones((obs_map_high.shape[0],1,1))
		reg_matrix = reg_matrix*ones
		# print(reg_matrix[14,14])
		# print('sigma total cond numner: ', jnp.linalg.cond(sigma_f_total))
		# fluxes = jnp.linalg.solve(sigma_f_total*scaling_factor_sigma + alpha*reg_matrix, second_half*scaling_factor_second)

		#trying the tikhonov regularization
		sigma_f_total = sigma_f_total*scaling_factor_sigma
		# print(reg_matrix.shape)
		# fluxes = jnp.linalg.solve(jnp.einsum('ijk,ikl->ijl',jnp.transpose(sigma_f_total), sigma_f_total) + alpha*reg_matrix, jnp.einsum('ijk,ik->ij',jnp.transpose(sigma_f_total), second_half*scaling_factor_second))

		fluxes = np.zeros((14,14))
		for i in range(14):
			fluxes[i],conv = scipy.sparse.linalg.gmres(np.array(sigma_f_total[i] + alpha*reg_matrix[i]), np.array(second_half[i]*scaling_factor_second), M = np.linalg.inv(np.diag(np.diag(sigma_f_total[i]))))
		# print(second_half.shape)
		sigma_f_inv = jnp.linalg.inv(sigma_f_total*scaling_factor_sigma +alpha*reg_matrix)
		# print(sigma_f_inv[14,14])
		# plt.imshow(jnp.matmul(sigma_f_inv[14],sigma_f_total[14]),	origin='lower')
		# plt.show()
		# fluxes = jnp.einsum('ijk,ik->ij', sigma_f_inv, second_half*scaling_factor_second)
		fluxes /= (scaling_factor*scaling_factor_second)
		fluxes *= scaling_factor_sigma
		fluxes *= scaling_factor_gaussian
		blocks = fluxes.reshape((int(fluxes.shape[0]/(factor)), factor, int(fluxes.shape[1]/factor), factor))
		fluxes_low = jnp.sum(blocks, axis=(1,3))
		# flux_uncertainty = jnp.zeros((fluxes.shape[0],fluxes.shape[1]))
		# for i in range(fluxes.shape[0]):
		# 	for j in range(fluxes.shape[1]):
		# 		flux_uncertainty= flux_uncertainty.at[i,j].set(jnp.sqrt(sigma_f_inv[i,j,j]))
		
		#jnp.sqrt(jnp.diag(sigma_f_inv))	
		# print(jnp.linalg.eigvals(sigma_f_inv))

		return fluxes, fluxes_low, gaussian, sigma_f_inv*scaling_factor_sigma
	
	# def compute_MLE(self, obs_map_high, cov_matrix, velocities, dispersions, factor, alpha, wave_factor):
	# 	gaussian = self.grism_object.compute_gaussian(velocities, dispersions)
	# 	blocks = gaussian.reshape((velocities.shape[0], velocities.shape[0], int(gaussian.shape[2] / wave_factor), wave_factor))
	# 	gaussian = jnp.sum(blocks, axis=-1)
	# 	gaussian_transpose = jnp.swapaxes(gaussian, 1, 2)
	# 	cov_matrix_inv = jnp.linalg.inv(cov_matrix)

	# 	# Calculate the scaling factor for the covariance matrix
	# 	scaling_factor_cov = jnp.max(jnp.abs(jnp.diag(cov_matrix[0])))
	# 	# Apply scaling to the covariance matrix
	# 	cov_matrix_scaled = cov_matrix*scaling_factor_cov


	# 	first_half = jnp.einsum('ijk,ikl->ijl', gaussian, jnp.linalg.inv(cov_matrix_scaled))
	# 	sigma_f_total = jnp.einsum('ijk,ikl->ijl', first_half, gaussian_transpose)
	# 	# Regularized inversion
	# 	sigma_f_inv = jnp.linalg.pinv(sigma_f_total + alpha * np.eye(obs_map_high.shape[0]))

	# 	scaling_factor = jnp.sqrt(jnp.linalg.cond(obs_map_high))
	# 	second_half = jnp.einsum('ijk,ik->ij', first_half, obs_map_high * scaling_factor)

	# 	# Regularized solution
	# 	fluxes = jnp.linalg.solve(sigma_f_total + alpha * np.eye(obs_map_high.shape[0]), second_half)
	# 	fluxes /= scaling_factor

	# 	blocks = fluxes.reshape((int(fluxes.shape[0] / (factor)), factor, int(fluxes.shape[1] / factor), factor))
	# 	fluxes_low = jnp.sum(blocks, axis=(1, 3))

	# 	return fluxes, fluxes_low

# -----------------------------------------------------------plotting results-----------------------------------------------------------------------------------


	def plot_model(self, factor=1, wave_factor=1, save=False, model_name = 'two_component_model'):
		"""
				Retrieve mean values from the fitting run and plot the resulting maps

		"""

		if model_name == 'one_component_model':
		#find indices of highest log likelihood sample

			best_indices = np.unravel_index(self.data['sample_stats']['lp'].argmin(), self.data['sample_stats']['lp'].shape)

		# rescale all of the posteriors from uniform to the actual parameter space

			# print(self.mu[15], self.sigma[15], self.low[15], self.high[15])
			# plt.imshow(self.mu, origin='lower', vmin = self.flux_prior.min(), vmax = self.flux_prior.max())
			# plt.show()
			# # print(self.x)

			# self.data.posterior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.posterior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			# self.data.prior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.prior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			# self.fluxes_mean = jnp.array(self.data.posterior['fluxes'].median(dim=["chain", "draw"]))

			# only fit for fluxes in the region of the mask
			self.data.posterior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.posterior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			self.data.prior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.prior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			#take the posterior median
			# self.fluxes_sample_mean = jnp.array(self.data.posterior['fluxes'].median(dim=["chain", "draw"]))
			#take the maximum likelihood sample
			self.fluxes_sample_mean = jnp.array(self.data.posterior['fluxes'].isel(chain=best_indices[0], draw=best_indices[1]))

			self.fluxes_mean = jnp.zeros_like(self.flux_prior)
			self.fluxes_mean = self.fluxes_mean.at[self.masked_indices].set(self.fluxes_sample_mean)

			# self.fluxes_mean = self.flux_prior
			# self.data.posterior['PA'].data = norm.ppf(  norm.cdf(self.low_PA) + self.data.posterior['PA'].data*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA
			# self.data.prior['PA'].data = norm.ppf(  norm.cdf(self.low_PA) + self.data.prior['PA'].data*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA
			if self.PA_bounds[1] != 'const':
				rotation = float(self.data.posterior['rotation'].median(dim=["chain", "draw"]))
				PA_morph = self.mu_PA + round(rotation)*180
				self.data.posterior['PA'].data = norm.ppf(self.data.posterior['PA'].data)*self.sigma_PA + PA_morph
				self.data.prior['PA'].data = norm.ppf(self.data.prior['PA'].data)*self.sigma_PA +  PA_morph
				# self.PA_mean = jnp.array(self.data.posterior['PA'].median())
				#take the maximum likelihood sample
				self.PA_mean = jnp.array(self.data.posterior['PA'].isel(chain=best_indices[0], draw=best_indices[1]))

			else:
				self.PA_mean = 30
			# self.data.posterior['sigma0'].data = norm.ppf(  norm.cdf(self.low_sigma0) + self.data.posterior['sigma0'].data*(norm.cdf(self.high_sigma0)-norm.cdf(self.low_sigma0)) )*self.sigma_sigma0 + self.mu_sigma0
			# self.data.prior['sigma0'].data = norm.ppf(  norm.cdf(self.low_sigma0) + self.data.prior['sigma0'].data*(norm.cdf(self.high_sigma0)-norm.cdf(self.low_sigma0)) )*self.sigma_sigma0 + self.mu_sigma0
			# self.data.posterior['sigma0'].data = self.data.posterior['sigma0'].data* (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			if self.sigma0_bounds != 'const':
				# self.data.posterior['sigma0'].data = norm.ppf(self.data.posterior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0
				# self.data.prior['sigma0'].data = norm.ppf(self.data.prior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0
				self.data.prior['sigma0'].data = self.data.prior['sigma0'].data * (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
				self.data.posterior['sigma0'].data = self.data.posterior['sigma0'].data * (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
				# self.sigma0_mean_model = jnp.array(self.data.posterior['sigma0'].median())
				#take the maximum likelihood sample
				self.sigma0_mean_model = jnp.array(self.data.posterior['sigma0'].isel(chain=best_indices[0], draw=best_indices[1]))
				# self.sigma0_mean_model= 100
			else:
				self.sigma0_mean_model= 100
			if self.i_bounds != 'const':	
				self.data.posterior['i'].data = self.data.posterior['i'].data * (self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
				self.data.prior['i'].data = self.data.prior['i'].data * (self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
				# self.i_mean = jnp.array(self.data.posterior['i'].median())
				#take the maximum likelihood sample
				self.i_mean = jnp.array(self.data.posterior['i'].isel(chain=best_indices[0], draw=best_indices[1]))
				# self.i_mean = 41
			else:
				self.i_mean = 60
			if self.Va_bounds != 'const':
				self.data.posterior['Va'].data = self.data.posterior['Va'].data * (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
				self.data.prior['Va'].data = self.data.prior['Va'].data * (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
				# self.Va_mean = jnp.array(self.data.posterior['Va'].median())
				#take the maximum likelihood sample
				self.Va_mean = jnp.array(self.data.posterior['Va'].isel(chain=best_indices[0], draw=best_indices[1]))
				# self.Va_mean = 400

			else:
				self.Va_mean = 600
			if self.r_t_bounds != 'const':
				self.data.posterior['r_t'].data = self.data.posterior['r_t'].data * (self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
				self.data.prior['r_t'].data = self.data.prior['r_t'].data * (self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
				# self.r_t_mean = jnp.array(self.data.posterior['r_t'].median())
				#take the maximum likelihood sample
				self.r_t_mean = jnp.array(self.data.posterior['r_t'].isel(chain=best_indices[0], draw=best_indices[1]))
			else:
				self.r_t_mean = 2
			
			self.data.posterior['v_r'] =  self.data.posterior['Va'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t'].data)
			self.data.prior['v_r'] = self.data.prior['Va'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t'].data)
			# print('v_r mean: ', self.Va_mean * (2/pi) * jnp.arctan(2/self.r_t_mean))
			# self.data.posterior['v_{rot}'] =  2*self.data.posterior['Va']
			self.model_flux = oversample(self.fluxes_mean, factor, factor)
			# plt.imshow(self.model_flux, origin='lower', vmin = self.flux_prior.min(), vmax = self.flux_prior.max())
			# plt.show()
			# delta_wave = self.grism_object.wave_scale
			# wave_space = self.grism_object.wave_space

			# self.data.posterior['x0'].data = norm.ppf(self.data.posterior['x0'].data)*(2) + self.x0
			# self.data.prior['x0'].data = norm.ppf(self.data.prior['x0'].data)*(2) + self.x0
			# self.data.posterior['y0'].data = norm.ppf(self.data.posterior['y0'].data)*(2) + self.y0
			# self.data.prior['y0'].data = norm.ppf(self.data.prior['y0'].data)*(2) + self.y0

			# self.x0_mean = jnp.array(self.data.posterior['x0'].median())
			# self.y0_mean = jnp.array(self.data.posterior['y0'].median())
			self.x0_mean = self.x0
			self.y0_mean = self.y0

			# x = jnp.linspace(0 - self.x0_mean, self.grism_object.direct.shape[1] - 1 - self.x0_mean, self.grism_object.direct.shape[1]*self.factor*10)
			# y = jnp.linspace(0 - self.y0_mean, self.grism_object.direct.shape[0] - 1 - self.y0_mean, self.grism_object.direct.shape[0]*self.factor*10)
			# X,Y = jnp.meshgrid(x,y)

			self.model_velocities = jnp.array(v(self.x,self.y, jnp.radians(self.PA_mean), jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
			# print(self.model_velocities[15*factor])
			# self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

			# self.model_dispersions = jnp.array(sigma(self.x, self.y, self.sigma0_mean_model))
			self.model_dispersions = self.sigma0_mean_model*jnp.ones_like(self.model_velocities)
			# plt.imshow(self.model_dispersions)
			# self.grism_object.wavelength = 3.5612
			
			self.data.posterior['wavelength'].data = norm.ppf(self.data.posterior['wavelength'].data)*0.001 + self.wavelength
			self.data.prior['wavelength'].data = norm.ppf(self.data.prior['wavelength'].data)*0.001 + self.wavelength
			#take the maximum likelihood sample
			corrected_wavelength = float(jnp.array(self.data.posterior['wavelength'].isel(chain=best_indices[0], draw=best_indices[1])))
			# corrected_wavelength = float(self.data.posterior['wavelength'].median(dim=["chain", "draw"]))
			self.grism_object.set_wavelength(corrected_wavelength)

			self.model_map_high = self.grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
			self.model_map = resample(self.model_map_high, self.y_factor*self.factor, self.wave_factor)
			# self.model_map = resample(self.model_map_high, self.factor, self.wave_factor
			return self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions, self.y

# #PLOT MODEL FOR THE REAL CASE
# 		elif model_name == 'MLE':
# 			print('MLE model')

# 			self.data.posterior['PA'].data = norm.ppf(self.data.posterior['PA'].data)*self.sigma_PA + self.mu_PA
# 			self.data.prior['PA'].data = norm.ppf(self.data.prior['PA'].data)*self.sigma_PA + self.mu_PA

# 			self.data.posterior['i'].data = self.data.posterior['i'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
# 			self.data.prior['i'].data = self.data.prior['i'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

# 			self.data.posterior['Va'].data = self.data.posterior['Va'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
# 			self.data.prior['Va'].data = self.data.prior['Va'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

# 			self.data.posterior['r_t'].data = self.data.posterior['r_t'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
# 			self.data.prior['r_t'].data = self.data.prior['r_t'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

# 			# self.data.posterior['sigma0'].data = norm.ppf(self.data.posterior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0
# 			# self.data.prior['sigma0'].data = norm.ppf(self.data.prior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0

# 			self.data.posterior['sigma0'].data = self.data.posterior['sigma0'].data * (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
# 			self.data.prior['sigma0'].data = self.data.prior['sigma0'].data * (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

# 			self.data.posterior['v_r'] =  self.data.posterior['Va'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t'].data)
# 			self.data.prior['v_r'] = self.data.prior['Va'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t'].data)

# 			self.data.posterior['alpha'].data = self.data.posterior['alpha'].data*(100-0.00001) +0.00001
# 			self.data.prior['alpha'].data = self.data.prior['alpha'].data*(100-0.00001) +0.00001

# 			# self.data.posterior['alpha'].data = self.data.posterior['alpha'].data*(jnp.log10(100)-jnp.log10(0.0001)) + jnp.log10(0.0001)
# 			# self.data.prior['alpha'].data = self.data.prior['alpha'].data*(jnp.log10(100)-jnp.log10(0.0001)) + jnp.log10(0.0001)

# 			# self.data.posterior['alpha'].data = 10**self.data.posterior['alpha'].data
# 			# self.data.prior['alpha'].data = 10**self.data.prior['alpha'].data
# 			# self.data.posterior['error_scaling_high'].data = self.data.posterior['error_scaling_high'].data*1000 #9.99 + 0.01
# 			# self.data.prior['error_scaling_high'].data = self.data.prior['error_scaling_high'].data*1000  #9.99 + 0.01

# 			# self.data.posterior['error_scaling_high'].data = self.data.posterior['error_scaling_high'].data*(jnp.log10(1000)-jnp.log10(0.001)) + jnp.log10(0.001)
# 			# self.data.prior['error_scaling_high'].data = self.data.prior['error_scaling_high'].data*(jnp.log10(1000)-jnp.log10(0.001)) + jnp.log10(0.001)

# 			# self.data.posterior['error_scaling_high'].data = 10**self.data.posterior['error_scaling_high'].data
# 			# self.data.prior['error_scaling_high'].data = 10**self.data.prior['error_scaling_high'].data


# 			# self.data.posterior['error_scaling_low'].data = self.data.posterior['error_scaling_low'].data*9 +1
# 			# self.data.prior['error_scaling_low'].data = self.data.prior['error_scaling_low'].data*9 +1

# 			# self.data.posterior['error_scaling_fluxes'].data = self.data.posterior['error_scaling_fluxes'].data
# 			# self.data.prior['error_scaling_fluxes'].data = self.data.prior['error_scaling_fluxes'].data

# 			self.PA_mean = jnp.array(self.data.posterior['PA'].median(dim=["chain", "draw"]))
# 			self.i_mean = jnp.array(self.data.posterior['i'].median(dim=["chain", "draw"]))
# 			self.Va_mean = jnp.array(self.data.posterior['Va'].median(dim=["chain", "draw"]))
# 			self.r_t_mean = jnp.array(self.data.posterior['r_t'].median(dim=["chain", "draw"]))
# 			self.sigma0_mean_model = jnp.array(self.data.posterior['sigma0'].median(dim=["chain", "draw"]))
# 			self.v_r_mean = jnp.array(self.data.posterior['v_r'].median(dim=["chain", "draw"]))
# 			self.alpha_mean = jnp.array(self.data.posterior['alpha'].median(dim=["chain", "draw"]))
# 			# self.error_scaling_high = jnp.array(self.data.posterior['error_scaling_high'].median(dim=["chain", "draw"]))
# 			# self.error_scaling_low = jnp.array(self.data.posterior['error_scaling_low'].median(dim=["chain", "draw"]))
# 			# self.error_scaling_fluxes = jnp.array(self.data.posterior['error_scaling_fluxes'].median(dim=["chain", "draw"]))
# 			self.alpha_mean =100
# 			# self.i_mean = 60
# 			# self.PA_mean = 30
# 			# self.Va_mean = 600
# 			# self.r_t_mean = 2
# 			# self.sigma0_mean_model = 100
# 			# print('self.x = ', self.x[1])
# 			x = jnp.linspace(0 - self.grism_object.jcenter,  self.grism_object.direct.shape[1]-1 -self.grism_object.jcenter,  self.grism_object.direct.shape[1]*10)
# 			y = jnp.linspace(0 - self.grism_object.icenter,  self.grism_object.direct.shape[0]-1 -self.grism_object.icenter,  self.grism_object.direct.shape[0]*10)
# 			# X_high, Y_high = jnp.meshgrid(x, y)
# 			# x = jnp.linspace(0 - 3, 7-1 -3, 7)
# 			# y = jnp.linspace(0 - 163, 7-1 -3, 7)
# 			X_high, Y_high = jnp.meshgrid(x, y)

# 			velocities = jnp.array(v(X_high, Y_high, jnp.radians(self.PA_mean),jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
# 			# velocities = jnp.array(v(self.x, self.y, jnp.radians(self.PA_mean),jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
# 			# print('velocities = ', velocities[7])
# 			# velocities = Va*jnp.ones_like(self.x)
# 			velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='bicubic')
			
# 			dispersions = self.sigma0_mean_model*jnp.ones_like(velocities)
# 			# velocities = jnp.where(self.mask == 0, 0, velocities)
# 			# dispersions = jnp.where(self.mask == 0, 1e-6, dispersions)

# 			# print(self.obs_map.shape)
			
# 			self.fluxes, self.model_flux_low = self.compute_MLE(self.obs_map,self.cov_matrix*(100**2), velocities, dispersions, self.factor,self.alpha_mean,self.wave_factor)
# 			# self.fluxes, self.model_flux_low = self.compute_MLE(self.obs_map,self.cov_matrix*self.error_scaling_mean**2, velocities, dispersions, self.factor,self.alpha_mean)

# 			self.truth_flux_high = oversample(self.flux_prior, self.factor, self.factor)
# 			plt.imshow(self.truth_flux_high)
# 			print(self.truth_flux_high.max(), self.fluxes.max())

# 			plot_image(self.fluxes, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])

# 			plot_image_residual(self.truth_flux_high, self.fluxes,self.truth_flux_high, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])
# 			self.fluxes = jnp.where(self.flux_prior< self.flux_prior.max()*0.1, 0, self.fluxes)
# 			self.model_map_high = self.grism_object.disperse(self.fluxes, velocities, dispersions)
# 			self.model_map = resample(self.model_map_high, self.y_factor*self.factor, self.wave_factor)
			
# 			# error_high = 0.29763962506647706
# 			# self.obs_map_low = resample(self.obs_map, self.y_factor*self.factor, self.wave_factor)
# 			# self.error_map_low = (error_high*math.sqrt(self.factor*self.y_factor*self.wave_factor))*jnp.ones_like(self.obs_map_low)
# 			# self.error_map_low = resample_errors(self.obs_error, self.y_factor*self.factor, self.wave_factor)
			
# 			blocks = self.fluxes.reshape((int(self.fluxes.shape[0]/(self.factor)), self.factor, int(self.fluxes.shape[1]/self.factor), self.factor))
# 			self.model_flux_low = jnp.sum(blocks, axis=(1,3))
# 			# print(self.fluxes.shape, self.model_flux_low.shape)
# 			# plot_image(self.model_flux_low, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])
# 			return self.model_map,self.fluxes,self.model_flux_low , velocities, dispersions, self.y

#PLOT MODEL FOR THE MODEL SIM CASE
		elif model_name == 'MLE':
			print('MLE model')

			self.data.posterior['PA'].data = norm.ppf(self.data.posterior['PA'].data)*self.sigma_PA + self.mu_PA
			self.data.prior['PA'].data = norm.ppf(self.data.prior['PA'].data)*self.sigma_PA + self.mu_PA

			self.data.posterior['i'].data = self.data.posterior['i'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
			self.data.prior['i'].data = self.data.prior['i'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

			self.data.posterior['Va'].data = self.data.posterior['Va'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
			self.data.prior['Va'].data = self.data.prior['Va'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

			self.data.posterior['r_t'].data = self.data.posterior['r_t'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
			self.data.prior['r_t'].data = self.data.prior['r_t'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

			# self.data.posterior['sigma0'].data = norm.ppf(self.data.posterior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0
			# self.data.prior['sigma0'].data = norm.ppf(self.data.prior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0

			self.data.posterior['sigma0'].data = self.data.posterior['sigma0'].data * (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			self.data.prior['sigma0'].data = self.data.prior['sigma0'].data * (self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

			self.data.posterior['v_r'] =  self.data.posterior['Va'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t'].data)
			self.data.prior['v_r'] = self.data.prior['Va'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t'].data)
			# self.data.posterior['alpha'].data = self.data.posterior['alpha'].data*(0.1-0.00001) +0.00001
			# self.data.prior['alpha'].data = self.data.prior['alpha'].data*(0.1-0.00001) +0.00001

			# self.data.posterior['alpha'].data = self.data.posterior['alpha'].data*(jnp.log10(1)-jnp.log10(0.00001)) + jnp.log10(0.00001)
			# self.data.prior['alpha'].data = self.data.prior['alpha'].data*(jnp.log10(1)-jnp.log10(0.00001)) + jnp.log10(0.00001)

			# self.data.posterior['alpha'].data = 10**self.data.posterior['alpha'].data
			# # self.data.prior['alpha'].data = 10**self.data.prior['alpha'].data
			# self.data.posterior['error_scaling_high'].data = self.data.posterior['error_scaling_high'].data*10 #9.99 + 0.01
			# self.data.prior['error_scaling_high'].data = self.data.prior['error_scaling_high'].data*10  #9.99 + 0.01

			
			# self.data.posterior['error_scaling_high'].data = self.data.posterior['error_scaling_high'].data*(jnp.log10(1000)-jnp.log10(1)) + jnp.log10(1)
			# self.data.prior['error_scaling_high'].data = self.data.prior['error_scaling_high'].data*(jnp.log10(1000)-jnp.log10(1)) + jnp.log10(1)

			# self.data.posterior['error_scaling_high'].data = 10**self.data.posterior['error_scaling_high'].data
			# self.data.prior['error_scaling_high'].data = 10**self.data.prior['error_scaling_high'].data


			# self.data.posterior['error_scaling_low'].data = self.data.posterior['error_scaling_low'].data*9+1
			# self.data.prior['error_scaling_low'].data = self.data.prior['error_scaling_low'].data*9+1

			# self.data.posterior['error_scaling_fluxes'].data = self.data.posterior['error_scaling_fluxes'].data*9+1
			# self.data.prior['error_scaling_fluxes'].data = self.data.prior['error_scaling_fluxes'].data*9+1

			# self.PA_mean = jnp.array(self.data.posterior['PA'].median(dim=["chain", "draw"]))
			# self.i_mean = jnp.array(self.data.posterior['i'].median(dim=["chain", "draw"]))
			# self.Va_mean = jnp.array(self.data.posterior['Va'].median(dim=["chain", "draw"]))
			# self.r_t_mean = jnp.array(self.data.posterior['r_t'].median(dim=["chain", "draw"]))
			# self.sigma0_mean_model = jnp.array(self.data.posterior['sigma0'].median(dim=["chain", "draw"]))
			# self.v_r_mean = jnp.array(self.data.posterior['v_r'].median(dim=["chain", "draw"]))
			# self.alpha_mean = 0.0 # jnp.array(self.data.posterior['alpha'].median(dim=["chain", "draw"]))
			# self.error_scaling_high = jnp.array(self.data.posterior['error_scaling_high'].median(dim=["chain", "draw"]))
			# self.error_scaling_low = 4 #jnp.array(self.data.posterior['error_scaling_low'].median(dim=["chain", "draw"]))
			# self.error_scaling_fluxes = jnp.array(self.data.posterior['error_scaling_fluxes'].median(dim=["chain", "draw"]))
			# self.alpha_mean =0.1
			# self.erro
			# r_scaling_low = 1
			# self.i_mean = 60
			# self.PA_mean = 30
			# self.Va_mean = 600
			# self.r_t_mean  =3
			# self.sigma0_mean_model =140
			# print('self.x = ', self.x[1])
			# x = jnp.linspace(0 - 14, 31-1 -14, 31)
			# y = jnp.linspace(0 - 16, 31-1 -16, 31)
			# X_high, Y_high = jnp.meshgrid(x, y)
			# x = jnp.linspace(0 - 8, 16-1 -8, 16*self.factor*10)
			# y = jnp.linspace(0 - 8, 16-1 -8, 16*self.factor*10)
			# x = jnp.linspace(0 - 28, 56-1 -28, 56)
			# y = jnp.linspace(0 - 28, 56-1 -28, 56)
			x = jnp.linspace(0 - 14, 28-1 -14, 28*10)
			y = jnp.linspace(0 - 14, 28-1 -14, 28*10)
			# x = jnp.linspace(0 - self.grism_object.jcenter, self.grism_object.direct.shape[1]-1 -self.grism_object.jcenter, self.grism_object.direct.shape[1]*10)
			# y = jnp.linspace(0 - self.grism_object.icenter, self.grism_object.direct.shape[0]-1 -self.grism_object.icenter, self.grism_object.direct.shape[0]*10)
			X_high, Y_high = jnp.meshgrid(x, y)

			velocities = jnp.array(v(X_high, Y_high, jnp.radians(self.PA_mean),jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
			# velocities = jnp.array(v(X_high, Y_high, jnp.radians(self.PA_mean),jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
			# print('velocities = ', velocities[7])
			# velocities = Va*jnp.ones_like(self.x)
			velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')
			dispersions = self.sigma0_mean_model*jnp.ones_like(velocities)
			# velocities = jnp.where(self.flux_prior < self.flux_prior.max()*0.02, 10, velocities)
			# dispersions = jnp.where(self.flux_prior < self.flux_prior.max()*0.02, 10, dispersions)
			fluxes_errors = self.fluxes_errors #jnp.max(jnp.abs(self.flux_prior))/50*jnp.ones_like(self.flux_prior) #+ jnp.sqrt(jnp.abs(self.flux_prior))/5 #error calculated as a S/N

			# velocities = jnp.where(self.flux_prior == 0.0, 0, velocities)
			# dispersions = jnp.where(self.flux_prior == 0.0, 0, dispersions)


			print(self.obs_map.shape)
			self.fluxes, self.model_flux_low, gaussian, self.covariance = self.compute_MLE(self.obs_map,self.cov_matrix, velocities, dispersions, self.factor,self.alpha, self.wave_factor)

			# self.fluxes = self.fluxes*self.flux_prior.max()/self.fluxes.max()

			# self.fluxes, self.model_flux_low = self.compute_MLE(self.obs_map,self.cov_matrix*self.error_scaling_mean**2, velocities, dispersions, self.factor,self.alpha_mean)
			# self.fluxes_masked = jnp.where(self.flux_prior<self.flux_prior.max()*0.02, 0.0, self.fluxes)
			# model_map = self.grism_object.disperse(self.fluxes, velocities, dispersions)
			# self.truth_flux_high = oversample(self.flux_prior, self.factor, self.factor)
			# print(self.truth_flux_high.max(), self.fluxes.max())
			plot_image(self.fluxes, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])

			# plot_image(self.truth_flux_high, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])
			plt.imshow(self.obs_map, origin='lower')
			plt.show()
			# plot_image_residual(self.truth_flux_high, self.fluxes,self.truth_flux_high, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])

			# self.model_map_high = self.grism_object.disperse(self.fluxes, velocities, dispersions)
			self.model_map_high = jnp.matmul( self.fluxes[:,None, :] , gaussian)[:,0,:]
			self.model_map = resample(self.model_map_high, self.y_factor*self.factor, 1)
			
			# error_high = 0.29763962506647706
			# self.obs_map_low = resample(self.obs_map, self.y_factor*self.factor, self.wave_factor)
			# self.error_map_low = (error_high*math.sqrt(self.factor*self.y_factor*self.wave_factor))*jnp.ones_like(self.obs_map_low)
			# self.error_map_low = resample_errors(self.obs_error, self.y_factor*self.factor, self.wave_factor)
			
			blocks = self.fluxes.reshape((int(self.fluxes.shape[0]/(self.factor)), self.factor, int(self.fluxes.shape[1]/self.factor), self.factor))
			self.model_flux_low = jnp.sum(blocks, axis=(1,3))
			# print(self.fluxes.shape, self.model_flux_low.shape)
			plotting.plot_image(self.model_flux_low, self.grism_object.jcenter, self.grism_object.jcenter, self.grism_object.direct.shape[0])
			return self.model_map,self.fluxes,self.model_flux_low , velocities, dispersions, self.y

		elif model_name == 'two_component_model':
			print('two-component model')
			self.data.posterior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.posterior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			self.data.prior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.prior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu

			self.data.posterior['PA'].data = norm.ppf(self.data.posterior['PA'].data)*self.sigma_PA + self.mu_PA
			self.data.prior['PA'].data = norm.ppf(self.data.prior['PA'].data)*self.sigma_PA + self.mu_PA

			self.data.posterior['i'].data = self.data.posterior['i'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
			self.data.prior['i'].data = self.data.prior['i'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

			self.data.posterior['Va'].data = self.data.posterior['Va'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
			self.data.prior['Va'].data = self.data.prior['Va'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

			self.data.posterior['r_t'].data = self.data.posterior['r_t'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
			self.data.prior['r_t'].data = self.data.prior['r_t'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

			self.data.posterior['sigma0'].data = norm.ppf(self.data.posterior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0
			self.data.prior['sigma0'].data = norm.ppf(self.data.prior['sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0

			self.data.posterior['clump_fluxes'].data = -self.clump_exp_width*jnp.log(1-self.data.posterior['clump_fluxes'].data)
			self.data.prior['clump_fluxes'].data = -self.clump_exp_width*jnp.log(1-self.data.prior['clump_fluxes'].data)

			self.data.posterior['clump_velocities'].data = norm.ppf(self.data.posterior['clump_velocities'].data)*self.sigma_v + self.mu_v
			self.data.prior['clump_velocities'].data = norm.ppf(self.data.prior['clump_velocities'].data)*self.sigma_v + self.mu_v
			
			self.data.posterior['clump_sigma0'].data = norm.ppf(self.data.posterior['clump_sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0
			self.data.prior['clump_sigma0'].data = norm.ppf(self.data.prior['clump_sigma0'].data)*self.sigma_sigma0 + self.mu_sigma0

			self.data.posterior['v_r'] =  self.data.posterior['Va'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t'].data)
			self.data.prior['v_r'] = self.data.prior['Va'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t'].data)
			# take all of the medians
			self.fluxes_mean = jnp.array(self.data.posterior['fluxes'].median(dim=["chain", "draw"]))
			self.PA_mean = jnp.array(self.data.posterior['PA'].median(dim=["chain", "draw"]))
			# self.PA_mean = 30
			self.i_mean = jnp.array(self.data.posterior['i'].median(dim=["chain", "draw"]))
			self.Va_mean = jnp.array(self.data.posterior['Va'].median(dim=["chain", "draw"]))
			# self.Va_mean = 400
			self.r_t_mean = jnp.array(self.data.posterior['r_t'].median(dim=["chain", "draw"]))
			self.sigma0_mean = jnp.array(self.data.posterior['sigma0'].median(dim=["chain", "draw"]))
			# self.sigma0_mean = 100
			self.clump_fluxes_mean = jnp.array(self.data.posterior['clump_fluxes'].median(dim=["chain", "draw"]))
			self.clump_velocities_mean = jnp.array(self.data.posterior['clump_velocities'].median(dim=["chain", "draw"]))
			self.clump_sigma0_mean = jnp.array(self.data.posterior['clump_sigma0'].median(dim=["chain", "draw"]))

			self.model_fluxes = oversample(self.fluxes_mean, self.factor, self.factor)
			self.model_clump_fluxes = oversample(self.clump_fluxes_mean, self.factor, self.factor)
			self.model_clump_velocities= image.resize(self.clump_velocities_mean, (int(self.clump_velocities_mean.shape[0]*factor), int(self.clump_velocities_mean.shape[1]*factor)), method='bicubic')

			self.clump_dispersions = self.clump_sigma0_mean*jnp.ones_like(self.model_clump_velocities)
			self.disc_velocities = v(self.x, self.y, jnp.radians(self.PA_mean),jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean)
			# self.disc_velocities = image.resize(self.disc_velocities, (int(self.disc_velocities.shape[0]/10), int(self.disc_velocities.shape[1]/10)), method='bicubic')
			self.disc_dispersions = self.sigma0_mean*jnp.ones_like(self.disc_velocities)
			self.model_disc_model_map = self.grism_object.disperse(self.model_fluxes, self.disc_velocities, self.disc_dispersions)

			self.clump_model_map = self.grism_object.disperse(self.model_clump_fluxes, self.model_clump_velocities, self.clump_dispersions)

			self.model_map = self.model_disc_model_map + self.clump_model_map

			self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)

			self.model_flux = self.model_fluxes + self.model_clump_fluxes

			# self.model_velocities = self.disc_velocities + self.model_clump_velocities

			self.model_dispersions = self.disc_dispersions

			return self.model_map, self.model_flux, self.fluxes_mean+self.clump_fluxes_mean, self.disc_velocities, self.model_clump_velocities ,self.model_dispersions, self.y

		elif model_name == 'two_disc_model':
			print('two-disc model')
			self.data.posterior['fluxes_1'].data = norm.ppf(norm.cdf(self.low_1) + self.data.posterior['fluxes_1'].data*(norm.cdf(self.high_1)-norm.cdf(self.low_1)))*self.sigma_1 + self.mu_1
			self.data.prior['fluxes_1'].data = norm.ppf(norm.cdf(self.low_1) + self.data.prior['fluxes_1'].data*(norm.cdf(self.high_1)-norm.cdf(self.low_1)))*self.sigma_1 + self.mu_1

			self.data.posterior['fluxes_2'].data = norm.ppf(norm.cdf(self.low_2) + self.data.posterior['fluxes_2'].data*(norm.cdf(self.high_2)-norm.cdf(self.low_2)))*self.sigma_2 + self.mu_2
			self.data.prior['fluxes_2'].data = norm.ppf(norm.cdf(self.low_2) + self.data.prior['fluxes_2'].data*(norm.cdf(self.high_2)-norm.cdf(self.low_2)))*self.sigma_2 + self.mu_2

			self.data.posterior['PA_1'].data = norm.ppf(self.data.posterior['PA_1'].data)*self.sigma_PA_1 + self.mu_PA_1
			self.data.prior['PA_1'].data = norm.ppf(self.data.prior['PA_1'].data)*self.sigma_PA_1 + self.mu_PA_1

			self.data.posterior['PA_2'].data = norm.ppf(self.data.posterior['PA_2'].data)*self.sigma_PA_2 + self.mu_PA_2
			self.data.prior['PA_2'].data = norm.ppf(self.data.prior['PA_2'].data)*self.sigma_PA_2 + self.mu_PA_2

			self.data.posterior['i_1'].data = self.data.posterior['i_1'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
			self.data.prior['i_1'].data = self.data.prior['i_1'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

			self.data.posterior['i_2'].data = self.data.posterior['i_2'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
			self.data.prior['i_2'].data = self.data.prior['i_2'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

			self.data.posterior['Va_1'].data = self.data.posterior['Va_1'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
			self.data.prior['Va_1'].data = self.data.prior['Va_1'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

			self.data.posterior['Va_2'].data = self.data.posterior['Va_2'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
			self.data.prior['Va_2'].data = self.data.prior['Va_2'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

			self.data.posterior['r_t_1'].data = self.data.posterior['r_t_1'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
			self.data.prior['r_t_1'].data = self.data.prior['r_t_1'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

			self.data.posterior['r_t_2'].data = self.data.posterior['r_t_2'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
			self.data.prior['r_t_2'].data = self.data.prior['r_t_2'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

			# self.data.posterior['sigma0_1'].data = norm.ppf(self.data.posterior['sigma0_1'].data)*self.sigma_sigma0 + self.mu_sigma0
			# self.data.prior['sigma0_1'].data = norm.ppf(self.data.prior['sigma0_1'].data)*self.sigma_sigma0 + self.mu_sigma0

			# self.data.posterior['sigma0_2'].data = norm.ppf(self.data.posterior['sigma0_2'].data)*self.sigma_sigma0 + self.mu_sigma0
			# self.data.prior['sigma0_2'].data = norm.ppf(self.data.prior['sigma0_2'].data)*self.sigma_sigma0 + self.mu_sigma0

			self.data.posterior['sigma0_1'].data = self.data.posterior['sigma0_1'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			self.data.prior['sigma0_1'].data = self.data.prior['sigma0_1'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

			self.data.posterior['sigma0_2'].data = self.data.posterior['sigma0_2'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			self.data.prior['sigma0_2'].data = self.data.prior['sigma0_2'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

			self.data.posterior['v_offset'].data = self.data.posterior['v_offset'].data*1000 - 500

			self.data.posterior['v_r_1'] =  self.data.posterior['Va_1'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t_1'].data)
			self.data.prior['v_r_1'] = self.data.prior['Va_1'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t_1'].data)

			self.data.posterior['v_r_2'] =  self.data.posterior['Va_2'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t_2'].data)
			self.data.prior['v_r_2'] = self.data.prior['Va_2'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t_2'].data)

			self.fluxes_1_mean = jnp.array(self.data.posterior['fluxes_1'].median(dim=["chain", "draw"]))
			self.fluxes_2_mean = jnp.array(self.data.posterior['fluxes_2'].median(dim=["chain", "draw"]))
			self.PA_1_mean = jnp.array(self.data.posterior['PA_1'].median(dim=["chain", "draw"]))
			self.PA_2_mean = jnp.array(self.data.posterior['PA_2'].median(dim=["chain", "draw"]))
			self.i_1_mean = jnp.array(self.data.posterior['i_1'].median(dim=["chain", "draw"]))
			self.i_2_mean = jnp.array(self.data.posterior['i_2'].median(dim=["chain", "draw"]))
			self.Va_1_mean = jnp.array(self.data.posterior['Va_1'].median(dim=["chain", "draw"]))
			self.Va_2_mean = jnp.array(self.data.posterior['Va_2'].median(dim=["chain", "draw"]))
			self.r_t_1_mean = jnp.array(self.data.posterior['r_t_1'].median(dim=["chain", "draw"]))
			self.r_t_2_mean = jnp.array(self.data.posterior['r_t_2'].median(dim=["chain", "draw"]))
			self.sigma0_1_mean = jnp.array(self.data.posterior['sigma0_1'].median(dim=["chain", "draw"]))
			self.sigma0_2_mean = jnp.array(self.data.posterior['sigma0_2'].median(dim=["chain", "draw"]))
			self.v_offset_mean = jnp.array(self.data.posterior['v_offset'].median(dim=["chain", "draw"]))
			

			self.model_fluxes_1 = oversample(self.fluxes_1_mean, self.factor, self.factor)
			self.model_fluxes_2 = oversample(self.fluxes_2_mean, self.factor, self.factor)

			self.disc_velocities_1 = v(self.x, self.y, jnp.radians(self.PA_1_mean),jnp.radians(self.i_1_mean), self.Va_1_mean, self.r_t_1_mean)
			self.disc_velocities_2 = v(self.x_2, self.y_2, jnp.radians(self.PA_2_mean),jnp.radians(self.i_2_mean), self.Va_2_mean, self.r_t_2_mean)

			self.disc_velocities_2 = self.disc_velocities_2 + self.v_offset_mean

			self.disc_dispersions_1 = self.sigma0_1_mean*jnp.ones_like(self.disc_velocities_1)
			self.disc_dispersions_2 = self.sigma0_2_mean*jnp.ones_like(self.disc_velocities_2)

			self.model_disc_model_map_1 = self.grism_object.disperse(self.model_fluxes_1, self.disc_velocities_1, self.disc_dispersions_1)
			self.model_disc_model_map_2 = self.grism_object.disperse(self.model_fluxes_2, self.disc_velocities_2, self.disc_dispersions_2)

			self.model_map = self.model_disc_model_map_1 + self.model_disc_model_map_2

			self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)

			self.model_map1 = resample(self.model_disc_model_map_1, self.y_factor*self.factor, self.wave_factor)
			self.model_map2 = resample(self.model_disc_model_map_2, self.y_factor*self.factor, self.wave_factor)

			self.model_flux = self.model_fluxes_1 + self.model_fluxes_2
			self.fluxes_mean = self.fluxes_1_mean + self.fluxes_2_mean

			return self.model_map, self.model_flux, self.fluxes_mean, self.disc_velocities_1, self.disc_velocities_2,self.disc_dispersions_1 + self.disc_dispersions_2, self.y

		elif model_name == 'unified_two_discs_model' :
			print('unified two-disc model')
			self.data.posterior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.posterior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			self.data.prior['fluxes'].data = norm.ppf(norm.cdf(self.low) + self.data.prior['fluxes'].data*(norm.cdf(self.high)-norm.cdf(self.low)))*self.sigma + self.mu
			fluxes_mean = jnp.array(self.data.posterior['fluxes'].median(dim=["chain", "draw"]))
			f1_mean = jnp.array(self.data.posterior['f1'].median(dim=["chain", "draw"]))

			model_fluxes = oversample(fluxes_mean, self.factor, self.factor)
			model_f1 = oversample(f1_mean, self.factor, self.factor)*self.factor**2

			# define the two flux components
			self.model_fluxes_1 = model_fluxes*model_f1
			self.model_fluxes_2 = model_fluxes*(1-model_f1)

			self.data.posterior['x0_1'].data = norm.ppf(self.data.posterior['x0_1'].data)*2 + self.x0_vel[0]
			self.data.prior['x0_1'].data = norm.ppf(self.data.prior['x0_1'].data)*2 + self.x0_vel[0]

			self.data.posterior['y0_1'].data = norm.ppf(self.data.posterior['y0_1'].data)*2 + self.y0_vel[0]
			self.data.prior['y0_1'].data = norm.ppf(self.data.prior['y0_1'].data)*2 + self.y0_vel[0]

			self.data.posterior['x0_2'].data = norm.ppf(self.data.posterior['x0_2'].data)*2 + self.x0_vel[1]
			self.data.prior['x0_2'].data = norm.ppf(self.data.prior['x0_2'].data)*2 + self.x0_vel[1]

			self.data.posterior['y0_2'].data = norm.ppf(self.data.posterior['y0_2'].data)*2 + self.y0_vel[1]
			self.data.prior['y0_2'].data = norm.ppf(self.data.prior['y0_2'].data)*2 + self.y0_vel[1]

			self.x0_1_mean = jnp.array(self.data.posterior['x0_1'].median(dim=["chain", "draw"]))
			self.y0_1_mean = jnp.array(self.data.posterior['y0_1'].median(dim=["chain", "draw"]))
			self.x0_2_mean = jnp.array(self.data.posterior['x0_2'].median(dim=["chain", "draw"]))
			self.y0_2_mean = jnp.array(self.data.posterior['y0_2'].median(dim=["chain", "draw"]))

			# make the grids for both components
			x_1 = jnp.linspace(0 - self.x0_1_mean,self.grism_object.direct.shape[1]-1 - self.x0_1_mean, self.grism_object.direct.shape[1]*self.factor)
			y_1 = jnp.linspace(0 - self.y0_1_mean,self.grism_object.direct.shape[0]-1 - self.y0_1_mean, self.grism_object.direct.shape[0]*self.factor)
			x_1, y_1 = jnp.meshgrid(x_1, y_1)

			x_2 = jnp.linspace(0 - self.x0_2_mean,self.grism_object.direct.shape[1]-1 - self.x0_2_mean, self.grism_object.direct.shape[1]*self.factor)
			y_2 = jnp.linspace(0 - self.y0_2_mean,self.grism_object.direct.shape[0]-1 - self.y0_2_mean, self.grism_object.direct.shape[0]*self.factor)
			x_2, y_2 = jnp.meshgrid(x_2, y_2)

			# sample the kinematic parameters of the two components
			self.data.posterior['PA_1'].data = norm.ppf(self.data.posterior['PA_1'].data)*self.sigma_PA_1 + self.mu_PA_1
			self.data.prior['PA_1'].data = norm.ppf(self.data.prior['PA_1'].data)*self.sigma_PA_1 + self.mu_PA_1

			self.data.posterior['PA_2'].data = norm.ppf(self.data.posterior['PA_2'].data)*self.sigma_PA_2 + self.mu_PA_2
			self.data.prior['PA_2'].data = norm.ppf(self.data.prior['PA_2'].data)*self.sigma_PA_2 + self.mu_PA_2

			self.data.posterior['i_1'].data = self.data.posterior['i_1'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
			self.data.prior['i_1'].data = self.data.prior['i_1'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

			self.data.posterior['i_2'].data = self.data.posterior['i_2'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
			self.data.prior['i_2'].data = self.data.prior['i_2'].data*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

			self.data.posterior['Va_1'].data = self.data.posterior['Va_1'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
			self.data.prior['Va_1'].data = self.data.prior['Va_1'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

			self.data.posterior['Va_2'].data = self.data.posterior['Va_2'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
			self.data.prior['Va_2'].data = self.data.prior['Va_2'].data*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

			self.data.posterior['r_t_1'].data = self.data.posterior['r_t_1'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
			self.data.prior['r_t_1'].data = self.data.prior['r_t_1'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

			self.data.posterior['r_t_2'].data = self.data.posterior['r_t_2'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]
			self.data.prior['r_t_2'].data = self.data.prior['r_t_2'].data*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

			self.data.posterior['sigma0_1'].data = self.data.posterior['sigma0_1'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			self.data.prior['sigma0_1'].data = self.data.prior['sigma0_1'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

			self.data.posterior['sigma0_2'].data = self.data.posterior['sigma0_2'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
			self.data.prior['sigma0_2'].data = self.data.prior['sigma0_2'].data*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

			self.data.posterior['v_offset'].data = self.data.posterior['v_offset'].data*1000 - 500
			self.data.prior['v_offset'].data = self.data.prior['v_offset'].data*1000 - 500

			self.data.posterior['v_r_1'] =  self.data.posterior['Va_1'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t_1'].data)
			self.data.prior['v_r_1'] = self.data.prior['Va_1'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t_1'].data)
			
			self.data.posterior['v_r_2'] =  self.data.posterior['Va_2'] * (2/pi) * jnp.arctan(2/self.data.posterior['r_t_2'].data)
			self.data.prior['v_r_2'] = self.data.prior['Va_2'] * (2/pi) * jnp.arctan(2/self.data.prior['r_t_2'].data)


			PA_1_mean = jnp.array(self.data.posterior['PA_1'].median(dim=["chain", "draw"]))
			PA_2_mean = jnp.array(self.data.posterior['PA_2'].median(dim=["chain", "draw"]))
			i_1_mean = jnp.array(self.data.posterior['i_1'].median(dim=["chain", "draw"]))
			i_2_mean = jnp.array(self.data.posterior['i_2'].median(dim=["chain", "draw"]))
			Va_1_mean = jnp.array(self.data.posterior['Va_1'].median(dim=["chain", "draw"]))
			Va_2_mean = jnp.array(self.data.posterior['Va_2'].median(dim=["chain", "draw"]))
			r_t_1_mean = jnp.array(self.data.posterior['r_t_1'].median(dim=["chain", "draw"]))
			r_t_2_mean = jnp.array(self.data.posterior['r_t_2'].median(dim=["chain", "draw"]))
			sigma0_1_mean = jnp.array(self.data.posterior['sigma0_1'].median(dim=["chain", "draw"]))
			sigma0_2_mean = jnp.array(self.data.posterior['sigma0_2'].median(dim=["chain", "draw"]))
			v_offset_mean = jnp.array(self.data.posterior['v_offset'].median(dim=["chain", "draw"]))

			# compute the velocities and dispersions of the two components
			model_velocities_1 = v(x_1, y_1, jnp.radians(PA_1_mean),jnp.radians(i_1_mean), Va_1_mean, r_t_1_mean)
			model_velocities_2 = v(x_2, y_2, jnp.radians(PA_2_mean),jnp.radians(i_2_mean), Va_2_mean, r_t_2_mean)
			model_velocities_2 = model_velocities_2 + v_offset_mean

			model_dispersions_1 = sigma0_1_mean*jnp.ones_like(model_velocities_1)
			model_dispersions_2 = sigma0_2_mean*jnp.ones_like(model_velocities_2)

			# compute the model map
			self.model_map = self.grism_object.disperse(self.model_fluxes_1, model_velocities_1, model_dispersions_1) + self.grism_object.disperse(self.model_fluxes_2, model_velocities_2, model_dispersions_2)

			self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)
			return self.model_map, model_fluxes, fluxes_mean, model_velocities_1, model_velocities_2, model_dispersions_1 + model_dispersions_2, self.y
		
		self.model_map = resample(self.model_map, self.y_factor*self.factor, self.wave_factor)
		


	def diverging_parameters(self, chain_number, divergence_number):
		divergences = az.convert_to_dataset(self.data, group="sample_stats").diverging.transpose("chain", "draw")
		PA_div = self.data.posterior['PA'][chain_number,:][divergences[chain_number,:]]
		i_div = self.data.posterior['i'][chain_number,:][divergences[chain_number,:]]
		Va_div = self.data.posterior['Va'][chain_number,:][divergences[chain_number,:]]
		sigma0_div = self.data.posterior['sigma0'][chain_number,:][divergences[chain_number,:]]
		r_t_div = self.data.posterior['r_t'][chain_number,:][divergences[chain_number,:]]
		fluxes_div = self.data.posterior['fluxes'][chain_number,:][divergences[chain_number,:]]

		return jnp.array(fluxes_div[divergence_number].data),jnp.array(PA_div[divergence_number]), jnp.array(i_div[divergence_number]), jnp.array(Va_div[divergence_number]), jnp.array(r_t_div[divergence_number]), jnp.array(sigma0_div[divergence_number])
	

# ---------------------------define the main functions defining an observation of an exponential disk----------


def x_int(x, y, PA, i):
	return x*jnp.cos(PA) - y*jnp.sin(PA)


def y_int(x, y, PA, i):
	return jnp.where(jnp.cos(i) != 0., (x*jnp.sin(PA) + y*jnp.cos(PA))/(jnp.cos(i)), 0.)


def r_int(x, y, PA, i):

	return jnp.sqrt((x_int(x, y, PA, i))**2 + (y_int(x, y, PA, i))**2)


def phi_int(x, y, PA, i):
	return jnp.where(r_int(x, y, PA, i) != 0, jnp.arccos(x_int(x, y, PA, i)/r_int(x, y, PA, i)), 0.)


def v(x, y, PA, i, Va, r_t):
	return (2/pi)*Va*jnp.arctan(r_int(x, y, PA, i)/r_t)*jnp.sin(i)*jnp.cos(phi_int(x, y, PA, i))


def flux(x, y, r_eff, I0, PA, i):
	fluxes  = I0*jnp.exp(-r_int(x, y, PA, i)/r_eff)
	# fluxes = jnp.where(r_int(x, y, PA, i) > r_eff*1.5, 0.0, fluxes)
	return fluxes


def sigma(x, y, sigma0, r_eff=None, I0=None, PA=None, i=None):
	return sigma0

# -----------------------------------------------------------over/re-sampling tools-----------------------------------------------------------------------------------
# figure out where is the best place to store these functions
# maybe in a separate module?? could be with the plotting too so that this one is not so busy?


def oversample(image_low_res, factor, wave_factor, method='nearest'):
	image_high_res = image.resize(image_low_res, (int(image_low_res.shape[0]*factor), int(image_low_res.shape[1]*wave_factor)), method=method)
	image_high_res /= factor*wave_factor

	return image_high_res

def oversample_errors(error_map, factor, wave_factor):
	# Repeat each element in the original array along both dimensions
	repeated_errors = jnp.kron(error_map, np.ones((factor, wave_factor)))

	# Divide each element by the respective oversampling factors
	oversampled_errors = repeated_errors / jnp.sqrt(factor*wave_factor)
	
	return oversampled_errors

def resample_errors(error_map, factor, wave_factor):

	blocks = error_map.reshape((int(error_map.shape[0]/(factor)), factor, int(error_map.shape[1]/wave_factor), wave_factor))
	# resampled_errors = jnp.sqrt(jnp.sum(blocks**2, axis=(1,3)))
	resampled_errors = jnp.linalg.norm(blocks, axis=(1,3))
	return resampled_errors

def resample(grism_spectrum, factor, wave_factor):
	# blocks = grism_spectrum[int(factor/4):int(grism_spectrum.shape[0]-factor/4)].reshape((int(grism_spectrum.shape[0]/(factor)), factor, int(grism_spectrum.shape[1]/wave_factor), wave_factor))
	blocks = grism_spectrum.reshape((int(grism_spectrum.shape[0]/(factor)), factor, int(grism_spectrum.shape[1]/wave_factor), wave_factor))
	grism_obs_res = jnp.sum(blocks, axis=(1,3))
	return grism_obs_res

# -----------------------------------------------------------plotting model at point parameter space-----------------------------------------------------------------------------------

def generate_map(grism_object,fluxes, PA, i, Va, r_t, sigma0, x0, y0, factor = 2, wave_factor = 10, y_factor =1):
	x = jnp.linspace(0 - x0, fluxes.shape[1]- 1 - x0, fluxes.shape[1]*10*factor)
	y = jnp.linspace(0 - y0, fluxes.shape[0]- 1 - y0, fluxes.shape[0]*10*factor)
	x, y = jnp.meshgrid(x, y)

	highdim_flux = oversample(fluxes, factor, factor)

	velocities = jnp.array(v(x, y, jnp.radians(PA), jnp.radians(i), Va, r_t))
	velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='bicubic')

	dispersions = jnp.array(sigma(x, y, sigma0))

	model_map_high = grism_object.disperse(highdim_flux, velocities, dispersions)

	generated_map = resample(model_map_high, y_factor*factor,wave_factor)

	return generated_map


# -----------------------------------------------------------matching output (F,V,sigma) to an IFU-----------------------------------------------------------------------------------

def match_to_IFU(F1, V1, D1, F2 , V2, D2, full_PSF, FWHM_PSF, R_LSF, wavelength, IFU = 'kmos'):
	# if there is no second component than you should input zero arrays in the F2,V2,D2
	# create a 3D cube with the model best fit, basically with f[i,j,k]=F[i,j]*gaussian(k)
		# make the gaussian with stats.norm.pdf(velocity_space, v[i,j],sigma_v[i,j])
	velocity_space = jnp.linspace(-1000, 1000, 2001)
	broadcast_velocity_space = np.broadcast_to(velocity_space[:, np.newaxis,np.newaxis],(velocity_space.size,F1.shape[0],F1.shape[0]))
	if F2 is not None:
		cube = F1*stats.norm.pdf(broadcast_velocity_space, V1, D1) + F2*stats.norm.pdf(broadcast_velocity_space, V2, D2)
	else:
		cube = F1*stats.norm.pdf(broadcast_velocity_space, V1, D1)

	print('cube created')
	# if no full PSF given, use the FWHM to create a 2D gaussian kernel
	if full_PSF is None:
		FWHM_PSF_pixels = FWHM_PSF/0.0629
		sigma_PSF = FWHM_PSF_pixels/(2*np.sqrt(2*np.log(2)))
		full_PSF = np.array( Gaussian2DKernel(sigma_PSF))
	print('PSF created')
	# create the LSF kernel - assuming a constant one for now since only emission around the same wavelength
	# compute the std for the velocities from the spectral resolution
	sigma_LSF = wavelength/(2.355*R_LSF) 
	sigma_LSF_v = (c/1000)*sigma_LSF/wavelength
	LSF = np.array( Gaussian1DKernel(sigma_LSF_v))
	print('LSF created')
	# create a 3D kernel with 2D PSF and 1D LSF
	full_kernel = np.array(full_PSF) * np.broadcast_to(np.array(LSF)[:, np.newaxis,np.newaxis],(np.array(LSF).size,np.array(full_PSF).shape[0],np.array(full_PSF).shape[0]))
	print('kernel created')
	# convolve the cube with the kernel
	print('convolving cube')
	convolved_cube = signal.fftconvolve(cube, full_kernel, mode='same')
	print('convolution done')
	# from the convolved cube, obtain the flux, velocity and dispersion maps
	F_kmos = np.sum(convolved_cube, axis=0)
	# fit a gaussian to the convolved cube to obtain the velocity and dispersion maps
	V_kmos = np.zeros_like(F_kmos)
	D_kmos = np.zeros_like(F_kmos)

	def gauss(x, A, mu, sigma):
		return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)*A

	for i in range(V_kmos.shape[0]):
		for j in range(V_kmos.shape[1]):
			popt, pcov = curve_fit(gauss, velocity_space, convolved_cube[:,i,j], p0=[F_kmos[i,j],V1[i,j], D1[i,j]])
			V_kmos[i,j] = popt[1]
			D_kmos[i,j] = popt[2]
	# popt, pcov = curve_fit(gauss, velocity_space, convolved_cube, p0=[F_kmos,V1, D1])
	# V_kmos = popt[1]
	# D_kmos = popt[2]

	# finally resample to IFU pixel scale
	IFU_scale_arcseconds = 0.1799
	NC_LW_scale_arcseconds = 0.0629
	# rescale the maps to the IFU pixel scale

	if IFU == 'kmos':
		IFU_scale_pixels = int(IFU_scale_arcseconds/NC_LW_scale_arcseconds)
		# blocks = F_kmos.reshape((int(F_kmos.shape[0]/(IFU_scale_pixels)), IFU_scale_pixels, int(F_kmos.shape[1]/IFU_scale_pixels), IFU_scale_pixels))
		# F_kmos = jnp.sum(blocks, axis=(1,3))
		F_kmos = image.resize(F_kmos, (int(F_kmos.shape[0]/IFU_scale_pixels), int(F_kmos.shape[1]/IFU_scale_pixels)), method='nearest')
		F_kmos*= IFU_scale_pixels**2
		V_kmos = image.resize(V_kmos, (int(V_kmos.shape[0]/IFU_scale_pixels), int(V_kmos.shape[1]/IFU_scale_pixels)), method='bicubic')
		D_kmos = image.resize(D_kmos, (int(D_kmos.shape[0]/IFU_scale_pixels), int(D_kmos.shape[1]/IFU_scale_pixels)), method='bicubic')

		print('rescaling done')
	return F_kmos, V_kmos, D_kmos
# -----------------------------------------------------------running the inference-----------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--choice', type=str, default='model', help='model or real')
parser.add_argument('--output', type=str, default='', help='output folder name')

if __name__ == "__main__":

	args = parser.parse_args()
	choice = args.choice
	output = args.output + '/'

	if choice == 'model':
		print('Running model')
		with open('fitting_results/' + output+'config_model.yaml', 'r') as file:
			input = yaml.load(file, Loader=yaml.FullLoader)
			print('Read inputs successfully')
		# ----------------------------------set fixed galaxy properties (PA and inclination in degrees)-----------------
		params = input[0]
		inference = input[1]
		priors = input[2]

		factor = params['Params']['factor']
		wave_factor = params['Params']['wave_factor']
		y_factor = params['Params']['y_factor']

		Pa = params['Params']['Pa']
		i = params['Params']['i']
		PA_truth = jnp.radians(Pa)
		i_truth = jnp.radians(i)

		Va_truth = params['Params']['Va_truth']
		I0 = params['Params']['I0']
		r_t_truth = params['Params']['r_t_truth']
		r_eff = params['Params']['r_eff']
		sigma0 = params['Params']['sigma0']
		redshift = params['Params']['redshift']

		direct_image_size = params['Params']['direct_image_size']
		x0 = params['Params']['x0']
		y0 = params['Params']['y0']

		print(factor, wave_factor, y_factor)

		# set the size of the grid
		# x = jnp.linspace(0 - x0, direct_image_size-1 -x0, direct_image_size*factor*10)
		# y = jnp.linspace(0 - y0, direct_image_size-1 -y0, direct_image_size*factor*10)
		# X_high, Y_high = jnp.meshgrid(x, y)
		# x = jnp.linspace(0 - 7, 14-1 -7, 14*10)
		# y = jnp.linspace(0 - 7, 14-1 -7, 14*10)
		# x = jnp.linspace(0 - 8, 16-1 -8, 16*10)
		# y = jnp.linspace(0 - 8, 16-1 -8, 16*10)
		x = jnp.linspace(0 - x0, direct_image_size-1 - x0, direct_image_size*factor)
		y = jnp.linspace(0 - y0, direct_image_size-1 - y0, direct_image_size*factor)
		X_high, Y_high = jnp.meshgrid(x, y)

		x = jnp.linspace(0 - x0, direct_image_size-1 -x0, direct_image_size*factor*10)
		y = jnp.linspace(0 - y0, direct_image_size-1 -y0, direct_image_size*factor*10)
		X_vel, Y_vel = jnp.meshgrid(x, y)

		x = jnp.linspace(0 - x0, direct_image_size-1-x0, direct_image_size)
		y = jnp.linspace(0 - y0, direct_image_size-1-y0, direct_image_size)
		X, Y = jnp.meshgrid(x, y)

		truth_flux = jnp.array(flux(X, Y, r_eff, I0, PA_truth, i_truth))
		flux_model = Sersic2D(amplitude=100, r_eff=r_eff, n=1, x_0=0, y_0=0,
               ellip=1-math.cos(math.radians(i)), theta=math.pi - math.radians(Pa))
		# flux_model = Sersic2D(amplitude=I0, r_eff=r_eff, n=4, x_0=x0, y_0=y0, ellip=(1-jnp.cos(i_truth)), theta=PA_truth)
		truth_flux = flux_model(X, Y)

		# ===========================MAKE GAL IMAGE WITH GALSIM===========================
		# truth_flux = galsim.InclinedSersic(n=0.5, inclination = galsim.Angle(i_truth,galsim.AngleUnit(1)) , half_light_radius=r_eff*0.0629, flux=I0).drawImage(nx=direct_image_size, ny=direct_image_size, scale=0.0629).array
		#rotate the galaxy by angle PA
		# truth_flux = truth_flux.rotate(PA_truth, galsim.degrees)
		# ================================================================================

# ======================= ADDING ZERO FLUX BORDERS TO INCREASE SIZE ======================


		# truth_flux = jnp.pad(truth_flux, pad_width=7, mode='constant', constant_values=0)
		# plt.imshow(truth_flux, origin = 'lower')
		# plt.show()
		# direct_image_size += 14
		# x0 += 7
		# y0 += 7

		# x = jnp.linspace(0 - x0, direct_image_size-1 - x0, direct_image_size*factor)
		# y = jnp.linspace(0 - y0, direct_image_size-1 - y0, direct_image_size*factor)
		# X_high, Y_high = jnp.meshgrid(x, y)

		# x = jnp.linspace(0 - x0, direct_image_size-1 -x0, direct_image_size*factor*10)
		# y = jnp.linspace(0 - y0, direct_image_size-1 -y0, direct_image_size*factor*10)
		# X_vel, Y_vel = jnp.meshgrid(x, y)

		# x = jnp.linspace(0 - x0, direct_image_size-1-x0, direct_image_size)
		# y = jnp.linspace(0 - y0, direct_image_size-1-y0, direct_image_size)
		# X, Y = jnp.meshgrid(x, y)


#==========================================================================================


		# truth_flux_high = jnp.array(
		# 	flux(X_high, Y_high, r_eff, I0, PA_truth, i_truth))
		# truth_flux_high /= factor**2

		# truth_flux_high = oversample(truth_flux, factor, factor)
		# print(truth_flux_high.shape, truth_flux.shape)
		# truth_flux_high = truth_flux

		# truth_flux_high *= truth_flux.sum()/truth_flux_high.sum()

		plotting.plot_image(truth_flux, x0, y0, direct_image_size)

		len_wave = int((5.068-3.835)/(0.001/wave_factor))
		wave_space = jnp.linspace(3.835, 5.068, len_wave+1)

		lower_index = params['Params']['lower_index']
		upper_index = params['Params']['upper_index']

		snr_tot = params['Params']['snr_tot']
		snr_obs = params['Params']['snr_obs']
		snr_flux = params['Params']['snr_flux']

		grism_object = model.Grism(direct=truth_flux, direct_scale=0.0629/y_factor, factor=factor, icenter=y0, jcenter=x0, segmentation=None,
								   xcenter_detector=1024, ycenter_detector=1024, wavelength=4.2, redshift=redshift,
								   wave_space=wave_space, wave_scale=0.001/wave_factor, wave_factor = wave_factor, index_min=int(lower_index*wave_factor), index_max=int(upper_index*wave_factor),
								   grism_filter='F444W', grism_module='A', grism_pupil='R')

		truth_velocities = jnp.array(v(X_vel, Y_vel, PA_truth, i_truth, Va_truth, r_t_truth))
		# truth_velocities = Va_truth*jnp.ones_like(X_vel)
		truth_velocities = image.resize(truth_velocities, (int(truth_velocities.shape[0]/10), int(truth_velocities.shape[1]/10)), method='nearest')
		# print(truth_velocities[28])
		truth_dispersions = sigma0*jnp.ones_like(truth_velocities)
		key = jax.random.PRNGKey(5)

		fluxes_errors = jnp.max(truth_flux)/100*jnp.ones_like(truth_flux)

		# truth_velocities = jnp.where(truth_flux/fluxes_errors<snr_flux, 0.0, truth_velocities)
		# truth_dispersions = jnp.where(truth_flux/fluxes_errors<snr_flux , 0.0, truth_dispersions)
		# truth_flux_high = jnp.where(truth_flux/fluxes_errors<snr_flux, 0.0, truth_flux)

		#uncomment this for MLE
		# truth_flux = truth_flux_high #+ jax.random.normal(key, truth_flux_high.shape)*fluxes_errors

		model_name = inference['Inference']['model']

		grism_full_high = grism_object.disperse(truth_flux, truth_velocities, truth_dispersions)
		
		grism_full = resample(grism_full_high, y_factor*factor, wave_factor)

		truth_map = grism_full


		
		error = jnp.max(grism_full)/snr_tot #+ jnp.sqrt(grism_full)/5


		# error = jnp.sum(truth_map)/snr_tot
		# error = (grism_full.sum()/2000)
		# error_high = error/math.sqrt(2*factor*wave_factor)
		# print(grism_full.max()/error, error)
		# pixel_inv_var = (1/((error_high)**2))


		random_noise = (jnp.ones_like(grism_full)*(error)) *jax.random.normal(key, truth_map.shape)
		obs_map = jnp.array(truth_map+ random_noise)
		# obs_map = jnp.where(obs_map<0, 0, obs_map)
		delta_wave2 = int((5.068-3.835)/(0.001))
		wave_space2 = jnp.linspace(3.835, 5.068, delta_wave2+1)
		wave_axis2 = wave_space2[lower_index:upper_index+1]
		# plot_grism(truth_map, y0, direct_image_size, wave_axis2)
		print('Cond number:', jnp.linalg.cond(obs_map))
		
		# obs_map_high = oversample(obs_map, grism_full_high.shape[0]/(grism_full.shape[0]), 1,method = 'nearest')
		# kernel = Box2DKernel(2)
		# obs_map_high = convolve(obs_map_high, kernel)
		# obs_map_high = jnp.where(obs_map_high<0, 0, obs_map_high)
		obs_map_high = obs_map
		
		# print(obs_map.sum(), obs_map_high.sum())
		# obs_map_high = grism_full_high + (jnp.ones_like(grism_full_high)*(error_high) + (grism_full_high/10)) * jax.random.normal(key, grism_full_high.shape)
		# obs_map_high = grism_full_high + (jnp.ones_like(grism_full_high)*(error_high)) * jax.random.normal(key, grism_full_high.shape)
		# obs_map_high = grism_full_high
		# obs_map = jnp.array(truth_map)

		error_map = jnp.ones_like(grism_full)*(error) #- (grism_full/50)
		obs_map = jnp.where(jnp.abs(obs_map/error_map) <snr_obs, 0.0, obs_map)
		# obs_map_high = obs_map
		error_map_high = error_map
		plotting.plot_grism(obs_map, y0, direct_image_size, wave_axis2)
		plotting.plot_grism(random_noise, y0, direct_image_size, wave_axis2)

		# print(truth_map[28], obs_map[28], error_map[28])

		# plot_grism(obs_map[7:-7,:], y0-7, direct_image_size-14, wave_axis2)
		# error_map = jnp.where(obs_map<obs_map.max()*0.5, error_map*10, error_map)
		# error_map_high = jnp.ones_like(grism_full_high)*(error_high) + (grism_full_high/10)
		# error_map_high = oversample_errors(error_map, y_factor*factor, 1)
		error_map = jnp.where(jnp.abs(obs_map/error_map) <snr_obs, error_map*1e6, error_map)
		# error_map_high = error_map

		var_map_high = error_map_high**2
		# print(error_map_high[28])
		# print(obs_map_high[28])

		# var_map_high = (jnp.ones_like(obs_map_high)*(error_high))**2
		# flux_mask = jnp.where(truth_flux/fluxes_errors < snr_flux, 0.0, truth_flux)
		# run_fit.plot_image(flux_mask, x0, y0, direct_image_size)
		# cov_matrix_obs = jnp.diag(jnp.ones(obs_map_high.shape[1])*error_high**2)
		cov_matrix_full = np.zeros((obs_map_high.shape[0],obs_map_high.shape[1],obs_map_high.shape[1]))
		for i in range(obs_map_high.shape[0]):
			cov_matrix_full[i] = jnp.diag(var_map_high[i])
		# print(fluxes_errors)
		# error_map = jnp.ones_like(grism_full)*(error)
		# plot_grism(obs_map_high, y0, direct_image_size, wave_space[int(349*wave_factor):int((382 +1)*wave_factor)])
	
		# print('Condition number:' , jnp.linalg.cond(cov_matrix_full))
		# make sure the MLE function is working okay:

		error_scaling_low = inference['Inference']['error_scaling_low']
		alpha = inference['Inference']['alpha']

		if model_name == 'MLE':
			run_fit = Fit_Numpyro(obs_map = obs_map_high, obs_error= error_map_high, obs_map_low = obs_map, obs_error_low = error_map, grism_object = grism_object, factor =factor, wave_factor = wave_factor,flux_prior=truth_flux, cov_matrix = cov_matrix_full, fluxes_errors = fluxes_errors, snr_flux = snr_flux, error_scaling_low = error_scaling_low, alpha=alpha)
		else:
			run_fit = Fit_Numpyro(obs_map = obs_map, obs_error = jnp.ones_like(obs_map)*error, grism_object = grism_object, factor = factor, wave_factor = wave_factor,flux_prior=truth_flux_high)

		# alpha = 0.05
		# truth_dispersions = 20*jnp.ones_like(truth_velocities)

		# ===========================================TESTING THE MLE====================================================================================
		fluxes, fluxes_low, gaussian, flux_uncertainty = run_fit.compute_MLE(obs_map_high, cov_matrix_full, truth_velocities, truth_dispersions, factor, alpha, wave_factor)


		model_map = jnp.matmul( fluxes[:,None, :] , gaussian)[:,0,:]
		#grism_object.disperse(fluxes, truth_velocities, truth_dispersions)
		plotting.plot_image(fluxes, x0, y0, direct_image_size)
		# plt.imshow(truth_flux/fluxes_errors,origin = 'lower')
		# plt.show()
		# plot_grism_residual(obs_map, model_map, error_map, y0, direct_image_size, wave_space[int(lower_index*wave_factor):int((upper_index +1)*wave_factor)])
		# fluxes_errors = jnp.where(jnp.abs(truth_flux/fluxes_errors) < snr_flux, 1e6*fluxes_errors , fluxes_errors)
		plotting.plot_image_residual(truth_flux, fluxes, fluxes_errors*error_scaling_low, x0, y0, direct_image_size)
		delta_wave = int((5.068-3.835)/(0.001))
		wave_space = jnp.linspace(3.835, 5.068, delta_wave+1)
		wave_axis = wave_space[lower_index:upper_index+1]
		run_fit.wave_axis = wave_axis

		plotting.plot_grism_residual(obs_map, resample(model_map,y_factor*factor,1), error_map, 14, obs_map.shape[0],wave_axis )
		 #sending with obs_map = obs_map_high bc i can't get the resampling to work correctly
		# fluxes = jnp.where(fluxes<0, 0, fluxes)
		# resample fluxes to fluxes prior shape

		# plt.imshow(fluxes_low, origin='lower', vmin = truth_flux.min(), vmax = truth_flux.max())
		# plt.title('MLE fluxes test')
		# plt.show()

		# fluxes_errors = jnp.where(truth_flux < 0.03*truth_flux.max(), 100*truth_flux , 0.1*truth_flux)
		# fluxes_errors = truth_flux.sum()/1500 #/450

		# plot_image_residual(truth_flux, fluxes_low, fluxes_errors, x0, y0, direct_image_size)

		# plot_image(truth_flux, x0, y0, direct_image_size)

		# plot_image_residual(truth_flux_high, fluxes, fluxes_errors/2, x0, y0, direct_image_size)

		# =================================================================================================================================
		print(run_fit)
		# setting all of the bounds for the fit
		flux_bounds = inference['Inference']['flux_bounds']
		flux_type = inference['Inference']['flux_type']
		PA_bounds = [Pa, inference['Inference']['PA_bounds']]
		PA_normal = inference['Inference']['PA_normal']
		i_bounds = inference['Inference']['i_bounds']
		Va_bounds = inference['Inference']['Va_bounds']
		r_t_bounds = inference['Inference']['r_t_bounds']
		sigma0_mean = inference['Inference']['sigma0_mean']
		sigma0_disp = inference['Inference']['sigma0_disp']
		sigma0_bounds = inference['Inference']['sigma0_bounds']
		obs_map_bounds = inference['Inference']['obs_map_bounds']
		y_factor = params['Params']['y_factor']
		# flux_threshold = inference['Inference']['flux_threshold']
		#get the prior for the width of the delta_V
		# kernel = Box2DKernel(20)
		# result = convolve(truth_flux, kernel)
		# threshold = 0.4*result.max()
		# clump = (truth_flux-result)
		# # clump[jnp.where(clump<threshold)] = 0
		# clump = clump.at[jnp.where(clump<threshold)].set(0)
		# x = jnp.linspace(0 - x0, direct_image_size-1 -
		# 				 x0, direct_image_size)
		# y = jnp.linspace(0 - y0, direct_image_size-1 -
		# 				 y0, direct_image_size)
		# X, Y = jnp.meshgrid(x, y)
		# #scale with radius
		# clump*= (X**2+Y**2)
		# #normalize the clump image so that the flux is between 0 and 1
		# clump = (clump-clump.min())/(clump.max()-clump.min())
		# # plt.imshow(clump, origin = 'lower')
		# # plt.show()]

		# if the boolean is False, then we don't want to fit for the clump so we input None in the set_bounds
		if not inference['Inference']['clump_bool']:
			clump = None

		# fluxes_errors = jnp.max(truth_flux)/50*jnp.ones_like(truth_flux) 
		# truth_flux += fluxes_errors*jax.random.normal(key, truth_flux.shape)
		# fluxes_errors = jnp.where(truth_flux/fluxes_errors <snr_flux, fluxes_errors*100, fluxes_errors)

		# plot_image(truth_flux, x0, y0, direct_image_size)


		run_fit.set_bounds(model = model_name, flux_prior=truth_flux, y_factor = y_factor, flux_bounds=flux_bounds,flux_type = flux_type ,PA_bounds=PA_bounds, 
		     i_bounds=i_bounds, Va_bounds=Va_bounds, r_t_bounds=r_t_bounds, sigma0_bounds=sigma0_bounds, obs_map_bounds=obs_map_bounds, clump=clump, sigma0_disp=sigma0_disp, sigma0_mean=sigma0_mean, PA_normal=PA_normal)


		rng_key = random.PRNGKey(0)
		if model_name == 'two_component_model':
			print('Running the 2 component model')
			prior_predictive= Predictive(run_fit.two_component_model, num_samples=5000)
		elif model_name == 'one_component_model': 
			print('Running the full parametric model')
			prior_predictive = Predictive(run_fit.full_parametric_renormalized, num_samples=5000)
		elif model_name == 'MLE': 
			print('Running the MLE model')
			prior_predictive = Predictive(run_fit.one_comp_MLE, num_samples=5000)
		prior = prior_predictive(rng_key)
		# plt.hist(norm.ppf(prior['sigma0'])*run_fit.sigma_sigma0 + run_fit.mu_sigma0, bins = 50)
		# plt.show()
		# print(prior['fluxes'])
		# prior['fluxes'] = norm.ppf(norm.cdf(run_fit.low) + prior['fluxes']*(norm.cdf(run_fit.high)-norm.cdf(run_fit.low)))*run_fit.sigma + run_fit.mu
		# plt.hist(prior['fluxes'][:,0,7], bins = 50)
		# plt.show()

		
		num_samples = inference['Inference']['num_samples']
		num_warmup = inference['Inference']['num_warmup']
	
		step_size = inference['Inference']['step_size']
		target_accept_prob = inference['Inference']['target_accept_prob']

		run_fit.run_inference(model_name = model_name, num_samples=num_samples, num_warmup=num_warmup, high_res=True, median=True, step_size=step_size, adapt_step_size=True, target_accept_prob=target_accept_prob,  num_chains=2, max_tree_depth=10)

		# rng_key = random.PRNGKey(0)
		# prior_predictive = Predictive(run_fit.two_component_model, num_samples=num_samples)

		inf_data = az.from_numpyro(run_fit.mcmc, prior=prior_predictive(rng_key))


		# inf_data.to_netcdf('fitting_results/'+ output +str(num_samples)+'_'+ str(num_warmup)+'_'+ str(factor) + '_'+ str(wave_factor) + '_' + str(Pa)+'_'+str(i)+'_' + str(Va_truth) + '_' + str(r_t_truth) + '_' + str(sigma0) + '_'
		#     + str(flux_bounds[0]) + '_' + str(flux_bounds[1]) + '_' + str(PA_bounds[0]) + '_' + str(PA_bounds[1]) + '_' + str(i_bounds[0]) + '_' + str(i_bounds[1]) + '_' + str(Va_bounds[0]) + '_' + str(Va_bounds[1]) + '_' + str(r_t_bounds[0]) + '_' + str(r_t_bounds[1]) + '_' + str(sigma0_bounds[0]) + '_' + str(sigma0_bounds[1]) + '_' +  str(obs_map_bounds[0]) + '_' +  str(obs_map_bounds[1]) + '_' +str(int(delta_V_bool)) + '_'
		# 	+ str(snr_tot) + '_' + str(step_size) + '_' + str(target_accept_prob))

		inf_data.to_netcdf('fitting_results/'+ output + 'output')
		


	if choice == 'real':
		print('Running the real data')
		#no ../ because the open() function reads from terminal directory (not module directory)
		with open('fitting_results/' + output+'config_real.yaml', 'r') as file:
			input = yaml.load(file, Loader=yaml.FullLoader)
			print('Read inputs successfully')
		
		#load of all the parameters from the configuration file
		data, params, inference, priors, ID, broad_filter, med_filter, med_band_path, broad_band_path, \
			grism_spectrum_path, field, wavelength, redshift, line, y_factor, res, to_mask, flux_threshold, factor, \
			wave_factor, x0, y0, x0_vel, y0_vel, model_name, flux_bounds, flux_type, PA_normal, i_bounds, Va_bounds, \
			r_t_bounds, sigma0_bounds, sigma0_mean, sigma0_disp, obs_map_bounds, clump_v_prior, clump_sigma_prior, \
			clump_flux_prior, clump_bool, num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff = pre.read_config_file(input, output)

		#preprocess the images and the grism spectrum
		obs_map, obs_error, direct, PA_truth, xcenter_detector, ycenter_detector, icenter, jcenter, \
		wave_space, delta_wave, index_min, index_max , factor = \
		model.preprocess_test(med_band_path, broad_band_path, grism_spectrum_path, redshift, line, wavelength, delta_wave_cutoff, field, res)


		#  ==============================================ALT ==============================================
		# data_path = 'fitting_results/' + output + data['Data']['data_path']
		# catalog_path = 'fitting_results/' + output+ data['Data']['catalog_path']
		# ID = data['Data']['ID']

		# catalog = fits.open(catalog_path)
		# catalog_table = Table(catalog[1].data)

		# RA = catalog_table[catalog_table['NUMBER'] == ID]['RA'][0]
		# DEC = catalog_table[catalog_table['NUMBER'] == ID]['DEC'][0]
		# redshift = catalog_table[catalog_table['NUMBER'] == ID]['Z_IF_HALPHA'][0]
		# x_center = catalog_table[catalog_table['NUMBER'] == ID]['X_IMAGE_LINE'][0]
		# wavelength = catalog_table[catalog_table['NUMBER'] == ID]['WAVELENGTH_LINE'][0]/1e4

		# delta_wave_cutoff=params['Params']['delta_wave_cutoff']
		# box_size=params['Params']['box_size']
		

		# obs_map, error_map, LW_image, PA_truth, icenter, jcenter, wave_space, delta_wave, index_min, index_max = model.preprocess_ALT(data_path, RA, DEC, x_center, delta_wave_cutoff, box_size, 2)

		# obs_map = jnp.array(obs_map)
		# obs_error = jnp.array(error_map)

		# direct = LW_image[10:41,10:41]
		# obs_map = obs_map[10:41,:]
		# obs_error = obs_error[10:41,:]

		# icenter,jcenter = jnp.unravel_index(jnp.argmax(direct), direct.shape)

		#  ================================================================================================

		#mask any hot or dead pixels, setting tolerance = 4 manually 
		obs_map, obs_error = pre.mask_bad_pixels(obs_map, obs_error)


		#i think this block is useless
		reshape = params['Params']['reshape']
		new_bounds_direct = params['Params']['new_bounds_direct']
		new_bounds_obs = params['Params']['new_bounds_obs']
		if reshape == True:
			direct = direct[new_bounds_direct[0]:new_bounds_direct[1], new_bounds_direct[2]:new_bounds_direct[3]]
			obs_map = obs_map[new_bounds_obs[0]:new_bounds_obs[1], new_bounds_obs[2]:new_bounds_obs[3]]
			obs_error = obs_error[new_bounds_obs[0]:new_bounds_obs[1], new_bounds_obs[2]:new_bounds_obs[3]]


		#renormalizing flux prior to EL map
		direct, normalization_factor, mask, mask_grism = pre.renormalize_image(direct, obs_map, flux_threshold, y_factor)

		# rescale the wave_space array and the direct image according to factor and wave_factor
		len_wave = int(
			(wave_space[len(wave_space)-1]-wave_space[0])/(delta_wave/wave_factor))
		wave_space = jnp.linspace(
			wave_space[0], wave_space[len(wave_space)-1], len_wave+1)
		
		#take x0 and y0 from the pre-processing unless specified otherwise in the config file
		if x0 == None:
			x0 = jcenter
		if y0 == None:
			y0 = icenter

		# if model_name == 'two_disc_model' or model_name == 'unified_two_discs_model':
		# 	x0_grism = x0[0]
		# 	y0_grism = y0[0]
		# 	print(x0_grism, y0_grism)
		# else:
		# 	x0_grism = x0
		# 	y0_grism = y0
		x0_grism = x0
		y0_grism = y0

		#initialize grism object
		grism_object = model.Grism(direct=direct, direct_scale=0.0629/y_factor, icenter=y0_grism, jcenter=x0_grism, segmentation=None, factor=factor,
								   xcenter_detector=xcenter_detector, ycenter_detector=ycenter_detector, wavelength=wavelength, redshift=redshift,
								   wave_space=wave_space, wave_factor=wave_factor, wave_scale=delta_wave/wave_factor, index_min=(index_min)*wave_factor, index_max=(index_max)*wave_factor,
								   grism_filter=broad_filter, grism_module='A', grism_pupil='R')

		# test the dispersion

		direct_image_size = direct.shape
		# # # print(direct_image_size)


		# x = jnp.linspace(0-12,25-13,26*4)
		# y = jnp.linspace(0-13,25-13,26*4)
		# X,Y = jnp.meshgrid(x,y)

		# extent = -13, 12, -13, 12
		# plt.imshow(direct, origin = 'lower', extent = extent, cmap = 'binary')
		# CS = plt.contour(X,Y,mock_velocities, 7, cmap = 'RdBu', origin = 'lower')
		# cbar =plt.colorbar(CS)
		# cbar.ax.set_ylabel('velocity [km/s]')
		# cbar.add_lines(CS)
		# plt.show()
		# print(obs_map.shape, resampled_EL_line.shape)
		# print('flux difference: ' , obs_map.sum() - EL_line.sum())

		#get the prior for the width of the delta_V
		kernel = Box2DKernel(20)
		result = convolve(direct, kernel)
		threshold = 0.5*result.max()
		clump = jnp.array(direct-result)
		clump = clump.at[jnp.where(clump<threshold)].set(0)
		x = jnp.linspace(0 - x0_grism, direct_image_size[1]-1 -
						 x0_grism, direct_image_size[1])
		y = jnp.linspace(0 - y0_grism, direct_image_size[0]-1 -
						 y0_grism, direct_image_size[0])
		X, Y = jnp.meshgrid(x, y)
		#scale with radius
		clump*= (X**2+Y**2)
		#normalize the clump image so that the flux is between 0 and 1
		clump = (clump-clump.min())/(clump.max()-clump.min())
		# plt.imshow(clump, origin = 'lower')
		# plt.show()


		# ----------------------------------------------------------running the inference------------------------------------------------------------------------
		
		# obs_map_high = oversample(obs_map, y_factor*factor, wave_factor,method = 'nearest')
		# obs_error_high = oversample_errors(obs_error, y_factor*factor, wave_factor)
		# obs_error = obs_error + obs_map.max()*0.001/np.abs(obs_map)
		# obs_error = obs_error - (obs_map/5)
		var_map_high = obs_error**2

		# var_map_high = (jnp.ones_like(obs_map_high)*(error_high))**2

		# cov_matrix_obs = jnp.diag(jnp.ones(obs_map_high.shape[1])*error_high**2)
		cov_matrix_full = np.zeros((obs_map.shape[0],obs_map.shape[1],obs_map.shape[1]))
		for i in range(obs_map.shape[0]):
			cov_matrix_full[i] = jnp.diag(var_map_high[i])

		# plt.imshow(obs_map, origin='lower')
		# plt.title('obs_map input to Fit_Numpyro')
		# plt.show()

		# if model_name == 'MLE':
		
		run_fit = Fit_Numpyro(obs_map = obs_map, obs_error = obs_error, grism_object = grism_object,
							  flux_prior=direct, factor=factor, wave_factor=wave_factor, x0 = x0_grism, y0 = y0_grism, cov_matrix=cov_matrix_full, x0_vel=x0_vel, y0_vel = y0_vel,wavelength = wavelength)

		if not clump_bool:
			clump = None
		
		# # plot the masked direct image 
		# direct_image_size = direct.shape[1]
		# x = jnp.linspace(0 - x0, direct.shape[1]-1 -x0, direct.shape[1])
		# y = jnp.linspace(0 - y0, direct.shape[0]-1 -y0, direct.shape[0])
		# X, Y = jnp.meshgrid(x, y)
		# fig, ax = plt.subplots(figsize = (8,6))
		# cp = ax.pcolormesh(X,Y,direct,shading= 'nearest') #RdBu
		# cbar = fig.colorbar(cp)
		# plt.show()

		# run_fit.obs_map_low = obs_map
		# run_fit.obs_error_low = obs_error

		if model_name == 'two_disc_model' or model_name == 'unified_two_discs_model':
			# split the direct image into 2 regions
			#threshold image
			threshold = threshold_otsu(direct)*4
			bw = closing(direct > threshold)
			cleared = clear_border(bw)
			#label image regions
			label_image = label(cleared)
			regions = regionprops(label_image, direct)

			#only continue if there are 2 or more regions:
			if len(regions) < 2:
				print('Not enough regions found')
			else:
				PA_1 = 90 + regions[0].orientation * (180/np.pi)
				PA_2 = 90 + regions[1].orientation * (180/np.pi)
				mask_1 = label_image == regions[0].label
				mask_2 = label_image == regions[1].label
				# dilate the masks 
				mask_1 = dilation(mask_1, disk(2.5))
				mask_2 = dilation(mask_2, disk(2.5))
				# mask_1 = dilation(mask_1)
				# mask_2 = dilation(mask_2)
				plt.imshow(mask_1, origin = 'lower')
				plt.show()
				plt.imshow(mask_2, origin = 'lower')
				plt.show()
				plt.imshow(jnp.where((mask_1) == 0, 0.0, direct), origin = 'lower')
				plt.show()
				plt.imshow(jnp.where((mask_2) == 0, 0.0, direct), origin = 'lower')
				plt.show()
				PA_bounds = [[PA_1-inference['Inference']['PA_bounds'][0], inference['Inference']['PA_bounds'][1]], [PA_2-inference['Inference']['PA_bounds'][0], inference['Inference']['PA_bounds'][1]]]
			if model_name == 'unified_two_discs_model':
				run_fit.set_bounds(model= model_name, flux_prior=direct, mask = mask, y_factor = y_factor, flux_bounds=flux_bounds, flux_type = flux_type, PA_bounds=PA_bounds, i_bounds=i_bounds, Va_bounds=Va_bounds, r_t_bounds=r_t_bounds, sigma0_bounds=sigma0_bounds, obs_map_bounds = obs_map_bounds, clump =clump, sigma0_disp=sigma0_disp, sigma0_mean=sigma0_mean, PA_normal=PA_normal, clump_v_prior=clump_v_prior, clump_sigma_prior=clump_sigma_prior, clump_flux_prior=clump_flux_prior)
			else:
				run_fit.set_bounds(model= model_name, flux_prior=direct, mask = [mask_1,mask_2], y_factor = y_factor, flux_bounds=flux_bounds, flux_type = flux_type, PA_bounds=PA_bounds, i_bounds=i_bounds, Va_bounds=Va_bounds, r_t_bounds=r_t_bounds, sigma0_bounds=sigma0_bounds, obs_map_bounds = obs_map_bounds, clump =clump, sigma0_disp=sigma0_disp, sigma0_mean=sigma0_mean, PA_normal=PA_normal, clump_v_prior=clump_v_prior, clump_sigma_prior=clump_sigma_prior, clump_flux_prior=clump_flux_prior)

		else:
			PA_bounds = [PA_truth-inference['Inference']['PA_bounds'][0], inference['Inference']['PA_bounds'][1]]
			run_fit.set_bounds(model= model_name, flux_prior=direct, mask = mask, y_factor = y_factor, flux_bounds=flux_bounds, flux_type = flux_type, PA_bounds=PA_bounds, i_bounds=i_bounds, Va_bounds=Va_bounds, r_t_bounds=r_t_bounds, sigma0_bounds=sigma0_bounds, obs_map_bounds = obs_map_bounds, clump =clump, sigma0_disp=sigma0_disp, sigma0_mean=sigma0_mean, PA_normal=PA_normal, clump_v_prior=clump_v_prior, clump_sigma_prior=clump_sigma_prior, clump_flux_prior=clump_flux_prior)

		# plot the masked direct image 
		# direct_image_size = direct.shape[1]
		# x = jnp.linspace(0 - x0_grism, direct.shape[1]-1 -x0_grism, direct.shape[1])
		# y = jnp.linspace(0 - y0_grism, direct.shape[0]-1 -y0_grism, direct.shape[0])
		# X, Y = jnp.meshgrid(x, y)
		# fig, ax = plt.subplots(figsize = (8,6))
		# full_mu = jnp.zeros_like(direct)
		# full_mu = full_mu.at[jnp.where(run_fit.mask ==1)].set(run_fit.mu)
		# cp = ax.pcolormesh(X,Y,full_mu, shading= 'nearest') #RdBu
		# cbar = fig.colorbar(cp)
		# plt.title('Masked direct image = flux prior mu')
		# plt.show()


		rng_key = random.PRNGKey(0)

		if model_name == 'two_component_model':
			print('Running the 2 component model')
			prior_predictive = Predictive(run_fit.two_component_model, num_samples=num_samples)
		elif model_name == 'one_component_model': 
			print('Running the full parametric model')
			prior_predictive = Predictive(run_fit.full_parametric_renormalized, num_samples=num_samples)
		elif model_name == 'two_disc_model':
			print('Running the two disc model')
			prior_predictive = Predictive(run_fit.two_disc_model, num_samples=num_samples)
		elif model_name == 'unified_two_discs_model':
			print('Running the unified two disc model')
			prior_predictive = Predictive(run_fit.unified_two_discs_model, num_samples=num_samples)
		elif model_name == 'MLE':
			print('Running the MLE model')
			prior_predictive = Predictive(run_fit.one_comp_MLE, num_samples=num_samples)
		prior = prior_predictive(rng_key)


		#testing the velocity field
		# velocities = jnp.array(v(run_fit.x, run_fit.y, jnp.radians(PA_truth-180),jnp.radians(30), 600, 4))
		# print(velocities.min(), velocities.max())
		# dispersions = 100*jnp.ones_like(velocities)

		# model_map_high = grism_object.disperse(oversample(run_fit.flux_prior,run_fit.factor,run_fit.factor),velocities, dispersions)
		# model_map_low = resample(model_map_high, run_fit.factor, run_fit.wave_factor)
		# plt.imshow(model_map_low, origin = 'lower', vmin = obs_map.min(), vmax = obs_map.max())
		# plt.show()
		# wave_factor = 1
		# # rescale the wave_space array and the direct image according to factor and wave_factor
		# len_wave = int(
		# 	(wave_space[len(wave_space)-1]-wave_space[0])/(delta_wave/wave_factor))
		# wave_space = jnp.linspace(wave_space[0], wave_space[len(wave_space)-1], len_wave+1)[index_min:index_max+1]
		# plot_grism_residual(model_map_low, obs_map, obs_error, y0, obs_map.shape[0], wave_space)

		# x = jnp.linspace(0 - x0, direct.shape[1]-1 -
        #          x0, direct.shape[1]*factor)
		# y = jnp.linspace(0 - y0, direct.shape[0]-1 -
        #          y0, direct.shape[0]*factor)
		# X_high, Y_high = jnp.meshgrid(x, y)

		# # mock_direct = oversample(direct, factor)

		# mock_velocities = jnp.array(v(X_high, Y_high, jnp.radians(PA_truth), 20, 300, 3))
		# mock_dispersions = 100*jnp.ones_like(mock_velocities)

		# fluxes, fluxes_low = run_fit.compute_MLE(obs_map, cov_matrix_full, mock_velocities, mock_dispersions, 1, 100, 1)

		# plt.imshow(fluxes, origin = 'lower')
		# plt.show()

		# EL_line = grism_object.disperse(
		# 	fluxes, mock_velocities, mock_dispersions)
		# plt.imshow(EL_line, origin='lower')
		# plt.title('Dispersed')
		# plt.show()

		run_fit.run_inference(model_name = model_name, num_samples=num_samples, num_warmup=num_warmup, high_res=True, median=True, step_size=step_size, adapt_step_size=True, target_accept_prob=target_accept_prob,  num_chains=2)

		inf_data = az.from_numpyro(run_fit.mcmc, prior=prior_predictive(rng_key))

		#no ../ because the open() function reads from terminal directory (not module directory)
		inf_data.to_netcdf('fitting_results/'+ output + 'output')
