__all__ = ["Fit_Numpyro"]

# imports

# importing my own modules
# from . import grism_dev
from . import preprocess as pre
from . import postprocess as post

import os

import jax
import jax.numpy as jnp
import numpy as np


import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS, BarkerMH, SA
# from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer.initialization import init_to_median, init_to_sample, init_to_uniform, init_to_value, init_to_feasible

import statistics as st
import math

# useful for plotting
from numpyro.infer import Predictive
from jax import random

import arviz as az

import argparse
import yaml
import corner

from astropy.table import Table

from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D
from astropy.convolution import convolve as convolve_astropy

from astropy.cosmology import Planck18 as cosmo


jax.config.update('jax_enable_x64', True)
numpyro.set_host_device_count(2)
numpyro.enable_validation()
# numpyro.set_platform('gpu')

# np.set_printoptions(precision=15, floatmode='maxprec')
# jnp.set_printoptions(precision=15, floatmode='maxprec')
# import figure_setup as setup

XLA_FLAGS = "--xla_gpu_force_compilation_parallelism=1"

# plotting settings

# setup.configure_plots()


class Fit_Numpyro():
	def __init__(self, obs_map, obs_error, grism_object, kin_model, inference_data, parametric):
		""" Class to fit model to data

						Parameters
						----------

						Attributes
						----------
		"""

		self.obs_map = obs_map
		self.obs_error = obs_error

		self.mask = (jnp.where(obs_map/obs_error < 5.0, 0, 1)).astype(bool)


		self.grism_object = grism_object

		self.kin_model = kin_model

		self.inference_data = inference_data

		self.parametric = parametric

	def __str__(self):
		# print all of the attributes of the class
		return 'Fit_Numpyro Class: \n' + ' - factor = ' + str(self.factor) + '\n - wave_factor = ' + str(self.wave_factor) + '\n grism object = ' + str(grism_object)

	def run_inference(self, num_samples=2000, num_warmup=2000, high_res=False, median=True, step_size=1, adapt_step_size=True, target_accept_prob=0.8, max_tree_depth=10, num_chains=5, init_vals = None):

		if self.parametric:
			inference_model = self.kin_model.inference_model_parametric
		else:
			inference_model = self.kin_model.inference_model
		self.nuts_kernel = NUTS(inference_model,  step_size=step_size, adapt_step_size=adapt_step_size, init_strategy=init_to_median(num_samples=2000),
								target_accept_prob=target_accept_prob, find_heuristic_step_size=True, max_tree_depth=10, dense_mass=False, adapt_mass_matrix=True) 
		
		print('max tree: ', max_tree_depth)
		print('step size: ', step_size)
		print('warmup: ', num_warmup)
		print('samples: ', num_samples)


		self.mcmc = MCMC(self.nuts_kernel, num_samples=num_samples,
						 num_warmup=num_warmup, num_chains=num_chains)
		self.rng_key = random.PRNGKey(100)

		new_mask = self.create_mask()

		with numpyro.validation_enabled():
			self.mcmc.run(self.rng_key, grism_object = self.grism_object, obs_map = self.obs_map, obs_error = self.obs_error, mask =new_mask) #, extra_fields=("potential_energy", "accept_prob"))

		print('done')

		self.mcmc.print_summary()
	
	def run_inference_ns(self, num_samples=2000, num_warmup=2000, high_res=False, median=True, step_size=1, adapt_step_size=True, target_accept_prob=0.8, max_tree_depth=10, num_chains=5, init_vals = None):

		constructor_kwargs = {"max_samples": 1000}
		self.ns = NestedSampler(model = self.kin_model.inference_model, constructor_kwargs= constructor_kwargs)

		self.ns.run(random.PRNGKey(0), grism_object = self.grism_object, obs_map = self.obs_map, obs_error = self.obs_error, mask =self.mask)
		self.ns.print_summary()

		ns_samples = self.ns.get_samples(random.PRNGKey(1), num_samples=num_samples)
		print(ns_samples)

		print('done')

	def diverging_parameters(self, chain_number, divergence_number):
		divergences = az.convert_to_dataset(
		    self.inference_data, group="sample_stats").diverging.transpose("chain", "draw")
		PA_div = self.inference_data.posterior['PA'][chain_number,
		    :][divergences[chain_number, :]]
		i_div = self.inference_data.posterior['i'][chain_number,
		    :][divergences[chain_number, :]]
		Va_div = self.inference_data.posterior['Va'][chain_number,
		    :][divergences[chain_number, :]]
		sigma0_div = self.inference_data.posterior['sigma0'][chain_number,
		    :][divergences[chain_number, :]]
		r_t_div = self.inference_data.posterior['r_t'][chain_number,
		    :][divergences[chain_number, :]]
		fluxes_div = self.inference_data.posterior['fluxes'][chain_number,
		    :][divergences[chain_number, :]]

		return jnp.array(fluxes_div[divergence_number].data), jnp.array(PA_div[divergence_number]), jnp.array(i_div[divergence_number]), jnp.array(Va_div[divergence_number]), jnp.array(r_t_div[divergence_number]), jnp.array(sigma0_div[divergence_number])


	def create_mask(self):
		'''
		Create a mask for the grism object based on the obs_map and obs_error
		'''
		sigma_rms = jnp.minimum((self.obs_map/self.obs_error).max(),5)
		im_conv = convolve_astropy(self.obs_map, make_2dgaussian_kernel(3.0, size=5))

		bkg = Background2D(self.obs_map, (15, 15), filter_size=(5, 5), exclude_percentile=99.0)


		segment_map = detect_sources(im_conv, sigma_rms*np.abs(bkg.background_median), npixels=10)

		main_label = segment_map.data[int(0.5*self.obs_map.shape[0]), int(0.5*self.obs_map.shape[1])]
		
		# construct mask
		mask = segment_map.data
		new_mask = np.zeros_like(mask)
		new_mask[mask == main_label] = 1.0
		return new_mask
# -----------------------------------------------------------running the inference-----------------------------------------------------------------------------------

def run_geko_fit(output, master_cat, line, parametric=False):

	# ----------------------------------------------------------preprocessing the data------------------------------------------------------------------------
	z_spec, wavelength, wave_space, obs_map, obs_error, model_name, kin_model, grism_object,\
	num_samples, num_warmup, step_size, target_accept_prob, delta_wave, factor = pre.run_full_preprocessing(output, master_cat, line)
	
	if parametric:
		with open('fitting_results/' + output + 'config_real.yaml', 'r') as file:
			input = yaml.load(file, Loader=yaml.FullLoader)
		ID = input[0]['Data']['ID']
		#get the field 
		field = input[0]['Data']['field']

		try:
			pysersic_summary = Table.read('PysersicFits_v1/summary_' + str(ID) + '_image_F150W_svi.cat', format='ascii')
		except:
			pysersic_summary = Table.read('PysersicFits_v1/summary_' + str(ID) + '_image_F182M_svi.cat', format='ascii')
		master_cat_table =Table.read(master_cat, format="ascii")
		#load the prior for the total emission line flux 
		log_int_flux = master_cat_table['fit_flux_cgs'][master_cat_table['ID'] == ID][0] #in log(ergs/s/cm2)
		int_flux = 10**log_int_flux #in ergs/s/cm2
		log_int_flux_err = master_cat_table['fit_flux_cgs_e'][master_cat_table['ID'] == ID][0] #in log(ergs/s/cm2)
		int_flux_err_high = 10**(log_int_flux + log_int_flux_err) - 10**log_int_flux #in ergs/s/cm2
		int_flux_err_low = 10**log_int_flux - 10**(log_int_flux - log_int_flux_err) #in ergs/s/cm2
		int_flux_err = np.mean([int_flux_err_high, int_flux_err_low]) #in ergs/s/cm2
		
		#based on the field, set the correction rotation to match JADES to the grism survey
		if field == 'GOODS-S-FRESCO':
			theta_rot = jnp.radians(0)
		elif field == 'GOODS-N': #fresco
			theta_rot = jnp.radians(230.5098)
		elif field == 'GOODS-N-CONGRESS':
			theta_rot = jnp.radians(228.22379)
		else:
			raise ValueError("Field not recognized. Please check the field name in the config file.")
 
		kin_model.disk.set_parametric_priors(pysersic_summary, [int_flux, int_flux_err], z_spec, wavelength, delta_wave, theta_rot, shape = obs_map.shape[0])
	else:
		#raise non-parametric fitting not implemented error
		raise ValueError("Non-parametric fitting is not implemented yet. Please set --parametric to True to use the parametric fitting.")

	# ----------------------------------------------------------running the inference------------------------------------------------------------------------

	run_fit = Fit_Numpyro(obs_map=obs_map, obs_error=obs_error, grism_object=grism_object, kin_model=kin_model, inference_data=None, parametric=parametric)

	rng_key = random.PRNGKey(4)
	if parametric:
		inference_model = run_fit.kin_model.inference_model_parametric
	else:
		inference_model = run_fit.kin_model.inference_model
	prior_predictive = Predictive(inference_model, num_samples=num_samples)

	prior = prior_predictive(rng_key, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error)

	run_fit.run_inference(num_samples=num_samples, num_warmup=num_warmup, high_res=True,
		                      median=True, step_size=step_size, adapt_step_size=True, target_accept_prob=target_accept_prob,  num_chains=2)

	inf_data = az.from_numpyro(run_fit.mcmc, prior=prior)

	# no ../ because the open() function reads from terminal directory (not module directory)
	inf_data.to_netcdf('fitting_results/' + output + 'output')

	#figure out how to make this work well
	v_re_16, v_re_med, v_re_84, kin_model, inf_data = post.process_results(output, master_cat, line,parametric=parametric, ID = ID)




#==============================EXTRA STUFF = NEED TO PUT SOMEWHERE ELSE==================================================

# -----------------------------------------------------------plotting model at point parameter space-----------------------------------------------------------------------------------

#this should go in the kin models module
# def generate_map(grism_object, fluxes, PA, i, Va, r_t, sigma0, x0, y0, factor=2, wave_factor=10, y_factor=1):
# 	x = jnp.linspace(0 - x0, fluxes.shape[1] - 1 - x0, fluxes.shape[1]*10*factor)
# 	y = jnp.linspace(0 - y0, fluxes.shape[0] - 1 - y0, fluxes.shape[0]*10*factor)
# 	x, y = jnp.meshgrid(x, y)

# 	highdim_flux = utils.oversample(fluxes, factor, factor)

# 	velocities = jnp.array(v(x, y, jnp.radians(PA), jnp.radians(i), Va, r_t))
# 	velocities = image.resize(velocities, (int(
# 	    velocities.shape[0]/10), int(velocities.shape[1]/10)), method='bicubic')

# 	dispersions = jnp.array(sigma(x, y, sigma0))

# 	model_map_high = grism_object.disperse(highdim_flux, velocities, dispersions)

# 	generated_map = utils.resample(model_map_high, y_factor*factor, wave_factor)

# 	return generated_map
