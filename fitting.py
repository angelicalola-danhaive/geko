# imports

# importing my own modules
import grism
import preprocess as pre
import postprocess as post
import plotting
import utils
import models

import os

import jax
import jax.numpy as jnp
import numpy as np


import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_sample, init_to_uniform, init_to_value

import statistics as st
import math

# useful for plotting
from numpyro.infer import Predictive
from jax import random

import arviz as az

import argparse
import yaml


jax.config.update('jax_enable_x64', True)
numpyro.set_host_device_count(2)
numpyro.set_platform('gpu')

# np.set_printoptions(precision=15, floatmode='maxprec')
# jnp.set_printoptions(precision=15, floatmode='maxprec')
# import figure_setup as setup

XLA_FLAGS = "--xla_gpu_force_compilation_parallelism=1"

# plotting settings

# setup.configure_plots()


class Fit_Numpyro():
	def __init__(self, obs_map, obs_error, grism_object, kin_model, inference_data):
		""" Class to fit model to data

						Parameters
						----------

						Attributes
						----------
		"""

		self.obs_map = obs_map
		self.obs_error = obs_error

		self.grism_object = grism_object

		self.kin_model = kin_model

		self.inference_data = inference_data

	def __str__(self):
		# print all of the attributes of the class
		return 'Fit_Numpyro Class: \n' + ' - factor = ' + str(self.factor) + '\n - wave_factor = ' + str(self.wave_factor) + '\n grism object = ' + str(grism_object)

	def run_inference(self, num_samples=2000, num_warmup=2000, high_res=False, median=True, step_size=1, adapt_step_size=True, target_accept_prob=0.8, max_tree_depth=10, num_chains=5, init_vals = None):

		self.nuts_kernel = NUTS(self.kin_model.inference_model, init_strategy=init_to_median(num_samples=2000), step_size=step_size, adapt_step_size=adapt_step_size,
								target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=10, find_heuristic_step_size=True)
		# init_to_value(values = init_vals)
		print('max tree: ', max_tree_depth)
		print('step size: ', step_size)
		print('warmup: ', num_warmup)
		print('samples: ', num_samples)

		self.mcmc = MCMC(self.nuts_kernel, num_samples=num_samples,
						 num_warmup=num_warmup, num_chains=num_chains)
		self.rng_key = random.PRNGKey(100)
		self.mcmc.run(self.rng_key, grism_object = self.grism_object, obs_map = self.obs_map, obs_error = self.obs_error, extra_fields=("potential_energy", "accept_prob"))

		print('done')

		self.mcmc.print_summary()

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



# -----------------------------------------------------------running the inference-----------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='',
                    help='output folder name')
parser.add_argument('--line', type=str, default='H_alpha',
		    		help='line to fit')
parser.add_argument('--master_cat', type=str, default='CONGRESS_FRESCO/master_catalog.cat',
		    		help='master catalog file name')			

if __name__ == "__main__":


	args = parser.parse_args()
	output = args.output + '/'
	master_cat = args.master_cat
	line = args.line

	print('Running the real data')

	direct, direct_error, obs_map, obs_error, model_name, kin_model, grism_object, y0_grism, x0_grism,\
	num_samples, num_warmup, step_size, target_accept_prob, \
	wave_space, delta_wave, index_min, index_max, factor = pre.run_full_preprocessing(output, master_cat, line)
	

	# ----------------------------------------------------------running the inference------------------------------------------------------------------------

	run_fit = Fit_Numpyro(obs_map=obs_map, obs_error=obs_error, grism_object=grism_object, kin_model=kin_model, inference_data=None)

	rng_key = random.PRNGKey(4)

	prior_predictive = Predictive(run_fit.kin_model.inference_model, num_samples=num_samples)

	prior = prior_predictive(rng_key, grism_object = run_fit.grism_object, obs_map = run_fit.obs_map, obs_error = run_fit.obs_error)

	run_fit.run_inference(num_samples=num_samples, num_warmup=num_warmup, high_res=True,
		                      median=True, step_size=step_size, adapt_step_size=True, target_accept_prob=target_accept_prob,  num_chains=2)

	inf_data = az.from_numpyro(run_fit.mcmc, prior=prior)

	# no ../ because the open() function reads from terminal directory (not module directory)
	inf_data.to_netcdf('fitting_results/' + output + 'output')

	#figure out how to make this work well
	post.process_results(output, master_cat, line)


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
