__all__ = ["Fit_Numpyro"]

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

		sigma_rms = jnp.minimum((self.obs_map/self.obs_error).max(),5)
		im_conv = convolve_astropy(self.obs_map, make_2dgaussian_kernel(3.0, size=5))

		bkg = Background2D(self.obs_map, (15, 15), filter_size=(5, 5), exclude_percentile=99.0)


		segment_map = detect_sources(im_conv, sigma_rms*np.abs(bkg.background_median), npixels=10)

		main_label = segment_map.data[int(0.5*self.obs_map.shape[0]), int(0.5*self.obs_map.shape[1])]
		
		# construct mask
		mask = segment_map.data
		new_mask = np.zeros_like(mask)
		new_mask[mask == main_label] = 1.0

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



# -----------------------------------------------------------running the inference-----------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='',
                    help='output folder name')
parser.add_argument('--line', type=str, default='H_alpha',
		    		help='line to fit')
parser.add_argument('--master_cat', type=str, default='CONGRESS_FRESCO/master_catalog.cat',
		    		help='master catalog file name')	
parser.add_argument('--parametric', type=bool, default=False,
		    		help='parametric flux model or not')		

if __name__ == "__main__":


	args = parser.parse_args()
	output = args.output + '/'
	master_cat = args.master_cat
	line = args.line
	parametric = args.parametric

	print('Running geko for the galaxy ID: ', output, ' with the line: ', line, ' and the master catalog: ', master_cat, ' and parametric: ', parametric)

	z_spec, wavelength, direct, direct_error, obs_map, obs_error, model_name, kin_model, grism_object, y0_grism, x0_grism,\
	num_samples, num_warmup, step_size, target_accept_prob, \
	wave_space, delta_wave, index_min, index_max, factor = pre.run_full_preprocessing(output, master_cat, line)
	
	if parametric:
		with open('fitting_results/' + output + 'config_real.yaml', 'r') as file:
			input = yaml.load(file, Loader=yaml.FullLoader)
		ID = input[0]['Data']['ID']
		#get the redshift from the master catalog
		try:
			pysersic_summary = Table.read('fitting_results/' + output + 'summary_' + str(ID) + '_image_F182M_svi.cat', format='ascii')
		except:
			pysersic_summary = Table.read('fitting_results/' + output + 'summary_' + str(ID) + '_image_F115W_svi.cat', format='ascii')
		try:	
			pysersic_grism_summary = Table.read('fitting_results/' + output + 'summary_' + str(ID) + '_grism_F356W_svi.cat', format='ascii')
		except:
			pysersic_grism_summary = Table.read('fitting_results/' + output + 'summary_' + str(ID) + '_grism_F444W_svi.cat', format='ascii')

		kin_model.disk.set_parametric_priors(pysersic_summary, pysersic_grism_summary, z_spec, wavelength, delta_wave)

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
	v_re_16, v_re_med, v_re_84, kin_model, inf_data = post.process_results(output, master_cat, line,parametric=parametric)

	#compute v/sigma posterior and quantiles
	inf_data.posterior['v_sigma'] = inf_data.posterior['v_re'] / inf_data.posterior['sigma0']
	v_sigma_16 = jnp.array(inf_data.posterior['v_sigma'].quantile(0.16, dim=["chain", "draw"]))
	v_sigma_med = jnp.array(inf_data.posterior['v_sigma'].median(dim=["chain", "draw"]))
	v_sigma_84 = jnp.array(inf_data.posterior['v_sigma'].quantile(0.84, dim=["chain", "draw"]))

	#compute Mdyn posterior and quantiles
	pressure_cor = 3.35 #= 2*re/rd
	inf_data.posterior['v_circ2'] = inf_data.posterior['v_re']**2 + inf_data.posterior['sigma0']**2*pressure_cor
	inf_data.posterior['v_circ'] = np.sqrt(inf_data.posterior['v_circ2'])
	ktot = 1.8 #for q0 = 0.2
	G = 4.3009172706e-3 #gravitational constant in pc*M_sun^-1*(km/s)^2
	DA = cosmo.angular_diameter_distance(z_spec).to('m')
	meters_to_pc = 3.086e16
	# Convert arcseconds to radians and calculate the physical size
	inf_data.posterior['r_eff_pc'] = np.deg2rad(inf_data.posterior['r_eff']*0.06/3600)*DA.value/meters_to_pc
	inf_data.posterior['M_dyn'] = np.log10(ktot*inf_data.posterior['v_circ2']*inf_data.posterior['r_eff_pc']/G)

	M_dyn_16 = jnp.array(inf_data.posterior['M_dyn'].quantile(0.16, dim=["chain", "draw"]))
	M_dyn_med = jnp.array(inf_data.posterior['M_dyn'].median(dim=["chain", "draw"]))
	M_dyn_84 = jnp.array(inf_data.posterior['M_dyn'].quantile(0.84, dim=["chain", "draw"]))

	v_circ_16 = jnp.array(inf_data.posterior['v_circ'].quantile(0.16, dim=["chain", "draw"]))
	v_circ_med = jnp.array(inf_data.posterior['v_circ'].median(dim=["chain", "draw"]))
	v_circ_84 = jnp.array(inf_data.posterior['v_circ'].quantile(0.84, dim=["chain", "draw"]))

	#save results to a file
	params= ['ID', 'PA_50', 'i_50', 'Va_50', 'r_t_50', 'sigma0_50', 'v_re_50', 'amplitude_50', 'r_eff_50', 'n_50','PA_morph_50', 'PA_16', 'i_16', 'Va_16', 'r_t_16', 'sigma0_16', 'v_re_16', 'PA_84', 'i_84', 'Va_84', 'r_t_84', 'sigma0_84', 'v_re_84', 'v_sigma_16', 'v_sigma_50', 'v_sigma_84', 'M_dyn_16', 'M_dyn_50', 'M_dyn_84', 'vcirc_16', 'vcirc_50', 'vcirc_84', 'r_eff_16', 'r_eff_84', 'ellip_50', 'ellip_16', 'ellip_84']
	t_empty = np.zeros((len(params), 3))
	res = Table(t_empty.T, names=params)
	res['ID'] = ID
	res['PA_50'] = kin_model.PA_mean
	res['i_50'] = kin_model.i_mean
	res['Va_50'] = kin_model.Va_mean
	res['r_t_50'] = kin_model.r_t_mean
	res['sigma0_50'] = kin_model.sigma0_mean_model
	res['v_re_50'] = v_re_med
	res['amplitude_50'] = kin_model.amplitude_mean
	res['r_eff_50'] = kin_model.r_eff_mean
	res['n_50'] = kin_model.n_mean
	res['PA_morph_50'] = kin_model.PA_morph_mean
	res['v_sigma_50'] = v_sigma_med

	res['PA_16'] = kin_model.PA_16
	res['i_16'] = kin_model.i_16
	res['Va_16'] = kin_model.Va_16
	res['r_t_16'] = kin_model.r_t_16
	res['sigma0_16'] = kin_model.sigma0_16
	res['v_re_16'] = v_re_16
	res['v_sigma_16'] = v_sigma_16

	res['PA_84'] = kin_model.PA_84
	res['i_84'] = kin_model.i_84
	res['Va_84'] = kin_model.Va_84
	res['r_t_84'] = kin_model.r_t_84
	res['sigma0_84'] = kin_model.sigma0_84
	res['v_re_84'] = v_re_84
	res['v_sigma_84'] = v_sigma_84

	res['M_dyn_16'] = M_dyn_16
	res['M_dyn_50'] = M_dyn_med
	res['M_dyn_84'] = M_dyn_84

	res['vcirc_16'] = v_circ_16
	res['vcirc_50'] = v_circ_med
	res['vcirc_84'] = v_circ_84

	res['r_eff_16'] = kin_model.r_eff_16
	res['r_eff_84'] = kin_model.r_eff_84

	res['ellip_50'] = kin_model.ellip_mean
	res['ellip_16'] = kin_model.ellip_16
	res['ellip_84'] = kin_model.ellip_84

	res.write('fitting_results/' + output + 'results', format='ascii', overwrite=True)
	
	#save a cornerplot of the v_sigma and sigma posteriors
	import smplotlib
	fig = plt.figure(figsize=(10, 10))
	CORNER_KWARGS = dict(
		smooth=4,
		label_kwargs=dict(fontsize=20),
		title_kwargs=dict(fontsize=20),
		quantiles=[0.16, 0.5, 0.84],
		plot_density=False,
		plot_datapoints=False,
		fill_contours=True,
		plot_contours=True,
		show_titles=True,
		labels=[r'$v_{re}/\sigma$', r'$\sigma_0$ [km/s]',  r'$\log ( M_{dyn} [M_{\odot}])$', r'$v_{circ}$ [km/s]'],
		titles= [r'$v_{re}/\sigma$ ', r'$\sigma_0$', r'$\log M_{dyn}$',r'$v_{circ}$'],
		max_n_ticks=3,
		divergences=False)

	figure = corner.corner(inf_data, group='posterior', var_names=['v_sigma','sigma0', 'M_dyn', 'v_circ'],
						color='dodgerblue', **CORNER_KWARGS)
	plt.tight_layout()
	plt.savefig('fitting_results/' + output + 'v_sigma_corner.png', dpi=300)
    
	v_re_16, v_re_med, v_re_84, kin_model, inf_data = post.process_results(output, master_cat, 'H_alpha', parametric = True)


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
