"""
Put all of the necessary post-processing functions here

	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ['process_results']

# imports
from . import  preprocess_dev as pre
from . import  fitting as fit

from . import  utils

from matplotlib import pyplot as plt

import jax.numpy as jnp
from jax import image
import numpyro

import arviz as az

import yaml

import numpy as np

from jax.scipy.signal import convolve

import argparse

from astropy.cosmology import Planck18 as cosmo

import xarray as xr

from astropy.table import Table

import corner

# import smplotlib

def save_fit_results(output, inf_data, kin_model, z_spec, ID, v_re_med, v_re_16, v_re_84):
	''' 
		Save all of the best-fit parameters in a table
	'''
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


def process_results(output, master_cat, line,  mock_params = None, test = None, j = None, parametric = False, ID = None):
	"""
		Main function that automatically post-processes the inference data and saves all of the relevant plots
		Returns the main data products so that data can be analyzed separately
	"""

	#pre-process the galaxy data
	z_spec, wavelength, wave_space, obs_map, obs_error, model_name, kin_model, grism_object,  num_samples, num_warmup, step_size, target_accept_prob, delta_wave, factor  = pre.run_full_preprocessing(output, master_cat, line, mock_params)

	#load inference data
	if mock_params is None:
		# inf_data = az.InferenceData.from_netcdf('FrescoHa/Runs-Final/' + output + '/'+ 'output')
		inf_data = az.InferenceData.from_netcdf('fitting_results/' + output + '/'+ 'output')
		j=0
	else:
		inf_data = az.InferenceData.from_netcdf('testing/' + str(test) + '/' + str(test) + '_' + str(j) + '_'+ 'output')
		
	num_samples = inf_data.posterior['sigma0'].shape[1]
	data = fit.Fit_Numpyro(obs_map = obs_map, obs_error = obs_error, grism_object = grism_object, kin_model = kin_model, inference_data = inf_data , parametric = parametric)
	inf_data, model_map,  model_flux, fluxes_mean, model_velocities, model_dispersions = kin_model.compute_model(inf_data, grism_object,parametric)
	#define the wave_space
	index_min = grism_object.index_min
	index_max = grism_object.index_max
	len_wave = int((wave_space[len(wave_space)-1] - wave_space[0])/(delta_wave))
	wave_space = jnp.linspace(wave_space[0], wave_space[len(wave_space)-1], len_wave+1)
	wave_space = wave_space[index_min:index_max]

	#save the posterior of the velocity at the effective radius
	inf_data, v_re_16, v_re_med, v_re_84 = utils.add_v_re(inf_data, kin_model, grism_object, num_samples)



	# compute v/sigma posterior and quantiles

	inf_data.posterior['sigma0_trunc'] = xr.DataArray(np.zeros((2,num_samples)), dims = ('chain', 'draw'))
	inf_data.prior['sigma0_trunc'] = xr.DataArray(np.zeros((1,num_samples)), dims = ('chain', 'draw'))
	for i in [0,1]:
		for sample in range(num_samples):
			if inf_data.posterior['sigma0'].quantile(0.16) <= 30:
				inf_data.posterior['sigma0_trunc'][i,sample] = np.random.uniform(inf_data.posterior['sigma0'].quantile(0.84), 0.5*inf_data.posterior['sigma0'].quantile(0.16))
				if i == 0:
					inf_data.prior['sigma0_trunc'][0,sample] = np.random.uniform(inf_data.prior['sigma0'].quantile(0.84), 0.5*inf_data.prior['sigma0'].quantile(0.16))
			else:
				inf_data.posterior['sigma0_trunc'][i,sample] = inf_data.posterior['sigma0'][i,sample]
				if i == 0:
					inf_data.prior['sigma0_trunc'][0,sample] = inf_data.prior['sigma0'][0,sample]
				
	inf_data.posterior['v_sigma'] = inf_data.posterior['v_re'] / inf_data.posterior['sigma0_trunc']
	inf_data['prior']['v_sigma'] = inf_data.prior['v_re'] / inf_data.prior['sigma0_trunc']
	v_sigma_16 = jnp.array(inf_data.posterior['v_sigma'].quantile(0.16, dim=["chain", "draw"]))
	v_sigma_med = jnp.array(inf_data.posterior['v_sigma'].median(dim=["chain", "draw"]))
	v_sigma_84 = jnp.array(inf_data.posterior['v_sigma'].quantile(0.84, dim=["chain", "draw"]))
	
	#save the best fit parameters in a table

	save_fit_results(output, inf_data, kin_model, z_spec, ID, v_re_med, v_re_16, v_re_84)
	
	kin_model.plot_summary(obs_map, obs_error, inf_data, wave_space, save_to_folder = output, name = 'summary', v_re = v_re_med)

	return  v_re_16, v_re_med, v_re_84, kin_model, inf_data


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='',
					help='folder of the galaxy you want to postprocess')
parser.add_argument('--line', type=str, default='H_alpha',
					help='line to fit')        
parser.add_argument('--master_cat', type=str, default='CONGRESS_FRESCO/master_catalog.cat',
					help = 'master catalog file to use for the post-processing')                                                                                  	

if __name__ == "__main__":

	#run the post-processing hands-off 
	args = parser.parse_args()
	output = args.output
	line = args.line
	master_cat = args.master_cat

	inf_data = az.InferenceData.from_netcdf('fitting_results/' + output + '/'+ 'output')
	process_results(output,master_cat,line)








# def match_to_IFU(F1, V1, D1, F2, V2, D2, full_PSF, FWHM_PSF, R_LSF, wavelength, IFU='kmos'):
# 	# if there is no second component than you should input zero arrays in the F2,V2,D2
# 	# create a 3D cube with the model best fit, basically with f[i,j,k]=F[i,j]*gaussian(k)
# 		# make the gaussian with stats.norm.pdf(velocity_space, v[i,j],sigma_v[i,j])
# 	velocity_space = jnp.linspace(-1000, 1000, 2001)
# 	broadcast_velocity_space = np.broadcast_to(
# 	    velocity_space[:, np.newaxis, np.newaxis], (velocity_space.size, F1.shape[0], F1.shape[0]))
# 	if F2 is not None:
# 		cube = F1*stats.norm.pdf(broadcast_velocity_space, V1, D1) + \
# 		                         F2*stats.norm.pdf(broadcast_velocity_space, V2, D2)
# 	else:
# 		cube = F1*stats.norm.pdf(broadcast_velocity_space, V1, D1)

# 	print('cube created')
# 	# if no full PSF given, use the FWHM to create a 2D gaussian kernel
# 	if full_PSF is None:
# 		FWHM_PSF_pixels = FWHM_PSF/0.0629
# 		sigma_PSF = FWHM_PSF_pixels/(2*np.sqrt(2*np.log(2)))
# 		full_PSF = np.array(Gaussian2DKernel(sigma_PSF))
# 	print('PSF created')
# 	# create the LSF kernel - assuming a constant one for now since only emission around the same wavelength
# 	# compute the std for the velocities from the spectral resolution
# 	sigma_LSF = wavelength/(2.355*R_LSF)
# 	sigma_LSF_v = (c/1000)*sigma_LSF/wavelength
# 	LSF = np.array(Gaussian1DKernel(sigma_LSF_v))
# 	print('LSF created')
# 	# create a 3D kernel with 2D PSF and 1D LSF
# 	full_kernel = np.array(full_PSF) * np.broadcast_to(np.array(LSF)[:, np.newaxis, np.newaxis], (np.array(
# 	    LSF).size, np.array(full_PSF).shape[0], np.array(full_PSF).shape[0]))
# 	print('kernel created')
# 	# convolve the cube with the kernel
# 	print('convolving cube')
# 	convolved_cube = signal.fftconvolve(cub     e, full_kernel, mode='same')
# 	print('convolution done')
# 	# from the convolved cube, obtain the flux, velocity and dispersion maps
# 	F_kmos = np.sum(convolved_cube, axis=0)
# 	# fit a gaussian to the convolved cube to obtain the velocity and dispersion maps
# 	V_kmos = np.zeros_like(F_kmos)
# 	D_kmos = np.zeros_like(F_kmos)

# 	def gauss(x, A, mu, sigma):
# 		return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)*A

# 	for i in range(V_kmos.shape[0]):
# 		for j in range(V_kmos.shape[1]):
# 			popt, pcov = curve_fit(gauss, velocity_space, convolved_cube[:, i, j], p0=[
# 			                       F_kmos[i, j], V1[i, j], D1[i, j]])
# 			V_kmos[i, j] = popt[1]
# 			D_kmos[i, j] = popt[2]
# 	# popt, pcov = curve_fit(gauss, velocity_space, convolved_cube, p0=[F_kmos,V1, D1])
# 	# V_kmos = popt[1]
# 	# D_kmos = popt[2]

# 	# finally resample to IFU pixel scale
# 	IFU_scale_arcseconds = 0.1799
# 	NC_LW_scale_arcseconds = 0.0629
# 	# rescale the maps to the IFU pixel scale

# 	if IFU == 'kmos':
# 		IFU_scale_pixels = int(IFU_scale_arcseconds/NC_LW_scale_arcseconds)
# 		# blocks = F_kmos.reshape((int(F_kmos.shape[0]/(IFU_scale_pixels)), IFU_scale_pixels, int(F_kmos.shape[1]/IFU_scale_pixels), IFU_scale_pixels))
# 		# F_kmos = jnp.sum(blocks, axis=(1,3))
# 		F_kmos = image.resize(F_kmos, (int(
# 		    F_kmos.shape[0]/IFU_scale_pixels), int(F_kmos.shape[1]/IFU_scale_pixels)), method='nearest')
# 		F_kmos *= IFU_scale_pixels**2
# 		V_kmos = image.resize(V_kmos, (int(
# 		    V_kmos.shape[0]/IFU_scale_pixels), int(V_kmos.shape[1]/IFU_scale_pixels)), method='bicubic')
# 		D_kmos = image.resize(D_kmos, (int(
# 		    D_kmos.shape[0]/IFU_scale_pixels), int(D_kmos.shape[1]/IFU_scale_pixels)), method='bicubic')

# 		print('rescaling done')
# 	return F_kmos, V_kmos, D_kmos