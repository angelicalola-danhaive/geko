"""

	Module holding all of the kinematic models used in the fitting process.

	Written by A L Danhaive: ald66@cam.ac.uk
"""
__all__ = ["KinModels"]

# imports
import numpy as np
# geko related imports
from . import  utils
from . import  plotting

# jax and its functions
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.signal import convolve
# from scipy.signal import convolve
from jax import image

# from skimage.morphology import dilation, disk

from astropy.modeling.models import GeneralSersic2D, Sersic2D

# scipy and its functions
from scipy.constants import pi
from scipy.ndimage import measurements

# numpyro and its functions
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam, CircularReparam, LocScaleReparam

from matplotlib import pyplot as plt

from photutils import centroids

from astropy.cosmology import Planck18 as cosmo

import time

from scipy.constants import c

import math
import xarray as xr
jax.config.update('jax_enable_x64', True)
numpyro.enable_validation()

class KinModels:
	'''
		This top level class only contains the functions to make the velocity maps. The rest will be specific to each sub class.
	'''


	def __init__(self):
		print('New kinematic model created')
	
	def v_rad(self, x, y, PA, i, Va, r_t, r):
		return (2/pi)*Va*jnp.arctan(r/r_t)*jnp.sin(i)


	def v(self, x, y, PA, i, Va, r_t):
		i_rad = i / 180 * jnp.pi
		PA_rad = PA / 180 * jnp.pi
		
		# Precompute trigonometric values
		sini = jnp.sin(i_rad)
		cosi = jnp.cos(i_rad)
		
		# Rotate coordinates
		x_rot = x * jnp.cos(PA_rad) - y * jnp.sin(PA_rad)
		y_rot = x * jnp.sin(PA_rad) + y * jnp.cos(PA_rad)
		
		# Safeguard for cases where cosi is zero
		i_rad_safe = jnp.where(cosi != 0, i_rad, 0)
		cosi_safe = jnp.where(cosi != 0, cosi, 1e-6)  # Use a small epsilon to avoid division by zero
		
		# Calculate r, handling x_rot = 0 and y_rot = 0 cases separately
		r_squared = x_rot**2 / cosi_safe**2 + y_rot**2
		r_safe_squared = jnp.where(r_squared != 0.0, r_squared, 1e-12)  # Small epsilon to avoid sqrt(0)
		r = jnp.sqrt(r_safe_squared)
		
		# Handle the special case where r = 0 or where both x_rot and y_rot are 0 explicitly
		r_safe = jnp.where((x_rot != 0) | (y_rot != 0), r, 1e-6)  # Use a small epsilon for r when both x_rot and y_rot are zero
		
		# Calculate observed velocity
		vel_obs = jnp.where(cosi != 0, self.vel1d(r_safe, Va, r_t) * sini, self.vel1d(x_rot, Va, r_t))
		
		# Final velocity computation, handling r = 0 or x_rot = y_rot = 0 case
		vel_obs_final = jnp.where(r_safe != 0.0, vel_obs * (y_rot / r_safe), 0.0)
		
		return vel_obs_final
	
	def v_int(self, x, y, PA, i, Va, r_t):
		i_rad = i / 180 * jnp.pi
		PA_rad = PA / 180 * jnp.pi
		
		# Precompute trigonometric values
		sini = jnp.sin(i_rad)
		cosi = jnp.cos(i_rad)
		
		# Rotate coordinates
		x_rot = x * jnp.cos(PA_rad) - y * jnp.sin(PA_rad)
		y_rot = x * jnp.sin(PA_rad) + y * jnp.cos(PA_rad)
		
		# Safeguard for cases where cosi is zero
		i_rad_safe = jnp.where(cosi != 0, i_rad, 0)
		cosi_safe = jnp.where(cosi != 0, cosi, 1e-6)  # Use a small epsilon to avoid division by zero
		
		# Calculate r, handling x_rot = 0 and y_rot = 0 cases separately
		r_squared = x_rot**2 / cosi_safe**2 + y_rot**2
		r_safe_squared = jnp.where(r_squared != 0.0, r_squared, 1e-12)  # Small epsilon to avoid sqrt(0)
		r = jnp.sqrt(r_safe_squared)
		
		# Handle the special case where r = 0 or where both x_rot and y_rot are 0 explicitly
		r_safe = jnp.where((x_rot != 0) | (y_rot != 0), r, 1e-6)  # Use a small epsilon for r when both x_rot and y_rot are zero
		
		# Calculate observed velocity
		vel_obs = jnp.where(cosi != 0, self.vel1d(r_safe, Va, r_t), self.vel1d(x_rot, Va, r_t))
		
		# Final velocity computation, handling r = 0 or x_rot = y_rot = 0 case
		vel_obs_final = jnp.where(r_safe != 0.0, vel_obs * (y_rot / r_safe), 0.0)
		
		return vel_obs_final
	


	def vel1d(self, r, Va, r_t):
		r_t_safe = jnp.where(r_t != 0.0, r_t, 1.0)
		r_safe = jnp.where(r != 0.0, r, 1.0)
		v_out = jnp.where((r_t!=0.0) & (r!=0.0), (2.0/jnp.pi)*Va*jnp.arctan2(r_safe,r_t_safe), 0.0)
		# v_out = 40
		# v_out = jnp.where(jnp.cos(r/r_t), (2.0/jnp.pi)*Va*jnp.arctan2(r,jnp.where(jnp.abs(r_t)>0, r_t, 1.0)), 0.0)

		return jnp.array(v_out)


		# return dispersions

		

	def set_main_bounds(self, factor, wave_factor, PA_sigma, Va_bounds, r_t_bounds, sigma0_bounds, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph,inclination,r_eff, r_eff_grism):
		"""
		Set the bounds for the model parameters by reading the ones from the config file.
		The more specific bounds computations for the different models will be done inside their
		class.
		"""

		self.factor = factor

		self.wave_factor = wave_factor

		self.PA_sigma = PA_sigma
		self.PA_grism = PA_grism
		self.PA_morph = PA_morph
		self.inclination = inclination
		self.r_eff = r_eff
		self.r_eff_grism = r_eff_grism
		# these are in the form (low, high)
		self.Va_bounds = Va_bounds
		self.r_t_bounds = r_t_bounds
		self.sigma0_bounds = sigma0_bounds


		self.x0 = x0
		self.y0 = y0

		self.x0_vel = x0_vel
		self.mu_y0_vel = y0_vel

		if self.mu_y0_vel == None:
			self.x0_vel = x0
			self.mu_y0_vel = y0

	def rescale_to_mask(self, array, mask):
		"""
			Rescale the bounds to the mask
		"""
		rescaled_array = []
		for a in array:
			a = a[jnp.where(mask == 1)]
			rescaled_array.append(a)
		return rescaled_array

@numpyro.handlers.reparam(
	config={"PA_radians": CircularReparam()} #, "i_radians": CircularReparam()} #, "y0_vel": TransformReparam()}
)

class Disk():
	"""
		Class for 1 disk object. Combinations of this will be used for the single disk model, 
		then 2 disks for the 2 component ones etc
	"""
	def __init__(self, direct_shape, factor,  x0_vel, mu_y0_vel, r_eff):
		print('Disk object created')

		#initialize all attributes with function parameters
		self.direct_shape = direct_shape


		self.x0_vel = direct_shape[1]//2
		self.mu_y0_vel = mu_y0_vel

		self.r_eff = r_eff

		self.factor = factor


		# self.print_priors()
	
	def print_priors(self):
		print('Priors for disk model')
		print('fluxes --- Truncated Normal w/ flux scaling')
		print('fluxes scaling --- Uniform w/ bounds: ' + str(0.05) + ' ' + str(2))
		print( 'PA --- Normal w/ mu: ' + str(self.mu_PA) + ' and sigma: ' + str(self.sigma_PA))
		print( 'i --- Truncated Normal w/ mu: ' + str(self.mu_i) + ' and sigma: ' + str(self.sigma_i) + ' and bounds: ' + str(self.i_bounds))
		print( 'Va --- Uniform w/ bounds: ' + str(self.Va_bounds))
		print('r_t --- Normal w/ mu: ' + str(self.r_t_bounds[1]) + ' and bounds: ' + str(self.r_t_bounds[0]) + ' ' + str(self.r_t_bounds[2]))
		print('sigma0 --- Uniform w/ bounds: ' + str(self.sigma0_bounds))
		print('y0_vel --- Truncated Normal w/ mu: ' + str(self.mu_y0_vel) + ' and sigma: ' + str(self.y0_std) + ' and bounds: ' + str(self.y_low) + ' ' + str(self.y_high))
		print('v0 --- Normal w/ mu: 0 and sigma: 100')

	def set_parametric_priors(self,py_table, py_grism_table, redshift, wavelength, delta_wave):
		"""
		Set the priors for the parametric model
		"""
		#need to set sizes in kpc before converting to arcsecs then pxs
		arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z=redshift).value
		kpc_per_pixel = 0.06/arcsec_per_kpc

		theta = py_table['theta_q50'][0]
		PA = (theta-jnp.pi/2) * (180/jnp.pi) #convert to degrees
		if PA < 0:
			print('Converting pysersic PA from ', PA, ' to ', PA + 180, ' degrees')
			PA += 180
		elif PA > 180:
			print('Converting pysersic PA from ', PA, ' to ', PA - 180, ' degrees')
			PA -= 180
		PA = 90 - PA #for the kinematics
		if PA < 0:
			PA += 180
		PA_mean_err = ((py_table['theta_q84'][0] - py_table['theta_q50'][0]) + (py_table['theta_q50'][0] - py_table['theta_q16'][0]))/2
		PA_std = (PA_mean_err*2)*(180/jnp.pi) #convert to degrees

		ellip = py_table['ellip_q50'][0] 
		# inclination = jnp.arccos(1-ellip)*180/jnp.pi
		inclination = utils.compute_inclination(ellip=ellip, q0 = 0.2) #q0=0.2 for a thick disk
		# ellip_err = ((py_table['ellip_q84'][0] - py_table['ellip_q50'][0]) + (py_table['ellip_q50'][0] - py_table['ellip_q16'][0]))/2
		# inclination_err = (((jnp.arccos(1-py_table['ellip_q84'][0]) - jnp.arccos(1-py_table['ellip_q50'][0])) + (jnp.arccos(1-py_table['ellip_q50'][0]) - jnp.arccos(1-py_table['ellip_q16'][0])))/2)*180/jnp.pi
		inclination_err = ( (utils.compute_inclination(ellip = py_table['ellip_q84'][0], q0 = 0.2) - inclination) + (inclination - utils.compute_inclination(ellip = py_table['ellip_q16'][0], q0 = 0.2)) )/2

		inclination_std = inclination_err #no x2 bc this is an accurate measurement!
		# ellip_std =   ellip_err/2.36

		#because the F115W is fit with the 0.03 resolution, r_eff is twice too big
		#also using Natalia's fit to convert from near-UV to Ha sizes and for the scatter/uncertainty
		nUV_to_Ha = 10**0.2
		nUV_to_Ha_std = 10**0.171
		#put the r_eff in kpc
		r_eff_kpc = py_table['r_eff_q50'][0]*kpc_per_pixel
		r_eff_kpc_err = (((py_table['r_eff_q84'][0] - py_table['r_eff_q50'][0]) + (py_table['r_eff_q50'][0] - py_table['r_eff_q16'][0]))/2)*kpc_per_pixel

		r_eff_UV = py_table['r_eff_q50'][0]/2
		r_eff_Ha = r_eff_UV*nUV_to_Ha
		r_eff_UV_err = ((py_table['r_eff_q84'][0] - py_table['r_eff_q50'][0]) + (py_table['r_eff_q50'][0] - py_table['r_eff_q16'][0]))/4
		#combine the uncertainties from measurements and scaling relation
		r_eff_std = r_eff_Ha*np.sqrt((r_eff_UV_err/r_eff_UV)**2 + (nUV_to_Ha_std/nUV_to_Ha)**2)

		#compute hard bounds for r_eff in kpc to not be too small or too big
		r_eff_min_kpc = 0.1 
		r_eff_max_kpc = 10
		#convert to pixels
		r_eff_min = r_eff_min_kpc/kpc_per_pixel
		r_eff_max = r_eff_max_kpc/kpc_per_pixel

		n = py_table['n_q50'][0]
		n_err = ((py_table['n_q84'][0] - py_table['n_q50'][0]) + (py_table['n_q50'][0] - py_table['n_q16'][0]))/2
		n_std = n_err*2

		#try taking n from grism too
		# n = py_grism_table['n_q50'][0]
		# n_err = ((py_grism_table['n_q84'][0] - py_grism_table['n_q50'][0]) + (py_grism_table['n_q50'][0] - py_grism_table['n_q16'][0]))/2
		# n_std = n_err/2.36

		#take the flux from the grism
		flux = py_grism_table['flux_q50'][0]
		amplitude = flux #utils.flux_to_Ie(flux,n, r_eff, ellip)
		amplitude_high = py_grism_table['flux_q84'][0] #utils.flux_to_Ie(py_grism_table['flux_q84'][0],n, r_eff, ellip)
		amplitude_low = py_grism_table['flux_q16'][0] #utils.flux_to_Ie(py_grism_table['flux_q16'][0],n, r_eff, ellip)
		amplitude_err = ((amplitude_high - amplitude) + (amplitude - amplitude_low))/2
		amplitude_std = amplitude_err #no x2 because this is an accurate measurement of the flux in the grism data!

		#central pixel from image
		 #because the F115W is fit with the 0.03 resolution, the centroids are twice too big
		xc_morph = py_table['xc_q50'][0]/2
		xc_err = ((py_table['xc_q84'][0] - py_table['xc_q50'][0]) + (py_table['xc_q50'][0] - py_table['xc_q16'][0]))/4
		xc_std = xc_err*2

		yc_morph = py_table['yc_q50'][0]/2
		yc_err = ((py_table['yc_q84'][0] - py_table['yc_q50'][0]) + (py_table['yc_q50'][0] - py_table['yc_q16'][0]))/4
		yc_std = yc_err*2

		print('Setting parametric priors: ', PA, inclination, r_eff_Ha, n, amplitude, xc_morph, yc_morph)

		#compute velocity bounds using the grism size
		r_eff_grism = py_grism_table['r_eff_q50'][0]
		r_s_grism = r_eff_grism/1.676
		r22_grism = 2.2*r_s_grism
		PA_grism = py_grism_table['theta_q50'][0]
		#project r22 on the x axis using the PA
		r22_x = jnp.abs(r22_grism*jnp.cos(PA_grism))
		#compute the velocity gradient
		vel_pix_scale = (delta_wave/wavelength)*(c/1000) #put c in km/s
		# self.V_max = r22_x*vel_pix_scale
		self.V_max = 800
		self.D_max = 500

		#set class attributes for all of these values
		self.PA_morph_mu = PA
		self.PA_morph_std = PA_std
		# self.ellip_mu = ellip
		# self.ellip_std = ellip_std
		self.inc_mu = inclination
		self.inc_std = inclination_std

		self.r_eff_mu = r_eff_Ha
		self.r_eff_std = r_eff_std
		self.r_eff_min = r_eff_min
		self.r_eff_max = r_eff_max

		self.amplitude_mu = amplitude
		self.amplitude_std = amplitude_std
		self.n_mu = n #*2 #just testing for Erica's
		self.n_std = n_std
		self.xc_morph = xc_morph
		self.xc_std = xc_std
		self.yc_morph = yc_morph
		self.yc_std = yc_std
	

	def set_parametric_priors_test(self,priors):
		#set class attributes for all of these values
		self.PA_morph_mu = priors['PA']
		self.PA_morph_std = 5 #0.2*priors['PA']
		# self.ellip_mu = ellip
		# self.ellip_std = ellip_std
		self.inc_mu = priors['i']
		self.inc_std = 0.2*priors['i']
		self.r_eff_mu = (1.676/0.4)*priors['r_t']
		self.r_eff_std = 2 #0.5*self.r_eff_mu
		self.r_eff_min = 0
		self.r_eff_max = 15
		self.n_mu = priors['n'] #1 #*2 #just testing for Erica's
		self.n_std = 2
		self.xc_morph = 15
		self.xc_std = 1
		self.yc_morph = 15
		self.yc_std = 1
		ellip = 1 - utils.compute_axis_ratio(60, 0.2)
		self.amplitude_mu = 200 #utils.Ie_to_flux(1, self.n_mu, self.r_eff_mu, ellip)
		self.amplitude_std = 40 #0.1*self.amplitude_mu

		self.V_max = 800
		self.D_max = 600

		print('Set mock kinematic priors: ', self.PA_morph_mu, self.inc_mu, self.r_eff_mu, self.amplitude_mu, self.n_mu, self.xc_morph, self.yc_morph)


	def sample_fluxes_parametric(self):

		#sample the parameters needed for a disc model

		#amplitude
		# amplitude_mu= 0.0011 #1.0
		# amplitude_std = 0.5*amplitude_mu
		unscaled_amplitude = numpyro.sample('unscaled_amplitude', dist.TruncatedNormal(low = (0.0 - self.amplitude_mu)/self.amplitude_std))
		amplitude = numpyro.deterministic('amplitude', unscaled_amplitude*self.amplitude_std + self.amplitude_mu)
		# amplitude = 1

		# r_eff_mu = 5.52 #self.r_eff
		# r_eff_std = 1
		unscaled_r_eff = numpyro.sample('unscaled_r_eff', dist.TruncatedNormal(low = (self.r_eff_min - self.r_eff_mu)/self.r_eff_std, high = (self.r_eff_max - self.r_eff_mu)/self.r_eff_std))
		r_eff = numpyro.deterministic('r_eff', unscaled_r_eff*self.r_eff_std + self.r_eff_mu)
		# r_eff = 4.19

		# n_mu = 1.778 #1.0
		# n_std = 0.4
		unscaled_n = numpyro.sample('unscaled_n', dist.TruncatedNormal(low = (0.36 - self.n_mu)/self.n_std, high = (8.0 - self.n_mu)/self.n_std))
		n = numpyro.deterministic('n', unscaled_n*self.n_std + self.n_mu)
		# n = 1

		# ellip_mu = 0.369 #0.5
		# ellip_std = 0.1
		# unscaled_ellip = numpyro.sample('unscaled_ellip', dist.TruncatedNormal(low = (0.0 - self.ellip_mu)/self.ellip_std, high = (1.0 - self.ellip_mu)/self.ellip_std))
		# ellip = numpyro.deterministic('ellip', unscaled_ellip*self.ellip_std + self.ellip_mu)
		# ellip = 0.5

		i_low = (0-self.inc_mu)/self.inc_std
		i_high = (90-self.inc_mu)/self.inc_std
		unscaled_i = numpyro.sample('unscaled_i', dist.TruncatedNormal(low = i_low, high = i_high))
		i = numpyro.deterministic('i', unscaled_i*self.inc_std + self.inc_mu)
		# ellip = 1 - jnp.cos(i*jnp.pi/180)
		ellip = 1 - utils.compute_axis_ratio(inc = i, q0 = 0.2)
		# PA_morph_mu = self.mu_PA
		# PA_morph_std = 10

		# low_PA = (-10 - self.PA_morph_mu)/(self.PA_morph_std)
		# high_PA = ( 190- self.PA_morph_mu)/(self.PA_morph_std)
		# unscaled_PA_morph = numpyro.sample('unscaled_PA_morph' + self.number, dist.TruncatedNormal(low = low_PA, high = high_PA)) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
		unscaled_PA_morph = numpyro.sample('unscaled_PA_morph', dist.Normal()) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
		PA_morph = numpyro.deterministic('PA_morph', unscaled_PA_morph*self.PA_morph_std + self.PA_morph_mu)
		# r_eff = numpyro.sample('r_eff', dist.TruncatedNormal(self.r_eff, 0.5, low = 0.0))
		# n = numpyro.sample('n', dist.TruncatedNormal(1.0, 0.5, low = 0.36))
		# ellip = (numpyro.sample('ellip', dist.TruncatedNormal(0.5, 0.1, low = 0.0)))
		# PA_morph = (numpyro.sample('PA_morph', dist.TruncatedNormal(self.mu_PA, 5.0, low = 0.0)))

		unscaled_xc_morph = numpyro.sample('unscaled_xc_morph', dist.Normal())
		xc_morph = numpyro.deterministic('xc_morph', unscaled_xc_morph*self.xc_std + self.xc_morph)
		# xc_morph = self.xc_morph

		unscaled_yc_morph = numpyro.sample('unscaled_yc_morph', dist.Normal())
		yc_morph = numpyro.deterministic('yc_morph', unscaled_yc_morph*self.yc_std + self.yc_morph)  
		# yc_morph = self.yc_morph                                 
						

		#create a mock galaxy using these parameters
		# galaxy_model = GeneralSersic2D(amplitude=amplitude, r_eff =r_eff*27, n = n, x_0 = self.direct_shape[0]//2*27 + 13 , y_0 = self.direct_shape[0]//2*27 +13, ellip = ellip, theta=(90 - PA_morph)*math.pi/180) #function takes theta in rads

		factor = self.factor
				
		sersic_factor = 25
		image_shape = self.direct_shape[0]

		x = jnp.linspace(0 - xc_morph, image_shape - xc_morph - 1, image_shape)
		y = jnp.linspace(0 - yc_morph, image_shape - yc_morph - 1, image_shape)
		x,y = jnp.meshgrid(x,y)

		amplitude_re = utils.flux_to_Ie(amplitude, n, r_eff, ellip)

		#-------------------------constant oversampling---------------------------------

		x_grid = image.resize(x, (image_shape*factor*sersic_factor, image_shape*factor*sersic_factor), method='linear')
		y_grid = image.resize(y, (image_shape*factor*sersic_factor, image_shape*factor*sersic_factor), method='linear')
		#the center is set at 0,0 because the grid is already centered at xc_morph, yc_morph
		model_image_highres = utils.sersic_profile(x_grid, y_grid, amplitude_re/(sersic_factor*factor)**2, r_eff, n,0.0,0.0, ellip, (90 - PA_morph)*jnp.pi/180)
		model_image = utils.resample(model_image_highres, int(sersic_factor), int(sersic_factor))

		#-------------------------adaptive oversampling---------------------------------
		# x_grid = image.resize(x, (image_shape*factor, image_shape*factor), method='linear')
		# y_grid = image.resize(y, (image_shape*factor, image_shape*factor), method='linear')
		# model_image = utils.compute_adaptive_sersic_profile(x_grid, y_grid, amplitude/factor**2, r_eff, n, 0.0,0.0, ellip, (90 - PA_morph)*jnp.pi/180)
		#------------------------------------------------------------------------------

		#mask the low fluxes of the model image
		model_image_masked = model_image #jnp.where(model_image>0.01*model_image.max(), model_image, 0.0)

		#the returned image has a shape of image_shape*factor
		return model_image_masked, r_eff, i, xc_morph, yc_morph


	def sample_params_parametric(self,r_eff = 0.0):
		"""
			Sample all of the parameters needed to model a disk velocity field
		"""


		unscaled_PA = numpyro.sample('unscaled_PA', dist.Normal())
		Pa = numpyro.deterministic('PA', unscaled_PA*self.PA_morph_std + self.PA_morph_mu)

		unscaled_Va = numpyro.sample('unscaled_Va', dist.Uniform())  #* (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		Va = numpyro.deterministic('Va', unscaled_Va*(2*self.V_max) - self.V_max)

		unscaled_r_t = numpyro.sample('unscaled_r_t', dist.Uniform())
		r_t = numpyro.deterministic('r_t', unscaled_r_t*r_eff)
	

		unscaled_sigma0 = numpyro.sample('unscaled_sigma0', dist.Uniform())
		sigma0 = numpyro.deterministic('sigma0', unscaled_sigma0*self.D_max)

		y0_vel = 0
		

		x0_vel = 0

		unscaled_v0 = numpyro.sample('unscaled_v0', dist.Normal())
		v0 = numpyro.deterministic('v0', unscaled_v0*50)
		# v0 = 0


		return Pa, Va, r_t,sigma0, y0_vel, x0_vel, v0
	

	def compute_posterior_means_parametric(self, inference_data):
		"""
			Retreive the best sample from the MCMC chains for the main disk variables
		"""

		self.PA_mean = jnp.array(inference_data.posterior['PA'].median(dim=["chain", "draw"]))
		self.y0_vel_mean = 15 
		self.x0_vel_mean = 15 
		self.v0_mean = jnp.array(inference_data.posterior['v0'].median(dim=["chain", "draw"]))
		self.r_t_mean = jnp.array(inference_data.posterior['r_t'].median(dim=["chain", "draw"]))
		self.sigma0_mean_model = jnp.array(inference_data.posterior['sigma0'].median(dim=["chain", "draw"]))
		self.Va_mean = jnp.array(inference_data.posterior['Va'].median(dim=["chain", "draw"]))

		#save the percentiles as well
		self.PA_16 = jnp.array(inference_data.posterior['PA'].quantile(0.16, dim=["chain", "draw"]))
		self.PA_84 = jnp.array(inference_data.posterior['PA'].quantile(0.84, dim=["chain", "draw"]))

		self.v0_16 = jnp.array(inference_data.posterior['v0'].quantile(0.16, dim=["chain", "draw"]))
		self.v0_84 = jnp.array(inference_data.posterior['v0'].quantile(0.84, dim=["chain", "draw"]))
		self.r_t_16 = jnp.array(inference_data.posterior['r_t'].quantile(0.16, dim=["chain", "draw"]))
		self.r_t_84 = jnp.array(inference_data.posterior['r_t'].quantile(0.84, dim=["chain", "draw"]))
		self.sigma0_16 = jnp.array(inference_data.posterior['sigma0'].quantile(0.16, dim=["chain", "draw"]))
		self.sigma0_84 = jnp.array(inference_data.posterior['sigma0'].quantile(0.84, dim=["chain", "draw"]))
		self.Va_16 = jnp.array(inference_data.posterior['Va'].quantile(0.16, dim=["chain", "draw"]))
		self.Va_84 = jnp.array(inference_data.posterior['Va'].quantile(0.84, dim=["chain", "draw"]))

		return  self.PA_mean,self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.x0_vel_mean, self.v0_mean

	def compute_parametrix_flux_posterior(self, inference_data):
		#compute means for parametric flux model
		self.amplitude_mean = jnp.array(inference_data.posterior['amplitude'].median(dim=["chain", "draw"]))
		self.r_eff_mean = jnp.array(inference_data.posterior['r_eff'].median(dim=["chain", "draw"]))
		self.n_mean = jnp.array(inference_data.posterior['n'].median(dim=["chain", "draw"]))
		# self.ellip_mean = jnp.array(inference_data.posterior['ellip'].median(dim=["chain", "draw"]))
		self.PA_morph_mean = jnp.array(inference_data.posterior['PA_morph'].median(dim=["chain", "draw"])) #- 45
		#compute the inclination prior posterior and median from the ellipticity
		num_samples = inference_data.posterior['i'].shape[1]

		inference_data.posterior['ellip'] = xr.DataArray(np.zeros((2,num_samples)), dims = ('chain', 'draw'))
		inference_data.prior['ellip'] = xr.DataArray(np.zeros((1,num_samples)), dims = ('chain', 'draw'))
		for i in [0,1]:
			for sample in range(num_samples-1):
				inference_data.posterior['ellip'][i,int(sample)] = 1 - utils.compute_axis_ratio(inc = float(inference_data.posterior['i'][i,int(sample)].values), q0 = 0.2)
				inference_data.prior['ellip'][0,int(sample)] = 1 - utils.compute_axis_ratio(inc = float(inference_data.prior['i'][0,int(sample)].values), q0 = 0.2)
		
		self.i_mean = jnp.array(inference_data.posterior['i'].median(dim=["chain", "draw"]))
		self.i_16 = jnp.array(inference_data.posterior['i'].quantile(0.16, dim=["chain", "draw"]))
		self.i_84 = jnp.array(inference_data.posterior['i'].quantile(0.84, dim=["chain", "draw"]))
		# self.ellip_mean = 1 - jnp.cos(jnp.radians(self.i_mean))
		self.ellip_mean = jnp.array(inference_data.posterior['ellip'].median(dim=["chain", "draw"]))
		self.ellip_16 = jnp.array(inference_data.posterior['ellip'].quantile(0.16, dim=["chain", "draw"]))
		self.ellip_84 = jnp.array(inference_data.posterior['ellip'].quantile(0.84, dim=["chain", "draw"]))
		#compute the fluxes for the parametric model
		# y, x = np.mgrid[0:self.direct_shape[0]*27, 0:self.direct_shape[1]*27]
		# fluxes_mean_high = utils.sersic_profile(x,y,amplitude=self.amplitude_mean, r_eff = self.r_eff_mean*27, n = self.n_mean, x_0 = self.direct_shape[0]//2*27 + 13 , y_0 = self.direct_shape[0]//2*27 +13, ellip = self.ellip_mean, theta=(90 - self.PA_morph_mean)*np.pi/180)/27**2 #function takes theta in rads
		# self.fluxes_mean = utils.resample(fluxes_mean_high, 27,27)

		self.r_eff_16 = jnp.array(inference_data.posterior['r_eff'].quantile(0.16, dim=["chain", "draw"]))
		self.r_eff_84 = jnp.array(inference_data.posterior['r_eff'].quantile(0.84, dim=["chain", "draw"]))

		self.xc_morph_mean = jnp.array(inference_data.posterior['xc_morph'].median(dim=["chain", "draw"]))
		self.yc_morph_mean = jnp.array(inference_data.posterior['yc_morph'].median(dim=["chain", "draw"]))

		#compute the fluxes in the sersic way
		factor = self.factor
				
		sersic_factor = 25
		image_shape = self.direct_shape[0]

		amplitude_re_mean = utils.flux_to_Ie(self.amplitude_mean,self.n_mean, self.r_eff_mean, self.ellip_mean)

		x = jnp.linspace(0 - self.xc_morph_mean, image_shape - self.xc_morph_mean - 1, image_shape)
		y = jnp.linspace(0 - self.yc_morph_mean, image_shape - self.yc_morph_mean - 1, image_shape)
		x,y = jnp.meshgrid(x,y)
		x_grid = image.resize(x, (image_shape*factor*sersic_factor, image_shape*factor*sersic_factor), method='linear')
		y_grid = image.resize(y, (image_shape*factor*sersic_factor, image_shape*factor*sersic_factor), method='linear')
		#testing the adaptive oversampling of the sersic profile
		fluxes_mean_high = utils.sersic_profile(x_grid, y_grid, amplitude_re_mean/(sersic_factor*factor)**2, self.r_eff_mean, self.n_mean, 0.0,0.0, self.ellip_mean, (90 - self.PA_morph_mean)*jnp.pi/180)
		self.fluxes_mean_high = utils.resample(fluxes_mean_high, sersic_factor, sersic_factor)
		self.fluxes_mean = utils.resample(fluxes_mean_high, factor*sersic_factor, factor*sersic_factor)
		self.fluxes_mean_masked = jnp.where(self.fluxes_mean>0.01*self.fluxes_mean.max(), self.fluxes_mean, 0.0)
		return inference_data, self.fluxes_mean_masked, self.fluxes_mean_high, self.amplitude_mean, self.r_eff_mean, self.n_mean, self.ellip_mean, self.PA_morph_mean, self.i_mean, self.xc_morph_mean, self.yc_morph_mean
	
	def v_rot(self, fluxes_mean, model_velocities, i_mean,factor):
		"""
			Compute the rotational velocity of the disk component

			If called from multiple component model, the 3 attributes of this function should be only from that component
		"""
		plt.imshow(fluxes_mean, origin='lower')
		plt.colorbar()
		plt.title('Fluxes mean')
		plt.show()
		print(fluxes_mean.max())
		threshold = 0.4*fluxes_mean.max()
		mask = jnp.zeros_like(fluxes_mean)
		mask = mask.at[jnp.where(fluxes_mean>threshold)].set(1)
		model_velocities_low = jax.image.resize(model_velocities, (int(model_velocities.shape[0]/factor), int(model_velocities.shape[1]/factor)), method='nearest')
		model_v_rot = 0.5*(jnp.nanmax(jnp.where(mask == 1, model_velocities_low, jnp.nan)) - jnp.nanmin(jnp.where(mask == 1, model_velocities_low, jnp.nan)))/ jnp.sin( jnp.radians(i_mean)) 
		plt.imshow(jnp.where(mask ==1, fluxes_mean, np.nan), origin = 'lower')
		plt.title('Mask for v_rot comp')
		plt.show()
		return model_v_rot

	def plot(self):

		"""
			Plot the disk model
		"""
		
		#plot the fluxes within the mask and the velocity centroid
		fluxes = jnp.zeros(self.direct_shape)
		fluxes = fluxes.at[self.masked_indices].set(self.mu)
		# fluxes = self.mu
		plt.imshow(fluxes, origin='lower')
		plt.colorbar()
		plt.scatter(self.x0_vel, self.mu_y0_vel, color='red')
		plt.title('Disk')
		plt.show()



class DiskModel(KinModels):
	"""
		Class for the one component exponential disk model
	"""

	def __init__(self):
		print('Disk model created')

		#declare var and label names for plotting

		self.var_names = [ 'i', 'Va', 'sigma0'] #, 'fluxes_scaling']
		self.labels = [ r'$i$', r'$V_a$', r'$\sigma_0$'] #, r'$f_{scale}$']
		# self.var_names = ['PA', 'i', 'Va', 'r_t', 'sigma0_max', 'sigma0_scale', 'sigma0_const']
		# self.labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_{max}$', r'$\sigma_{scale}$', r'$\sigma_{const}$']

	def set_bounds(self, im_shape, factor, wave_factor, PA_sigma,Va_bounds, r_t_bounds, sigma0_bounds, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph, inclination, r_eff, r_eff_grism):
		"""

		Compute all of the necessary bounds for the disk model sampling distributions

		"""
		# first set all of the main bounds taken from the config file
		self.set_main_bounds(factor, wave_factor, PA_sigma, Va_bounds, r_t_bounds, sigma0_bounds, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph,inclination,r_eff, r_eff_grism)

		self.im_shape = im_shape
			
		self.disk = Disk(self.im_shape, self.factor,self.x0_vel, self.mu_y0_vel, self.r_eff )
		
		# self.disk.plot()



	
	def inference_model_parametric(self, grism_object, obs_map, obs_error, mask = None):
		"""

		Model used to infer the disk parameters from the data => called in fitting.py as the forward
		model used for the inference

		"""
		# fluxes = self.disk.sample_fluxes()
		fluxes,r_eff, i, xc_morph, yc_morph = self.disk.sample_fluxes_parametric() 
		Pa, Va, r_t, sigma0, y0_vel, x0_vel, v0 = self.disk.sample_params_parametric(r_eff=r_eff)      
		# i = utils.compute_inclination(ellip = ellip) 

		fluxes_high = fluxes #utils.oversample(fluxes, grism_object.factor, grism_object.factor, method= 'bilinear')

		image_shape = self.im_shape[0]
		# print(image_shape//2)
		# x= jnp.linspace(0 - x0_vel, image_shape - x0_vel - 1, image_shape)
		# y = jnp.linspace(0 - y0_vel, image_shape - y0_vel - 1, image_shape)
		x= jnp.linspace(0 - xc_morph, image_shape - xc_morph - 1, image_shape)
		y = jnp.linspace(0 - yc_morph, image_shape - yc_morph - 1, image_shape)
		X, Y = jnp.meshgrid(x,y)

		X_grid = image.resize(X, (int(X.shape[0]*grism_object.factor), int(X.shape[1]*grism_object.factor)), method='linear')
		Y_grid = image.resize(Y, (int(Y.shape[0]*grism_object.factor), int(Y.shape[1]*grism_object.factor)), method='linear')

		velocities = jnp.asarray(self.v(X_grid, Y_grid, Pa, i, Va, r_t))

		velocities_scaled = velocities + v0

		dispersions = sigma0*jnp.ones_like(velocities_scaled)

		self.model_map = grism_object.disperse(fluxes_high, velocities_scaled, dispersions)

		self.model_map = utils.resample(self.model_map, grism_object.factor, self.wave_factor)


		self.error_scaling = 1 #numpyro.sample('error_scaling', dist.Uniform(0, 1))*5
		# SN_min = jnp.minimum((obs_map/obs_error).max()/10,5)
		# mask = jnp.where(obs_map/obs_error < SN_min, 0, 1)
		# model_masked = jnp.where(mask == 1, self.model_map, 0)
		# obs_masked = jnp.where(mask == 1, obs_map, 0)
		# obs_error_masked = jnp.where(mask == 1, obs_error, 1e6)

		obs_error_masked = jnp.where(mask == 1, obs_error, 1e6)



		# numpyro.sample('obs', dist.Normal(self.model_map[5:26,:], self.error_scaling*obs_error[5:26,:]), obs=obs_map[5:26,:])
		numpyro.sample('obs', dist.Normal(self.model_map, self.error_scaling*obs_error_masked), obs=obs_map)


	def compute_model_nonparam(self, inference_data, grism_object):
		"""

		Function used to post-process the MCMC samples and plot results from the model

		"""

		self.PA_mean,self.Va_mean, self.i_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean,self.x0_vel_mean, self.v0_mean = self.disk.compute_posterior_means(inference_data)
		print('posterior means kinematics: ', self.PA_mean,self.Va_mean, self.i_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean,self.x0_vel_mean, self.v0_mean)
		self.fluxes_mean, self.fluxes_scaling_mean = self.disk.compute_flux_posterior(inference_data, self.flux_type)

		self.model_flux = utils.oversample(self.fluxes_mean, grism_object.factor, grism_object.factor, method= 'bilinear')


		image_shape =  self.im_shape[0]
		x_10 = jnp.linspace(0 - self.x0_vel_mean, image_shape - self.x0_vel_mean - 1, image_shape*grism_object.factor)
		y_10 = jnp.linspace(0 - self.y0_vel_mean, image_shape - self.y0_vel_mean - 1, image_shape*grism_object.factor)
		X, Y = jnp.meshgrid(x_10,y_10)


		self.model_velocities = jnp.asarray(self.v(X, Y, self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean))
		# self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')
		print('posterior velocities: ', self.model_velocities)
		self.model_velocities = self.model_velocities  + self.v0_mean

		self.model_dispersions = self.sigma0_mean_model *jnp.ones_like(self.model_velocities) #self.sigma0_mean_model *jnp.ones_like(self.model_velocities)

		print(self.model_flux.shape, self.model_velocities.shape, self.model_dispersions.shape)
		self.model_map_high = grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
		# self.model_map_high = grism_object.disperse(self.convolved_fluxes, self.convolved_velocities, self.convolved_dispersions)

		self.model_map = utils.resample(self.model_map_high, grism_object.factor, self.wave_factor)

		#compute velocity grid in flux image resolution for plotting velocity maps
		self.model_velocities_low = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/grism_object.factor), int(self.model_velocities.shape[1]/grism_object.factor)), method='nearest')
		self.model_dispersions_low = image.resize(self.model_dispersions, (int(self.model_dispersions.shape[0]/grism_object.factor), int(self.model_dispersions.shape[1]/grism_object.factor)), method='nearest')

		return inference_data, self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions

	def compute_model_parametric(self, inference_data, grism_object):
		"""

		Function used to post-process the MCMC samples and plot results from the model

		"""

		self.PA_mean,self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean,self.x0_vel_mean, self.v0_mean = self.disk.compute_posterior_means_parametric(inference_data)
		#save all of the percentile values
		self.PA_16 = self.disk.PA_16
		self.PA_84 = self.disk.PA_84
		self.Va_16 = self.disk.Va_16
		self.Va_84 = self.disk.Va_84
		self.r_t_16 = self.disk.r_t_16
		self.r_t_84 = self.disk.r_t_84
		self.sigma0_16 = self.disk.sigma0_16
		self.sigma0_84 = self.disk.sigma0_84
		# self.y0_vel_16 = self.disk.y0_vel_16
		# self.y0_vel_84 = self.disk.y0_vel_84
		# self.x0_vel_16 = self.disk.x0_vel_16
		# self.x0_vel_84 = self.disk.x0_vel_84
		self.v0_16 = self.disk.v0_16
		self.v0_84 = self.disk.v0_84

		# self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean,self.y0_vel_mean, self.v0_mean = self.disk.compute_posterior_means(inference_data)

		# self.fluxes_mean, self.fluxes_scaling_mean = self.disk.compute_flux_posterior(inference_data, self.flux_type)
		inference_data,self.fluxes_mean, self.fluxes_mean_high, self.amplitude_mean, self.r_eff_mean, self.n_mean, self.ellip_mean, self.PA_morph_mean, self.i_mean, self.xc_morph_mean, self.yc_morph_mean = self.disk.compute_parametrix_flux_posterior(inference_data)
		self.r_eff_16 = self.disk.r_eff_16
		self.r_eff_84 = self.disk.r_eff_84

		self.ellip_mean = self.disk.ellip_mean
		self.ellip_16 = self.disk.ellip_16
		self.ellip_84 = self.disk.ellip_84
		self.i_16 = self.disk.i_16
		self.i_84 = self.disk.i_84
		# self.model_flux = utils.oversample(self.fluxes_mean, grism_object.factor, grism_object.factor, method= 'bicubic')
		self.model_flux = self.fluxes_mean_high

		image_shape =  self.im_shape[0]
		x= jnp.linspace(0 - self.xc_morph_mean, image_shape - self.xc_morph_mean - 1, image_shape)
		y = jnp.linspace(0 - self.yc_morph_mean, image_shape - self.yc_morph_mean - 1, image_shape)
		X, Y = jnp.meshgrid(x,y)

		X_grid = image.resize(X, (int(X.shape[0]*grism_object.factor), int(X.shape[1]*grism_object.factor)), method='nearest')
		Y_grid = image.resize(Y, (int(Y.shape[0]*grism_object.factor), int(Y.shape[1]*grism_object.factor)), method='nearest')

		self.model_velocities = jnp.asarray(self.v(X_grid, Y_grid, self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean))
		# self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

		self.model_velocities = self.model_velocities  + self.v0_mean

		self.model_dispersions = self.sigma0_mean_model *jnp.ones_like(self.model_velocities) #self.sigma0_mean_model *jnp.ones_like(self.model_velocities)

		self.model_map_high = grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
		# self.model_map_high = grism_object.disperse(self.convolved_fluxes, self.convolved_velocities, self.convolved_dispersions)

		self.model_map = utils.resample(self.model_map_high, grism_object.factor, self.wave_factor)
		# print('Model vels:', self.model_velocities)
		#compute velocity grid in flux image resolution for plotting velocity maps
		self.model_velocities_low = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/grism_object.factor), int(self.model_velocities.shape[1]/grism_object.factor)), method='nearest')
		# print(self.fluxes_mean)
		self.model_velocities_low = np.where(self.fluxes_mean == 0, np.nan, self.model_velocities_low)
		self.model_dispersions_low = image.resize(self.model_dispersions, (int(self.model_dispersions.shape[0]/grism_object.factor), int(self.model_dispersions.shape[1]/grism_object.factor)), method='nearest')
		self.model_dispersions_low = jnp.where(self.fluxes_mean == 0, np.nan, self.model_dispersions_low)
		return inference_data, self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions
	
	def compute_model(self,inference_data, grism_object, parametric = False):
		"""

		Function used to post-process the MCMC samples and plot results from the model

		"""

		if parametric:
			return self.compute_model_parametric(inference_data, grism_object)
		else:
			return self.compute_model_nonparam(inference_data, grism_object)

	def log_likelihood(self, grism_object, obs_map, obs_error, values = {}):
		Pa = values['PA']
		i = values['i']
		Va = values['Va']
		r_t = values['r_t']
		sigma0 = values['sigma0']

		fluxes = jnp.where(self.mask ==1, self.flux_prior, 0.0)

		fluxes_high = utils.oversample(fluxes, grism_object.factor, grism_object.factor)

		image_shape = fluxes.shape[0]
		# print(image_shape//2)
		x_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*grism_object.factor)
		y_10 = jnp.linspace(0 - image_shape//2, image_shape - image_shape//2 - 1, image_shape*grism_object.factor)
		X_10, Y_10 = jnp.meshgrid(x_10,y_10)
		# sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)

		# start  = time.time()
		
		# self.compute_factors(Pa, i,X_10, Y_10)
		velocities = jnp.array(self.v(X_10, Y_10, Pa, i, Va, r_t))
		# velocities = velocities.at[15,15].set(3e-14)
		# velocities = utils.resample(velocities, 10, 10)/10**2

		velocities_scaled = velocities

		dispersions = sigma0*jnp.ones_like(velocities_scaled)

		self.model_map = grism_object.disperse(fluxes_high, velocities_scaled, dispersions)


		self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, self.wave_factor)
		# #plot the residuals
		# plt.imshow((obs_map - self.model_map)/obs_error, origin = 'lower')
		# plt.colorbar()
		# plt.show()
		# plt.close()

		mask_obs = jnp.where(obs_map/obs_error > 5, 1, 0)
		model_mask = jnp.where(mask_obs == 1, self.model_map, 0.0)
		obs_mask = jnp.where(mask_obs == 1, obs_map, 0.0)
		obs_error_mask = jnp.where(mask_obs == 1, obs_error, 1e6)

		#compute the gaussian likelihood for the model
		log_likelihood = dist.Normal(model_mask, obs_error_mask).log_prob(obs_mask)


		# const = jnp.log(jnp.sqrt(2*jnp.pi)*obs_error[0])
		# value_scaled = (self.model_map - obs_map)/obs_error
		# log_likelihood = -0.5*(value_scaled**2) - const

		# print('Const: ', const)
		# print('Value scaled: ', value_scaled)
		# print('Log likelihood: ', log_likelihood)
		# plot the log L for each point
		# plt.imshow(log_likelihood, origin = 'lower')
		# plt.colorbar()
		# plt.show()
		# plt.close()

		# print(log_likelihood)

		log_likelihood_sum = jnp.sum(log_likelihood)

		print('Log likelihood: ', log_likelihood_sum)

		return log_likelihood_sum
	
	def log_prior(self, values = {}):
		Pa = values['PA']
		i = values['i']
		Va = values['Va']
		r_t = values['r_t']
		sigma0 = values['sigma0']
		fluxes = values['fluxes']
		fluxes_errors = values['fluxes_error']

		log_prior_PA = dist.TruncatedNormal(Pa, 5,low = -10,high = 100).log_prob(self.mu_PA)
		log_prior_i = dist.TruncatedNormal(i, 5,low = 0,high = 90).log_prob(self.mu_i)
		log_prior_Va = dist.Uniform(self.Va_bounds[0], self.Va_bounds[1]).log_prob(Va)
		log_prior_r_t = dist.Normal(0,4).log_prob(r_t)
		log_prior_sigma0 = dist.Uniform(0, 400).log_prob(sigma0)
		log_prior_fluxes = dist.Normal(fluxes,fluxes_errors).log_prob(self.flux_prior)
		log_prior_fluxes_tot = jnp.sum(log_prior_fluxes)
		log_prior = log_prior_PA + log_prior_i + log_prior_Va + log_prior_r_t + log_prior_sigma0 + log_prior_fluxes_tot

		print('Log prior: ', log_prior)

		return log_prior
	
	def log_posterior(self, grism_object, obs_map, obs_error,values = {}):
		return -(self.log_likelihood(grism_object, obs_map, obs_error,values) + self.log_prior(values))
	def plot_summary(self, obs_map, obs_error, inf_data, wave_space, save_to_folder = None, name = None, v_re = None, PA = None, i = None, Va = None, r_t = None, sigma0 = None, obs_radius = None, ellip = None, theta_obs = None, theta_Ha =None, n = None):
		ymin,ymax = plotting.plot_disk_summary(obs_map, self.model_map, obs_error, self.model_velocities_low, self.model_dispersions_low, v_re, self.fluxes_mean, inf_data, wave_space, x0 = self.x0, y0 = self.y0, factor = 1, direct_image_size = self.im_shape[0], save_to_folder = save_to_folder, name = name, PA = PA, i = i, Va = Va, r_t = r_t, sigma0 = sigma0, obs_radius = obs_radius, ellip = ellip, theta_obs = theta_obs, theta_Ha =theta_Ha, n = n)
		return ymin, ymax



##### BIN OF OLD NON-PARAM MODEL FUNCTIONS #####

	# def sample_fluxes(self):
	# 	# f = open("timing_total.txt", "a")
	# 	#sample the fluxes within the mask
	# 	fluxes_scaling = numpyro.sample('fluxes_scaling' + self.number, dist.Uniform())*(4-0.05) + 0.05
	# 	# unscaled_fluxes_sample = numpyro.sample('unscaled_fluxes'+ self.number, dist.TruncatedNormal(jnp.zeros(int(len(self.masked_indices[0]))),jnp.ones(int(len(self.masked_indices[0]))),low = low, high = high), sample_shape=(int(len(self.masked_indices[0])),))
	# 	# fluxes_sample = numpyro.deterministic('fluxes', unscaled_fluxes_sample*self.std + self.mu*fluxes_scaling)
	# 	# fluxes_sample = numpyro.sample('unscaled_fluxes'+ self.number, dist.Uniform(), sample_shape=(int(len(self.masked_indices[0])),))
	# 	# fluxes_sample = norm.ppf(norm.cdf(self.low) + fluxes_sample*(norm.cdf(self.high)-norm.cdf(self.low)))*self.std + self.mu*fluxes_scaling
	# 	# reparam_config = {"fluxes": TransformReparam()}
	# 	# with numpyro.handlers.reparam(config=reparam_config):
	# 	#     fluxes_sample = numpyro.sample("fluxes",dist.TransformedDistribution(
	# 	#             dist.TruncatedNormal(0.0, 1.0, low=low, high=high),
	# 	#             AffineTransform(self.mu*fluxes_scaling, self.std*fluxes_scaling),
	# 	#         ),
	# 	#     )
	# 	fluxes_error_scaling = 1 #numpyro.sample('fluxes_error_scaling' + self.number, dist.Uniform())*9 + 1
	# 	#just sample from normal distribution
	# 	# unscaled_fluxes = numpyro.sample('unscaled_fluxes'+ self.number, dist.Normal(jnp.zeros(int(len(self.masked_indices[0]))),jnp.ones(int(len(self.masked_indices[0]))))  )
	# 	# unscaled_fluxes = numpyro.sample('unscaled_fluxes'+ self.number, dist.Normal(jnp.zeros(self.mu.shape),jnp.ones(self.mu.shape))  )

	# 	# fluxes_sample = numpyro.deterministic('fluxes', unscaled_fluxes*self.std*fluxes_error_scaling + self.mu*fluxes_scaling)  
	# 	# fluxes_sample = numpyro.sample('fluxes', dist.TruncatedNormal(self.mu*fluxes_scaling, self.std*fluxes_scaling, low = self.low))

	# 	low = (jnp.zeros(self.mu.shape) - self.mu*fluxes_scaling)/(self.std*fluxes_error_scaling)
	# 	high = (2*self.mu - self.mu*fluxes_scaling)/(self.std*fluxes_error_scaling)
	# 	# reparam_config = {"fluxes": TransformReparam()}
	# 	# with numpyro.handlers.reparam(config=reparam_config):
	# 	#     fluxes_sample = numpyro.sample("fluxes",dist.TransformedDistribution(
	# 	#             dist.TruncatedNormal(0.0, 1.0, low=low),
	# 	#             AffineTransform(self.mu*fluxes_scaling, self.std*fluxes_error_scaling),
	# 	#         ),
	# 	#     )
	# 	unscaled_fluxes_sample = numpyro.sample("unscaled_fluxes",dist.Normal(loc = jnp.zeros_like(self.mu), scale = jnp.ones_like(self.mu)))
	# 	# unscaled_fluxes_sample = numpyro.sample("unscaled_fluxes",dist.MultivariateNormal(jnp.zeros_like(jnp.reshape(self.mu,(31*31))),jnp.diag(jnp.ones((31*31,)) ,k=0)))
	# 	# fluxes_sample = numpyro.sample("fluxes",dist.Normal(self.mu,self.std))
	# 	# print(unscaled_fluxes_sample.shape)
	# 	fluxes_sample = numpyro.deterministic('fluxes', unscaled_fluxes_sample*self.std*fluxes_error_scaling + self.mu*fluxes_scaling)
	# 	# f.write('fluxes_sample' + str(fluxes_sample[0]) +' \n')
	# 	# fluxes_sample = self.mu
	# 	fluxes = jnp.zeros(self.direct_shape)
	# 	fluxes = fluxes.at[self.masked_indices].set(fluxes_sample)
	# 	# fluxes = jnp.reshape(fluxes_sample, (31,31))
	# 	# print(fluxes.shape)

	# 	return fluxes


		# def sample_params(self):
		# """
		# 	Sample all of the parameters needed to model a disk velocity field
		# """
		# # f = open("timing_total.txt", "a")
		# # start = time.time()
		# # unscaled_Pa = numpyro.sample('unscaled_PA'+ self.number, dist.Normal())
		# # sample the mu_PA + 0 or 180 (orientation of velocity field)
		# # rotation = numpyro.sample('rotation'+ self.number, dist.Uniform())

		# # simulate a bernouilli discrete distribution
		# # PA_morph = self.mu_PA + round(rotation)*180
		# # Pa = numpyro.deterministic('PA', unscaled_Pa*self.sigma_PA + PA_morph)
		# # unscaled_Pa = numpyro.sample('unscaled_Pa'+ self.number, dist.Uniform())
		# # Pa = numpyro.deterministic('PA', unscaled_Pa*180)
		# # Pa = norm.ppf(Pa)*self.sigma_PA + PA_morph

		# #sample from circular normal distribution (in radians)


		# # Pa_rad = numpyro.sample('PA_radians' + self.number, dist.VonMises(self.mu_PA*jnp.pi/180,1/((self.sigma_PA*jnp.pi/180)**2))) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
		# # Pa = numpyro.deterministic('PA' + self.number, Pa_rad*180/jnp.pi)
		# low_PA = (-10 - self.PA_morph_mu)/(self.PA_morph_std)
		# high_PA = ( 100- self.PA_morph_mu)/(self.PA_morph_std)
		# unscaled_PA = numpyro.sample('unscaled_PA' + self.number, dist.TruncatedNormal(low = low_PA, high = high_PA)) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
		# # unscaled_PA = numpyro.sample('unscaled_PA' + self.number, dist.Normal()) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
		# Pa = numpyro.deterministic('PA' + self.number, unscaled_PA*self.PA_morph_std + self.PA_morph_mu)
		# # Pa = 0
		# # Pa = numpyro.sample('PA', dist.Normal(self.mu_PA, self.sigma_PA))
		# # Pa = 1.57
		# # end = /time.time()
		# # f.write('PA sampling time: '+ str(end-start)+ '\n')
		# # Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA

		# #could probably use utils for this too
		# # start = time.time()
		# unscaled_i = numpyro.sample('unscaled_i' + self.number, dist.TruncatedNormal(low = self.i_low, high = self.i_high))


		# # unscaled_i = numpyro.sample('unscaled_i' + self.number, dist.Normal())
		# i = numpyro.deterministic('i', unscaled_i*self.sigma_i + self.mu_i)
		# # i = 60


		# # i = numpyro.deterministic('i', unscaled_i*self.sigma_i + self.mu_i)

		# # unscaled_i = numpyro.sample('unscaled_i' + self.number, dist.Uniform())
		# # i = numpyro.deterministic('i', unscaled_i*90)
		# # i = numpyro.sample('i', dist.Normal(self.mu_i, self.sigma_i))
		# # i = numpyro.sample('i', dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

		# # i = 1.05
		# # i = numpyro.sample('i' + self.number, dist.Uniform())*(self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]
		# # end = time.time()
		# # f.write('i sampling time: '+ str(end-start)+ '\n')

		# #sample from circular normal distribution (in radians)
		# # i_rad = numpyro.sample('i_radians' + self.number, dist.VonMises(self.mu_i*jnp.pi/180, 1/((self.sigma_i*jnp.pi/180)**2)))
		# # i = numpyro.deterministic('i' + self.number, i_rad*180/jnp.pi)

		# # start = time.time()
		# unscaled_Va = numpyro.sample('unscaled_Va', dist.Uniform())  #* (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
		# Va = numpyro.deterministic('Va', unscaled_Va*(self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0])
		# # Va = 300
		# # end = time.time()
		# # f.write('Va sampling time: '+ str(end-start)+ '\n')

		# # ------- log normal distribution-------
		# # r_t = numpyro.sample('r_t' + self.number, dist.Uniform(
		# # ))*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

		# # r_t_sigma = jnp.log(self.r_t_bounds[1]/2)
		# # r_t_high = (jnp.log(self.r_t_bounds[2]) - jnp.log(self.r_t_bounds[1]))/r_t_sigma
		# # reparam_config = {"log_r_t": TransformReparam()}
		# # with numpyro.handlers.reparam(config=reparam_config):
		# #     log_r_t = numpyro.sample("log_r_t",dist.TransformedDistribution(
		# #             dist.TruncatedNormal(0.0, 1.0, high=r_t_high),
		# #             AffineTransform(jnp.log(self.r_t_bounds[1]), r_t_sigma),
		# #         ),
		# #     )

		# # ------- normal distribution -------
		# r_t_sigma = self.r_t_bounds[1]/2
		# r_t_mu = self.r_t_bounds[1]
		# r_t_max = self.r_t_bounds[2]
		# r_t_high = (r_t_max - r_t_mu)/r_t_sigma
		# r_t_low = (0.0 - r_t_mu)/r_t_sigma
		# # reparam_config = {"r_t": TransformReparam()}
		# # with numpyro.handlers.reparam(config=reparam_config):
		# #     r_t = numpyro.sample("r_t",dist.TransformedDistribution(
		# #             dist.TruncatedNormal(0.0, 1.0, high=r_t_high, low = r_t_low),
		# #             AffineTransform(r_t_mu,r_t_sigma),
		# #         ),
		# #     )
		# # unscaled_r_t = numpyro.sample('unscaled_r_t'+ self.number, dist.Normal())
		# # r_t = numpyro.deterministic('r_t', unscaled_r_t*r_t_sigma + r_t_mu)
		# # r_t = numpyro.sample('r_t', dist.Normal(r_t_mu, r_t_sigma))
		# # r_t = 1
		# # r_t = numpyro.deterministic('r_t', jnp.exp(log_r_t)) 
		# # end = time.time()
		# # f.write('r_t sampling time: '+ str(end-start)+ '\n')
		# unscaled_r_t = numpyro.sample('unscaled_r_t', dist.Uniform())
		# r_t = numpyro.deterministic('r_t', unscaled_r_t*4)
		# # r_t = 1

		# # start = time.time()
		# # sigma0 = numpyro.sample('sigma0'+ self.number, dist.Uniform(
		# # ))*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# # sigma0 = numpyro.sample('sigma0', dist.TruncatedDistribution(dist.Logistic(70, 20), low = 0, high = 250))
		# # reparam_config = {"sigma0": TransformReparam()}
		# # with numpyro.handlers.reparam(config=reparam_config):
		# #     sigma0 = numpyro.sample("sigma0",dist.TransformedDistribution(
		# #             dist.TruncatedDistribution(dist.Logistic(0, 1),low = (0-70)/20, high = (250-70)/20),
		# #             AffineTransform(70,20),
		# #         ),
		# #     )

		# unscaled_sigma0 = numpyro.sample('unscaled_sigma0'+ self.number, dist.Uniform())
		# sigma0 = numpyro.deterministic('sigma0', unscaled_sigma0*(600-self.sigma0_bounds[0]) + self.sigma0_bounds[0])
		# # sigma0 = 80
		# # unscaled_sigma0 = numpyro.sample('unscaled_sigma0'+ self.number,  dist.TruncatedDistribution(dist.Logistic(0, 1),low = (0-70)/20, high = (250-70)/20))
		# # unscaled_sigma0 = numpyro.sample('unscaled_sigma0'+ self.number, dist.Normal())
		# # sigma0 = numpyro.deterministic('sigma0', unscaled_sigma0*40 + 70)
		# # sigma0 = numpyro.deterministic('sigma0', sigma0_unit*70)
		# # sigma0 = 100
		# # end = time.time()
		# # f.write('sigma0 sampling time: '+ str(end-start)+ '\n')

		# # sigma0_max = numpyro.sample('sigma0_max'+ self.number, dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
		# # sigma0_scale =  numpyro.sample('sigma0_scale'+ self.number, dist.Uniform())*10
		# # sigma0_const = numpyro.sample('sigma0_const'+ self.number, dist.Uniform())*sigma0_max

		# # sampling the y axis velocity centroids
  

		# # start = time.time()
		# # unscaled_y0_vel = numpyro.sample('unscaled_y0_vel'+ self.number, dist.TruncatedNormal(high=(self.y_high -self.mu_y0_vel )/self.y0_std, low = (self.y_low -self.mu_y0_vel )/self.y0_std))
		# unscaled_y0_vel = numpyro.sample('unscaled_y0_vel'+ self.number, dist.Normal())

		# y0_vel = numpyro.deterministic('y0_vel', unscaled_y0_vel*self.y0_std + self.mu_y0_vel)
		
		# unscaled_x0_vel = numpyro.sample('unscaled_x0_vel'+ self.number, dist.Normal())

		# x0_vel = numpyro.deterministic('x0_vel', unscaled_x0_vel*self.y0_std + self.mu_y0_vel)
		# # y0_vel = numpyro.deterministic('y0_vel', unscaled_y0_vel*self.y0_std + self.mu_y0_vel)
		# # y0_vel = 15
		# # y0_vel = numpyro.sample("y0_vel", dist.TruncatedNormal(self.mu_y0_vel, self.y0_std, low=self.y_low, high=self.y_high ))
		# # reparam_config = {"y0_vel": TransformReparam()}
		# # with numpyro.handlers.reparam(config=reparam_config):
		# #     y0_vel = numpyro.sample("y0_vel",dist.TransformedDistribution(
		# #             dist.TruncatedNormal(0.0, 1.0, high=(self.y_high -self.mu_y0_vel )/self.y0_std, low = (self.y_low -self.mu_y0_vel )/self.y0_std),
		# #             AffineTransform(self.mu_y0_vel,self.y0_std),
		# #         ),
		# #     )


		# # end = time.time()
		# # f.write('y0_vel sampling time: '+ str(end-start)+ '\n')
		# # sample a global velicity shift v0:
		# # start = time.time()
		# # unscaled_v0 = numpyro.sample('unscaled_v0'+ self.number, dist.Normal())
		# # # v0 = norm.ppf(v0)*100
		# # v0 = numpyro.deterministic('v0', unscaled_v0*100)
		# v0 = 0
		# # end = time.time()
		# # f.write('v0 sampling time: '+ str(end-start)+ '\n')

		# return Pa, i, Va, r_t,sigma0, y0_vel, x0_vel, v0



# def compute_posterior_means(self, inference_data):
# 		"""
# 			Retreive the best sample from the MCMC chains for the main disk variables
# 		"""
# 		# best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
# 		# ), inference_data['sample_stats']['lp'].shape)

# 		# rescale all of the posteriors from uniform to the actual parameter space
# 		# rotation = float(inference_data.posterior['rotation' + self.number].median(dim=["chain", "draw"]))
# 		#create lists with variables and their scaling parameters 
# 		# variables = ['PA'+ self.number, 'i'+ self.number, 'Va'+ self.number, 'r_t'+ self.number, 'sigma0'+ self.number, 'y0_vel'+ self.number, 'v0'+ self.number]
# 		variables = ['Va'+ self.number]
# 		# variables = ['PA' + self.number, 'i'+ self.number, 'Va'+ self.number, 'r_t'+ self.number, 'sigma0_max'+ self.number,'sigma0_scale'+ self.number , 'sigma0_const'+ self.number, 'y0_vel'+ self.number, 'v0'+ self.number]
# 		#for variables drawn from uniform dist, the scaling parameters are (low, high) so mu and sigma are set to none
# 		# mus = [self.mu_PA + round(rotation)*180, self.mu_i, None, None, None, self.y0_vel, 0.0]
# 		# sigmas = [self.sigma_PA, self.sigma_i, None, None, None,2.0, 100.0]
# 		#when using numpyro.deterministic, don't need to change posteriors
# 		mus = [ None]
# 		sigmas = [None]

# 		#for variables drawn from normal dist, the scaling parameters are (mu, sigma) so low and high are set to none
# 		# highs = [None, self.i_high, self.Va_bounds[1], self.r_t_bounds[1], self.sigma0_bounds[1],None, None]
# 		# lows = [None,self.i_low, self.Va_bounds[0], self.r_t_bounds[0], self.sigma0_bounds[0], None, None]
# 		highs = [self.Va_bounds[1]]
# 		lows = [self.Va_bounds[0]]

# 		#find the best sample for each variable in the list of variables
# 		# best_sample = utils.find_best_sample(inference_data, variables, mus, sigmas, highs, lows, best_indices)
# 		#taking the median:
# 		# best_sample = utils.find_best_sample(inference_data, variables, mus, sigmas, highs, lows, MLS = None)


# 		# self.Va_mean = best_sample[0]
# 		# self.sigma0_mean_model*=0.5
# 		# self.Va_mean*=0.05
# 				# self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean, self.y0_vel_mean, self.v0_mean = best_sample

# 		#taking best sample
# 		# self.PA_mean =  jnp.array(inference_data.posterior['PA'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		# self.i_mean = jnp.array(inference_data.posterior['i'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		# self.y0_vel_mean = jnp.array(inference_data.posterior['y0_vel'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		# self.v0_mean = jnp.array(inference_data.posterior['v0'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))


# 		#update PA and i distributions to degrees
# 		# inference_data.posterior['PA'+ self.number] = inference_data.posterior['PA'+ self.number]
# 		# inference_data.posterior['i'+ self.number] = inference_data.posterior['i'+ self.number]

# 		# inference_data.prior['PA'+ self.number] = inference_data.prior['PA'+ self.number]
# 		# inference_data.prior['i'+ self.number] = inference_data.prior['i'+ self.number]
# 		#taking the median:
# 		self.PA_mean = jnp.array(inference_data.posterior['PA'+ self.number].median(dim=["chain", "draw"]))
# 		self.i_mean = jnp.array(inference_data.posterior['i'+ self.number].median(dim=["chain", "draw"]))
# 		self.y0_vel_mean = jnp.array(inference_data.posterior['y0_vel'+ self.number].median(dim=["chain", "draw"]))
# 		self.x0_vel_mean = jnp.array(inference_data.posterior['x0_vel'+ self.number].median(dim=["chain", "draw"]))
# 		self.v0_mean = 0 #jnp.array(inference_data.posterior['v0'+ self.number].median(dim=["chain", "draw"]))
# 		self.r_t_mean = jnp.array(inference_data.posterior['r_t'+ self.number].median(dim=["chain", "draw"]))
# 		self.sigma0_mean_model = jnp.array(inference_data.posterior['sigma0'+ self.number].median(dim=["chain", "draw"]))
# 		self.Va_mean = jnp.array(inference_data.posterior['Va'+ self.number].median(dim=["chain", "draw"]))
# 		# log_r_t_mean = jnp.array(inference_data.posterior['log_r_t'].median(dim=["chain", "draw"]))
# 		# print('r_t mean: ', self.r_t_mean)
# 		# print('log_r_t mean: ', log_r_t_mean)





# 		# self.PA_mean = 180 - self.PA_mean
# 				# self.Va_mean = jnp.array(inference_data.posterior['Va'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		# self.r_t_mean = jnp.array(inference_data.posterior['r_t'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		# self.sigma0_mean_model = jnp.array(inference_data.posterior['sigma0'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))


# 		return  self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.x0_vel_mean, self.v0_mean
# 		# return  self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean,  self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean, self.y0_vel_mean, self.v0_mean


# def compute_flux_posterior(self, inference_data, flux_type = 'auto'):

# 		best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
# 		), inference_data['sample_stats']['lp'].shape)

# 		inference_data.posterior['fluxes_scaling'+ self.number].data = inference_data.posterior['fluxes_scaling'+ self.number].data*(4-0.05) + 0.05 #*(1-0.1) + 0.1
# 		inference_data.prior['fluxes_scaling'+ self.number].data = inference_data.prior['fluxes_scaling'+ self.number].data*(4-0.05) + 0.05 #*(1-0.1) + 0.1

# 		# inference_data.posterior['fluxes_error_scaling'+ self.number].data = inference_data.posterior['fluxes_error_scaling'+ self.number].data*4 + 1
# 		# inference_data.prior['fluxes_error_scaling'+ self.number].data = inference_data.prior['fluxes_error_scaling'+ self.number].data*4 + 1

# 		# inference_data.posterior['regularization_strength'+ self.number].data = inference_data.posterior['regularization_strength'+ self.number].data*(1-0.01) + 0.01 #*(1-0.1) + 0.1
# 		# inference_data.prior['regularization_strength'+ self.number].data = inference_data.prior['regularization_strength'+ self.number].data*(1-0.01) + 0.01  #*(1-0.1) + 0.1
# 		#if the fluxes are manually rescaled in the prior, then rescale them
# 		# self.fluxes_scaling_mean = jnp.array(inference_data.posterior['fluxes_scaling'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		self.fluxes_scaling_mean = jnp.array(inference_data.posterior['fluxes_scaling'+ self.number].median(dim=["chain", "draw"]))
# 		print('Flux scaling mean: ', self.fluxes_scaling_mean)
# 		self.fluxes_error_scaling_mean = 1 #jnp.array(inference_data.posterior['fluxes_error_scaling'+ self.number].median(dim=["chain", "draw"]))
# 		# print(flux_type)
# 		if flux_type == 'manual':
# 			best_flux_sample = utils.find_best_sample(inference_data, ['fluxes'+ self.number], [self.mu*self.fluxes_scaling_mean], [self.std], [self.high], [self.low], best_indices)

# 		# self.fluxes_sample_mean = jnp.array(inference_data.posterior['fluxes'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
# 		self.fluxes_sample_mean = jnp.array(inference_data.posterior['fluxes'+ self.number].median(dim=["chain", "draw"]))

# 		# self.fluxes_sample_mean = self.mu
# 		self.fluxes_mean = jnp.zeros(self.direct_shape)
# 		self.fluxes_mean =self.fluxes_mean.at[self.masked_indices].set(self.fluxes_sample_mean)
# 		return self.fluxes_mean, self.fluxes_scaling_mean



# def inference_model(self, grism_object, obs_map, obs_error, mask = None):
# 		"""

# 		Model used to infer the disk parameters from the data => called in fitting.py as the forward
# 		model used for the inference

# 		"""
# 		# f = open("timing_total.txt", "a")
# 		# sample the fluxes within the mask from a truncated normal distribution
# 		# start = time.time()
		
# 		# end =  time.time()
# 		# Pa, i, Va, r_t, sigma0_max, sigma0_scale, sigma0_const, y0_vel, v0 = self.disk.sample_params()
# 		# f.write("Time to sample params: " + str(end-start) + "\n")


# 		fluxes = self.disk.sample_fluxes()
# 		# fluxes,r_eff, ellip = self.disk.sample_fluxes_parametric() 
# 		Pa, i, Va, r_t, sigma0, y0_vel, x0_vel, v0 = self.disk.sample_params()      

# 		# sample_fluxes_reparam = numpyro.handlers.reparam(self.disk.sample_fluxes, config={'fluxes': LocScaleReparam(centered = 0)})
# 		# fluxes = sample_fluxes_reparam()
# 		# end =  time.time()
# 		# f.write("Time to sample fluxes: " + str(end-start) + "\n")

# 		# oversample the fluxes to match the grism object
# 		# start = time.time()
# 		fluxes_high = utils.oversample(fluxes, grism_object.factor, grism_object.factor, method= 'bilinear')
# 		# end =  time.time()
# 		# f.write("Time to oversample fluxes: " + str(end-start) + "\n")

# 		# create new grid centered on those centroids
# 		# x = jnp.linspace(0 - self.x0_vel, self.im_shape[1]-1 - self.x0_vel, self.im_shape[1]*grism_object.factor)
# 		# y = jnp.linspace(0 - 15, self.im_shape[0]-1 -15, self.im_shape[0]*grism_object.factor)
# 		# X, Y = jnp.meshgrid(x, y)
# 		# x_10 = jnp.linspace(0 - 15, self.im_shape[1]-1 - 15, self.im_shape[1]*grism_object.factor*1)
# 		# y_10 = jnp.linspace(0 - 15, self.im_shape[0]-1 -15, self.im_shape[0]*grism_object.factor*1)
# 		image_shape = self.im_shape[0]
# 		# print(image_shape//2)
# 		x_10 = jnp.linspace(0 - x0_vel, image_shape - x0_vel - 1, image_shape*grism_object.factor)
# 		y_10 = jnp.linspace(0 - y0_vel, image_shape - y0_vel - 1, image_shape*grism_object.factor)
# 		X_10, Y_10 = jnp.meshgrid(x_10,y_10)
# 		# sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)

# 		# start  = time.time()
		
# 		# self.compute_factors(Pa, i,X_10, Y_10)
# 		velocities = jnp.asarray(self.v(X_10, Y_10, Pa, i, Va, r_t))
# 		# velocities = velocities.at[15,15].set(3e-14)
# 		# velocities = utils.resample(velocities, 10, 10)/10**2
# 		# end =  time.time()
# 		# f.write("Time to compute velocities: " + str(end-start) + "\n")

# 		# velocities = jnp.array(v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
# 		# velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')

# 		velocities_scaled = velocities + v0

# 		dispersions = sigma0*jnp.ones_like(velocities_scaled)
# 		# dispersions = self.sigma_disk(sigma0_max, sigma0_scale, sigma0_const, fluxes)

		
# 		# #make a data cube with the 'spectra' in each pixel
# 		# broadcast_velocity_space = grism_object.velocity_space
# 		# cube = fluxes*norm.pdf(broadcast_velocity_space, velocities, dispersions)

# 		# #the PSF is already oversampled here and in 3D form
# 		# PSF_kernel = grism_object.PSF

# 		# convolved_cube = convolve(cube, PSF_kernel, mode='same')

# 		# convolved_fluxes = jnp.sum(convolved_cube, axis=0)

# 		# convolved_velocities = jnp.mean(convolved_cube, axis = 2)
# 		# convolved_dispersions = jnp.std(convolved_cube, axis = 2)

# 		# start = time.time()
# 		self.model_map = grism_object.disperse(fluxes_high, velocities_scaled, dispersions)
# 		# end =  time.time()
# 		# f.write("Time to disperse: " + str(end-start) + "\n")
# 		# self.model_map = grism_object.disperse(convolved_fluxes, convolved_velocities, convolved_dispersions)

# 		# start = time.time()
# 		self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, self.wave_factor)
# 		# end = time.time()
# 		# f.write("Time to resample: " + str(end-start) + "\n")

# 		self.error_scaling = 1 #numpyro.sample('error_scaling', dist.Uniform(0, 1))*5
# 		#regularize the fluxes
		
# 		# reg_strength = numpyro.sample('regularization_strength', dist.Uniform()) #*(1-0.00001) + 0.00001

# 		# laplace_fluxes = convolve(fluxes, self.Laplace_kernel, mode='same')
# 		# sum and renormalize regularization term
# 		# threshold_grism = 0.2*obs_map.max()
# 		# sum_reg = jnp.sum(jnp.abs(laplace_fluxes))
# 		# #use a cut to be consistent  with grism/direct renormalization + avoid negative pixels
# 		# sum_reg_norm = sum_reg/jnp.sum(jnp.where(obs_map>threshold_grism,obs_map, 0.0)) #*self.model_map.shape[0]*self.model_map.shape[1]
# 		# error_reg = jnp.sqrt(jnp.sum(jnp.where(obs_map>threshold_grism,obs_error, 0.0)**2))/jnp.sum(jnp.where(obs_map>threshold_grism,obs_error, 0.0))
# 		# # error_reg_norm = error_reg*self.model_map.shape[0]*self.model_map.shape[1]/self.model_map.sum()

# 		#new renormalization of reg term

# 		# sum_reg = jnp.sum(jnp.abs(laplace_fluxes))/jnp.sum(jnp.abs(fluxes))
# 		# sum_reg_norm = sum_reg*jnp.max(obs_map)

# 		# error_reg = obs_error[jnp.unravel_index(jnp.argmax(obs_map), (obs_map.shape[0], obs_map.shape[1]))]


# 		# laplace_fluxes_r = jnp.reshape(sum_reg_norm, (1,1))
# 		# laplace_fluxes_err_r = jnp.reshape(error_reg, (1,1))

# 		# reshaping the grism items to add the regularization term
# 		# model_map_r = jnp.reshape(self.model_map, (1,self.model_map.shape[0]*self.model_map.shape[1]))
# 		# obs_error_r = jnp.reshape(obs_error, (1,obs_error.shape[0]*obs_error.shape[1]))
# 		# obs_map_r = jnp.reshape(obs_map, (1,obs_map.shape[0]*obs_map.shape[1]))

# 		# self.error_scaling = 1
# 		# start = time.time()
# 		#make a mask to only fit high sn regions in the grism


# 		# numpyro.sample('obs', dist.Normal(self.model_map[:,self.model_map.shape[1]//2-fluxes.shape[1]//2], self.error_scaling*obs_error[:,self.model_map.shape[1]//2-fluxes.shape[1]//2]), obs=obs_map[:,self.model_map.shape[1]//2-fluxes.shape[1]//2])
# 		mask = jnp.where(obs_map/obs_error < 5, 0, 1)
# 		model_masked = jnp.where(mask == 1, self.model_map, 0)
# 		obs_masked = jnp.where(mask == 1, obs_map, 0)
# 		obs_error_masked = jnp.where(mask == 1, obs_error, 1e6)

# 		numpyro.sample('obs', dist.Normal(model_masked, self.error_scaling*obs_error_masked), obs=obs_masked)

# 		# numpyro.sample('obs', dist.Normal(jnp.concatenate((model_map_r,reg_strength*laplace_fluxes_r), axis = 1),
# 		#             jnp.concatenate((self.error_scaling*obs_error_r, laplace_fluxes_err_r), axis = 1)), 
# 		#             obs=jnp.concatenate((obs_map_r, jnp.zeros_like(laplace_fluxes_r)), axis = 1))

# 		# end = time.time()
# 		# f.write("Time to sample obs: " + str(end-start) + "\n")
# 		# end_all = time.time()
# 		# f.write("Total time: " + str(end_all-start_all) + "\n")
