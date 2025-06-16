"""
	This Module contains functions that deal with the modelling (both image and its grism) 
	Differs from model because the functions used in the fitting are written with JAX instead of numpy
	
	Contains:
	----------------------------------------
	class Grism

		__init__ 
			

	class Image


	----------------------------------------
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ["Grism"]

from . import utils

import numpy as np
from astropy.io import ascii
from scipy import interpolate
from scipy.constants import c #in m/s
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import math
from scipy.constants import c
import matplotlib.pyplot as plt
from jax.scipy import special

import jax.numpy as jnp
from jax import image

from astropy.convolution import Gaussian1DKernel
from jax.scipy.signal import fftconvolve

#import time
import time 
import jax
jax.config.update('jax_enable_x64', True)



class Grism:
	def __init__(self, im_shape, im_scale = 0.031, icenter = 5, jcenter = 5, wavelength = 4.2 , wave_space = None, index_min = None, index_max = None, grism_filter = 'F444W', grism_module = 'A', grism_pupil = 'R', PSF = None):


		self.im_shape = im_shape #used to be self.im_shape
		self.im_scale = im_scale #used to be self.direct_scale

		self.set_detector_scale(0.0629) #setting to JWST resolution but made it a function so that it's easy to change as the outside user

		self.factor = int(self.detector_scale/self.im_scale) #factor between the model space and the observation space, spatially

		#this is the RA,DEC center of the galaxy as used in the grism data reduction to define the wavelength space
		#in order to remain accurate, self.factor should be uneven so that the centroids can be easiliy calculated
		#these centers are expressed in pixel INDICES on the cutout (original res) image
		self.icenter = icenter
		self.jcenter = jcenter


		#initialize attributes

		#center of the object on the detector image
		self.xcenter_detector = 1024 #xcenter_detector 
		self.ycenter_detector =  1024 # ycenter_detector

		#create the detector space 
		self.init_detector()
		
		self.wave_space = wave_space #already in the model resolution

		self.wave_scale = jnp.diff(self.wave_space)[0] #in microns, this is the scale of the wavelength space in the model

		self.index_min = index_min
		self.index_max = index_max
		self.wavelength = wavelength

		self.filter = grism_filter
		self.module = 'A' 
		self.module_lsf = grism_module
		self.pupil = grism_pupil

		#load the coefficients needed for the trace and dispersion
		self.load_coefficients()

		self.load_poly_factors(*self.w_opt)
		self.load_poly_coefficients()

		#initialize model grism detector cutout
		self.sh_beam = (self.im_shape,self.wave_space.shape[0])
		#full grism array
		self.grism_full = jnp.zeros(self.sh_beam)


		self.get_trace()

		self.compute_lsf_new()

		self.compute_PSF(PSF)

		self.set_wave_array()
	
	def __str__(self):
		return 'Grism object: \n' + ' - direct shape: ' + str(self.im_shape) + '\n - grism shape: ' + str(self.sh_beam) + '\n - wave_scale = ' + str(self.wave_scale) + '\n - factor = ' + str(self.factor) + '\n - i_center = ' + str(self.icenter) + '\n - j_center = ' + str(self.jcenter)

	def init_detector(self):
		self.detector_xmin = self.xcenter_detector - self.jcenter
		self.detector_xmax = self.xcenter_detector + (self.im_shape//self.factor-1-self.jcenter) #need to check that this still works but should do!

		detector_x_space_low = jnp.linspace(self.detector_xmin, self.detector_xmax, int(self.im_shape//self.factor))

		detector_x_space_low_2d = jnp.reshape(detector_x_space_low, (1, int(self.im_shape//self.factor)))

		#oversampling it to high resolution
		self.detector_space_1d = image.resize(detector_x_space_low_2d, (1,self.im_shape), method = 'linear')[0] #taking only the first row since all rows are the same - and I want it in 1D

		#the position on the detector of each pixel in the high res model
		self.detector_space_2d = self.detector_space_1d * jnp.ones_like(jnp.zeros( (self.im_shape,1) ))

	def load_coefficients(self):
		'''
			Loading the parameters for the tracing, dispersion, and sensitivity functions

			Parameters
			----------

			Returns
			----------
			parameters
		'''

		### select the filter that we will work on:
		tmp_filter = self.filter

		### Interested wavelength Range
		if tmp_filter == 'F444W': WRANGE = jnp.array([3.8, 5.1])
		elif tmp_filter == 'F322W2': WRANGE = jnp.array([2.4, 4.1])
		elif tmp_filter == 'F356W':  WRANGE = jnp.array([3.1, 4.0])
		elif tmp_filter == 'F277W':  WRANGE = jnp.array([2.4, 3.1])

		### Spectral tracing parameters:
		if tmp_filter in ['F277W', 'F356W']: disp_filter = 'F322W2'
		else: disp_filter = tmp_filter
		#no ../ because the open() function reads from terminal directory (not module directory)
		tb_order23_fit_AR = ascii.read('nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'A', 'R'))
		fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
		tb_order23_fit_BR = ascii.read('nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'B', 'R'))
		fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
		tb_order23_fit_AC = ascii.read('nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'A', 'C'))
		fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
		tb_order23_fit_BC = ascii.read('nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'B', 'C'))
		fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data

		### grism dispersion parameters:
		tb_fit_displ_AR = ascii.read('nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('A', "R"))
		w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
		tb_fit_displ_BR = ascii.read('nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('B', "R"))
		w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
		tb_fit_displ_AC = ascii.read('nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('A', "C"))
		w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
		tb_fit_displ_BC = ascii.read('nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('B', "C"))
		w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data

		### list of module/pupil and corresponding tracing/dispersion function:
		list_mod_pupil   = np.array(['AR', 'BR', 'AC', 'BC'])
		list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
		list_w_opt       = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])

		### Sensitivity curve:
		dir_fluxcal = 'nircam_grism/all_wfss_sensitivity/'
		tb_sens_AR = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat' % (disp_filter, 'A', 'R'))
		tb_sens_BR = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat'% (disp_filter, 'B', 'R'))
		tb_sens_AC = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat' % (disp_filter, 'A', 'C'))
		tb_sens_BC = ascii.read(dir_fluxcal + '%s_mod%s_grism%s_sensitivity.dat'% (disp_filter, 'B', 'C'))
		f_sens_AR = interpolate.UnivariateSpline(tb_sens_AR['wavelength'], tb_sens_AR['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
		f_sens_BR = interpolate.UnivariateSpline(tb_sens_BR['wavelength'], tb_sens_BR['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
		f_sens_AC = interpolate.UnivariateSpline(tb_sens_AC['wavelength'], tb_sens_AC['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
		f_sens_BC = interpolate.UnivariateSpline(tb_sens_BC['wavelength'], tb_sens_BC['DN/s/Jy'], ext = 'zeros', k = 1, s = 1e2)
		list_f_sens = np.array([f_sens_AR, f_sens_BR, f_sens_AC, f_sens_BC])

		self.fit_opt_fit = list_fit_opt_fit[list_mod_pupil == self.module + self.pupil][0] #module and pupil have to be taken from the grism image that we want to model
		self.w_opt = list_w_opt[list_mod_pupil == self.module + self.pupil][0]
		self.WRANGE = WRANGE
		self.f_sens = list_f_sens[list_mod_pupil == self.module + self.pupil][0]


	def get_trace(self):

		'''
		Assuming a position on the detector for the galaxy (which we assume to be the center of the detector because we are looking at combined grism data), 
		compute the dispersion space of the grism image, i.e. where does the central pixel end up on the detector if emitting at each wavelength in wave_space

		Return/Compute:

		-self.wavs: the wavelength corresponding to each pixel in the grism image, in the ref frame of the central pixel
		'''
		xpix = self.xcenter_detector
		ypix = self.ycenter_detector
		wave = self.wave_space

		xpix -= 1024
		ypix -= 1024
		wave -= 3.95
		xpix2 = xpix**2
		ypix2 = ypix**2
		wave2 = wave**2
		wave3 = wave**3

		#disperse the central pixel at each wavelength of your filter
		self.disp_space=  ((self.a01 + (self.a02 * xpix + self.a03 * ypix) + (self.a04 * xpix2 + self.a05 * xpix * ypix + self.a06 * ypix2)) +
	  					(self.b01 + (self.b02 * xpix + self.b03 * ypix) + (self.b04 * xpix2 + self.b05 * xpix * ypix + self.b06 * ypix2)) * wave +
						(self.c01 + (self.c02 * xpix + self.c03 * ypix)) * wave2 + (self.d01 ) * wave3)	
		self.disp_space = jnp.array(self.disp_space)
		# print('disp space: ', self.disp_space)
		wave += 3.95

		#create a dx space centered on 0 where your pixel is in the direct image, evenly spaced
		delta_dx = (jnp.max(self.disp_space) - jnp.min(self.disp_space))/ (self.disp_space.shape[0]-1) #the -2 is because you need to divide by the number of INTERVALS
		self.dxs = jnp.arange(jnp.min(self.disp_space), jnp.max(self.disp_space) + delta_dx, delta_dx)
		self.inverse_wave_disp = InterpolatedUnivariateSpline(self.disp_space[jnp.argsort(self.disp_space)], wave[jnp.argsort(self.disp_space)], k = 1)	
		self.wavs = self.inverse_wave_disp(self.dxs)

		return self.dxs, self.disp_space

	def set_wave_array(self):
		'''
			Compute the effective central wavelength of each pixel on the plane of the central pixel (i.e. how much are they spatially separated in wavelength space)
			Steps:
			1. Disperse each pixel with wavelength self.wavelength (and zero velocity) and see where it ends up on the detector
			2. Find the wavelength corresponding to each of those pixels (in the ref frame of the central pixels)

		'''

		#disperse each pixel with wavelength self.wavelength (and zero velocity)
		dispersion_indices = self.grism_dispersion(self.wavelength)

		#put the dxs in the rest frame of the central pixel (since otherwise they are dx wrt to their original pixel in self.detector_space_1d)
		dispersion_indices += (self.detector_space_1d - self.xcenter_detector)
		#for each dx, find the closest in the uniformly distrubuted dxs
		wave_indices = np.argmin(np.abs(self.dxs[np.newaxis,np.newaxis,:] - dispersion_indices[:,:,np.newaxis]), axis = 2)
		#translate this to a wavelength in the rest frame of the central pixel
		self.wave_array = self.wavs[wave_indices]

	def load_poly_factors(self,a01, a02, a03, a04, a05, a06, b01, b02, b03, b04, b05, b06, c01, c02, c03, d01):
		self.a01 = a01
		self.a02 = a02
		self.a03 = a03
		self.a04 = a04
		self.a05 = a05
		self.a06 = a06
		self.b01 = b01
		self.b02 = b02
		self.b03 = b03
		self.b04 = b04
		self.b05 = b05
		self.b06 = b06
		self.c01 = c01
		self.c02 = c02
		self.c03 = c03
		self.d01 = d01
	
	def load_poly_coefficients(self):

		xpix = self.detector_space_2d 
		# print('xpix: ', xpix[0])
		ypix = self.ycenter_detector * jnp.ones_like(xpix) #bcause we are not considering vertical dys, we can set this as if they are all on the same row
		xpix -= 1024
		ypix -= 1024
		xpix2 = xpix**2
		ypix2 = ypix**2
		#setitng coefficients for the whole grid of the cutout detector
		self.coef1 = self.a01 + (self.a02 * xpix + self.a03 * ypix) + (self.a04 * xpix2 + self.a05 * xpix * ypix + self.a06 * ypix2)
		self.coef2 = self.b01 + (self.b02 * xpix + self.b03 * ypix) + (self.b04 * xpix2 + self.b05 * xpix * ypix + self.b06 * ypix2)
		self.coef3 = self.c01 + (self.c02 * xpix + self.c03 * ypix)
		self.coef4 = self.d01*jnp.ones_like(xpix)



	def grism_dispersion(self, wave):
		'''
			from x0,y0 and the lamda of one pixel in direct image, get the dx in the grism image

			Parameters
			----------
			data 
				(x0,y0,lambda)

			Returns
			----------
			dx
				in the grism image
		'''

		wave -=3.95

		return ((self.coef1 + self.coef2 * wave + self.coef3 * wave**2 + self.coef4 * wave**3))

	def set_detector_scale(self, scale):
		self.detector_scale = scale


	def compute_lsf(self):

		#compute the sigma lsf for the wavelength of interest, wavelength must be in MICRONS
		R = 3.35*self.wavelength**4 - 41.9*self.wavelength**3 + 95.5*self.wavelength**2 + 536*self.wavelength - 240

		self.sigma_lsf = (1/2.36)*self.wavelength/R 
		# print('LSF: ', self.sigma_lsf)
		self.sigma_v_lsf = (1/2.36)*(c/1000)/R #put c in km/s #0.5*
		# print('LSF vel: ', self.sigma_v_lsf)

		#returning R for testing purposes
		return R


	def effective_sigma(self,kernel):
		"""Estimate the effective standard deviation of a normalized LSF kernel."""
		x = jnp.arange(kernel.shape[0]) - kernel.shape[0] // 2  # symmetric axis
		mean = jnp.sum(x * kernel)
		variance = jnp.sum(((x - mean) ** 2) * kernel)
		return jnp.sqrt(variance)
	def compute_lsf_new(self):   
		'''
			New LSF computed by Fengwu from new data - sum of two Gaussians
		'''
		if self.module_lsf == 'A':
			frac_1 = 0.679*jnp.log10(self.wavelength/4) + 0.604
			fwhm_1 = (2.23*jnp.log10(self.wavelength/4) + 2.22)/1000 #in microns
			fwhm_2 = (8.75*jnp.log10(self.wavelength/4) + 5.97)/1000 #in microns
		else:
			frac_1 = 1.584*jnp.log10(self.wavelength/4) + 0.557
			fwhm_1 = (3.5*jnp.log10(self.wavelength/4) + 2.22)/1000 #in microns
			fwhm_2 = (11.27*jnp.log10(self.wavelength/4) + 5.78)/1000 #in microns
		
		sigma_1 = fwhm_1/(2*math.sqrt(2*math.log(2)))
		sigma_2 = fwhm_2/(2*math.sqrt(2*math.log(2)))

		#compute the LSF kernel as the sum of the two gaussians
		kernel_size = int(6*max(sigma_1, sigma_2)/self.wave_scale) + 1 #in pixels, 6 sigma is the full width of the kernel
		#the std is divided by the wavelength/pixel to get it in units of pixels
		lsf_kernel = jnp.array(float(frac_1)*Gaussian1DKernel(float(sigma_1/self.wave_scale), x_size = kernel_size) + float(1-frac_1)*Gaussian1DKernel(float(sigma_2/self.wave_scale), x_size = kernel_size)) #make it into a jax array so it is jax-compatible
		#normalize the kernel to sum = 1
		self.lsf_kernel = lsf_kernel/jnp.sum(lsf_kernel)

		#compute the effective LSF for testing purposes
		self.sigma_lsf = self.effective_sigma(lsf_kernel)* self.wave_scale #in microns
		print(self.sigma_lsf, ' microns')

		self.sigma_v_lsf = self.sigma_lsf/(self.wavelength/(c/1000))

		R =self.wavelength/(self.sigma_lsf*(2*math.sqrt(2*math.log(2))))

		# Plot the resulting LSF
		x = np.arange(kernel_size) - kernel_size // 2
		#convert x from pixels to velocity space 
		vel_space  = x*self.wave_scale/(self.wavelength/(c/1000)) #in km/s

		# plt.figure(figsize=(6, 3))
		# plt.plot(vel_space, lsf_kernel, label='Composite LSF')
		# #plot the effective LSF 
		# # plt.axvline(x=0, color='k', linestyle='--', label='Central Pixel')
		# # plt.axvline(x=-self.sigma_lsf/self.wave_scale, color='r', linestyle='--', label='Effective LSF')
		# plt.xlabel('Pixels')
		# plt.ylabel('Amplitude')
		# plt.title('Instrument Line Spread Function (Sum of Gaussians)')
		# plt.legend()
		# plt.tight_layout()
		# plt.show()


		#returning R for testing purposes
		return float(R)
	
	def compute_PSF(self, PSF):

		#sets the grism object's oversampled PSF and the velocity space needed for the cube
		if self.factor == 1:
			self.oversampled_PSF = PSF
		else:
			# self.oversampled_PSF = utils.oversample_PSF(PSF, self.factor)
			self.oversampled_PSF = utils.oversample(PSF, self.factor, self.factor, method = 'bilinear')
			#crop it down to the central 9x9 pixels
			self.oversampled_PSF = self.oversampled_PSF[self.oversampled_PSF.shape[0]//2 - 5:self.oversampled_PSF.shape[0]//2 + 6, self.oversampled_PSF.shape[1]//2 - 5:self.oversampled_PSF.shape[1]//2 +6]
			#normalize the PSF to sum = 1
			self.oversampled_PSF = self.oversampled_PSF/jnp.sum(self.oversampled_PSF)
		# print('oversampled PSF sum = ', jnp.sum(self.oversampled_PSF ))
		# plt.imshow(self.oversampled_PSF)
		# plt.title('PSF')
		# plt.colorbar()
		# plt.show()
		self.PSF =  self.oversampled_PSF[:,:, jnp.newaxis]

		self.full_kernel = jnp.array(self.PSF) * self.lsf_kernel

	
	def disperse(self, F, V, D):
		'''
			Dispersion function going from flux space F to grism space G
		'''

		J_min = self.index_min
		J_max = self.index_max

		#self.wave_array contains the wavelength of each pixel in the grism image, in the ref frame of the central pixel
		wave_centers = self.wavelength*( V/(c/1000) ) + self.wave_array
		wave_sigmas = self.wavelength*(D/(c/1000) ) #the velocity dispersion doesn't need to be translated to the ref frame of the central pixel

		sigma_LSF = self.sigma_lsf

		#set the effective dispersion which also accounts for the LSF
		# wave_sigmas_eff = jnp.sqrt(jnp.square(wave_sigmas) + jnp.square(sigma_LSF)) 
		wave_sigmas_eff = wave_sigmas

		#make a 3D cube (spacial, spectral, wavelengths)
		mu = wave_centers[:,:,jnp.newaxis]
		sigma = wave_sigmas_eff[:,:,jnp.newaxis]
		
		#compute the edges of the wave space in order to evaluate the gaussian at those points - focusing only on the region of interest
		wave_space_crop = self.wave_space[J_min:J_max]
		wave_space_edges_prov= wave_space_crop[1:] - jnp.diff(wave_space_crop)/2
		wave_space_edges_prov2 = jnp.insert(wave_space_edges_prov, 0, wave_space_edges_prov[0] - jnp.diff(wave_space_crop)[0])
		wave_space_edges = jnp.append(wave_space_edges_prov2, wave_space_edges_prov2[-1] + jnp.diff(wave_space_crop)[-1])


		cdf = 0.5* (1 + special.erf( (wave_space_edges[jnp.newaxis,jnp.newaxis,:] - mu)/ (sigma*math.sqrt(2.)) ))
		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]

		cube = F[:,:,jnp.newaxis]*gaussian_matrix

		psf_cube = fftconvolve(cube, self.full_kernel, mode='same') 
		#collapse across the x axis
		grism_full = jnp.sum(psf_cube, axis = 1) 
		return grism_full
	
	#unused functions - if not needed REMOVE
	#if keep - then they need to be updated accordingly becasue they are not up to date!!!


	# def disperse_mock(self, F, V, D):
	# 	'''
	# 		Dispersion function going from flux space F to grism space G
	# 	'''

	# 	J_min = self.index_min
	# 	J_max = self.index_max

	# 	# wave_centers = self.wavelength*(1  + V/(c/1000) )
	# 	wave_centers = self.wavelength*( V/(c/1000) ) + self.wave_array
	# 	wave_sigmas = self.wavelength*(D/(c/1000) ) #*(1/2.36)
	# 	# print('wave_centers: ', wave_centers[0,0])
	# 	# print('wave_sigmas: ', wave_sigmas[0,0])
	# 	sigma_LSF = self.sigma_lsf
	# 	# print('sigma_LSF: ', sigma_LSF)
	# 	wave_sigmas_eff = jnp.sqrt(jnp.square(wave_sigmas) + jnp.square(sigma_LSF)) #*(1/2.36)
	# 	mu = wave_centers[:,:,jnp.newaxis]
	# 	sigma = wave_sigmas_eff[:,:,jnp.newaxis]

	# 	wave_space_crop = self.wave_space[J_min:J_max+1*self.wave_factor]
	# 	wave_space_edges_prov= wave_space_crop[1:] - jnp.diff(wave_space_crop)/2
	# 	wave_space_edges_prov2 = jnp.insert(wave_space_edges_prov, 0, wave_space_edges_prov[0] - jnp.diff(wave_space_crop)[0])
	# 	wave_space_edges = jnp.append(wave_space_edges_prov2, wave_space_edges_prov2[-1] + jnp.diff(wave_space_crop)[-1])
	# 	# print(sigma) 
	# 	# print(jnp.exp(-0.5*((self.wave_space[jnp.newaxis, jnp.newaxis, :]-mu)**2/sigma**2)))
	# 	# print(jnp.sqrt(2.0*jnp.pi)*sigma)
	# 	# cube= F[:, :, jnp.newaxis]*jnp.exp(-0.5*(((self.wave_space[jnp.newaxis, jnp.newaxis, J_min:J_max+1*self.wave_factor]-mu)/sigma)**2))/(jnp.sqrt(2.0*jnp.pi)*sigma)*1e-3
	# 	cdf = 0.5* (1 + special.erf( (wave_space_edges[jnp.newaxis,jnp.newaxis,:] - mu)/ (sigma*math.sqrt(2.)) ))
	# 	gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
	# 	gaussian_matrix_cut = gaussian_matrix # [:,:, J_min:J_max+1*self.wave_factor]
	# 	cube = F[:,:,jnp.newaxis]*gaussian_matrix_cut
	# 	# plt.plot(self.wave_space,jnp.exp(-0.5*((self.wave_space-mu)**2/sigma**2))/(jnp.sqrt(2.0*jnp.pi)*sigma) )
	# 	# plt.show()
	# 	# print(self.wave_space.shape)
	# 	# print(self.wavs.shape)
	# 	# cube = cube_full[:,:,J_min:J_max+1*self.wave_factor]
	# 	psf_cube = fftconvolve(cube, self.PSF, mode='same') #convolve(cube, self.full_kernel)

	# 	# grism_spectra = jnp.zeros((cube.shape[1],cube.shape[0]))
	# 	# grism_spectra_padded = jnp.pad(grism_spectra, [(0, 0), (cube.shape[2],cube.shape[2])], mode='constant')
	# 	# # at this stage, we need to use the instrument dispersion function

		
	# 	grism_full = jnp.sum(psf_cube, axis = 1) #grism_spectra_padded[:, cube.shape[0]:-cube.shape[0]] #jnp.sum(psf_cube, axis = 1)
	# 	 #grism_spectra_padded[:, 62:-62]
	# 	#remove all pixels with flux below 1e-6
	# 	# grism_full = jnp.where(grism_full < 1e-5, 0, grism_full)
	# 	# plt.imshow(grism_full, origin = 'lower')
	# 	# plt.colorbar()
	# 	# plt.show()
	# 	return grism_full
	
	# def compute_gaussian(self, V, D):
	# 	'''
	# 		Add explanation here
	# 	'''

	# 	# print('V: ', V)
	# 	# print('D: ', D)
	# 	#compute the dispersion dx0 of the central pixel 
	# 	# print(self.wave_space.shape, self.disp_space.shape)
	# 	# dx0 = jnp.interp(self.wavelength*(1  + V/(c/1000) ), self.wave_space[jnp.argsort(self.disp_space)], self.disp_space[jnp.argsort(self.disp_space)])
	# 	dx0 = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + V/(c/1000) )), *self.w_opt)
	# 	DX = self.detector_space_2d - self.xcenter_detector + dx0 
	# 	# print(DX.type(), self.disp_space.type(), self.wave_space.type())

	# 	# J = self.compute_disp_index(DX, self.disp_space)
	# 	# J0 = self.compute_disp_index(dx0, self.disp_space)

	# 	# J= jnp.argmin(jnp.abs(self.disp_space - DX[:,:, None]), axis = 2)
	# 	# J0 = jnp.argmin(jnp.abs(self.disp_space - dx0[:,:, None]),axis =2)

	# 	J_min = self.index_min
	# 	J_max = self.index_max

	# 	# J = (jnp.rint(J)).astype(int)
	# 	# J0 = (jnp.rint(J0)).astype(int)

	# 	# print(J)


	# 	#compute the dispersion due to the LSF
	# 	D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
	# 	# print('D_LSF = ', D_LSF)

	# 	#add the LSF to the dispersion: 
	# 	CORR_D = jnp.sqrt( D**2 +  D_LSF**2)
	# 	DX_disp = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + (V+CORR_D)/(c/1000) )), *self.w_opt)
	# 	DX_disp_final = self.detector_space_2d - self.xcenter_detector + DX_disp - DX



	# 	NEW_J, NEW_K = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor-1, J_max-J_min+1*self.wave_factor))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
	# 	# print(NEW_J)
	# 	# grism_full = jnp.zeros(self.sh_beam)
	# 	# print(self.sh_beam)

	# 	# mu = self.wave_space[J[:,NEW_K]]
	# 	# sigma = CORR_D[:, NEW_K]/(c/1000)*self.wave_space[J[:,NEW_K]]

	# 	# sigma = CORR_D[:, NEW_K]/(c/1000)*self.wave_space[J[:,NEW_K]]
	# 	# compute the edges of the disp_space array

	# 	disp_space_edges_prov= self.disp_space[1:] - jnp.diff(self.disp_space)/2
	# 	disp_space_edges_prov2 = jnp.insert(disp_space_edges_prov, 0, disp_space_edges_prov[0] - jnp.diff(self.disp_space)[0])
	# 	disp_space_edges = jnp.append(disp_space_edges_prov2, disp_space_edges_prov2[-1] + jnp.diff(self.disp_space)[-1])

	# 	# print(disp_space_edges)
	# 	# cdf = 0.5* (1 + special.erf( (self.wave_space[NEW_J] - mu)/ (sigma*math.sqrt(2.)) ))
	# 	# prev_wave = self.wave_space[NEW_J] - self.wave_scale
	# 	# cdf_prev = 0.5* (1 + special.erf( (prev_wave - mu)/ (sigma*math.sqrt(2.)) ))
	# 	# gaussian_matrix = cdf[:,NEW_K,NEW_J-J_min] - cdf_prev[:,NEW_K,NEW_J-J_min]
	# 	# print(self.disp_space[NEW_J].shape)
	# 	# print(mu.shape)
	# 	# print(sigma.shape)

	# 	# compute the gaussian matrix in 10 times higher resolution and then sample it down

	# 	# dx0 = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + V/(c/1000) )), *self.w_opt)
	# 	# DX = self.detector_space_2d - self.xcenter_detector + dx0 

	# 	# J_min = self.index_min
	# 	# J_max = self.index_max

	# 	# D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
	# 	# # print('D_LSF = ', D_LSF)

	# 	# #add the LSF to the dispersion: 
	# 	# CORR_D = jnp.sqrt( D**2 +  D_LSF**2)
	# 	# DX_disp = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + (V+CORR_D)/(c/1000) )), *self.w_opt)
	# 	# DX_disp_final = self.detector_space_2d - self.xcenter_detector + DX_disp - DX

	# 	# disp_space_high = jnp.linspace(self.disp_space[0], self.disp_space[-1], 10*self.disp_space.shape[0])
	# 	# disp_space_edges_prov= disp_space_high[1:] - jnp.diff(disp_space_high)/2
	# 	# disp_space_edges_prov2 = jnp.insert(disp_space_edges_prov, 0, disp_space_edges_prov[0] - jnp.diff(disp_space_high)[0])
	# 	# disp_space_edges = jnp.append(disp_space_edges_prov2, disp_space_edges_prov2[-1] + jnp.diff(disp_space_high)[-1])


	# 	# NEW_J_cdf, NEW_K_cdf = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor, 34*self.wave_factor*10+1))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
	# 	# mu = DX[:,NEW_K_cdf]
	# 	# sigma = DX_disp_final[:,NEW_K_cdf]
	# 	# cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
	# 	# gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
	# 	# # gaussian_matrix_low = np.zeros((14,14,34))
	# 	# # for i in range(14):
	# 	# # 	blocks = gaussian_matrix[i].reshape(int(gaussian_matrix[i].shape[0]/(1)), 1, int(gaussian_matrix[i].shape[1]/10), 10)
	# 	# # 	gaussian_matrix_low[i] = jnp.sum(blocks, axis=(1,3))
	# 	# plt.imshow(gaussian_matrix[1])
	# 	# plt.show()
	# 	# # print('Gauss shape pre :' ,gaussian_matrix.shape)
	# 	# blocks = gaussian_matrix.reshape((V.shape[0],V.shape[0], int(gaussian_matrix.shape[2]/10), 10))
	# 	# gaussian_matrix = jnp.sum(blocks, axis=-1)
	# 	# # print('Gauss shape post :' ,gaussian_matrix.shape)


	# 	NEW_J_cdf, NEW_K_cdf = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor, J_max-J_min+1*self.wave_factor+1))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
	# 	# print(NEW_K_cdf)
	# 	mu = DX[:,NEW_K_cdf]
	# 	sigma = DX_disp_final[:,NEW_K_cdf]
	# 	cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
	# 	gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
	# 	# plt.imshow(gaussian_matrix[14], origin = 'lower')
	# 	# plt.show()
	# 	# print(gaussian_matrix.shape)
	# 	return gaussian_matrix

	# def get_beam_cutout(self, wave_min, wave_max, wave_space, grism_full):
	# 	'''
	# 		get cutout of the full grism spectrum from wave_min to wave_max
	# 		in the implementation these can be taken by finding wave_min and wave_max on the observed 2D image

	# 	'''

	# 	j_min, j_max = self.compute_wave_index(jnp.array([wave_min, wave_max]), wave_space)
	# 	j_min = 360
	# 	j_max = 370
	# 	self.grism_cutout = grism_full[:, j_min:j_max]

	# 	return self.grism_cutout

	# def set_sensitivity(self):
	

	# 	self.sensitivity = self.f_sens(self.wave_space)
	# 	self.compute_lsf()

	# def set_wavelength(self, wavelength):

	# 	self.wavelength = wavelength

	# def set_wave_scale(self, scale):

		'''
			Sets the scale of the grism image to 'scale'. Modifies all relevant parameters in the class so that when the disperse function is called, 
			the grism wavelength space has the appropriate scale

			Parameters
			----------

			Returns
			----------
		'''

		self.wave_scale = scale
		self.sh_beam = (self.sh[0],((self.WRANGE[1]-self.WRANGE[0])/self.wave_scale).astype(int))
		#full grism array
		self.grism_full = jnp.zeros(self.sh_beam)

		self.get_trace()

		self.compute_lsf()

		self.set_sensitivity()