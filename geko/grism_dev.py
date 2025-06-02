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
from astropy import wcs
from astropy.io import fits
from scipy import interpolate
from scipy.constants import c #in m/s
from astropy.coordinates import SkyCoord
# from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from astropy import units as u
import astropy
import math
from scipy.constants import c
import matplotlib.pyplot as plt
from jax.scipy import special
from jax.scipy.stats import norm
from scipy import signal

from scipy import ndimage

import jax.numpy as jnp
from jax import image

from reproject import reproject_adaptive
from photutils.centroids import centroid_1dg
from numpy.fft import rfftn, irfftn, fftshift
from astropy.convolution import Gaussian1DKernel
from jax.scipy.signal import convolve, fftconvolve

#import time
import time 
import jax
jax.config.update('jax_enable_x64', True)



class Grism:
	def __init__(self, direct = None, direct_scale = 0.03, factor = 1, y_factor = 1, icenter = 5, jcenter = 5, segmentation = None, xcenter_detector = 1024, ycenter_detector = 1024, 
		wavelength = 4.2 , redshift = 7.2, wave_space = None, wave_factor = 1, wave_scale = 0.001, index_min = None, index_max = None, grism_filter = 'F444W', grism_module = 'A', grism_pupil = 'R', higher_res = False, PSF = None):


		#direct is the image getting dispersed
		self.direct = jnp.array(direct) # initiliazed with the real image (not the high res model image)
		# self.sh = direct.shape
		self.sh_high = [direct.shape[0]*factor,	direct.shape[1]*factor]  #this is the updated shape of the flux image that will be dispersed (so high res model image)
		self.direct_scale_high = direct_scale/factor #this is the updated scale of the flux image that will be dispersed (so high res model image)
		# self.direct_scale = direct_scale

		self.detector_scale = 0.0629

		self.direct_factor_high = round(0.0629/self.direct_scale_high)

		self.icenter = icenter
		self.jcenter = jcenter

		self.icenter_high = icenter*factor
		self.jcenter_high = jcenter*factor 

		self.factor = factor
		self.wave_factor = wave_factor
		self.y_factor = y_factor

		#segmentation map over the object pixels that need to be considered
		# self.seg = jnp.array(segmentation)

		# #full 1048x1048 image of the dispersion field
		# self.detector_image = detector_image

		#initialize attributes

		#center of the object on the detector image
		self.xcenter_detector = xcenter_detector 
		self.ycenter_detector =  ycenter_detector

		# self.detector_xmin = self.xcenter_detector - (self.jcenter_high-1)/self.direct_factor_high
		# self.detector_xmax = self.xcenter_detector + (self.sh_high[1]-1-self.jcenter_high)/self.direct_factor_high

		self.detector_xmin = self.xcenter_detector - self.jcenter/y_factor
		self.detector_xmax = self.xcenter_detector + (self.direct.shape[0]-1-self.jcenter)/y_factor
		detector_x_space_low = jnp.linspace(self.detector_xmin, self.detector_xmax, direct.shape[0])
		spacing = detector_x_space_low[1] - detector_x_space_low[0]
		# print(detector_x_space_low)
		# Create new positions at twice the resolution
		# new_spacing = spacing / factor
		# detector_x_space_high = np.arange(detector_x_space_low[0] - new_spacing/2, 
		# 						detector_x_space_low[-1] + new_spacing, 
		# 						new_spacing)
		# if factor == 1:
		# 	self.detector_x_space = detector_x_space_low
		# else:
		# 	self.detector_x_space = detector_x_space_high

		# self.detector_x_space = jnp.linspace(self.detector_xmin, self.detector_xmax, self.sh_high[0])
		detector_x_space_low_2d = jnp.reshape(detector_x_space_low, (1, direct.shape[0]))

		self.detector_x_space = image.resize(detector_x_space_low_2d, (1,self.sh_high[0]), method = 'linear')[0]

		# print('det x space: ', self.detector_x_space)

		# self.detector_position = self.detector_x_space * jnp.ones_like(np.zeros( (self.sh[0]*factor,1) ))
		# self.detector_position = self.detector_x_space * jnp.ones_like(jnp.zeros( (self.sh[0]*factor,1) ))
		self.detector_position = self.detector_x_space * jnp.ones_like(jnp.zeros( (self.sh_high[0],1) ))


		# print(self.detector_position)

		#these are already in the high model
		self.wave_scale = wave_scale
		
		self.wave_space = wave_space 
		# print('wave space grism: ', wave_space)
		#make a higer res wave_space => the computed disp space will also be higher res
		# wave_min = wave_space[0]
		# wave_max = wave_space[len(self.wave_space)-1]
		# self.wave_space = jnp.linspace(wave_min, wave_max + wave_scale, int((wave_max-wave_min)/ self.wave_scale))

		self.index_min = index_min
		self.index_max = index_max
		self.wavelength = wavelength
		self.redshift = redshift

		self.filter = grism_filter
		self.module = 'A' 
		self.module_lsf = grism_module
		self.pupil = grism_pupil

		self.load_coefficients()

		self.load_poly_factors(*self.w_opt)
		self.load_poly_coefficients()

		# self.sh_beam = (self.sh[0]*factor,self.wave_space.shape[0])
		self.sh_beam = (self.sh_high[0],self.wave_space.shape[0])
		#full grism array
		self.grism_full = jnp.zeros(self.sh_beam)

		

		self.get_trace()

		# self.dx_of_lambda = interpolate.UnivariateSpline(self.wave_space[jnp.argsort(self.disp_space)], self.disp_space[jnp.argsort(self.disp_space)], s = 0, k = 1)

		# self.dx_to_wave = interpolate.UnivariateSpline(self.disp_space[jnp.argsort(self.disp_space)], self.wave_space[jnp.argsort(self.disp_space)], s = 0, k = 1)

		self.compute_lsf_new()
		# print(self.sigma_v_lsf)

		self.set_sensitivity()

		self.compute_PSF(PSF)

		# disp_grid = self.grism_dispersion(self.wavelength)
		# print(np.diff(disp_grid))
		self.set_wave_array()
	
	def __str__(self):
		return 'Grism object: \n' + ' - direct shape: ' + str(self.sh_high) + '\n - grism shape: ' + str(self.sh_beam) + '\n - wave_scale = ' + str(self.wave_scale) + '\n - factor = ' + str(self.factor) + '\n - wave_factor = ' + str(self.wave_factor) + '\n - i_center = ' + str(self.icenter) + '\n - j_center = ' + str(self.jcenter)


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
		tb_order23_fit_AR = ascii.read('../nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'A', 'R'))
		fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
		tb_order23_fit_BR = ascii.read('../nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'B', 'R'))
		fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
		tb_order23_fit_AC = ascii.read('../nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'A', 'C'))
		fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
		tb_order23_fit_BC = ascii.read('../nircam_grism/FS_grism_config/DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'B', 'C'))
		fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data

		### grism dispersion parameters:
		tb_fit_displ_AR = ascii.read('../nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('A', "R"))
		w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
		tb_fit_displ_BR = ascii.read('../nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('B', "R"))
		w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
		tb_fit_displ_AC = ascii.read('../nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('A', "C"))
		w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
		tb_fit_displ_BC = ascii.read('../nircam_grism/FS_grism_config/DISPL_mod%s_grism%s.txt' % ('B', "C"))
		w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data

		### list of module/pupil and corresponding tracing/dispersion function:
		list_mod_pupil   = np.array(['AR', 'BR', 'AC', 'BC'])
		list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
		list_w_opt       = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])

		### Sensitivity curve:
		dir_fluxcal = '../nircam_grism/all_wfss_sensitivity/'
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

	def set_wave_array(self):

		#disperse each pixel with wavelength self.wavelength (and zero velocity)
		dispersion_indices = self.grism_dispersion(self.wavelength)
		# print('dispersion indices: ', dispersion_indices)
		# print(self.detector_x_space[0])
		# print(self.xcenter_detector)
		dispersion_indices += (self.detector_x_space - self.xcenter_detector)
		wave_indices = np.argmin(np.abs(self.dxs[np.newaxis,np.newaxis,:] - dispersion_indices[:,:,np.newaxis]), axis = 2)
		self.wavs = self.wave_space[jnp.argsort(self.wave_space)] #temporary for testing - need to sort this
		self.wave_array = self.wavs[wave_indices]
		# print(self.wave_scale*self.direct.shape[0]//2)
		# self.wave_array = jnp.linspace(self.wavelength - (self.wave_scale*self.wave_factor)*(self.direct.shape[0]//2), self.wavelength + (self.wave_scale*self.wave_factor)*(self.direct.shape[0]//2), self.direct.shape[0]*self.factor)
		# self.wave_array = jnp.linspace(self.wavelength - (self.wave_scale*self.wave_factor)*(self.direct.shape[0]//2), self.wavelength + (self.wave_scale*self.wave_factor)*(self.direct.shape[0]//2), self.direct.shape[0]*self.factor)
		# wave_array_low = jnp.linspace(self.wavelength - (self.wave_scale*self.wave_factor)*(self.jcenter), self.wavelength + (self.wave_scale*self.wave_factor)*(self.direct.shape[0]-self.jcenter-1), self.direct.shape[0]) 
		# spacing = wave_array_low[1] - wave_array_low[0]
    
		# # Create new positions at twice the resolution
		# new_spacing = spacing / self.factor
		# wave_array_high = np.arange(wave_array_low[0] - new_spacing/2, 
		# 						wave_array_low[-1] + new_spacing, 
		# 						new_spacing)
		# if self.factor == 1:
		# 	self.wave_array = wave_array_low
		# else:
		# 	self.wave_array = wave_array_high
		
		# self.wave_array = jnp.linspace(self.wavelength - (self.wave_scale*self.wave_factor)*(15), self.wavelength + (self.wave_scale*self.wave_factor)*(self.direct.shape[0]-15-1), self.direct.shape[0]*self.factor)
		# print('wave array: ', self.wave_array)
		# print('det space: ', self.detector_x_space)

	def get_trace(self):
		xpix = self.xcenter_detector
		ypix = self.ycenter_detector
		wave = self.wave_space

		# wave_low = jnp.arrange(wave_high[0], wave_high[-1], self.wave_scale*self.wave_factor)
		# wave= wave_low

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
		self.dxs = jnp.arange(jnp.min(self.disp_space), jnp.max(self.disp_space), 1/self.wave_factor)  #- self.xcenter_detector
		# print('dx central pixel: ', self.dxs)
		#obtain the wavelength corresponding to each of those pixels

		#commenting these 2 out for now because they are not being used and the spline from jax_cosmo gave me trouble in the past I think

		# self.inverse_wave_disp = InterpolatedUnivariateSpline(self.disp_space[jnp.argsort(self.disp_space)], wave[jnp.argsort(self.disp_space)], k = 1)	
		# self.wavs = self.inverse_wave_disp(self.dxs)
		# print('self.wavs: ', self.wavs)
		# print('wavelengths: ', np.diff(self.wavs))
		
		# self.disp_space = jnp.linspace(self.disp_space_low[0], self.disp_space_low[-1], self.disp_space_low.shape[0]*self.wave_factor)

		# print('disp_space min max: ', self.disp_space[0], self.disp_space[-1])

	def set_wavelength(self, wavelength):
		self.wavelength = wavelength

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

		xpix = self.detector_position
		# print('xpix: ', xpix[0])
		ypix = 1024 * jnp.ones_like(xpix)
		xpix -= 1024
		ypix -= 1024
		xpix2 = xpix**2
		ypix2 = ypix**2

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


	def set_sensitivity(self):

		self.sensitivity = self.f_sens(self.wave_space)
		self.compute_lsf()

	def set_wave_scale(self, scale):

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

	def compute_lsf(self):

		#compute the sigma lsf for the wavelength of interest, wavelength must be in MICRONS
		R = 3.35*self.wavelength**4 - 41.9*self.wavelength**3 + 95.5*self.wavelength**2 + 536*self.wavelength - 240

		self.sigma_lsf = (1/2.36)*self.wavelength/R 
		# print('LSF: ', self.sigma_lsf)
		self.sigma_v_lsf = (1/2.36)*(c/1000)/R #put c in km/s #0.5*
		# print('LSF vel: ', self.sigma_v_lsf)

		#returning R for testing purposes
		return R

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

		self.sigma_lsf = math.sqrt(frac_1*sigma_1**2 + (1-frac_1)*sigma_2**2)*0.617

		self.sigma_v_lsf = self.sigma_lsf/(self.wavelength/(c/1000))

		R =self.wavelength/(self.sigma_lsf*(2*math.sqrt(2*math.log(2))))

		#returning R for testing purposes
		return R
	
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

		# velocity_space = jnp.linspace(-1000, 1000, 2001)
		# self.velocity_space = jnp.broadcast_to(velocity_space[:, jnp.newaxis, jnp.newaxis], (velocity_space.size, self.sh_high[0], self.sh_high[0]))
		gaussian_kernel = Gaussian1DKernel(float(self.sigma_lsf/self.wave_scale))
		LSF = gaussian_kernel.array
		LSF = LSF[LSF.shape[0]//2 - 5:LSF.shape[0]//2 + 5]
		#normalize LSF kernel to sum = 1
		LSF = LSF/jnp.sum(LSF)
		# print(LSF)
		# create a 3D kernel with 2D PSF and 1D LSF
		# self.full_kernel = np.array(self.PSF) * np.broadcast_to(np.array(LSF)[:, np.newaxis,np.newaxis],(np.array(LSF).size,np.array(self.PSF).shape[0],np.array(self.PSF).shape[0]))
		self.full_kernel = jnp.array(self.PSF) * LSF


	def disperse_old(self, F, V, D):
		'''
			Dispersion function going from flux space F to grism space G
		'''

		J_min = self.index_min
		J_max = self.index_max

		DX = self.grism_dispersion(self.wavelength*(1  + V/(c/1000) ))
		x_grism = DX + self.detector_position
		# print(DX[15])
		# print(x_grism)

		#compute the dispersion due to the LSF
		D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
		# print('D_LSF: ', np.unique(D_LSF))
		# print('D: ', np.unique(D))
		#add the LSF to the dispersion: 
		CORR_D = jnp.sqrt( jnp.square(D)+  jnp.square(D_LSF))
		# CORR_D = D #jnp.where(V>0, D, D)
		# print('CORR_D: ', CORR_D[0,0])
		# print(self.wavelength)
		# print('V: ', V)

		# start = time.time()
		DX_disp= self.grism_dispersion(self.wavelength*(1  + (CORR_D[0,0])/(c/1000) )) 
		DX_null = self.grism_dispersion(self.wavelength*(1  + (0.0)/(c/1000) )) 
		x_grism_sigma = DX_disp - DX_null
		# print(DX_disp[15])
		# print(self.detector_position[15])
		# print(DX[15])
		# print(x_grism_sigma)
		
		grism_space = self.disp_space + 1024
		grism_space_edges_prov= grism_space[1:] - jnp.diff(grism_space)/2
		grism_space_edges_prov2 = jnp.insert(grism_space_edges_prov, 0, grism_space_edges_prov[0] - jnp.diff(grism_space)[0])
		grism_space_edges = jnp.append(grism_space_edges_prov2, grism_space_edges_prov2[-1] + jnp.diff(grism_space)[-1])


		mu = x_grism[:,:,jnp.newaxis]
		# print('mu shape: ', mu.shape)

		sigma = x_grism_sigma[:,:,jnp.newaxis]
		# print()

		cdf = 0.5* (1 + special.erf( (grism_space_edges[jnp.newaxis,jnp.newaxis,:] - mu)/ (sigma*math.sqrt(2.)) ))

		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		# print('gaussian_matrix shape: ', gaussian_matrix.shape)
		# cube = F[:, :, jnp.newaxis]*jnp.exp(-0.5*((grism_space[jnp.newaxis, jnp.newaxis, J_min:J_max+1*self.wave_factor]-mu)**2/sigma**2))/(jnp.sqrt(2.0*jnp.pi)*sigma) #*1e-4
		#crop and renormalize the gaussian so that it sums to 1
		gaussian_matrix_cut = gaussian_matrix[:,:, J_min:J_max+1*self.wave_factor] #/jnp.sum(gaussian_matrix[:,:, J_min:J_max+1*self.wave_factor], axis = 2)[:,:,jnp.newaxis]
		# print(gaussian_matrix_cut[1,1].sum())
		cube = F[:,:,jnp.newaxis]*gaussian_matrix_cut

		psf_cube = cube #fftconvolve(cube, self.PSF, mode='same') #convolve(cube, self.full_kernel)

		grism_full = jnp.sum(psf_cube, axis = 1)
		# end = time.time()
		
		# self.f.write("matmul time: " + str(end-start) + "\n")

		# self.f.close()

		return grism_full
	
	def disperse(self, F, V, D):
		'''
			Dispersion function going from flux space F to grism space G
		'''

		J_min = self.index_min
		J_max = self.index_max

		# wave_centers = self.wavelength*(1  + V/(c/1000) )
		wave_centers = self.wavelength*( V/(c/1000) ) + self.wave_array
		wave_sigmas = self.wavelength*(D/(c/1000) ) #*(1/2.36)
		# print('wave_centers: ', wave_centers[0,0])
		# print('wave_sigmas: ', wave_sigmas[0,0])
		sigma_LSF = self.sigma_lsf
		# print('sigma_LSF: ', sigma_LSF)
		# print('wave_sigma: ', wave_sigmas)
		# print('sigma_LSF: ', sigma_LSF)
		wave_sigmas_eff = jnp.sqrt(jnp.square(wave_sigmas) + jnp.square(sigma_LSF)) #*(1/2.36)
		mu = wave_centers[:,:,jnp.newaxis]
		sigma = wave_sigmas_eff[:,:,jnp.newaxis]

		wave_space_crop = self.wave_space[J_min:J_max+1*self.wave_factor]
		wave_space_edges_prov= wave_space_crop[1:] - jnp.diff(wave_space_crop)/2
		wave_space_edges_prov2 = jnp.insert(wave_space_edges_prov, 0, wave_space_edges_prov[0] - jnp.diff(wave_space_crop)[0])
		wave_space_edges = jnp.append(wave_space_edges_prov2, wave_space_edges_prov2[-1] + jnp.diff(wave_space_crop)[-1])
		# print(sigma) 
		# print(jnp.exp(-0.5*((self.wave_space[jnp.newaxis, jnp.newaxis, :]-mu)**2/sigma**2)))
		# print(jnp.sqrt(2.0*jnp.pi)*sigma)
		# cube= F[:, :, jnp.newaxis]*jnp.exp(-0.5*(((self.wave_space[jnp.newaxis, jnp.newaxis, J_min:J_max+1*self.wave_factor]-mu)/sigma)**2))/(jnp.sqrt(2.0*jnp.pi)*sigma)*1e-3
		cdf = 0.5* (1 + special.erf( (wave_space_edges[jnp.newaxis,jnp.newaxis,:] - mu)/ (sigma*math.sqrt(2.)) ))
		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		gaussian_matrix_cut = gaussian_matrix # [:,:, J_min:J_max+1*self.wave_factor]
		cube = F[:,:,jnp.newaxis]*gaussian_matrix_cut
		# plt.plot(self.wave_space,jnp.exp(-0.5*((self.wave_space-mu)**2/sigma**2))/(jnp.sqrt(2.0*jnp.pi)*sigma) )
		# plt.show()
		# print(self.wave_space.shape)
		# print(self.wavs.shape)
		# cube = cube_full[:,:,J_min:J_max+1*self.wave_factor]
		psf_cube = fftconvolve(cube, self.PSF, mode='same') #convolve(cube, self.full_kernel)

		grism_full = jnp.sum(psf_cube, axis = 1) #grism_spectra_padded[:, cube.shape[0]:-cube.shape[0]] #jnp.sum(psf_cube, axis = 1)
		 #grism_spectra_padded[:, 62:-62]
		#remove all pixels with flux below 1e-6
		# grism_full = jnp.where(grism_full < 1e-5, 0, grism_full)
		# plt.imshow(grism_full, origin = 'lower')
		# plt.colorbar()
		# plt.show()
		return grism_full
	
	def disperse_mock(self, F, V, D):
		'''
			Dispersion function going from flux space F to grism space G
		'''

		J_min = self.index_min
		J_max = self.index_max

		# wave_centers = self.wavelength*(1  + V/(c/1000) )
		wave_centers = self.wavelength*( V/(c/1000) ) + self.wave_array
		wave_sigmas = self.wavelength*(D/(c/1000) ) #*(1/2.36)
		# print('wave_centers: ', wave_centers[0,0])
		# print('wave_sigmas: ', wave_sigmas[0,0])
		sigma_LSF = self.sigma_lsf
		# print('sigma_LSF: ', sigma_LSF)
		wave_sigmas_eff = jnp.sqrt(jnp.square(wave_sigmas) + jnp.square(sigma_LSF)) #*(1/2.36)
		mu = wave_centers[:,:,jnp.newaxis]
		sigma = wave_sigmas_eff[:,:,jnp.newaxis]

		wave_space_crop = self.wave_space[J_min:J_max+1*self.wave_factor]
		wave_space_edges_prov= wave_space_crop[1:] - jnp.diff(wave_space_crop)/2
		wave_space_edges_prov2 = jnp.insert(wave_space_edges_prov, 0, wave_space_edges_prov[0] - jnp.diff(wave_space_crop)[0])
		wave_space_edges = jnp.append(wave_space_edges_prov2, wave_space_edges_prov2[-1] + jnp.diff(wave_space_crop)[-1])
		# print(sigma) 
		# print(jnp.exp(-0.5*((self.wave_space[jnp.newaxis, jnp.newaxis, :]-mu)**2/sigma**2)))
		# print(jnp.sqrt(2.0*jnp.pi)*sigma)
		# cube= F[:, :, jnp.newaxis]*jnp.exp(-0.5*(((self.wave_space[jnp.newaxis, jnp.newaxis, J_min:J_max+1*self.wave_factor]-mu)/sigma)**2))/(jnp.sqrt(2.0*jnp.pi)*sigma)*1e-3
		cdf = 0.5* (1 + special.erf( (wave_space_edges[jnp.newaxis,jnp.newaxis,:] - mu)/ (sigma*math.sqrt(2.)) ))
		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		gaussian_matrix_cut = gaussian_matrix # [:,:, J_min:J_max+1*self.wave_factor]
		cube = F[:,:,jnp.newaxis]*gaussian_matrix_cut
		# plt.plot(self.wave_space,jnp.exp(-0.5*((self.wave_space-mu)**2/sigma**2))/(jnp.sqrt(2.0*jnp.pi)*sigma) )
		# plt.show()
		# print(self.wave_space.shape)
		# print(self.wavs.shape)
		# cube = cube_full[:,:,J_min:J_max+1*self.wave_factor]
		psf_cube = fftconvolve(cube, self.PSF, mode='same') #convolve(cube, self.full_kernel)

		# grism_spectra = jnp.zeros((cube.shape[1],cube.shape[0]))
		# grism_spectra_padded = jnp.pad(grism_spectra, [(0, 0), (cube.shape[2],cube.shape[2])], mode='constant')
		# # at this stage, we need to use the instrument dispersion function

		
		grism_full = jnp.sum(psf_cube, axis = 1) #grism_spectra_padded[:, cube.shape[0]:-cube.shape[0]] #jnp.sum(psf_cube, axis = 1)
		 #grism_spectra_padded[:, 62:-62]
		#remove all pixels with flux below 1e-6
		# grism_full = jnp.where(grism_full < 1e-5, 0, grism_full)
		# plt.imshow(grism_full, origin = 'lower')
		# plt.colorbar()
		# plt.show()
		return grism_full

	def disperse_interp(self, F, V, D):
		'''
			Dispersion function going from flux space F to grism space G
		'''

		J_min = 496
		J_max = 537
		#get the dispersion of each pixel with its given wavelength
		disps = self.grism_dispersion(self.wavelength*(1  + V/(c/1000) ))
		#compute the dx wrt the central pixel
		dxs= jnp.array(disps + self.detector_position - self.xcenter_detector)
		#obtain the wave for each of those dxs
		waves = self.inverse_wave_disp(dxs)
		# print('waves: ', waves)


		#compute the dispersion due to the LSF
		D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
		#add the LSF to the dispersion: 
		CORR_D = jnp.sqrt( jnp.square(D)+  jnp.square(D_LSF))

		sigma_disps= self.grism_dispersion(self.wavelength*(1  + (V + CORR_D[0,0])/(c/1000) )) 
		dxs_sigma = jnp.array(sigma_disps + self.detector_position - self.xcenter_detector)
		waves_sigma = self.inverse_wave_disp(dxs_sigma)
		# print('waves_sigma: ', waves_sigma)
		delta_wave_sigma = waves_sigma - waves
		# print('delta_wave_sigma: ', delta_wave_sigma)

		cube= F[:, :, jnp.newaxis]*jnp.exp(-0.5*((self.wavs[jnp.newaxis, jnp.newaxis, J_min:J_max]-waves[:,:, jnp.newaxis])**2/delta_wave_sigma[:,:, jnp.newaxis]**2))/(jnp.sqrt(2.0*jnp.pi)*delta_wave_sigma[:,:, jnp.newaxis])*1e-4


		psf_cube = cube #fftconvolve(cube, self.PSF, mode='same') #convolve(cube, self.full_kernel)

		grism_full = jnp.sum(psf_cube, axis = 1)
		# end = time.time()
		
		# self.f.write("matmul time: " + str(end-start) + "\n")

		# self.f.close()

		return grism_full

	def compute_gaussian(self, V, D):
		'''
			Add explanation here
		'''

		# print('V: ', V)
		# print('D: ', D)
		#compute the dispersion dx0 of the central pixel 
		# print(self.wave_space.shape, self.disp_space.shape)
		# dx0 = jnp.interp(self.wavelength*(1  + V/(c/1000) ), self.wave_space[jnp.argsort(self.disp_space)], self.disp_space[jnp.argsort(self.disp_space)])
		dx0 = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + V/(c/1000) )), *self.w_opt)
		DX = self.detector_position - self.xcenter_detector + dx0 
		# print(DX.type(), self.disp_space.type(), self.wave_space.type())

		# J = self.compute_disp_index(DX, self.disp_space)
		# J0 = self.compute_disp_index(dx0, self.disp_space)

		# J= jnp.argmin(jnp.abs(self.disp_space - DX[:,:, None]), axis = 2)
		# J0 = jnp.argmin(jnp.abs(self.disp_space - dx0[:,:, None]),axis =2)

		J_min = self.index_min
		J_max = self.index_max

		# J = (jnp.rint(J)).astype(int)
		# J0 = (jnp.rint(J0)).astype(int)

		# print(J)


		#compute the dispersion due to the LSF
		D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
		# print('D_LSF = ', D_LSF)

		#add the LSF to the dispersion: 
		CORR_D = jnp.sqrt( D**2 +  D_LSF**2)
		DX_disp = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + (V+CORR_D)/(c/1000) )), *self.w_opt)
		DX_disp_final = self.detector_position - self.xcenter_detector + DX_disp - DX



		NEW_J, NEW_K = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor-1, J_max-J_min+1*self.wave_factor))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
		# print(NEW_J)
		# grism_full = jnp.zeros(self.sh_beam)
		# print(self.sh_beam)

		# mu = self.wave_space[J[:,NEW_K]]
		# sigma = CORR_D[:, NEW_K]/(c/1000)*self.wave_space[J[:,NEW_K]]

		# sigma = CORR_D[:, NEW_K]/(c/1000)*self.wave_space[J[:,NEW_K]]
		# compute the edges of the disp_space array

		disp_space_edges_prov= self.disp_space[1:] - jnp.diff(self.disp_space)/2
		disp_space_edges_prov2 = jnp.insert(disp_space_edges_prov, 0, disp_space_edges_prov[0] - jnp.diff(self.disp_space)[0])
		disp_space_edges = jnp.append(disp_space_edges_prov2, disp_space_edges_prov2[-1] + jnp.diff(self.disp_space)[-1])

		# print(disp_space_edges)
		# cdf = 0.5* (1 + special.erf( (self.wave_space[NEW_J] - mu)/ (sigma*math.sqrt(2.)) ))
		# prev_wave = self.wave_space[NEW_J] - self.wave_scale
		# cdf_prev = 0.5* (1 + special.erf( (prev_wave - mu)/ (sigma*math.sqrt(2.)) ))
		# gaussian_matrix = cdf[:,NEW_K,NEW_J-J_min] - cdf_prev[:,NEW_K,NEW_J-J_min]
		# print(self.disp_space[NEW_J].shape)
		# print(mu.shape)
		# print(sigma.shape)

		# compute the gaussian matrix in 10 times higher resolution and then sample it down

		# dx0 = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + V/(c/1000) )), *self.w_opt)
		# DX = self.detector_position - self.xcenter_detector + dx0 

		# J_min = self.index_min
		# J_max = self.index_max

		# D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
		# # print('D_LSF = ', D_LSF)

		# #add the LSF to the dispersion: 
		# CORR_D = jnp.sqrt( D**2 +  D_LSF**2)
		# DX_disp = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + (V+CORR_D)/(c/1000) )), *self.w_opt)
		# DX_disp_final = self.detector_position - self.xcenter_detector + DX_disp - DX

		# disp_space_high = jnp.linspace(self.disp_space[0], self.disp_space[-1], 10*self.disp_space.shape[0])
		# disp_space_edges_prov= disp_space_high[1:] - jnp.diff(disp_space_high)/2
		# disp_space_edges_prov2 = jnp.insert(disp_space_edges_prov, 0, disp_space_edges_prov[0] - jnp.diff(disp_space_high)[0])
		# disp_space_edges = jnp.append(disp_space_edges_prov2, disp_space_edges_prov2[-1] + jnp.diff(disp_space_high)[-1])


		# NEW_J_cdf, NEW_K_cdf = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor, 34*self.wave_factor*10+1))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
		# mu = DX[:,NEW_K_cdf]
		# sigma = DX_disp_final[:,NEW_K_cdf]
		# cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
		# gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		# # gaussian_matrix_low = np.zeros((14,14,34))
		# # for i in range(14):
		# # 	blocks = gaussian_matrix[i].reshape(int(gaussian_matrix[i].shape[0]/(1)), 1, int(gaussian_matrix[i].shape[1]/10), 10)
		# # 	gaussian_matrix_low[i] = jnp.sum(blocks, axis=(1,3))
		# plt.imshow(gaussian_matrix[1])
		# plt.show()
		# # print('Gauss shape pre :' ,gaussian_matrix.shape)
		# blocks = gaussian_matrix.reshape((V.shape[0],V.shape[0], int(gaussian_matrix.shape[2]/10), 10))
		# gaussian_matrix = jnp.sum(blocks, axis=-1)
		# # print('Gauss shape post :' ,gaussian_matrix.shape)


		NEW_J_cdf, NEW_K_cdf = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor, J_max-J_min+1*self.wave_factor+1))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
		# print(NEW_K_cdf)
		mu = DX[:,NEW_K_cdf]
		sigma = DX_disp_final[:,NEW_K_cdf]
		cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		# plt.imshow(gaussian_matrix[14], origin = 'lower')
		# plt.show()
		# print(gaussian_matrix.shape)
		return gaussian_matrix

	def get_beam_cutout(self, wave_min, wave_max, wave_space, grism_full):
		'''
			get cutout of the full grism spectrum from wave_min to wave_max
			in the implementation these can be taken by finding wave_min and wave_max on the observed 2D image

		'''

		j_min, j_max = self.compute_wave_index(jnp.array([wave_min, wave_max]), wave_space)
		j_min = 360
		j_max = 370
		self.grism_cutout = grism_full[:, j_min:j_max]

		return self.grism_cutout
