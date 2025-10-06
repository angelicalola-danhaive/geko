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
import os
from pathlib import Path

# Get the directory where the geko package is installed
# This allows finding nircam_grism regardless of where code is run from
PACKAGE_DIR = Path(__file__).parent.parent.resolve()
NIRCAM_GRISM_DIR = PACKAGE_DIR / 'nircam_grism'
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
# jax.config.update('jax_enable_x64', True)  # Already set in fitting.py

# ============================================================================
# STANDALONE JIT-COMPILED FUNCTIONS (extracted from Grism class methods)
# ============================================================================

@jax.jit
def _disperse_gaussian_core(F, wave_centers, wave_sigmas_eff, wave_space_edges):
	"""
	Core Gaussian computation for grism dispersion.

	Integrates Gaussian profiles over wavelength bins to create spectral cube.
	Uses error functions (CDF) for accurate bin integration.

	Parameters
	----------
	F : jax.numpy.ndarray
		2D flux map (spatial y, spatial x)
	wave_centers : jax.numpy.ndarray
		2D array of central wavelengths for each pixel (spatial y, spatial x)
	wave_sigmas_eff : jax.numpy.ndarray
		2D array of effective wavelength dispersions (spatial y, spatial x)
	wave_space_edges : jax.numpy.ndarray
		1D array of wavelength bin edges

	Returns
	-------
	cube : jax.numpy.ndarray
		3D spectral cube (spatial y, spatial x, wavelength)

	Notes
	-----
	Extracted from Grism.disperse() for JIT compilation compatibility.
	Computes integral of Gaussian from edge[i] to edge[i+1] for each bin.
	"""
	from jax.scipy import special
	import math
	
	# Make 3D cube (spatial, spectral, wavelengths)
	mu = wave_centers[:,:,jnp.newaxis]
	sigma = wave_sigmas_eff[:,:,jnp.newaxis]
	
	# Compute CDF for Gaussian integration
	cdf = 0.5 * (1 + special.erf((wave_space_edges[jnp.newaxis,jnp.newaxis,:] - mu) / (sigma * math.sqrt(2.0))))
	gaussian_matrix = cdf[:,:,1:] - cdf[:,:,:-1]
	
	# Create spectral cube
	cube = F[:,:,jnp.newaxis] * gaussian_matrix
	
	return cube

# ============================================================================



class Grism:
	"""
	JWST NIRCam grism spectroscopy forward modeling class.

	Models the dispersion and convolution of 3D data cubes through a grism,
	including wavelength-dependent trace, LSF, and PSF effects.

	Parameters
	----------
	im_shape : int
		Size of the image (assumes square)
	im_scale : float, optional
		Pixel scale of the model image in arcsec (default: 0.031)
	icenter : int, optional
		i-coordinate of galaxy center in pixels (default: 5)
	jcenter : int, optional
		j-coordinate of galaxy center in pixels (default: 5)
	wavelength : float, optional
		Central wavelength in microns (default: 4.2)
	wave_space : numpy.ndarray, optional
		Wavelength array in microns
	index_min : int, optional
		Minimum wavelength index
	index_max : int, optional
		Maximum wavelength index
	grism_filter : str, optional
		Grism filter name (default: 'F444W')
	grism_module : str, optional
		JWST module 'A' or 'B' (default: 'A')
	grism_pupil : str, optional
		Grism pupil 'R' or 'C' (default: 'R')
	PSF : numpy.ndarray, optional
		Point spread function array

	Attributes
	----------
	im_shape : int
		Image size
	im_scale : float
		Model pixel scale
	detector_scale : float
		Detector pixel scale (0.0629" for JWST)
	factor : int
		Oversampling factor between model and detector
	"""
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

		self.compute_lsf() #_new()

		self.compute_PSF(PSF)

		self.set_wave_array()
	
	def __str__(self):
		"""
		String representation of the Grism object.

		Returns
		-------
		str
			Formatted string with grism configuration details
		"""
		return 'Grism object: \n' + ' - direct shape: ' + str(self.im_shape) + '\n - grism shape: ' + str(self.sh_beam) + '\n - wave_scale = ' + str(self.wave_scale) + '\n - factor = ' + str(self.factor) + '\n - i_center = ' + str(self.icenter) + '\n - j_center = ' + str(self.jcenter)

	def init_detector(self):
		"""
		Initialize detector coordinate system.

		Creates 1D and 2D arrays mapping model pixels to detector coordinates,
		accounting for oversampling and galaxy center position.

		Notes
		-----
		Sets attributes:
		- detector_xmin, detector_xmax: Detector coordinate bounds
		- detector_space_1d: 1D detector coordinate array
		- detector_space_2d: 2D detector coordinate grid
		"""
		self.detector_xmin = self.xcenter_detector - self.jcenter
		self.detector_xmax = self.xcenter_detector + (self.im_shape//self.factor-1-self.jcenter) #need to check that this still works but should do!

		detector_x_space_low = jnp.linspace(self.detector_xmin, self.detector_xmax, int(self.im_shape//self.factor))

		detector_x_space_low_2d = jnp.reshape(detector_x_space_low, (1, int(self.im_shape//self.factor)))

		#oversampling it to high resolution
		self.detector_space_1d = image.resize(detector_x_space_low_2d, (1,self.im_shape), method = 'linear')[0] #taking only the first row since all rows are the same - and I want it in 1D

		#the position on the detector of each pixel in the high res model
		self.detector_space_2d = self.detector_space_1d * jnp.ones_like(jnp.zeros( (self.im_shape,1) ))

	def load_coefficients(self):
		"""
		Load JWST NIRCam grism calibration coefficients.

		Loads tracing and dispersion polynomial coefficients from the
		nircam_grism calibration files for the specified filter, module,
		and pupil configuration.

		Notes
		-----
		Sets attributes:
		- fit_opt: Trace polynomial coefficients
		- w_opt: Dispersion polynomial coefficients
		- WRANGE: Valid wavelength range for the filter

		Calibration files are read from the nircam_grism/FS_grism_config directory.
		"""

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
		#Using NIRCAM_GRISM_DIR to find calibration files regardless of where code is run from
		tb_order23_fit_AR = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'A', 'R'))))
		fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
		tb_order23_fit_BR = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'B', 'R'))))
		fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
		tb_order23_fit_AC = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'A', 'C'))))
		fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
		tb_order23_fit_BC = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISP_%s_mod%s_grism%s.txt' % (disp_filter, 'B', 'C'))))
		fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data

		### grism dispersion parameters:
		tb_fit_displ_AR = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISPL_mod%s_grism%s.txt' % ('A', "R"))))
		w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
		tb_fit_displ_BR = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISPL_mod%s_grism%s.txt' % ('B', "R"))))
		w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
		tb_fit_displ_AC = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISPL_mod%s_grism%s.txt' % ('A', "C"))))
		w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
		tb_fit_displ_BC = ascii.read(str(NIRCAM_GRISM_DIR / 'FS_grism_config' / ('DISPL_mod%s_grism%s.txt' % ('B', "C"))))
		w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data

		### list of module/pupil and corresponding tracing/dispersion function:
		list_mod_pupil   = np.array(['AR', 'BR', 'AC', 'BC'])
		list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
		list_w_opt       = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])

		### Sensitivity curve:
		dir_fluxcal = NIRCAM_GRISM_DIR / 'all_wfss_sensitivity'
		tb_sens_AR = ascii.read(str(dir_fluxcal / ('%s_mod%s_grism%s_sensitivity.dat' % (disp_filter, 'A', 'R'))))
		tb_sens_BR = ascii.read(str(dir_fluxcal / ('%s_mod%s_grism%s_sensitivity.dat' % (disp_filter, 'B', 'R'))))
		tb_sens_AC = ascii.read(str(dir_fluxcal / ('%s_mod%s_grism%s_sensitivity.dat' % (disp_filter, 'A', 'C'))))
		tb_sens_BC = ascii.read(str(dir_fluxcal / ('%s_mod%s_grism%s_sensitivity.dat' % (disp_filter, 'B', 'C'))))
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
		"""
		Compute grism dispersion trace for the central galaxy pixel.

		Assuming the galaxy is at the detector center, computes where the central
		pixel appears on the detector when emitting at each wavelength in wave_space.
		Uses polynomial coefficients from NIRCam grism calibration.

		Returns
		-------
		dxs : jax.numpy.ndarray
			Uniformly spaced dispersion positions (detector x-coordinates)
		disp_space : jax.numpy.ndarray
			Dispersion positions for each wavelength in wave_space

		Notes
		-----
		Sets attributes:
		- disp_space: Dispersion positions from polynomial trace equation
		- dxs: Uniformly spaced array spanning min to max dispersion
		- wavs: Wavelengths corresponding to each position in dxs
		- inverse_wave_disp: Interpolator mapping dispersion to wavelength
		"""
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
		"""
		Compute effective wavelength for each model pixel.

		Determines the central wavelength of each pixel on the plane of the central
		pixel by computing spatial separation in wavelength space. This accounts for
		the wavelength shift each pixel experiences due to its position.

		Notes
		-----
		Algorithm:
		1. Disperse each pixel at self.wavelength with zero velocity
		2. Find corresponding wavelength in central pixel's reference frame
		3. Store result in self.wave_array

		Sets attribute:
		- wave_array: 2D array of effective wavelengths for each model pixel
		"""

		#disperse each pixel with wavelength self.wavelength (and zero velocity)
		dispersion_indices = self.grism_dispersion(self.wavelength)

		#put the dxs in the rest frame of the central pixel (since otherwise they are dx wrt to their original pixel in self.detector_space_1d)
		dispersion_indices += (self.detector_space_1d - self.xcenter_detector)
		#for each dx, find the closest in the uniformly distrubuted dxs
		wave_indices = np.argmin(np.abs(self.dxs[np.newaxis,np.newaxis,:] - dispersion_indices[:,:,np.newaxis]), axis = 2)
		#translate this to a wavelength in the rest frame of the central pixel
		self.wave_array = self.wavs[wave_indices]

	def load_poly_factors(self,a01, a02, a03, a04, a05, a06, b01, b02, b03, b04, b05, b06, c01, c02, c03, d01):
		"""
		Load polynomial dispersion coefficients.

		Parameters
		----------
		a01-a06 : float
			Constant term polynomial coefficients
		b01-b06 : float
			Linear (wavelength) term coefficients
		c01-c03 : float
			Quadratic (wavelength^2) term coefficients
		d01 : float
			Cubic (wavelength^3) term coefficient

		Notes
		-----
		These coefficients define the NIRCam grism dispersion polynomial:
		dx = (a_coeffs) + (b_coeffs)*wave + (c_coeffs)*wave^2 + d01*wave^3
		where each set of coefficients also depends on x,y detector position.
		"""
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
		"""
		Pre-compute position-dependent polynomial coefficients.

		Evaluates the polynomial coefficients at each detector position in the
		2D grid. This precomputation speeds up dispersion calculations by avoiding
		repeated polynomial evaluations.

		Notes
		-----
		Assumes horizontal dispersion only (all pixels on same detector row).
		Sets attributes:
		- coef1: Constant term coefficients (2D array)
		- coef2: Linear wavelength term coefficients (2D array)
		- coef3: Quadratic wavelength term coefficients (2D array)
		- coef4: Cubic wavelength term coefficients (2D array)
		"""
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
		"""
		Compute grism dispersion offset for given wavelength.

		Calculates the x-axis offset (dx) in the grism image for a pixel at
		wavelength wave using the NIRCam grism dispersion polynomial.

		Parameters
		----------
		wave : float or array_like
			Wavelength in microns

		Returns
		-------
		dx : jax.numpy.ndarray
			Dispersion offset in detector pixels

		Notes
		-----
		Uses pre-computed position-dependent coefficients (coef1-coef4) from
		load_poly_coefficients(). Wavelength is normalized by subtracting 3.95 μm.
		"""
		wave -=3.95

		return ((self.coef1 + self.coef2 * wave + self.coef3 * wave**2 + self.coef4 * wave**3))

	def set_detector_scale(self, scale):
		"""
		Set detector pixel scale.

		Parameters
		----------
		scale : float
			Detector pixel scale in arcsec/pixel
		"""
		self.detector_scale = scale


	def compute_lsf(self):
		"""
		Compute Line Spread Function (LSF) for NIRCam grism.

		Calculates spectral resolution and LSF width at self.wavelength using
		empirical polynomial fit to NIRCam grism resolving power.

		Returns
		-------
		R : float
			Spectral resolving power (R = λ/Δλ)

		Notes
		-----
		Sets attributes:
		- sigma_lsf: LSF width in wavelength units (microns)
		- sigma_v_lsf: LSF width in velocity units (km/s)

		The resolving power R is modeled as a 4th-order polynomial in wavelength.
		Conversion to sigma assumes Gaussian profile (FWHM = 2.36 * sigma).
		"""
		#compute the sigma lsf for the wavelength of interest, wavelength must be in MICRONS
		R = 3.35*self.wavelength**4 - 41.9*self.wavelength**3 + 95.5*self.wavelength**2 + 536*self.wavelength - 240

		self.sigma_lsf = (1/2.36)*self.wavelength/R
		# print('LSF: ', self.sigma_lsf)
		self.sigma_v_lsf = (1/2.36)*(c/1000)/R #put c in km/s #0.5*
		# print('LSF vel: ', self.sigma_v_lsf)

		#returning R for testing purposes
		return R

	
	def compute_lsf_new(self):
		"""
		Compute improved LSF using double-Gaussian model.

		Uses empirical two-component Gaussian model derived from NIRCam flight data.
		The LSF is modeled as a weighted sum of two Gaussians with module-dependent
		parameters (Module A vs B).

		Returns
		-------
		lsf_kernel : jax.numpy.ndarray
			Normalized 1D LSF convolution kernel

		Notes
		-----
		Model parameters (fraction, FWHM) depend on NIRCam module and wavelength.
		The kernel is constructed with 6-sigma width and normalized to unit sum.

		References
		----------
		LSF model from Fengwu Sun based on updated NIRCam grism calibration data.
		"""
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
		fwhm_lsf = np.sum(self.lsf_kernel >= np.max(self.lsf_kernel) / 2) * np.diff(self.wave_space)[0]

		# self.sigma_v_lsf = self.sigma_lsf/(self.wavelength/(c/1000))

		R = self.wavelength/fwhm_lsf    #self.wavelength/(self.sigma_lsf*(2*math.sqrt(2*math.log(2))))

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
		"""
		Prepare Point Spread Function for grism modeling.

		Oversamples the input PSF to match model resolution, crops to 11x11 pixels
		(9x9 for factor>1 after oversampling), normalizes, and reshapes for 3D
		convolution with spectral dimension.

		Parameters
		----------
		PSF : numpy.ndarray or jax.numpy.ndarray
			2D Point Spread Function at detector resolution

		Notes
		-----
		Sets attributes:
		- oversampled_PSF: PSF oversampled by self.factor and normalized
		- PSF: 3D array with shape (spatial_y, spatial_x, 1) for broadcasting

		If factor=1, uses input PSF directly. Otherwise oversamples using bilinear
		interpolation and crops to central 11x11 pixels.
		"""
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

		# self.full_kernel = jnp.array(self.PSF) * self.lsf_kernel

	
	def disperse(self, F, V, D):
		"""
		Disperse a 3D data cube (flux, velocity, dispersion) through the grism.

		Forward models the grism spectroscopy by:
		1. Shifting wavelengths based on velocity field
		2. Broadening by velocity dispersion and LSF
		3. Convolving with spatial PSF
		4. Collapsing to 2D grism spectrum

		Parameters
		----------
		F : jax.numpy.ndarray
			2D flux map (spatial y, spatial x)
		V : jax.numpy.ndarray
			2D velocity field in km/s (spatial y, spatial x)
		D : jax.numpy.ndarray
			2D velocity dispersion field in km/s (spatial y, spatial x)

		Returns
		-------
		jax.numpy.ndarray
			2D dispersed grism spectrum (spatial y, wavelength)

		Notes
		-----
		Uses Gaussian profile convolution for spectral dispersion and
		FFT convolution for spatial PSF. Velocity is converted to wavelength
		shift via Doppler formula.
		"""

		J_min = self.index_min
		J_max = self.index_max

		#self.wave_array contains the wavelength of each pixel in the grism image, in the ref frame of the central pixel
		wave_centers = self.wavelength*( V/(c/1000) ) + self.wave_array
		wave_sigmas = self.wavelength*(D/(c/1000) ) #the velocity dispersion doesn't need to be translated to the ref frame of the central pixel

		sigma_LSF = self.sigma_lsf

		#set the effective dispersion which also accounts for the LSF
		wave_sigmas_eff = jnp.sqrt(jnp.square(wave_sigmas) + jnp.square(sigma_LSF)) 
		# wave_sigmas_eff = wave_sigmas

		#make a 3D cube (spacial, spectral, wavelengths)
		mu = wave_centers[:,:,jnp.newaxis]
		sigma = wave_sigmas_eff[:,:,jnp.newaxis]
		
		#compute the edges of the wave space in order to evaluate the gaussian at those points - focusing only on the region of interest
		wave_space_crop = self.wave_space[J_min:J_max]
		wave_space_edges_prov= wave_space_crop[1:] - jnp.diff(wave_space_crop)/2
		wave_space_edges_prov2 = jnp.insert(wave_space_edges_prov, 0, wave_space_edges_prov[0] - jnp.diff(wave_space_crop)[0])
		wave_space_edges = jnp.append(wave_space_edges_prov2, wave_space_edges_prov2[-1] + jnp.diff(wave_space_crop)[-1])


		# Use JIT-compiled core for Gaussian computation
		cube = _disperse_gaussian_core(F, wave_centers, wave_sigmas_eff, wave_space_edges)

		# psf_cube = fftconvolve(cube, self.full_kernel, mode='same') 
		psf_cube = fftconvolve(cube, self.PSF, mode='same') 

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

		# '''
		# 	Sets the scale of the grism image to 'scale'. Modifies all relevant parameters in the class so that when the disperse function is called, 
		# 	the grism wavelength space has the appropriate scale

		# 	Parameters
		# 	----------

		# 	Returns
		# 	----------
		# '''

		# self.wave_scale = scale
		# self.sh_beam = (self.sh[0],((self.WRANGE[1]-self.WRANGE[0])/self.wave_scale).astype(int))
		# #full grism array
		# self.grism_full = jnp.zeros(self.sh_beam)

		# self.get_trace()

		# self.compute_lsf()

		# self.set_sensitivity()