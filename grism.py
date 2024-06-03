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

import utils

import numpy as np
from astropy.io import ascii
from astropy import wcs
from astropy.io import fits
from scipy import interpolate
from scipy.constants import c #in m/s
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy
import math
from scipy.constants import c
import matplotlib.pyplot as plt
from jax.scipy import special
from jax.scipy.stats import norm
from jax.scipy.signal import convolve
from scipy import ndimage

import jax.numpy as jnp

from reproject import reproject_adaptive
from photutils.centroids import centroid_1dg

#import time
import time 



class Grism:
	def __init__(self, direct = None, direct_scale = 0.03, factor = 1, y_factor = 1, icenter = 5, jcenter = 5, segmentation = None, xcenter_detector = 1024, ycenter_detector = 1024, 
		wavelength = 4.2 , redshift = 7.2, wave_space = None, wave_factor = 1, wave_scale = 0.001, index_min = None, index_max = None, grism_filter = 'F444W', grism_module = 'A', grism_pupil = 'R', higher_res = False, PSF = None):
		"""Object for computing dispersed model spectra (emission line)

		Parameters
		----------
		id : int
			Only consider pixels in the segmentation image with value `id`.
			Default of zero to match the default empty segmentation image.

		direct : `~numpy.ndarray`
			Direct image cutout in f_lambda units (i.e., e-/s times PHOTFLAM).
			Default is a trivial zeros array.

		segmentation : `~numpy.ndarray` (float32) or None
			Segmentation image.  If None, create a zeros array with the same
			shape as `direct`.

		origin : [int, int]
			`origin` defines the lower left pixel index (y,x) of the `direct`
			cutout from a larger detector-frame image

		xcenter, ycenter : float, float
			Sub-pixel centering of the exact center of the object, relative
			to the center of the thumbnail.  Needed for getting exact
			wavelength grid correct for the extracted 2D spectra.


		conf : [str, str, str] or `grismconf.aXeConf` object.
			Pre-loaded aXe-format configuration file object or if list of
			strings determine the appropriate configuration filename with
			`grismconf.get_config_filename` and load it.

		scale : float
			Multiplicative factor to apply to the modeled spectrum from
			`compute_model`.

		yoffset : float
			Cross-dispersion offset to apply to the trace
		
		xoffset : float
			Dispersion offset to apply to the trace
			
		Attributes
		----------
		sh : 2-tuple
			shape of the direct array

		sh_beam : 2-tuple
			computed shape of the 2D spectrum

		seg : `~numpy.array`
			segmentation array

		lam : `~numpy.array`
			wavelength along the trace

		ytrace : `~numpy.array`
			y pixel center of the trace.  Has same dimensions as sh_beam[1].

		sensitivity : `~numpy.array`
			conversion factor from native e/s to f_lambda flux densities

		modelf, model : `~numpy.array`, `~numpy.ndarray`
			2D model spectrum.  `model` is linked to `modelf` with "reshape",
			the later which is a flattened 1D array where the fast
			calculations are actually performed.

		model : `~numpy.ndarray`
			2D model spectrum linked to `modelf` with reshape.

		total_flux : float
			Total f_lambda flux in the thumbail within the segmentation
			region.
		"""

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
		self.seg = jnp.array(segmentation)

		# #full 1048x1048 image of the dispersion field
		# self.detector_image = detector_image

		#initialize attributes

		#center of the object on the detector image
		self.xcenter_detector = xcenter_detector 
		self.ycenter_detector =  ycenter_detector

		self.detector_xmin = self.xcenter_detector - (self.jcenter_high-1)/self.direct_factor_high
		self.detector_xmax = self.xcenter_detector + (self.sh_high[1]-1-self.jcenter_high)/self.direct_factor_high
		# self.detector_x_space = jnp.linspace(self.detector_xmin, self.detector_xmax, self.sh[1]*factor)
		# self.detector_x_space = jnp.linspace(self.detector_xmin, self.detector_xmax, self.sh[1]*factor)
		self.detector_x_space = jnp.linspace(self.detector_xmin, self.detector_xmax, self.sh_high[1])

		# print(self.detector_x_space.shape)

		# self.detector_position = self.detector_x_space * jnp.ones_like(np.zeros( (self.sh[0]*factor,1) ))
		# self.detector_position = self.detector_x_space * jnp.ones_like(jnp.zeros( (self.sh[0]*factor,1) ))
		self.detector_position = self.detector_x_space * jnp.ones_like(jnp.zeros( (self.sh_high[0],1) ))


		# print(self.detector_position.shape)
		#dispersed grism attributes

		#these are already in the high model
		self.wave_scale = wave_scale
		self.wave_space = wave_space 

		#make a higer res wave_space => the computed disp space will also be higher res
		# wave_min = wave_space[0]
		# wave_max = wave_space[len(self.wave_space)-1]
		# self.wave_space = jnp.linspace(wave_min, wave_max + wave_scale, int((wave_max-wave_min)/ self.wave_scale))

		self.index_min = index_min
		self.index_max = index_max
		self.wavelength = wavelength
		self.redshift = redshift

		self.filter = grism_filter
		self.module = grism_module
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

		self.compute_lsf()
		# print(self.sigma_v_lsf)

		self.set_sensitivity()

		self.compute_PSF(PSF)


	
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

		self.disp_space =  ((self.a01 + (self.a02 * xpix + self.a03 * ypix) + (self.a04 * xpix2 + self.a05 * xpix * ypix + self.a06 * ypix2)) +
	  					(self.b01 + (self.b02 * xpix + self.b03 * ypix) + (self.b04 * xpix2 + self.b05 * xpix * ypix + self.b06 * ypix2)) * wave +
						(self.c01 + (self.c02 * xpix + self.c03 * ypix)) * wave2 + (self.d01 ) * wave3)		

		# print('disp_space: ', self.disp_space)

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
		ypix = self.ycenter_detector
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
		# print(R)
		# self.sigma_lsf = 0.5*R
		# self.sigma_v_lsf = 0.5*(0.001*(c/1000)/self.wavelength)


		#the 0.5 in front is bc I think I am over-estimating LSF
		self.sigma_lsf = (1/2.36)*self.wavelength/R #0.5*
		# print(self.sigma_lsf)
		self.sigma_v_lsf = (1/2.36)*(c/1000)/R #put c in km/s #0.5*
		print(self.sigma_v_lsf)

		# print(0.001*(c/1000)/4.2)
	
	def compute_PSF(self, PSF):
		#load the PSF 
		PSF = jnp.array(fits.open(PSF)[0].data)
		print('PSF sum = ', jnp.sum(PSF))
		#sets the grism object's oversampled PSF and the velocity space needed for the cube
		oversampled_PSF = utils.oversample(PSF, self.factor, self.factor, method='bilinear')
		self.PSF = oversampled_PSF[jnp.newaxis,:,:]

		velocity_space = jnp.linspace(-1000, 1000, 2001)
		self.velocity_space = jnp.broadcast_to(velocity_space[:, jnp.newaxis, jnp.newaxis], (velocity_space.size, self.sh_high[0], self.sh_high[0]))

	def compute_disp_index(self, values, disp_space):

		array = jnp.array(disp_space)
		idxs = jnp.searchsorted(array, values, side="left")

		prev_idx_is_less = (idxs == len(array))|(jnp.fabs(values - array[jnp.maximum(idxs-1, 0)]) < jnp.fabs(values - array[jnp.minimum(idxs, len(array)-1)]))
		# idxs = idxs.at[prev_idx_is_less].set(idxs[prev_idx_is_less]-1)
		idxs = jnp.where( prev_idx_is_less,  idxs -1, idxs)
		return idxs

	def compute_wave_index(self, values,wave_space):

		return jnp.argmin(wave_space-values)


	def disperse(self, F, V, D):
		'''
			Dispersion function going from flux space F to grism space G
		'''
		# self.f = open("timing.txt", "a")
		# start = time.time()

		DX = self.grism_dispersion(self.wavelength*(1  + V/(c/1000) ))
		# print('DX shape: ', DX.shape)
		# end = time.time()
		# self.f.write("DX time: " + str(end-start) + "\n")

		J_min = self.index_min
		J_max = self.index_max


		#compute the dispersion due to the LSF
		D_LSF = self.sigma_v_lsf*jnp.ones_like(V)

		#add the LSF to the dispersion: 
		CORR_D = jnp.sqrt( D**2 +  D_LSF**2)

		# start = time.time()
		DX_disp_final = self.grism_dispersion(self.wavelength*(1  + (V+CORR_D)/(c/1000) )) -DX
		# end = time.time()
		# self.f.write("DX_disp_final time: " + str(end-start) + "\n")
		NEW_J, NEW_K = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor-1, J_max-J_min+1*self.wave_factor))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )

		disp_space_edges_prov= self.disp_space[1:] - jnp.diff(self.disp_space)/2
		disp_space_edges_prov2 = jnp.insert(disp_space_edges_prov, 0, disp_space_edges_prov[0] - jnp.diff(self.disp_space)[0])
		disp_space_edges = jnp.append(disp_space_edges_prov2, disp_space_edges_prov2[-1] + jnp.diff(self.disp_space)[-1])

		NEW_J_cdf, NEW_K_cdf = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor, J_max-J_min+1*self.wave_factor+1))).astype(int) ,  (jnp.rint(jnp.linspace(0, V.shape[1]-1, V.shape[1]))).astype(int)  )
		mu = DX[:,NEW_K_cdf]
		sigma = DX_disp_final[:,NEW_K_cdf]
		# start = time.time()
		cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
		# cdf = norm.cdf( (disp_space_edges[NEW_J_cdf] - mu) / (sigma*math.sqrt(2.)) )
		# end = time.time()
		# self.f.write("cdf time: " + str(end-start) + "\n")

		# start = time.time()
		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		# gaussian_matrix = norm.pdf((self.disp_space[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)))
		# end = time.time()
		# self.f.write("gaussian_matrix time: " + str(end-start) + "\n")

		# start = time.time()
		psf_gaussian_matrix = convolve(gaussian_matrix, self.PSF, mode='same')
		# end = time.time()
		# self.f.write("convolve time: " + str(end-start) + "\n")

		# start = time.time()
		grism_full = jnp.matmul( F[:,None, :] , psf_gaussian_matrix)[:,0,:]  
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
