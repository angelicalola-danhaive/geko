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

import numpy as np
from astropy.io import ascii
from astropy import wcs
from astropy.wcs.utils import fit_wcs_from_points
from astropy.io import fits
from scipy import interpolate
from scipy.constants import c #in m/s
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy
import math
from scipy.constants import c,pi
import matplotlib.pyplot as plt
from jax.scipy import special

from scipy import ndimage

import jax.numpy as jnp
from jax import random, vmap
from jax import image
from jax.scipy.special import logsumexp

from reproject import reproject_interp, reproject_adaptive
from photutils.centroids import centroid_com, centroid_quadratic,centroid_1dg



class Grism:
	def __init__(self, direct = None, direct_scale = 0.03, factor = 1, icenter = 5, jcenter = 5, segmentation = None, xcenter_detector = 1024, ycenter_detector = 1024, 
		wavelength = 4.2 , redshift = 7.2, wave_space = None, wave_factor = 1, wave_scale = 0.001, index_min = None, index_max = None, grism_filter = 'F444W', grism_module = 'A', grism_pupil = 'R', higher_res = False):
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
		# delta_wave = int((self.WRANGE[1]-self.WRANGE[0])/self.wave_scale)
		# self.wave_space = jnp.linspace(self.WRANGE[0],self.WRANGE[1],delta_wave+1)
		self.disp_space = self.grism_dispersion(jnp.vstack((self.xcenter_detector * jnp.ones_like(self.wave_space), self.ycenter_detector * jnp.ones_like(self.wave_space), self.wave_space)), *self.w_opt )


	def grism_dispersion(self, data, a01, a02, a03, a04, a05, a06, b01, b02, b03, b04, b05, b06, c01, c02, c03, d01):
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

		## data is an numpy array of the shape (3, N)
		##     - data[0]:  x_pixel      --> fit with second-degree polynomial
		##     - data[1]:  y_pixel      --> fit with second-degree polynomial
		##     - data[2]:  wavelength   --> fit with third-degree polynomial
		xpix, ypix, dx = data[0] - 1024, data[1] - 1024, data[2] - 3.95
		## return dx = dx(x_pixel, y_pixel, lambda)
		return ((a01 + (a02 * xpix + a03 * ypix) + (a04 * xpix**2 + a05 * xpix * ypix + a06 * ypix**2)) + 
				(b01 + (b02 * xpix + b03 * ypix) + (b04 * xpix**2 + b05 * xpix * ypix + b06 * ypix**2)) * dx +
				(c01 + (c02 * xpix + c03 * ypix)) * dx**2 + 
				(d01 ) * dx**3) 

	def set_sensitivity(self):

		self.sensitivity = self.f_sens(self.wave_space)

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

		self.sigma_lsf = 0.5*self.wavelength/R
		print(self.sigma_lsf)
		self.sigma_v_lsf = 0.5*(c/1000)/R #put c in km/s
		print(self.sigma_v_lsf)

		# print(0.001*(c/1000)/4.2)

	def compute_disp_index(self, values, disp_space):

		array = jnp.array(disp_space)
		idxs = jnp.searchsorted(array, values, side="left")

		prev_idx_is_less = (idxs == len(array))|(jnp.fabs(values - array[jnp.maximum(idxs-1, 0)]) < jnp.fabs(values - array[jnp.minimum(idxs, len(array)-1)]))
		# idxs = idxs.at[prev_idx_is_less].set(idxs[prev_idx_is_less]-1)
		idxs = jnp.where( prev_idx_is_less,  idxs -1, idxs)
		return idxs

	def compute_wave_index(self, values,wave_space):

		return jnp.argmin(wave_space-values)


	# def disperse(self, F, V, D):
	# 	'''
	# 		Disperse the full galaxy cube object pixel by pixel onto a grism space
	# 		Generates the 'model' for the dinesty fitting
			

	# 		Parameters
	# 		----------
	# 		F
	# 			2D array: flux map (direct image getting dispersed)
	# 		V
	# 			2D array: velocity map
	# 		D
	# 			2D array: vel dispersion map


		# 	Returns
		# 	----------
		# 	grism_data
		# 		2D array representing the flux in the grism image
		# 		Directly to be compared to the grism data (after some matching/scaling up )

		# '''

		# # print('V: ', V)
		# # print('D: ', D)
		# #compute the dispersion dx0 of the central pixel 
		# # print(self.wave_space.shape, self.disp_space.shape)
		# # dx0 = jnp.interp(self.wavelength*(1  + V/(c/1000) ), self.wave_space[jnp.argsort(self.disp_space)], self.disp_space[jnp.argsort(self.disp_space)])
		# dx0 = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + V/(c/1000) )), *self.w_opt)
		# DX = self.detector_position - self.xcenter_detector + dx0 
		# # print(DX.type(), self.disp_space.type(), self.wave_space.type())

		# # J = self.compute_disp_index(DX, self.disp_space)
		# # J0 = self.compute_disp_index(dx0, self.disp_space)

		# # J= jnp.argmin(jnp.abs(self.disp_space - DX[:,:, None]), axis = 2)
		# # J0 = jnp.argmin(jnp.abs(self.disp_space - dx0[:,:, None]),axis =2)

		# J_min = self.index_min
		# J_max = self.index_max

		# # J = (jnp.rint(J)).astype(int)
		# # J0 = (jnp.rint(J0)).astype(int)

		# # print(J)


		# #compute the dispersion due to the LSF
		# D_LSF = self.sigma_v_lsf*jnp.ones_like(V)
		# # print('D_LSF = ', D_LSF)

		# #add the LSF to the dispersion: 
		# CORR_D = jnp.sqrt( D**2 +  D_LSF**2)
		# DX_disp = self.grism_dispersion((self.xcenter_detector, self.ycenter_detector, self.wavelength*(1  + (V+CORR_D)/(c/1000) )), *self.w_opt)
		# DX_disp_final = self.detector_position - self.xcenter_detector + DX_disp - DX



		# NEW_J, NEW_K = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor-1, J_max-J_min+1*self.wave_factor))).astype(int) ,  (jnp.rint(jnp.linspace(0, F.shape[1]-1, F.shape[1]))).astype(int)  )
		# # print(NEW_J)
		# grism_full = jnp.zeros(self.sh_beam)

		# # mu = self.wave_space[J[:,NEW_K]]
		# # sigma = CORR_D[:, NEW_K]/(c/1000)*self.wave_space[J[:,NEW_K]]

		# # sigma = CORR_D[:, NEW_K]/(c/1000)*self.wave_space[J[:,NEW_K]]
		# # compute the edges of the disp_space array
		# disp_space_edges_prov= self.disp_space[1:] - jnp.diff(self.disp_space)/2
		# disp_space_edges_prov2 = jnp.insert(disp_space_edges_prov, 0, disp_space_edges_prov[0] - jnp.diff(self.disp_space)[0])
		# disp_space_edges = jnp.append(disp_space_edges_prov2, disp_space_edges_prov2[-1] + jnp.diff(self.disp_space)[-1])


		# # cdf = 0.5* (1 + special.erf( (self.wave_space[NEW_J] - mu)/ (sigma*math.sqrt(2.)) ))
		# # prev_wave = self.wave_space[NEW_J] - self.wave_scale
		# # cdf_prev = 0.5* (1 + special.erf( (prev_wave - mu)/ (sigma*math.sqrt(2.)) ))
		# # gaussian_matrix = cdf[:,NEW_K,NEW_J-J_min] - cdf_prev[:,NEW_K,NEW_J-J_min]
		# # print(self.disp_space[NEW_J].shape)
		# # print(mu.shape)
		# # print(sigma.shape)
		# NEW_J_cdf, NEW_K_cdf = jnp.meshgrid( (jnp.rint(jnp.linspace(J_min, J_max+1*self.wave_factor, J_max-J_min+1*self.wave_factor+1))).astype(int) ,  (jnp.rint(jnp.linspace(0, F.shape[1]-1, F.shape[1]))).astype(int)  )
		# mu = DX[:,NEW_K_cdf]
		# sigma = DX_disp_final[:,NEW_K_cdf]
		# cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
		# gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		# # print(gaussian_matrix.shape)
		# # plt.imshow(gaussian_matrix[15,:,:])
		# # plt.show()
		# grism_full = grism_full.at[:,J_min:J_max+1*self.wave_factor].set((jnp.matmul( F[:,None, :] , gaussian_matrix))[:,0,:])  #/self.sensitivity[J_min:J_max+1]) *1e3 #for mJy
		
		# # print('shape from model output: ',self.wave_factor)
		# return grism_full[:, int(J_min): int(J_max)+1*self.wave_factor]
		# # return grism_full


		# # grism_full = image.resize(grism_full[:, int(J_min): int(J_max)+1], (int(F.shape[0]/(self.direct_factor*factor)), int(grism_full[:, int(J_min): int(J_max)+1].shape[1])), 'cubic')
		# # grism_full = grism_full*(self.direct_factor*factor)
	def disperse(self, F, V, D):
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
		mu = DX[:,NEW_K_cdf]
		sigma = DX_disp_final[:,NEW_K_cdf]
		cdf = 0.5* (1 + special.erf( (disp_space_edges[NEW_J_cdf] - mu)/ (sigma*math.sqrt(2.)) ))
		gaussian_matrix = cdf[:,:,1:]-cdf[:,:,:-1]
		# plt.imshow(gaussian_matrix[60], origin = 'lower')
		# plt.show()
		# plt.imshow(F, origin = 'lower')
		# plt.show()
		# print(F.max())
		# print(gaussian_matrix.shape)
		# print('Gaussian cond number: ', jnp.linalg.cond(gaussian_matrix))
		grism_full = jnp.matmul( F[:,None, :] , gaussian_matrix)[:,0,:]  #/self.sensitivity[J_min:J_max+1]) *1e3 #for mJy
		# plt.imshow(grism_full, origin = 'lower')
		# plt.show()
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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u

def preprocess(med_band_path, LW_grism_path, grism_spectrum_path, RA, DEC, wavelength, delta_wave_cutoff = 0.02, box_size = (50,50)):
	if LW_grism_path == None:
	#load med band, LW grism , 2D spectrum (cont substraction)
		med_band_fits = fits.open(med_band_path)
		grism_spectrum_fits = fits.open(grism_spectrum_path)

		med_band_data = med_band_fits[1].data
		grism_spectrum_data = grism_spectrum_fits['SPEC2D'].data #NOT cont sub
		#do the cont subtraction
		L_box, L_mask = 25, 4
		mf_footprint = np.ones((1, L_box * 2 + 1))
		mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
		tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data, footprint=mf_footprint, mode='reflect')
		grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map
		# print('grism_spectrum_data shape: ', grism_spectrum_data.shape)


		wcs_LW_grism = wcs.WCS(LW_grism_fits[1].header)
		# wcs_LW_grism = wcs.WCS(LW_grism_fits[0].header)

		wcs_med_band = wcs.WCS(med_band_fits[1].header)

		#convert the med band units from MJy/sr to mJy
		#pixel area in sr
		pixel_area_sr = med_band_fits['SCI'].header['PIXAR_SR']
		med_band_data*= 10E9*pixel_area_sr 

		#saving these since they are inputs to the grism class
		xcenter_detector = 1024
		ycenter_detector = 1024


		#project med band on LW grism (but keep resolution)
		# factor = math.sqrt(LW_grism_fits['SCI'].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)
		# factor = math.sqrt(LW_grism_fits[0].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)

		center_med_band = jnp.argmax(med_band_data) #in (x,y) format
		center_med_band = [center_med_band%med_band_data.shape[1], center_med_band//med_band_data.shape[1]]
		# print('pixel center: ' ,center_med_band)
		#get the RA and DEC of the center pixel of the med band image
		center_world = wcs_med_band.pixel_to_world(center_med_band[0],center_med_band[1] )

		# wcs_med_band.wcs.crval[0]+= LW_world.ra.deg-center_world.ra.deg
		# wcs_med_band.wcs.crval[1]+= LW_world.dec.deg - center_world.dec.deg

		# factor = 2.088235294117647
		factor = 1


		# reproject_med_band, footprint = reproject_adaptive((med_band_data,wcs_med_band), LW_grism_cutout.wcs, (round(LW_grism_cutout.data.shape[0]*(factor)), round(LW_grism_cutout.data.shape[1]*(factor))), conserve_flux = True) #the extra pixels are nan

		#from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
		wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
		d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
		naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
		naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']

		wave_last = wave_first + d_wave*(naxis_x-1)

		wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave

		#choose which wavelengths will be the cutoff of the EL map and save those
		wave_min = wavelength - delta_wave_cutoff 
		wave_max = wavelength + delta_wave_cutoff 

		# print(wave_min, wave_max)

		index_min = round((wave_min - wave_first)/d_wave) #+10
		index_max = round((wave_max - wave_first)/d_wave) #-10
		index_wave = round((wavelength - wave_first)/d_wave)


		#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
		obs_map = grism_spectrum_data[:,index_min:index_max+1]
		#save the error array of the EL map cutout
		obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min:index_max+1], - 0.5)

		# plt.imshow(obs_map, origin='lower')
		# plt.show()

		#make a cutout around the source in the projected image (using centroid) => how to define a good size depending on galaxy size?
		#=> the y aperture is set by Fengwu = 15 pixels on either side of centroid 
		centroid_med_band = np.rint(centroid_1dg(reproject_med_band)).astype(int)  #in (x,y) format
		#calling it as the corresponding entry in the Grism class
		# direct = reproject_med_band[centroid_med_band[1]-16*4 : centroid_med_band[1]+ 17*4, centroid_med_band[0]-16*4 : centroid_med_band[0]+17*4]
		# direct = reproject_med_band
		# centroid_LW = jnp.argmax(LW_grism_cutout.data) 
		# centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
		icenter = 15
		jcenter = 15

		#compute PA
		PA_truth = compute_PA(direct)

		return obs_map, obs_error, direct, PA_truth, centroid_med_band, xcenter_detector, ycenter_detector, icenter, jcenter, wave_space, d_wave, index_min, index_max , factor
	else:
		#load med band, LW grism , 2D spectrum (cont substraction)
		med_band_fits = fits.open(med_band_path)
		LW_grism_fits = fits.open(LW_grism_path)
		grism_spectrum_fits = fits.open(grism_spectrum_path)

		med_band_data = med_band_fits[1].data
		LW_grism_data = LW_grism_fits[1].data
		grism_spectrum_data = grism_spectrum_fits['SPEC2D'].data #NOT cont sub
		#do the cont subtraction
		L_box, L_mask = 25, 4
		mf_footprint = np.ones((1, L_box * 2 + 1))
		mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
		tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data, footprint=mf_footprint, mode='reflect')
		grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map
		# print('grism_spectrum_data shape: ', grism_spectrum_data.shape)


		wcs_LW_grism = wcs.WCS(LW_grism_fits[1].header)
		# wcs_LW_grism = wcs.WCS(LW_grism_fits[0].header)

		wcs_med_band = wcs.WCS(med_band_fits[1].header)

		#convert the med band units from MJy/sr to mJy
		#pixel area in sr
		pixel_area_sr = med_band_fits['SCI'].header['PIXAR_SR']
		med_band_data*= 10E9*pixel_area_sr 

		#save filter, pupil, and module from LW grism header
		lw_filter = LW_grism_fits[0].header['FILTER']
		pupil = 'R' #LW_grism_fits[0].header['PUPIL'] #i dont actually care about this
		module = LW_grism_fits[0].header['MODULE']

		#using RA, DEC => make cutout around source in LW => find center using centroid => get detector coords of central pixels
		galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
		LW_grism_cutout = Cutout2D(LW_grism_data, galaxy_position, box_size, wcs=wcs_LW_grism)
		# plt.imshow(LW_grism_cutout.data, origin='lower')
		# plt.show()
		# print(LW_grism_cutout.data.max(), LW_grism_cutout.data.min())

		# centroid_LW = centroid_quadratic(LW_grism_cutout.data) #in (x,y) format
		centroid_LW = jnp.argmax(LW_grism_cutout.data) 
		centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
		# print('center LW :' , centroid_LW)
		LW_world = LW_grism_cutout.wcs.pixel_to_world(centroid_LW[0], centroid_LW[1]) #takes (x,y) zero-based (so column row basically)
		#add some kind of check that we got the right centroid (for ex by picking brightest pixel and comparing)
		center_detector = LW_grism_cutout.to_original_position(centroid_LW)
		#saving these since they are inputs to the grism class
		xcenter_detector = center_detector[0]
		ycenter_detector = center_detector[1]


		#project med band on LW grism (but keep resolution)
		factor = math.sqrt(LW_grism_fits['SCI'].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)
		# factor = math.sqrt(LW_grism_fits[0].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)

		center_med_band = jnp.argmax(med_band_data) #in (x,y) format
		center_med_band = [center_med_band%med_band_data.shape[1], center_med_band//med_band_data.shape[1]]
		# print('pixel center: ' ,center_med_band)
		#get the RA and DEC of the center pixel of the med band image
		center_world = wcs_med_band.pixel_to_world(center_med_band[0],center_med_band[1] )

		wcs_med_band.wcs.crval[0]+= LW_world.ra.deg-center_world.ra.deg
		wcs_med_band.wcs.crval[1]+= LW_world.dec.deg - center_world.dec.deg

		# factor = 2.088235294117647
		factor = 1


		reproject_med_band, footprint = reproject_adaptive((med_band_data,wcs_med_band), LW_grism_cutout.wcs, (round(LW_grism_cutout.data.shape[0]*(factor)), round(LW_grism_cutout.data.shape[1]*(factor))), conserve_flux = True) #the extra pixels are nan
		plt.imshow(LW_grism_cutout.data, origin='lower')
		plt.show()
		#from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
		wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
		d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
		naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
		naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']

		wave_last = wave_first + d_wave*(naxis_x-1)

		wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave

		#choose which wavelengths will be the cutoff of the EL map and save those
		wave_min = wavelength - delta_wave_cutoff 
		wave_max = wavelength + delta_wave_cutoff 

		# print(wave_min, wave_max)

		index_min = round((wave_min - wave_first)/d_wave) #+10
		index_max = round((wave_max - wave_first)/d_wave) #-10
		index_wave = round((wavelength - wave_first)/d_wave)


		#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
		obs_map = grism_spectrum_data[:,index_min:index_max+1]
		#save the error array of the EL map cutout
		obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min:index_max+1], - 0.5)

		# plt.imshow(obs_map, origin='lower')
		# plt.show()

		#make a cutout around the source in the projected image (using centroid) => how to define a good size depending on galaxy size?
		#=> the y aperture is set by Fengwu = 15 pixels on either side of centroid 
		centroid_med_band = np.rint(centroid_1dg(reproject_med_band)).astype(int)  #in (x,y) format
		#calling it as the corresponding entry in the Grism class
		# direct = reproject_med_band[centroid_med_band[1]-16*4 : centroid_med_band[1]+ 17*4, centroid_med_band[0]-16*4 : centroid_med_band[0]+17*4]
		direct = reproject_med_band

		plt.imshow(med_band_data, origin='lower')
		plt.show()
		# centroid_LW = jnp.argmax(LW_grism_cutout.data) 
		# centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
		icenter = 13
		jcenter = 14

		#compute PA
		PA_truth = compute_PA(direct)
		# PA_truth = 150

		obs_map = jnp.flip(obs_map, axis = 1)
		obs_error = jnp.flip(obs_error, axis = 1)

		return obs_map, obs_error, direct, reproject_med_band, LW_grism_cutout, PA_truth, centroid_med_band, xcenter_detector, ycenter_detector, icenter, jcenter, wave_space, d_wave, index_min, index_max , factor


def preprocess_new(med_band_path, LW_grism_path, grism_spectrum_path, RA, DEC, wavelength, delta_wave_cutoff = 0.02, box_size = (50,50)):
	#load med band, LW grism , 2D spectrum (cont substraction)
	med_band_fits = fits.open(med_band_path)
	LW_grism_fits = fits.open(LW_grism_path)
	grism_spectrum_fits = fits.open(grism_spectrum_path)

	med_band_data = med_band_fits[0].data
	LW_grism_data = LW_grism_fits[0].data
	grism_spectrum_data = grism_spectrum_fits['SPEC2D'].data #NOT cont sub
	plt.imshow(med_band_data[50:150,50:150], origin='lower')
	plt.show()
	# plt.imshow(LW_grism_data, origin='lower', vmin = -0.05, vmax = 0.05)
	# plt.show()
	med_band_data = med_band_data[50:150,50:150]
	#do the cont subtraction
	L_box, L_mask = 25, 4
	mf_footprint = np.ones((1, L_box * 2 + 1))
	mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
	tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data, footprint=mf_footprint, mode='reflect')
	grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map
	# print('grism_spectrum_data shape: ', grism_spectrum_data.shape)

	# plt.imshow(grism_spectrum_data, origin='lower', vmin =-0.01, vmax = 0.01)
	# plt.show()

	wcs_LW_grism = wcs.WCS(LW_grism_fits[0].header)
	# wcs_LW_grism = wcs.WCS(LW_grism_fits[0].header)

	wcs_med_band = wcs.WCS(med_band_fits[0].header)

	#convert the med band units from MJy/sr to mJy
	#pixel area in sr
	# pixel_area_sr = med_band_fits[0].header['PIXAR_SR']
	# med_band_data*= 10E9*pixel_area_sr 

	#save filter, pupil, and module from LW grism header
	lw_filter = LW_grism_fits[0].header['FILTER']
	pupil = 'R' #LW_grism_fits[0].header['PUPIL'] #i dont actually care about this
	module = 'A' #LW_grism_fits[0].header['MODULE']

	#using RA, DEC => make cutout around source in LW => find center using centroid => get detector coords of central pixels
	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
	LW_grism_cutout = Cutout2D(LW_grism_data, galaxy_position, box_size, wcs=wcs_LW_grism)
	plt.imshow(LW_grism_cutout.data, origin='lower')
	plt.show()
	# print(LW_grism_cutout.data.max(), LW_grism_cutout.data.min())

	# centroid_LW = centroid_quadratic(LW_grism_cutout.data) #in (x,y) format
	centroid_LW = jnp.argmax(LW_grism_cutout.data) 
	centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
	# print('center LW :' , centroid_LW)
	LW_world = LW_grism_cutout.wcs.pixel_to_world(centroid_LW[0], centroid_LW[1]) #takes (x,y) zero-based (so column row basically)
	#add some kind of check that we got the right centroid (for ex by picking brightest pixel and comparing)
	center_detector = LW_grism_cutout.to_original_position(centroid_LW)
	#saving these since they are inputs to the grism class
	xcenter_detector = center_detector[0]
	ycenter_detector = center_detector[1]


	#project med band on LW grism (but keep resolution)
	# factor = math.sqrt(LW_grism_fits[0].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)
	# factor = math.sqrt(LW_grism_fits[0].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)
	# factor = 2.1
	center_med_band = jnp.argmax(med_band_data) #in (x,y) format
	center_med_band = [center_med_band%med_band_data.shape[1], center_med_band//med_band_data.shape[1]]
	# print('pixel center: ' ,center_med_band)
	#get the RA and DEC of the center pixel of the med band image
	center_world = wcs_med_band.pixel_to_world(center_med_band[0],center_med_band[1] )

	wcs_med_band.wcs.crval[0]+= LW_world.ra.deg-center_world.ra.deg
	wcs_med_band.wcs.crval[1]+= LW_world.dec.deg - center_world.dec.deg

	factor = 2.088235294117647
	# factor = 1

	# print(round(LW_grism_cutout.data.shape[0]*(factor)))
	# print(wcs_med_band, LW_grism_cutout.wcs)
	reproject_LW, footprint = reproject_adaptive((med_band_data,wcs_med_band), LW_grism_cutout.wcs, (round(LW_grism_cutout.data.shape[0]), round(LW_grism_cutout.data.shape[1])), conserve_flux = True) #the extra pixels are nan
	plt.imshow(reproject_LW, origin = 'lower')
	plt.show()
	LW_grism_cutout.wcs.wcs.cd/= factor
	LW_grism_cutout.wcs.wcs.crpix *= factor
	reproject_med_band, footprint = reproject_adaptive((med_band_data,wcs_med_band), LW_grism_cutout.wcs, (round(LW_grism_cutout.data.shape[0]*(factor)), round(LW_grism_cutout.data.shape[1]*(factor))), conserve_flux = True) #the extra pixels are nan
	plt.imshow(reproject_med_band, origin = 'lower')
	plt.show()
	#from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
	wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
	d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
	naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
	naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']
	# print(wave_first, d_wave, naxis_x, naxis_y)

	wave_last = wave_first + d_wave*(naxis_x-1)

	wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave

	#choose which wavelengths will be the cutoff of the EL map and save those
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	# print(wave_min, wave_max)

	index_min = round((wave_min - wave_first)/d_wave) #+10
	index_max = round((wave_max - wave_first)/d_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - wave_first)/d_wave)


	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	obs_map = grism_spectrum_data[:,index_min+1:index_max+1+1]
	# obs_map = grism_spectrum_data[:,350:450]
	#save the error array of the EL map cutout
	obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min+1:index_max+1+1], - 0.5)

	# plt.imshow(obs_map, origin='lower')
	# plt.show()

	#make a cutout around the source in the projected image (using centroid) => how to define a good size depending on galaxy size?
	#=> the y aperture is set by Fengwu = 15 pixels on either side of centroid 
	centroid_med_band = np.rint(centroid_1dg(reproject_med_band)).astype(int)  #in (x,y) format
	#calling it as the corresponding entry in the Grism class
	# direct = reproject_med_band[centroid_med_band[1]-16*4 : centroid_med_band[1]+ 17*4, centroid_med_band[0]-16*4 : centroid_med_band[0]+17*4]
	direct = reproject_med_band
	centroid_LW = jnp.argmax(LW_grism_cutout.data) 
	centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
	# print(centroid_LW)
	icenter = 15
	jcenter = 15

	#compute PA
	PA_truth = compute_PA(direct)

	obs_map = jnp.flip(obs_map, axis = 1)
	obs_error = jnp.flip(obs_error, axis = 1)

	return obs_map, obs_error, direct, reproject_med_band, reproject_LW, LW_grism_cutout.data, PA_truth, centroid_med_band, xcenter_detector, ycenter_detector, icenter, jcenter, wave_space, d_wave, index_min, index_max , factor


def preprocess_mosaic(med_band_path, LW_grism_path, grism_spectrum_path, RA, DEC, wavelength, delta_wave_cutoff = 0.02, box_size = (50,50)):
	#load med band, LW grism , 2D spectrum (cont substraction)
	med_band_fits = fits.open(med_band_path)
	LW_grism_fits = fits.open(LW_grism_path)
	grism_spectrum_fits = fits.open(grism_spectrum_path)

	med_band_data = med_band_fits[0].data
	LW_grism_data = LW_grism_fits[0].data
	grism_spectrum_data = grism_spectrum_fits['SPEC2D'].data #NOT cont sub
	# plt.imshow(med_band_data[50:150,50:150], origin='lower')
	# plt.show()
	# plt.imshow(LW_grism_data, origin='lower', vmin = -0.05, vmax = 0.05)
	# plt.show()
	med_band_data = med_band_data[50:150,50:150]
	#do the cont subtraction
	L_box, L_mask = 25, 4
	mf_footprint = np.ones((1, L_box * 2 + 1))
	mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
	tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data, footprint=mf_footprint, mode='reflect')
	grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map
	# print('grism_spectrum_data shape: ', grism_spectrum_data.shape)

	# plt.imshow(grism_spectrum_data, origin='lower', vmin =-0.01, vmax = 0.01)
	# plt.show()

	wcs_LW_grism = wcs.WCS(LW_grism_fits[0].header)
	# wcs_LW_grism = wcs.WCS(LW_grism_fits[0].header)

	wcs_med_band = wcs.WCS(med_band_fits[0].header)

	#convert the med band units from MJy/sr to mJy
	#pixel area in sr
	# pixel_area_sr = med_band_fits[0].header['PIXAR_SR']
	# med_band_data*= 10E9*pixel_area_sr 

	#save filter, pupil, and module from LW grism header
	lw_filter = LW_grism_fits[0].header['FILTER']
	pupil = 'R' #LW_grism_fits[0].header['PUPIL'] #i dont actually care about this
	module = 'A' #LW_grism_fits[0].header['MODULE']

	#using RA, DEC => make cutout around source in LW => find center using centroid => get detector coords of central pixels
	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
	LW_grism_cutout = Cutout2D(LW_grism_data, galaxy_position, box_size, wcs=wcs_LW_grism)
	# plt.imshow(LW_grism_cutout.data, origin='lower')
	# plt.show()
	# print(LW_grism_cutout.data.max(), LW_grism_cutout.data.min())

	# centroid_LW = centroid_quadratic(LW_grism_cutout.data) #in (x,y) format
	centroid_LW = jnp.argmax(LW_grism_cutout.data) 
	centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
	# print('center LW :' , centroid_LW)
	LW_world = LW_grism_cutout.wcs.pixel_to_world(centroid_LW[0], centroid_LW[1]) #takes (x,y) zero-based (so column row basically)
	#add some kind of check that we got the right centroid (for ex by picking brightest pixel and comparing)
	center_detector = LW_grism_cutout.to_original_position(centroid_LW)
	#saving these since they are inputs to the grism class
	xcenter_detector = center_detector[0]
	ycenter_detector = center_detector[1]


	#project med band on LW grism (but keep resolution)
	# factor = math.sqrt(LW_grism_fits[0].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)
	# factor = math.sqrt(LW_grism_fits[0].header['PIXAR_SR'])/math.sqrt(pixel_area_sr)
	# factor = 2.1
	center_med_band = jnp.argmax(med_band_data) #in (x,y) format
	center_med_band = [center_med_band%med_band_data.shape[1], center_med_band//med_band_data.shape[1]]
	# print('pixel center: ' ,center_med_band)
	#get the RA and DEC of the center pixel of the med band image
	center_world = wcs_med_band.pixel_to_world(center_med_band[0],center_med_band[1] )

	wcs_med_band.wcs.crval[0]+= LW_world.ra.deg-center_world.ra.deg
	wcs_med_band.wcs.crval[1]+= LW_world.dec.deg - center_world.dec.deg

	factor = 2.088235294117647
	# factor = 1

	# print(round(LW_grism_cutout.data.shape[0]*(factor)))
	# print(wcs_med_band, LW_grism_cutout.wcs)
	reproject_LW, footprint = reproject_adaptive((med_band_data,wcs_med_band), LW_grism_cutout.wcs, (round(LW_grism_cutout.data.shape[0]), round(LW_grism_cutout.data.shape[1])), conserve_flux = True) #the extra pixels are nan
	# plt.imshow(reproject_LW, origin = 'lower')
	# plt.show()
	LW_grism_cutout.wcs.wcs.cd/= factor
	LW_grism_cutout.wcs.wcs.crpix *= factor
	reproject_med_band, footprint = reproject_adaptive((med_band_data,wcs_med_band), LW_grism_cutout.wcs, (round(LW_grism_cutout.data.shape[0]*(factor)), round(LW_grism_cutout.data.shape[1]*(factor))), conserve_flux = True) #the extra pixels are nan
	# plt.imshow(reproject_med_band, origin = 'lower')
	# plt.show()
	#from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
	wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
	d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
	naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
	naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']
	# print(wave_first, d_wave, naxis_x, naxis_y)

	wave_last = wave_first + d_wave*(naxis_x-1)

	wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave

	#choose which wavelengths will be the cutoff of the EL map and save those
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	# print(wave_min, wave_max)

	index_min = round((wave_min - wave_first)/d_wave) #+10
	index_max = round((wave_max - wave_first)/d_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - wave_first)/d_wave)


	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	obs_map = grism_spectrum_data[:,index_min+1:index_max+1+1]
	# obs_map = grism_spectrum_data[:,350:450]
	#save the error array of the EL map cutout
	obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min+1:index_max+1+1], - 0.5)

	# plt.imshow(obs_map, origin='lower')
	# plt.show()

	#make a cutout around the source in the projected image (using centroid) => how to define a good size depending on galaxy size?
	#=> the y aperture is set by Fengwu = 15 pixels on either side of centroid 
	centroid_med_band = np.rint(centroid_1dg(reproject_med_band)).astype(int)  #in (x,y) format
	#calling it as the corresponding entry in the Grism class
	# direct = reproject_med_band[centroid_med_band[1]-16*4 : centroid_med_band[1]+ 17*4, centroid_med_band[0]-16*4 : centroid_med_band[0]+17*4]
	direct = reproject_med_band
	centroid_LW = jnp.argmax(LW_grism_cutout.data) 
	centroid_LW = [centroid_LW%LW_grism_cutout.data.shape[1], centroid_LW//LW_grism_cutout.data.shape[1]]#in (x,y) format
	# print(centroid_LW)
	icenter = 14
	jcenter = 15

	#compute PA
	PA_truth = compute_PA(direct)

	obs_map = jnp.flip(obs_map, axis = 1)
	obs_error = jnp.flip(obs_error, axis = 1)

	return obs_map, obs_error, direct, reproject_med_band, reproject_LW, LW_grism_cutout.data, PA_truth, centroid_med_band, xcenter_detector, ycenter_detector, icenter, jcenter, wave_space, d_wave, index_min, index_max , factor


def make_EL_map(med_band, broad_band, z, line):
	#function based on the module make_EL_map.py to create a good approximation of the EL map from medium and broad band images

	#define the redshift ranges

	z_410M = [4.92, 5.61] #Halpha falls in F410M
	z_444W = [5.61, 6.59] #Hbeta falls in F444W but not 410M

	z_335M = [5.37, 6.15] #OIII 5007 falls in F335M
	z_356W = [6.15, 6.77] #OIII 4959 falls in F356W but not 335M

	#Hbeta limits for reference, but I don't think I can do much about Hbeta contamination
	z_335M_Hbeta = [5.537-6.276] #Hbeta falls in F335M
	z_356W_Hbeta = [5.449 - 7.1897] #Hbeta falls in F356W

	z_356W_halpha = [4.37,4.92] #Halpha falls in F356W but not 335M
	z_335M_halpha = [3.90,4.37] #Halpha falls in F335M

	#define the filter widths
	width_F410M = 0.436
	width_F444W = 1.101
	width_F335M = 0.360
	width_F356W = 0.846

	box_size = med_band.shape[0]

	EL_map = np.zeros((box_size, box_size))

	if line == 'Halpha' :
		if z_410M[0] < z < z_410M[1]:
			print('Halpha is in F410M and F444W')
			# Halpha_map = image_cutout_F410M - (image_cutout_F444W-image_cutout_F410M)/(width_F444W-width_F410M)*(width_F410M)
			EL_map = (med_band*width_F410M*width_F444W - broad_band*width_F444W*width_F410M)/(width_F444W-width_F410M)
			# Halpha_map = image_cutout_F410M - image_cutout_F444W
		#if the galaxy is in z_444W, then estimate Halpha map as F444W-F410M
		elif z_444W[0] < z < z_444W[1]:
			print('Halpha is only in F444W')
			EL_map = broad_band*width_F444W- med_band*width_F444W
		elif z_356W_halpha[0] < z < z_356W_halpha[1]:
			print('Halpha is only in F356W') 
			EL_map = broad_band*width_F356W - med_band*width_F356W 
		elif z_335M_halpha[0] < z < z_335M_halpha[1]:
			print('Halpha is in F335M and F356W')
			EL_map = (med_band*width_F335M*width_F356W - broad_band*width_F356W*width_F335M)/(width_F356W-width_F335M)
		else:
			print('Halpha is not in the observed range, returning zeros')
	
	elif line == 'OIII_5007':
		if z_335M[0] < z < z_335M[1]:
			print('OIII_5007 is in F335M and F356W')
			EL_map = (med_band*width_F335M*width_F356W - broad_band*width_F356W*width_F335M)/(width_F356W-width_F335M)
		#if the galaxy is in z_356W, then estimate OIII map as F356W-F335M
		elif z_356W[0] < z < z_356W[1]:
			print('OIII_5007 is only in F356W ')
			EL_map = broad_band*width_F356W - med_band*width_F356W 
		else:
			print('OIII_5007 is not in the observed range, returning zeros')

	return EL_map

	

def preprocess_test(med_band_path, broad_band_path, grism_spectrum_path, redshift, line, wavelength, delta_wave_cutoff = 0.02, field = 'GOODS-S', fitting = None):
	 

	med_band_fits = fits.open(med_band_path)
	med_band = med_band_fits[1].data

	broad_band_fits = fits.open(broad_band_path)
	broad_band = broad_band_fits[1].data

	wcs_med_band = wcs.WCS(med_band_fits[1].header)

	cutout_size = med_band.shape[0]

	grism_spectrum_fits = fits.open(grism_spectrum_path)

	RA = grism_spectrum_fits[0].header['RA0']
	DEC = grism_spectrum_fits[0].header['DEC0']


	#compute the EL map prior first
	EL_map = make_EL_map(med_band, broad_band, redshift, line)


	if field == 'GOODS-S':
		print('FRESCO PA is the same in GOODS-S, no correction needed')
		med_band_flip = med_band
	else:
		if field == 'GOODS-N':
			print('Correcting image for GOODS-N FRESCO PA')
			theta = 230.5098
		elif field == 'GOODS-N-CONGRESS':
			print('Correcting image for GOODS-N Congress PA')
			theta = 228.22379
		# print(wcs.WCS(wcs_med_band.to_header()).wcs.pc)
		rotated_wcs = wcs.WCS(med_band_fits[1].header)
		theta = np.deg2rad(theta)
		sinq = np.sin(theta)
		cosq = np.cos(theta)
		mrot = np.array([[cosq, -sinq],
                         [sinq, cosq]])
		print(mrot)
		if rotated_wcs.wcs.has_cd():    # CD matrix
			newcd = np.dot(mrot, rotated_wcs.wcs.cd)
			rotated_wcs.wcs.cd = newcd
			# rotated_wcs.wcs.set()
		elif rotated_wcs.wcs.has_pc():      # PC matrix + CDELT
			newpc = np.dot(rotated_wcs.wcs.get_pc(),mrot)
			rotated_wcs.wcs.pc = newpc
			# rotated_wcs.wcs.set()
		rotated_wcs.wcs.crpix = [cutout_size//2,cutout_size//2]
		rotated_wcs.wcs.crval = wcs_med_band.all_pix2world(cutout_size//2,cutout_size//2,0)
		rotated_wcs.wcs.set()
		print('Setting new central pixel and value: ', rotated_wcs.wcs.crpix, rotated_wcs.wcs.crval)

		# print('wcs med band : ', wcs_med_band)
		# print('rotated wcs : ', rotated_wcs)
		#reproject both images in the rotated WCS frame
		# med_band_flip, footprint = reproject_adaptive((med_band, wcs_med_band), rotated_wcs, shape_out = med_band.shape)
		# broad_band_flip, footprint = reproject_adaptive((broad_band, wcs_med_band), rotated_wcs, shape_out = broad_band.shape)
		EL_map_flip, footprint = reproject_adaptive((EL_map, wcs_med_band), rotated_wcs, shape_out = EL_map.shape)
		#flip this too so the PA can be computed
		med_band_flip, footprint = reproject_adaptive((med_band, wcs_med_band), rotated_wcs, shape_out = med_band.shape)

	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
	# LW_grism_cutout = Cutout2D(LW_grism, galaxy_position, (50,50), wcs=wcs_wide_band)

	# factor = 2 #2.088235294117647
	#rescaling the cutout to med band resolution
	# blocks = med_band_flip.reshape((int(med_band_flip.shape[0]/(factor)), factor, int(med_band_flip.shape[1]/factor), factor))
	# med_band_flipped_rescaled = np.sum(blocks, axis=(1,3))
	# plt.imshow(med_band_flipped_rescaled, origin='lower')
	# plt.title('Flipped and correct scale med band')
	# plt.show()

	#find the photometric center of the galaxy
	# print(np.argmax(med_band_flipped_rescaled))
	#the problem is that if there's other galaxy in the cutout then it will find the wrong max..
	#not sure how to go about this
	#for now, assume the galaxy max is in the central 40x40 pixels so compute the argmax in the central 40x40 pixels of med_band_flip
	# image_center = med_band_flip.shape[0]//2
	# icenter_medband, jcenter_medband = np.unravel_index(np.argmax(med_band_flip[image_center-20:image_center+20,image_center-20:image_center+20], axis=None), med_band_flip[image_center-20:image_center+20,image_center-20:image_center+20].shape)
	# icenter_medband+=image_center-20 #+2
	# jcenter_medband+=image_center-20

	#compute the med band center from the WCS coords:
	icenter_high, jcenter_high = rotated_wcs.world_to_array_index_values(galaxy_position.ra.deg, galaxy_position.dec.deg)
	if icenter_high < 0 or icenter_high > cutout_size or jcenter_high < 0 or jcenter_high > cutout_size:
		print('The galaxy is not in the cutout')
		return None
	else:
		print('Center of the galaxy in the high res image: ', icenter_high, jcenter_high)

	plt.imshow(EL_map_flip, origin='lower')
	plt.title('Correctly rotated image')
	plt.show()
	#cut the med band image to 62x62 around the photometric center: icenter_medband, jcenter_medband
	high_res_prior = EL_map_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]
	med_band_cutout = med_band_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]
	#med_band_prior = med_band_flipped_rescaled[25-15:25+15,25-15:25+15] #need to generalize this when the sizes will be different 
	# med_band_prior = med_band_flipped_rescaled
	plt.imshow(high_res_prior, origin='lower')
	plt.title('Flipped and correct scale image 62x62 cutout')
	plt.show()

	icenter_high, jcenter_high = 31, 31 #this is by defintion of how the cutout was made

	factor = 2
	#Rescale the med band cutout to 31x31 cutout by halphing the resolution to LW resolution
	blocks = high_res_prior.reshape((int(high_res_prior.shape[0]/(factor)), factor, int(high_res_prior.shape[1]/factor), factor))
	low_res_prior = np.sum(blocks, axis=(1,3))
	plt.imshow(low_res_prior, origin='lower')
	plt.title('Flipped and correct scale image 31x31 low res cutout')
	plt.show()

	# icenter_LWband, jcenter_LWband = np.unravel_index(np.argmax(LW_band_prior, axis=None), LW_band_prior.shape)
	icenter_low, jcenter_low = 15, 15 #this is by defintion of how the cutout was made
	print('Center of the galaxy in the LW band image: ', icenter_low, jcenter_low)

	#now moving on to the grism data
	grism_spectrum_data = grism_spectrum_fits['SPEC2D'].data #NOT cont sub

	#do the cont subtraction
	L_box, L_mask = 25, 4
	mf_footprint = np.ones((1, L_box * 2 + 1))
	mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
	tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data, footprint=mf_footprint, mode='reflect')
	grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map

	xcenter_detector = 1024
	ycenter_detector = 1024

	#from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
	wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
	d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
	naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
	naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']
	# print(wave_first, d_wave, naxis_x, naxis_y)

	wave_last = wave_first + d_wave*(naxis_x-1)

	wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave


	#choose which wavelengths will be the cutoff of the EL map and save those
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	# print(wave_min, wave_max)

	index_min = round((wave_min - wave_first)/d_wave) #+10
	index_max = round((wave_max - wave_first)/d_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - wave_first)/d_wave)


	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	obs_map = grism_spectrum_data[:,index_min+1:index_max+1+1]

	# obs_map = grism_spectrum_data[:,350:450]
	#save the error array of the EL map cutout
	obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min+1:index_max+1+1], - 0.5)

	#load number of module A and module B frames from the header
	module_A = grism_spectrum_fits[0].header['N_A']
	module_B = grism_spectrum_fits[0].header['N_B']

	if module_A == 0:
		print('Flipping map! (Mod B)')
		obs_error = jnp.flip(obs_error, axis = 1)
		obs_map = jnp.flip(obs_map, axis = 1)

	plt.imshow(obs_map, origin='lower')
	plt.title('EL map cutout')
	plt.show()
	
	if fitting == 'high':
		direct = high_res_prior
		icenter_prior = icenter_high
		jcenter_prior = jcenter_high
	elif fitting == 'low':
		direct = low_res_prior
		icenter_prior = icenter_low
		jcenter_prior = jcenter_low

	#compute PA from the cropped med band image (not the EL map)
	PA_truth = compute_PA(med_band_cutout)

	return jnp.array(obs_map), jnp.array(obs_error), jnp.array(direct), PA_truth, xcenter_detector, ycenter_detector, icenter_prior, jcenter_prior, wave_space, d_wave, index_min, index_max , factor

def preprocess_ALT(data_path, RA, DEC, x_center, delta_wave_cutoff = 0.02, box_size = (50,50), factor = 2):
	# resample the med band image to LW resolution
	data = fits.open(data_path)
	med_band_data = data['STAMP'].data
	med_band_wcs = wcs.WCS(data['STAMP'].header)
	RA = data[0].header['RA']
	DEC = data[0].header['DEC']
	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)

	cutout = Cutout2D(med_band_data.data, galaxy_position, box_size, wcs=med_band_wcs)

	blocks = cutout.data.reshape(int(cutout.data.shape[0]/factor),factor,int(cutout.data.shape[1]/factor),factor)
	LW_image = np.sum(blocks, axis = (1,3))

	grism_data = data['SCI'].data
	filtered_grism_data = data['EMLINEA'].data #for now hardcoded mod A
	error_data = data['ERRA'].data

	x = jnp.linspace(0,grism_data.shape[1]-1, grism_data.shape[1])
	wave_space = 3 + 0.000975*x #in microns
	delta_wave = 0.000975

	index_min = round(x_center - round(delta_wave_cutoff/0.000975))
	index_max = round(x_center + round(delta_wave_cutoff/0.000975))

	obs_map = filtered_grism_data[:, index_min:index_max+1]
	error_map = error_data[:, index_min:index_max+1]

	icenter,jcenter = jnp.unravel_index(jnp.argmax(LW_image), LW_image.shape)
	
	PA_truth = compute_PA(cutout.data)

	return obs_map, error_map, LW_image, PA_truth, icenter, jcenter, wave_space, delta_wave, index_min, index_max


from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

def compute_PA(image):
	'''
		compute PA of galaxy in the image using skimage measure regionprops

	'''
	threshold = threshold_otsu(image)
	bw = closing(image > threshold)

	cleared = clear_border(bw)

	#label image regions
	label_image = label(cleared)
	image_label_overlay = label2rgb(label_image, image=image)
	regions = regionprops(label_image, image)
	PA = 90 + regions[0].orientation * (180/np.pi)
	
	return PA



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Image:
	def __init__(self, image_fits =None, image_data = None, image_err = None, image_dq = None, image_wcs = None ):
		"""Object for a galaxy image container
		initialize with a fits file or manually
		assumes this is already a cutout around the galaxy

		Parameters
		----------

			
		Attributes
		----------

		"""

		if image_fits is None: 
			self.image_data = image_data 
			self.image_err = image_err
			self.image_dq = image_dq
			self.image_wcs = image_wcs
		else:
			self.image_sci = image_fits[1]
			self.image_data = self.image_sci.data
			self.image_err = image_fits[2].data
			self.image_dq = image_fits[3].data
			self.image_wcs = wcs.WCS(self.image_sci.header)

		#define the centroid 
		self.centroid = centroid_com(self.image_data) #in pixel order (x,y) 
		self.icenter = round(self.centroid[1])
		self.jcenter = round(self.centroid[0])

		if self.image_wcs is not None:
			self.centroid_RA = self.image_wcs.pixel_to_world(self.centroid[0], self.centroid[1]).ra.degree
			self.centroid_DEC = self.image_wcs.pixel_to_world(self.centroid[0], self.centroid[1]).dec.degree

			pixel_scale_degrees = wcs.utils.proj_plane_pixel_scales(self.image_wcs) #in degrees

			x_deg = Angle(pixel_scale_degrees[0],unit=u.deg) 
			y_deg = Angle(pixel_scale_degrees[1],unit=u.deg)

			self.pixel_scale = (x_deg.arcsecond, y_deg.arcsecond) 


	def resample_to_scale(self, scale, conserve_flux = False, boundary_mode = 'grid-constant', boundary_fill_value = 0.0):
		"""
			Resample image to a given scale, and modify wcs accordinfly
			this leaves err and dq untouched (bc doesn't make sense to resampled errors)


		Parameters
		----------

			
		Attributes
		----------

		"""
		factor = round(self.pixel_scale[0]/scale)

		new_wcs = self.image_wcs.deepcopy()
		#cd and cdelt represent the scale (which gets smaller w the resampling)
		new_wcs.wcs.cd = self.image_wcs.wcs.cd/factor
		#cr pix is the reference pixel, which has to be multiplied since there are now many more pixels
		new_wcs.wcs.crpix = self.image_wcs.wcs.crpix*factor

		array, footprint = reproject_adaptive((self.image_data,self.image_wcs), new_wcs, [np.shape(self.image_data)[0] * factor,np.shape(self.image_data)[1] * factor], conserve_flux = conserve_flux, boundary_mode = boundary_mode, boundary_fill_value = boundary_fill_value)
		

		#change all of the object attributes

		self.image_data = array
		self.image_wcs = new_wcs

		self.pixel_scale /= factor

		self.centroid *=  factor
		self.icenter *= factor
		self.jcenter *= factor

	# def project_on_wcs(self, new_wcs):

	# 	array, footprint = reproject_adaptive((self.image_data,self.image_wcs), new_wcs, [np.shape(self.image_data)[0] * factor,np.shape(self.image_data)[1] * factor], conserve_flux = True, boundary_mode = 'grid-constant', boundary_fill_value = 0.0)

	def reproject_and_align(self, wcs ):
		"""
			Project this image to another WCS given as input parameters (astropy.WCS object)
			Also need to align with dispersion direction (this is done automatically?)
			The change is done internally (can always copy the object?)
		

		Parameters
		----------

			
		Attributes
		----------

		"""
		#the shape is dictated by how many pixels we want => play with this but you could define an aperture in arcseconds to make it independent of scale
		shape_out = [30,30]
		
		self.reprojected_image_data, footprint = reproject_adaptive((self.image_data,self.image_wcs), wcs, shape_out, conserve_flux = conserve_flux, boundary_mode = 'grid-constant', boundary_fill_value = 0.0)



	def initialize_detector_properties(self,grism_direct_image_fits):
		"""
			Initializes all of the detector information from the grism image (wcs, center of our galaxy, pixel scale, etc, filter, module, pupil)
		

		Parameters
		----------

			
		Attributes
		----------

		"""
		self.grism_direct_image_sci = grism_direct_image_fits[1]
		self.grism_direct_image_data = grism_direct_image_fits[1].data
		self.grism_direct_image_wcs = wcs.WCS(self.grism_direct_image_sci.header)
		self.grism_direct_module = grism_direct_image_fits[0].header['MODULE']
		self.grism_direct_filter = grism_direct_image_fits[0].header['FILTER']
		self.grism_direct_pupil =  grism_direct_image_fits[0].header['PUPIL']
		self.telescope_pointing = (self.grism_direct_image_sci.header['RA_V1'], self.grism_direct_image_sci.header['DEC_V1'], self.grism_direct_image_sci.header['PA_V3'])


	def find_detector_center(self):

		"""
			Find the coordinates of the galaxy centroid on the detector image that we are forward modelling
		

		Parameters
		----------

			
		Attributes
		----------

		"""
		position = SkyCoord(ra=self.centroid_RA * u.deg, dec=self.centroid_DEC * u.deg)
		self.detector_centroid_pixels = self.grism_direct_image_wcs.world_to_pixel(position) #comes in (x,y) format
		self.detector_centroid_pixelx = self.detector_centroid_pixels[0]
		self.detector_centroid_pixely = self.detector_centroid_pixels[1]

	def convolve_with_PSF(self, PSF):
		"""
			To call after each sampling of the flux before dipsersing it. This has to be the NIRCam F444W PSF
		

		Parameters
		----------

			
		Attributes
		----------

		"""


	def deconvolve_with_PSF(self, PSF):
		"""
			To get best prior from Med band JADES image
		

		Parameters
		----------

			
		Attributes
		----------

		"""
