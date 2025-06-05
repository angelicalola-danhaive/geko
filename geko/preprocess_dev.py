"""
Put all of the necessary pre-processing functions here so that fit_numpyro is cleaner
Eventually should also add here scripts to automatically create folders for sources, with all of the right cutouts etc
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ['run_full_preprocessing', 'prep_grism']

#imports
from . import  utils
from . import  grism_dev #just for now since in active dev
from . import  models
# import run_pysersic as py

import numpy as np
import math
import matplotlib.pyplot as plt

import jax.numpy as jnp

import yaml

from scipy.ndimage import median_filter, sobel, center_of_mass
from scipy import ndimage

#for the masking
from skimage.morphology import  dilation, disk
from skimage.filters import threshold_otsu
from skimage.measure import centroid

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import wcs
from astropy.table import Table

from reproject import reproject_adaptive
from scipy.constants import c
from numpyro.infer.util import log_likelihood


from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.aperture import EllipticalAperture

# from skimage.morphology import dilation, disk, ellipse, binary_closing, closing

from photutils.segmentation import detect_sources, SourceCatalog, SegmentationImage, deblend_sources

from skimage import color, data, restoration

c_m = c*1e-9




def read_config_file(input, output, master_cat_path, line):
	"""

		Read the config file for the galaxy and returns all of the relevant parameters

	"""
	data = input[0]
	params = input[1]
	inference = input[2]

	ID = data['Data']['ID']

	master_cat =Table.read(master_cat_path, format="ascii")
	
	field = data['Data']['field']


	if field == 'GOODS-N':
		grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_GDN_' + 'F444W' + '_ID' + str(ID) + '_comb.fits'
		grism_filter = 'F444W'
	elif field == 'GOODS-N-CONGRESS':
		grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_GDN_' + 'F356W' + '_ID' + str(ID) + '_comb.fits'
		grism_filter = 'F356W'
	elif field == 'GOODS-S-FRESCO':
		grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_FRESCO_' + 'F444W' + '_ID' + str(ID) + '_comb.fits'
		grism_filter = 'F444W'
	else:
		#generate error that the input field is not supported
		raise ValueError(f"Field {field} is not supported. Supported fields are: GOODS-N, GOODS-N-CONGRESS, GOODS-S-FRESCO.")
		

	wavelength = master_cat[master_cat['ID'] == ID][str(line) + '_lambda'][0]

	redshift = master_cat[master_cat['ID'] == ID]['zspec'][0]

	flux_threshold = inference['Inference']['flux_threshold']
	
	delta_wave_cutoff = params['Params']['delta_wave_cutoff']
	
	factor = params['Params']['factor']
	wave_factor = params['Params']['wave_factor']
 

	model_name = inference['Inference']['model']
	
	#import all of the bounds needed for the priors

	PA_sigma = inference['Inference']['PA_bounds']
	i_bounds = inference['Inference']['i_bounds']
	Va_bounds = inference['Inference']['Va_bounds']
	r_t_bounds = inference['Inference']['r_t_bounds']
	sigma0_bounds = inference['Inference']['sigma0_bounds']

	num_samples = inference['Inference']['num_samples'] 
	num_warmup = inference['Inference']['num_warmup']
	step_size = inference['Inference']['step_size']
	target_accept_prob = inference['Inference']['target_accept_prob']
	
	#return all of the parameters
	return ID, redshift, grism_filter, grism_spectrum_path, field, wavelength, redshift, line,  flux_threshold, factor, wave_factor, model_name, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff


def renormalize_image(direct,direct_error, obs_map):
	"""
		Normalize the image to match the total flux in the EL map
	"""

	# threshold = 0.2*direct[31-10:31+10,31-10:31+10 ].max()
	# # threshold = threshold_otsu(direct)
	# mask = jnp.zeros_like(direct)
	# mask = mask.at[jnp.where(direct>threshold)].set(1)
	# # mask = dilation(mask, disk(2))

	# # plot the direct image found within the mask, the rest set to 0
	# plt.imshow(direct*mask, cmap='viridis', origin='lower')
	# plt.title('Direct image within the mask')
	# plt.show()

	# #create a mask for the grism map
	# threshold_grism = 0.2*obs_map.max()
	# # threshold_grism = threshold_otsu(obs_map)
	# mask_grism = jnp.zeros_like(obs_map)
	# mask_grism = mask_grism.at[jnp.where(obs_map>threshold_grism)].set(1)
	# mask_grism = dilation(mask_grism, disk(2))

	mask_image_seg, cat_image = utils.find_central_object(direct,4)
	mask_grism_seg, cat_grism = utils.find_central_object(obs_map, 1, 5)

	mask_image = mask_image_seg.data
	mask_grism = mask_grism_seg.data

	# plot the direct image found within the mask, the rest set to 0
	plt.imshow(direct*mask_image, cmap='viridis', origin='lower')
	plt.title('Direct image within the mask')
	plt.show()


	# plot the grism image found within the mask, the rest set to 0
	plt.imshow(obs_map*mask_grism, cmap='viridis', origin='lower')
	plt.title('Grism data within the mask')
	plt.show()

	#compute the normalization factor
	normalization_factor = obs_map[jnp.where(mask_grism == 1)].sum()/direct[jnp.where(mask_image == 1)].sum()
	#normalize the direct image to the grism image
	direct = direct*normalization_factor
	direct_error = direct_error*normalization_factor
	
	plt.imshow(direct, cmap='viridis', origin='lower')
	plt.title('Normalized direct image')
	plt.show()


	return direct, direct_error, normalization_factor, mask_grism

def mask_bad_pixels(image,errors,tolerance=3.5):
	"""
		Set the hot and dead pixels to zero and set their errors to a high value so they do not
		carry weight in the fit
	"""

	#find the hot/dead pixels by using the sobel filter image
	sobel_image_h = sobel(image, 0)
	sobel_image_v = sobel(image, 1)
	sobel_image = jnp.sqrt(sobel_image_h**2 + sobel_image_v**2)
	hot_pixels = jnp.where(sobel_image > tolerance)

	#show sobel image
	plt.imshow(sobel_image, cmap='viridis', origin='lower')
	plt.colorbar()
	plt.title('Sobel filter image')
	plt.show()

	if len(hot_pixels[0]) == 0:
		print('No hot pixels found')
		return image, errors
	
	#plot the hot pixels
	plt.imshow(image, cmap='viridis', origin='lower')
	plt.scatter(hot_pixels[1],hot_pixels[0],c='r',s=1)
	plt.title('Selected hot/dead pixels')
	plt.show()

	for pixels in [hot_pixels]:
		image = image.at[pixels[0],pixels[1]].set(0)
		errors = errors.at[pixels[0],pixels[1]].set(1e6)
	
	return image, errors



def contiuum_subtraction(grism_spectrum_data, min, max):
	'''
		Subtract the continuum from the EL map
	'''
	grism_spectrum_data = grism_spectrum_data[:,min:max] #NOT cont sub

	grism_spectrum_data = jnp.where(jnp.isnan(grism_spectrum_data), 0.0, jnp.array(grism_spectrum_data))
	
	# plt.imshow(grism_spectrum_data, origin='lower')
	# plt.title('EL map before continuum subtraction')
	# plt.colorbar()
	# plt.show()

	#do the cont subtraction
	L_box, L_mask = 25, 9
	mf_footprint = np.ones((1, L_box * 2 + 1))
	mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
	# last_index = grism_spectrum_data.shape[1] - 1
	# grism_spectrum_data[:,0:50] = jnp.nan 
	# grism_spectrum_data[:,last_index-50:last_index] = jnp.nan
	# print(grism_spectrum_data[0:2,12], grism_spectrum_data[0:2,-12])
	# tmp_grism_img_median = ndimage.generic_filter(grism_spectrum_data, np.nanmedian, footprint=mf_footprint, mode='reflect')
	tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data,footprint=mf_footprint, mode='reflect')


	grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map

	#second round of filtering but masking bright regions

	# mask_seg, cat = utils.find_central_object(grism_spectrum_data[:,140:160], 0.5)
	# mask = mask_seg.data

	# grism_spectrum_data_crop = grism_spectrum_data[:,140:160]
	# grism_spectrum_data_crop= grism_spectrum_data_crop.at[jnp.where(mask == 1)].set(jnp.nan)
	# grism_spectrum_data_masked = grism_spectrum_data
	# grism_spectrum_data_masked= grism_spectrum_data_masked.at[:,140:160].set(grism_spectrum_data_crop)
	# plt.imshow(mask, origin='lower')
	# plt.title('mask')
	# plt.colorbar()
	# plt.show()
	# L_box, L_mask = 10, 12
	# mf_footprint = np.ones((1, L_box * 2 + 1))
	# tmp_grism_img_median = ndimage.generic_filter(grism_spectrum_data_masked, np.nanmedian, footprint=mf_footprint, mode='reflect')
	# grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map
	# plt.imshow(grism_spectrum_data, origin='lower')
	# plt.title('EL map after continuum subtraction')
	# plt.colorbar()
	# plt.show()

	print("Cont sub done")


	return grism_spectrum_data

def prep_grism(grism_spectrum,grism_spectrum_error, wavelength, delta_wave_cutoff = 0.02, wave_first = 3.5, d_wave = 0.001):
	'''
		Crop the grism spectrum to a smaller size, centered on the wavelength of interest
		Remove the continuum
	'''
	#choose which wavelengths will be the cutoff of the EL map and save those
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	# print(wave_min, wave_max)

	index_min = round((wave_min - wave_first)/d_wave) #+10
	index_max = round((wave_max - wave_first)/d_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - wave_first)/d_wave)

	#subtract continuum and crop image by 200 on each size of EL
	crop_size = 150
	if index_min - crop_size < 0:
		crop_size = index_min
	grism_spectrum_data = contiuum_subtraction(jnp.array(grism_spectrum), index_wave - crop_size, index_wave + crop_size)
	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	obs_map = grism_spectrum_data[:,index_min-(index_wave -crop_size):index_max-(index_wave -crop_size)+1]

	obs_error = jnp.power(jnp.array(grism_spectrum_error[:,index_min:index_max+1]), - 0.5)
		
	return obs_map, obs_error, index_min, index_max

def preprocess_data(grism_spectrum_path,wavelength, delta_wave_cutoff = 0.02, field = 'GOODS-S'):

	grism_spectrum_fits = fits.open(grism_spectrum_path)


	RA = grism_spectrum_fits[0].header['RA0']
	DEC = grism_spectrum_fits[0].header['DEC0']


	#load number of module A and module B frames from the header
	if field == 'GOODS-S-FRESCO' or field == 'GOODS-N' or field == 'GOODS-N-CONGRESS':
		# print('Manually setting the module because the N_A and N_B params are not in the header')
		# module_A = 0
		# module_B = 1
		modules = grism_spectrum_fits[5].data['module']
		module_A = int('A' in modules)


	else:
		module_A = grism_spectrum_fits[0].header['N_A']
		module_B = grism_spectrum_fits[0].header['N_B']


	#from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
	wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
	d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
	naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
	naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']
	# print(wave_first, d_wave, naxis_x, naxis_y)

	wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave


	#crop and continuum subtract the grism spectrum
	grism_spectrum = grism_spectrum_fits['SPEC2D'].data
	grism_spectrum_error = grism_spectrum_fits['WHT2D'].data
	obs_map, obs_error, index_min, index_max = prep_grism(grism_spectrum,grism_spectrum_error, wavelength, delta_wave_cutoff, wave_first, d_wave)

	#mask bad pixels in obs/error map
	obs_error = jnp.where(jnp.isnan(obs_map)| jnp.isnan(obs_error) | jnp.isinf(obs_error), 1e10, obs_error)

	module = 'A'
	if module_A == 0:
		print('Flipping map! (Mod B)')
		obs_error = jnp.flip(obs_error, axis = 1)
		obs_map = jnp.flip(obs_map, axis = 1)
		module = 'B'


	return module,jnp.array(obs_map), jnp.array(obs_error),  wave_space, d_wave, index_min, index_max, wavelength




def define_mock_params():

	broad_filter = 'F444W'
	grism_filter = 'F444W'
	wavelength = 4.5
	redshift = 5.0
	line = 'H_alpha'
	y_factor = 1
	flux_threshold = 3
	factor = 5
	wave_factor = 9
	x0 = y0 = 31//2
	model_name = 'Disk'
	flux_bounds = None
	flux_type = 'auto'
	PA_sigma = None
	i_bounds = [0,90]
	Va_bounds = None
	r_t_bounds = None
	sigma0_bounds = None #can put similar bounds on this using the Va measured from 1D spectrum
	num_samples = 500
	num_warmup = 500
	step_size = 0.001
	target_accept_prob = 0.9

	
	return broad_filter, grism_filter, wavelength, redshift, line, y_factor, flux_threshold, factor, \
		wave_factor, x0, y0, model_name, flux_bounds, flux_type, PA_sigma, i_bounds, Va_bounds, \
		r_t_bounds, sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob

def preprocess_mock_data(mock_params):

	obs_map = mock_params['grism_spectrum_noise']
	obs_error = mock_params['grism_error']*2
	direct = mock_params['convolved_noise_image']
	direct_error = mock_params['image_error']
	broad_band = mock_params['convolved_noise_image']
	xcenter_detector = 1024
	ycenter_detector = 1024
	icenter = 31//2
	jcenter = 31//2
	icenter_low = None
	jcenter_low = None
	wave_space = mock_params['wave_space']
	delta_wave = wave_space[1] - wave_space[0]
	index_min = mock_params['index_min']
	index_max = mock_params['index_max']
	wavelength = mock_params['wavelength']
	theta = 0
	grism_object = mock_params['grism_object']

	plt.imshow(obs_map, origin='lower')
	plt.title('obs_map')
	plt.colorbar()
	plt.show()

	plt.imshow(obs_map/obs_error, origin='lower')
	plt.title('obs map S/N')
	plt.colorbar()
	plt.show()

	plt.imshow(direct, origin='lower')
	plt.title('Direct')
	plt.colorbar()
	plt.show()

	plt.imshow(direct/direct_error, origin='lower')
	plt.title('Direct S/N')
	plt.colorbar()
	plt.show()

	plt.close()

	
	return  obs_map, obs_error, direct, direct_error, broad_band, xcenter_detector, ycenter_detector, icenter, jcenter, icenter_low, jcenter_low, \
				wave_space, delta_wave, index_min, index_max, wavelength, theta, grism_object

def run_full_preprocessing(output,master_cat, line, mock_params = None, priors = None):
	"""
		Main function that automatically post-processes the inference data and saves all of the relevant plots
	"""
	
	if mock_params == None:
		with open('fitting_results/' + output + '/' + 'config_real.yaml', 'r') as file:
			input = yaml.load(file, Loader=yaml.FullLoader)
		print('Read inputs successfully')
		#load of all the parameters from the configuration file
		ID, redshift, grism_filter, grism_spectrum_path, field, wavelength, redshift, line,flux_threshold, factor, wave_factor, model_name, PA_sigma, i_bounds, Va_bounds, r_t_bounds,\
		sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff = read_config_file(input, output + '/', master_cat,line)
		#preprocess the images and the grism spectrum
		if field == 'ALT':
			#generate an error that says not updated
			raise ValueError("The field ALT is not updated in the preprocessing function. Please use a different field.")
		else:
			module,obs_map, obs_error,wave_space, delta_wave, index_min, index_max, wavelength = preprocess_data(grism_spectrum_path, wavelength, delta_wave_cutoff, field)
		grism_object = None
	else:
		broad_filter, grism_filter, wavelength, redshift, line, y_factor, flux_threshold, factor, \
		wave_factor, x0, y0, model_name, flux_bounds, flux_type, PA_sigma, i_bounds, Va_bounds, \
		r_t_bounds, sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob = define_mock_params()
		obs_map, obs_error, direct, direct_error, broad_band, xcenter_detector, ycenter_detector, icenter, jcenter, icenter_low, jcenter_low, \
				wave_space, delta_wave, index_min, index_max, wavelength, theta, grism_object = preprocess_mock_data(mock_params)


	PSF = utils.load_psf(filter = grism_filter, y_factor = 1) #y_factor = 1 because putting it at grism resolution and then will be oversampled accordingly in the grism module

	#run pysersic fit to get morphological parameters
	if mock_params == None:
		path_output = 'fitting_results/' + output
	else: 
		test = mock_params['test']
		j = mock_params['j']
		path_output = 'testing/' + str(test) + '/'
		ID = j
	
	
	if grism_object == None:

		half_step = (delta_wave / wave_factor)*(wave_factor//2)
		wave_space_model = np.arange(wave_space[0]- half_step, wave_space[-1] + delta_wave + half_step, delta_wave / wave_factor)

		im_shape = obs_map.shape[0]*factor
		im_scale = 0.0629/factor #in arcsec/pixel
	
		#the input index_max should be the index of the last element of the array +1 (since that is how array cropping works)
		#setting by default the dispersion center at the center of the image in its original resolution 
		icenter = jcenter = obs_map.shape[0]//2
		grism_object = grism_dev.Grism(im_shape, im_scale, icenter = icenter, jcenter = jcenter, wavelength = wavelength, wave_space = wave_space_model, index_min = (index_min)*wave_factor, index_max = (index_max+1)*wave_factor, 
					   grism_filter = grism_filter, grism_module = module, grism_pupil = 'R', PSF = PSF)



	x0_vel = jcenter
	y0_vel = icenter


	# take x0 and y0 from the pre-processing unless specified otherwise in the config file
	x0 = None
	y0 = None
	if x0 == None:
		x0 = jcenter
	if y0 == None:
		y0 = icenter


	
	# cute the S/N to max 20: #try without this
	SN_max = 20
	obs_error = obs_error #§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§11§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§jnp.where(jnp.abs(obs_map/obs_error) > SN_max, obs_map/SN_max, obs_error)

	if model_name == 'Disk':
		kin_model = models.DiskModel()
	else:
		raise ValueError(f"Model {model_name} is not supported. Supported models are: Disk.")
	


	# plt.show()
	inclination = None
	r_eff = None
	r_full_grism = None
	PA_grism = None
	PA_morph = None
	kin_model.set_bounds((obs_map.shape[0], obs_map.shape[0]), factor, wave_factor, PA_sigma, Va_bounds, r_t_bounds, sigma0_bounds, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph, inclination, r_eff, r_full_grism)

	return redshift, wavelength, wave_space, obs_map, obs_error, model_name, kin_model, grism_object,  num_samples, num_warmup, step_size, target_accept_prob, delta_wave, factor


########## unused functions bins below - typically relevant for the non-parametric fitting but useless in the parametric!! ###########

# def make_EL_map(med_band, broad_band, med_band_err,broad_band_err,z, line):
	#function based on the module make_EL_map.py to create a good approximation of the EL map from medium and broad band images

	#define the redshift ranges

	# z_410M = [4.92, 5.61] #Halpha falls in F410M
	# z_444W = [5.61, 6.59] #Hbeta falls in F444W but not 410M

	# z_335M = [5.37, 6.15] #OIII 5007 falls in F335M
	# z_356W = [6.15, 6.77] #OIII 4959 falls in F356W but not 335M

	# #Hbeta limits for reference, but I don't think I can do much about Hbeta contamination
	# z_335M_Hbeta = [5.537-6.276] #Hbeta falls in F335M
	# z_356W_Hbeta = [5.449 - 7.1897] #Hbeta falls in F356W

	# z_356W_halpha = [4.37,4.92] #Halpha falls in F356W but not 335M
	# z_335M_halpha = [3.90,4.37] #Halpha falls in F335M

	# z_444W_PAalpha = [1.2945, 1.6571] #PAalpha falls in F444W but not 410M
	# z_410M_PAalpha = [1.0699, 1.2944] #PAalpha falls in F410M

	# #filter bandwidths in um
	# width_F410M = 0.436
	# width_F444W = 1.024
	# width_F335M = 0.347
	# width_F356W = 0.787

	# #pivot wavelengths of the filters
	# l410M = 4.092
	# l444W = 4.421
	# l335M = 3.365
	# l356W = 3.563

	# box_size = med_band.shape[0]

	# EL_map = np.zeros((box_size, box_size))

	# if line == 'H_alpha' or line == 'H_beta': #use Halpha map as prior for Hbeta 
	# 	if 4.916158536585365 < z < 5.557926829268291:
	# 		print('Halpha is in F410M and F444W')
	# 		# Halpha_map = image_cutout_F410M - (image_cutout_F444W-image_cutout_F410M)/(width_F444W-width_F410M)*(width_F410M)
	# 		broad_band = broad_band*c_m/l444W**2
	# 		med_band = med_band*c_m/l410M**2
	# 		broad_band_err = broad_band_err*c_m/l444W**2
	# 		med_band_err = med_band_err*c_m/l410M**2
	# 		EL_map = (med_band*width_F410M*width_F444W - broad_band*width_F444W*width_F410M)/(width_F444W-width_F410M)
	# 		width_sum = width_F410M + width_F444W
	# 		EL_map_err = np.sqrt(med_band_err**2*width_F410M/width_sum + broad_band_err**2*width_F444W/width_sum)

	# 		#see if a source is detected in the map:
	# 		segment_map, cat = utils.find_central_object(EL_map[69:131, 69:131], 4)
	# 		if segment_map == None:
	# 			print('No sources detected in EL map, using the med band map')
	# 			EL_map = med_band
	# 			EL_map_err = med_band_err
	# 		EL_map = med_band
	# 		EL_map_err = med_band_err
	# 		# Halpha_map = image_cutout_F410M - image_cutout_F444W
	# 	#if the galaxy is in z_444W, then estimate Halpha map as F444W-F410M
	# 	elif 5.5579268292682915 < z < 6.594512195121951:
	# 		print('Halpha is only in F444W')
	# 		broad_band = broad_band*c_m/l444W**2
	# 		med_band = med_band*c_m/l410M**2
	# 		broad_band_err = broad_band_err*c_m/l444W**2
	# 		med_band_err = med_band_err*c_m/l410M**2
	# 		EL_map = broad_band*width_F444W- med_band*width_F444W
	# 		width_sum = width_F410M + width_F444W
	# 		EL_map_err = np.sqrt(med_band_err**2*width_F410M/width_sum + broad_band_err**2*width_F444W/width_sum)
	# 		#see if a source is detected in the map:
	# 		segment_map, cat = utils.find_central_object(EL_map[69:131, 69:131], 4)
	# 		if segment_map == None:
	# 			print('No sources detected in EL map, using the broad band map')
	# 			EL_map = broad_band
	# 			EL_map_err = broad_band_err
	# 		EL_map = broad_band
	# 		EL_map_err = broad_band_err

	# 	elif 3.15 < z < 4:
	# 		print('Halpha is only in F356W') 
	# 		broad_band = broad_band*c_m/l356W**2
	# 		med_band = med_band*c_m/l335M**2
	# 		broad_band_err = broad_band_err*c_m/l356W**2
	# 		med_band_err = med_band_err*c_m/l335M**2
	# 		EL_map = broad_band*width_F356W - med_band*width_F356W 
	# 		width_sum = width_F335M + width_F356W
	# 		EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
	# 		EL_map = broad_band
	# 		EL_map_err = broad_band_err
	# 	elif z_335M_halpha[0] < z < z_335M_halpha[1]:
	# 		print('Halpha is in F335M and F356W')
	# 		broad_band = broad_band*c_m/l356W**2
	# 		med_band = med_band*c_m/l335M**2
	# 		broad_band_err = broad_band_err*c_m/l356W**2
	# 		med_band_err = med_band_err*c_m/l335M**2
	# 		EL_map = (med_band*width_F335M*width_F356W - broad_band*width_F356W*width_F335M)/(width_F356W-width_F335M)
	# 		width_sum = width_F335M + width_F356W
	# 		EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
	# 		EL_map = med_band
	# 		EL_map_err = med_band_err
	# 	else:
	# 		print('Halpha is not in the observed range, returning zeros')
	
	# elif line == 'OIII_5008':
	# 	if 5.343849840255591 < z < 6.062699680511182 :
	# 		print('OIII_5008 is in F335M and F356W')

	# 		broad_band = broad_band*c_m/l356W**2
	# 		med_band = med_band*c_m/l335M**2
	# 		broad_band_err = broad_band_err*c_m/l356W**2
	# 		med_band_err = med_band_err*c_m/l335M**2
	# 		EL_map = (med_band*width_F335M*width_F356W - broad_band*width_F356W*width_F335M)/(width_F356W-width_F335M)
	# 		width_sum = width_F335M + width_F356W
	# 		EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
	# 		EL_map = med_band
	# 		EL_map_err = med_band_err
	# 	#if the galaxy is in z_356W, then estimate OIII map as F356W-F335M
	# 	elif (5.259984025559104 < z < 5.3438409840255591) or (6.062699680511182< z < 6.949281150159743) or z == 4.480:
	# 		print('OIII_5008 is only in F356W ')

	# 		broad_band = broad_band*c_m/l356W**2
	# 		med_band = med_band*c_m/l335M**2
	# 		broad_band_err = broad_band_err*c_m/l356W**2
	# 		med_band_err = med_band_err*c_m/l335M**2
	# 		EL_map = broad_band*width_F356W - med_band*width_F356W 
	# 		width_sum = width_F335M + width_F356W
	# 		EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
	# 		EL_map = broad_band
	# 		EL_map_err = broad_band_err
	# 	else:
	# 		print('OIII_508 is not in the observed range, returning zeros')
		# EL_map = med_band
		# EL_map_err = med_band_err
	
	# elif line == 'PAalpha':
	# 	if z_410M_PAalpha[0] < z < z_410M_PAalpha[1]:
	# 		print('PAalpha is in F410M and F444W')
	# 		EL_map = (med_band*width_F410M*width_F444W - broad_band*width_F444W*width_F410M)/(width_F444W-width_F410M)
	# 	#if the galaxy is in z_444W, then estimate PAalpha map as F444W-F410M
	# 	elif z_444W_PAalpha[0] < z < z_444W_PAalpha[1]:
	# 		print('PAalpha is only in F444W')
	# 		EL_map = broad_band*width_F444W- med_band*width_F444W

	
	# return EL_map, EL_map_err

# def rotate_wcs(med_band_fits, EL_map, EL_error, broad_map, field, cutout_size):
	# '''
	# 	rotate the WCS of the image to match the PA of the grism image
	# 	the med_band_fits is for the WCS
	# '''
	# med_band = med_band_fits[1].data
	# wcs_med_band = wcs.WCS(med_band_fits[1].header)

	# if field == 'GOODS-S' or field =='GOODS-S-FRESCO':
	# 		print('FRESCO PA is the same in GOODS-S, no correction needed')
	# 		med_band_flip = med_band
	# 		EL_map_flip = EL_map
	# 		EL_err_flip = EL_error
	# 		broad_map_flip = broad_map
	# 		rotated_wcs = wcs_med_band
	# 		theta = 0.0
	# else:
	# 	if field == 'GOODS-N':
	# 		print('Correcting image for GOODS-N FRESCO PA')
	# 		theta = 230.5098
	# 	elif field == 'GOODS-N-CONGRESS':
	# 		print('Correcting image for GOODS-N Congress PA')
	# 		theta = 228.22379
	# 	# print(wcs.WCS(wcs_med_band.to_header()).wcs.pc)
	# 	rotated_wcs = wcs.WCS(med_band_fits[1].header)
	# 	theta_rad = np.deg2rad(theta)
	# 	sinq = np.sin(theta_rad)
	# 	cosq = np.cos(theta_rad)
	# 	mrot = np.array([[cosq, -sinq],
	# 						[sinq, cosq]])
	# 	if rotated_wcs.wcs.has_cd():    # CD matrix
	# 		newcd = np.dot(mrot, rotated_wcs.wcs.cd)
	# 		rotated_wcs.wcs.cd = newcd
	# 		# rotated_wcs.wcs.set()
	# 	elif rotated_wcs.wcs.has_pc():      # PC matrix + CDELT
	# 		newpc = np.dot(rotated_wcs.wcs.get_pc(),mrot)
	# 		rotated_wcs.wcs.pc = newpc
	# 			# rotated_wcs.wcs.set()
	# 		rotated_wcs.wcs.crpix = [cutout_size//2,cutout_size//2]
	# 		rotated_wcs.wcs.crval = wcs_med_band.all_pix2world(cutout_size//2,cutout_size//2,0)
	# 		rotated_wcs.wcs.set()
	# 		print('Setting new central pixel and value: ', rotated_wcs.wcs.crpix, rotated_wcs.wcs.crval)

	# 		# print('wcs med band : ', wcs_med_band)
	# 		# print('rotated wcs : ', rotated_wcs)
	# 		#reproject both images in the rotated WCS frame
	# 		# med_band_flip, footprint = reproject_adaptive((med_band, wcs_med_band), rotated_wcs, shape_out = med_band.shape)
	# 		# broad_band_flip, footprint = reproject_adaptive((broad_band, wcs_med_band), rotated_wcs, shape_out = broad_band.shape)
	# 		EL_map_flip, footprint = reproject_adaptive((EL_map, wcs_med_band), rotated_wcs, shape_out = EL_map.shape)
	# 		EL_err_flip, footprint = reproject_adaptive((EL_error, wcs_med_band), rotated_wcs, shape_out = EL_error.shape)
	# 		#flip this too so the PA can be computed
	# 		med_band_flip, footprint = reproject_adaptive((med_band, wcs_med_band), rotated_wcs, shape_out = med_band.shape)
	# 		broad_map_flip, footprint = reproject_adaptive((broad_map, wcs_med_band), rotated_wcs, shape_out = broad_map.shape)

	# return rotated_wcs, med_band_flip, EL_map_flip, EL_err_flip, broad_map_flip, theta



# def rotate_wcs_ALT(wcs_orig, rot_wcs, EL_map, EL_error, broad_map,cutout_size, angle):
# 	'''
# 		rotate the WCS of the image to match the PA of the grism image
# 		the med_band_fits is for the WCS
# 	'''

# 	wcs_med_band = wcs_orig
# 	theta = -angle
# 	# print(wcs.WCS(wcs_med_band.to_header()).wcs.pc)
# 	rotated_wcs = rot_wcs
# 	theta_rad = np.deg2rad(theta)
# 	sinq = np.sin(theta_rad)
# 	cosq = np.cos(theta_rad)
# 	mrot = np.array([[cosq, -sinq],
# 							[sinq, cosq]])
# 	if rotated_wcs.wcs.has_cd():    # CD matrix
# 		# newcd = np.dot(mrot, rotated_wcs.wcs.cd)
# 		# rotated_wcs.wcs.cd = newcd
# 		newpc = np.dot(wcs.WCS(rotated_wcs.to_header()).wcs.get_pc(), mrot)
# 		rotated_wcs.wcs.pc = newpc
# 		# rotated_wcs.wcs.set()
# 	elif rotated_wcs.wcs.has_pc():      # PC matrix + CDELT
# 		newpc = np.dot(rotated_wcs.wcs.get_pc(),mrot)
# 		rotated_wcs.wcs.pc = newpc
# 			# rotated_wcs.wcs.set()
# 	rotated_wcs.wcs.crpix = [cutout_size//2,cutout_size//2]
# 	rotated_wcs.wcs.crval = wcs_med_band.all_pix2world(cutout_size//2,cutout_size//2,0)
# 	rotated_wcs.wcs.set()
# 	print('Setting new central pixel and value: ', rotated_wcs.wcs.crpix, rotated_wcs.wcs.crval)
# 		# print('wcs med band : ', wcs_med_band)
# 		# print('rotated wcs : ', rotated_wcs)
# 		#reproject both images in the rotated WCS frame
# 		# med_band_flip, footprint = reproject_adaptive((med_band, wcs_med_band), rotated_wcs, shape_out = med_band.shape)
# 		# broad_band_flip, footprint = reproject_adaptive((broad_band, wcs_med_band), rotated_wcs, shape_out = broad_band.shape)
# 	EL_map_flip, footprint = reproject_adaptive((EL_map, wcs_med_band), rotated_wcs, shape_out = EL_map.shape)
# 	EL_err_flip, footprint = reproject_adaptive((EL_error, wcs_med_band), rotated_wcs, shape_out = EL_error.shape)
# 	#flip this too so the PA can be computed
# 	broad_map_flip, footprint = reproject_adaptive((broad_map, wcs_med_band), rotated_wcs, shape_out = broad_map.shape)

# 	return rotated_wcs, EL_map_flip, EL_err_flip, broad_map_flip

# def preprocess_data_ALT(obj_id, output, line, delta_wave_cutoff = 0.02, field = 'ALT', RA_vel = -1, DEC_vel = -1):

# 	#no need to rotate this data, and not using EL prior bc no med band imaging
	 
# 	#load data from fits files
# 	image_hdu = fits.open(f"fitting_results/" + output + "j001420m3023_%s.full.fits"%(str(obj_id).zfill(5)))

# 	RA = RA_vel
# 	DEC = DEC_vel

# 	print('RA, DEC: ', RA, DEC)
# 	broad_band = image_hdu['DSCI','F356W-CLEAR'].data
# 	broad_band_mask = image_hdu['DSCI','F356W-CLEAR'].data

# 	broad_band_err = jnp.sqrt(1/image_hdu['DWHT','F356W-CLEAR'].data)
# 	broad_band_wcs = wcs.WCS(image_hdu['DSCI','F356W-CLEAR'].header)

# 	#prior from broad band image
# 	EL_map = image_hdu['DSCI','F356W-CLEAR'].data
# 	#for now will make the error map artificially - then will ask ALT chat
# 	EL_map_error = 0.25*EL_map
# 	# EL_map_error = np.power(image_hdu['DWHT','F356W-CLEAR'].data, - 0.5)

# 	plt.imshow(EL_map_error, origin = 'lower')
# 	plt.show()

# 	EL_map_error = np.where(np.isnan(EL_map)| np.isnan(EL_map_error) | np.isinf(EL_map_error), 1e10, EL_map_error)		
# 	grism_spectrum_fits = fits.open(f"fitting_results/" + output + "j001420m3023_%s.stack.fits"%(str(obj_id).zfill(5)))


# 	#load grism PA
# 	theta = grism_spectrum_fits['SCI'].header['PA']


# 	cutout_size = EL_map.shape[0]


# 	#rotate images to match grism PA
# 	rotated_wcs, EL_map_flip, EL_err_flip, broad_map_flip = rotate_wcs_ALT(wcs.WCS(image_hdu['DSCI','F356W-CLEAR'].header),wcs.WCS(image_hdu['DSCI','F356W-CLEAR'].header),EL_map, EL_map_error, broad_band_mask, cutout_size, theta)

# 	wavelength = image_hdu['LINE',line].header['WAVELEN']*1e-4

# 	cutout_size = broad_band.shape[0]

# 	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
# 	if RA_vel == None:
# 		velocity_centroid = galaxy_position
# 	else:
# 		velocity_centroid = SkyCoord(ra=RA_vel * u.deg, dec=DEC_vel * u.deg)

# 	#compute the med band center from the WCS coords:
# 	icenter_high, jcenter_high = rotated_wcs.world_to_array_index_values(galaxy_position.ra.deg, galaxy_position.dec.deg)
# 	icenter_vel, jcenter_vel = rotated_wcs.world_to_array_index_values(velocity_centroid.ra.deg, velocity_centroid.dec.deg)
# 	if icenter_high < 0 or icenter_high > cutout_size or jcenter_high < 0 or jcenter_high > cutout_size:
# 		print('The galaxy is not in the cutout')
# 		return None
# 	else:
# 		print('Center of the galaxy in the high res image: ', icenter_high, jcenter_high)
# 		print('Center of the velocity map in the high res image: ', icenter_vel, jcenter_vel)
# 	# plt.imshow(EL_map_flip, origin='lower')
# 	# plt.title('Correctly rotated image')
# 	# plt.show()

# 	#cut the broad band image to 31x31 around the center
# 	ny,nx = EL_map_flip.shape
# 	cx = int(0.5*(nx-1))
# 	dx = 32
# 	high_res_prior = EL_map_flip[cx-dx:cx+dx,cx-dx:cx+dx]
# 	high_res_error = EL_err_flip[cx-dx:cx+dx,cx-dx:cx+dx]
# 	broad_band_cutout = jnp.array(broad_map_flip[cx-dx:cx+dx,cx-dx:cx+dx])
# 	# save the broadband image needed to compute mask - using F444W for all ELs to have save mask
# 	broad_band_mask = broad_map_flip[cx-dx:cx+dx,cx-dx:cx+dx]
# 	# compute mask from med band image
# 	# threshold = threshold_otsu(med_band_cutout)/3
# 	# plt.imshow(jnp.where(med_band_cutout>threshold,med_band_cutout, 0.0) , origin  = 'lower')
# 	# plt.show()

# 	plt.imshow(high_res_prior, origin='lower')
# 	plt.title('Flipped and correct scale image 62x62 high res cutout')
# 	plt.colorbar()
# 	plt.show()

# 	plt.imshow(high_res_error, origin='lower')
# 	plt.title('Flipped and correct scale image 62x62 high res error cutout')
# 	plt.colorbar()
# 	plt.show()

# 	icenter_vel, jcenter_vel = icenter_vel - cx + dx , jcenter_vel - cx + dx
# 	icenter_high, jcenter_high = icenter_high - cx + dx , jcenter_high - cx + dx#this is by defintion of how the cutout was made
# 	#recompute the velocity centroid in the new 62x62 cutout arounf icenter_high, jcenter_high

# 	x0_vel = jcenter_vel
# 	y0_vel = icenter_vel

# 	xcenter_detector = 1024
# 	ycenter_detector = 1024

# 	#grism stuff
# 	h_i = grism_spectrum_fits[('SCI','F356W')].header
# 	wave_space = (np.arange(h_i['NAXIS1']+1)-h_i['CRPIX1']+1)*h_i['CD1_1'] + h_i['CRVAL1']


# 	#choose which wavelengths will be the cutoff of the EL map and save those
# 	wave_min = wavelength - delta_wave_cutoff 
# 	wave_max = wavelength + delta_wave_cutoff 

# 	d_wave = h_i['CD1_1']
# 	wave_first = wave_space[0]

# 	# print(wave_min, wave_max)

# 	index_min = round((wave_min - wave_first)/d_wave) #+10
# 	index_max = round((wave_max - wave_first)/d_wave) #-10
# 	# print(index_min, index_max)
# 	index_wave = round((wavelength - wave_first)/d_wave)


# 	#subtract continuum and crop image by 200 on each size of EL
# 	grism_spectrum_data = jnp.array(grism_spectrum_fits[('SCI','F356W')].data) #contiuum_subtraction(jnp.array(grism_spectrum_fits[('SCI','F356W')].data), index_wave - 150, index_wave + 150)

# 	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
# 	# obs_map = grism_spectrum_data[:,index_min-(index_wave -150) +1:index_max-(index_wave -150)+1+1]
# 	obs_map = grism_spectrum_data[:,index_min:index_max+1]

# 	obs_error = np.power(grism_spectrum_fits[('WHT','F356W')].data[:,index_min+1:index_max+1+1], - 0.5)
# 	#mask bad pixels in obs/error map
# 	obs_error = jnp.where(jnp.isnan(obs_map)| jnp.isnan(obs_error) | jnp.isinf(obs_error), 1e10, obs_error)

# 	plt.imshow(obs_map, origin='lower')
# 	plt.title('obs_map')
# 	plt.show()
	
# 	direct = high_res_prior
# 	direct_error = high_res_error
# 	icenter_prior = icenter_high
# 	jcenter_prior = jcenter_high

# 	icenter_low = None
# 	jcenter_low = None

# 	return jnp.array(obs_map), jnp.array(obs_error), jnp.array(direct), jnp.array(direct_error), jnp.array(broad_band_mask), xcenter_detector, ycenter_detector, icenter_prior, jcenter_prior,icenter_low, jcenter_low, wave_space, d_wave, index_min, index_max, x0_vel, y0_vel, wavelength, theta


# def preprocess_data_ALT_stacked(obj_id, output, line, delta_wave_cutoff = 0.02, field = 'ALT'):

	#no need to rotate this data, and not using EL prior bc no med band imaging
	 
	#load data from fits files
	# image_hdu = fits.open("fitting_results/" + output + "j001420m3023_%s.full.fits"%(str(obj_id).zfill(5)))

	# stacked_hdu = fits.open("fitting_results/" + output + "stacked_2D_ALT_%s.fits"%(str(obj_id).zfill(5)))

	# broad_hdu = fits.open("fitting_results/" + output + "%s_cutout.fits"%(str(obj_id).zfill(5)))

	# err_hdu = fits.open("fitting_results/" + output + "%s_wht_cutout.fits"%(str(obj_id).zfill(5)))

	# broad_band = broad_hdu[0].data
	# broad_band_mask = broad_hdu[0].data

	# broad_band_err = jnp.sqrt(1/err_hdu[0].data)
	# broad_band_wcs = wcs.WCS(broad_hdu[0].header)

	# #prior from broad band image
	# EL_map = broad_hdu[0].data
	# #for now will make the error map artificially - then will ask ALT chat
	# EL_map_error = jnp.sqrt(1/err_hdu[0].data)
	# # EL_map_error = np.power(image_hdu['DWHT','F356W-CLEAR'].data, - 0.5)

	# plt.imshow(EL_map_error, origin = 'lower')
	# plt.show()

	# EL_map_error = np.where(np.isnan(EL_map)| np.isnan(EL_map_error) | np.isinf(EL_map_error), 1e10, EL_map_error)	

	# # grism_spectrum_fits = stacked_hdu['EMLINE']


	# #load grism PA
	# theta = - stacked_hdu['STAMP'].header['PA_V3'] #- ROLL 2 angle 
	# print('PA_V3: ', -theta)


	# cutout_size = EL_map.shape[0]


	# #rotate images to match grism PA
	# rotated_wcs, EL_map_flip, EL_err_flip, broad_map_flip = rotate_wcs_ALT(wcs.WCS(broad_hdu[0].header),wcs.WCS(broad_hdu[0].header),EL_map, EL_map_error, broad_band_mask, cutout_size, theta)

	# wavelength = image_hdu['LINE',line].header['WAVELEN']*1e-4

	# cutout_size = broad_band.shape[0]

	# #project the image to the grism scale 0.0629
	# old_wcs = rotated_wcs.deepcopy()
	# rotated_wcs.wcs.pc = rotated_wcs.wcs.pc*(0.0629/0.04)
	# rotated_wcs.wcs.crpix = [cutout_size//2,cutout_size//2]
	# rotated_wcs.wcs.crval = old_wcs.all_pix2world(cutout_size//2,cutout_size//2,0)
	# rotated_wcs.wcs.set()

	# EL_map_flip, footprint = reproject_adaptive((EL_map_flip, old_wcs), rotated_wcs, shape_out = EL_map_flip.shape)
	# EL_err_flip, footprint = reproject_adaptive((EL_err_flip, old_wcs), rotated_wcs, shape_out = EL_err_flip.shape)
	# broad_map_flip, footprint = reproject_adaptive((broad_map_flip, old_wcs), rotated_wcs, shape_out = broad_map_flip.shape)
	# #cut the broad band image to 51x51 around the center, current size 100x100
	# ny,nx = EL_map_flip.shape
	# cx = int(0.5*(nx-1))
	# dx = 25
	# high_res_prior = jnp.array(EL_map_flip[cx-dx:cx+dx+1,cx-dx:cx+dx+1])
	# high_res_error = jnp.array(EL_err_flip[cx-dx:cx+dx+1,cx-dx:cx+dx+1])
	# broad_band_cutout = jnp.array(broad_map_flip[cx-dx:cx+dx+1,cx-dx:cx+dx+1])
	# # save the broadband image needed to compute mask - using F444W for all ELs to have save mask
	# broad_band_mask = jnp.array(broad_map_flip[cx-dx:cx+dx+1,cx-dx:cx+dx+1])

	# icenter_high, jcenter_high = dx, dx #this is by defintion of how the cutout was made
	# # compute mask from med band image
	# # threshold = threshold_otsu(med_band_cutout)/3
	# # plt.imshow(jnp.where(med_band_cutout>threshold,med_band_cutout, 0.0) , origin  = 'lower')
	# # plt.show()

	# plt.imshow(broad_band_mask, origin='lower')
	# plt.title('Broad band mask')
	# plt.colorbar()
	# plt.show()
	# xcenter_detector = 1024
	# ycenter_detector = 1024

	# #grism stuff
	# h_i =stacked_hdu['EMLINE'].header
	# wave_space = ((np.arange(h_i['NAXIS1']+1)-h_i['CRPIX1']+1)*h_i['CDELT1']+ h_i['CRVAL1'])*1e-4


	# #choose which wavelengths will be the cutoff of the EL map and save those
	# wave_min = wavelength - delta_wave_cutoff 
	# wave_max = wavelength + delta_wave_cutoff 

	# d_wave = h_i['CDELT1']*1e-4
	# wave_first = wave_space[0]
	# # print(wave_min, wave_max)

	# index_min = round((wave_min - wave_first)/d_wave) #+10
	# index_max = round((wave_max - wave_first)/d_wave) #-10
	# # print(index_min, index_max)
	# index_wave = round((wavelength - wave_first)/d_wave)

	# print(index_max, index_min)
	# #subtract continuum and crop image by 200 on each size of EL
	# grism_spectrum_data = jnp.array(stacked_hdu['EMLINE'].data) #contiuum_subtraction(jnp.array(grism_spectrum_fits[('SCI','F356W')].data), index_wave - 150, index_wave + 150)

	# #cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	# # obs_map = grism_spectrum_data[:,index_min-(index_wave -150) +1:index_max-(index_wave -150)+1+1]
	# obs_map = grism_spectrum_data[:,index_min+1:index_max+1+1]

	# obs_error = jnp.array(stacked_hdu['ERR'].data[:,index_min+1:index_max+1+1])
	# #mask bad pixels in obs/error map
	# obs_error = jnp.where(jnp.isnan(obs_map)| jnp.isnan(obs_error) | jnp.isinf(obs_error), 1e10, obs_error)

	# if np.all(np.isnan(stacked_hdu['EMLINEA'].data)) == True or np.all(stacked_hdu['EMLINEA'].data) == False:
	# #np.all returns False if there any elements = 0 or = False in the array
	# #here we test is either every element of Mod A is none of if it is all zeros => in this case the map is only
	# #in mod B and hence flipped
	# 	obs_map = jnp.flip(obs_map, axis = 1)
	# 	obs_error = jnp.flip(obs_error, axis = 1)
		
	# direct = high_res_prior
	# direct_error = high_res_error
	# icenter_prior = icenter_high
	# jcenter_prior = jcenter_high

	# #introduce error floor of max S/N in map + flux:
	# # print(jnp.max(obs_map/obs_error))
	# SN_max = 20 #jnp.minimum(20, jnp.max(obs_map/obs_error))
	# obs_error = jnp.where(jnp.abs(obs_map/obs_error) > SN_max, obs_map/SN_max, obs_error)
	# direct_error = jnp.where(jnp.abs(high_res_prior/high_res_error) > SN_max, high_res_prior/SN_max, high_res_error)
	# plt.imshow(obs_map, origin='lower')
	# plt.title('obs_map')
	# plt.colorbar()
	# plt.show()

	# plt.imshow(obs_map/obs_error, origin='lower')
	# plt.title('obs map S/N')
	# plt.colorbar()
	# plt.show()

	# plt.imshow(direct, origin='lower')
	# plt.title('Direct')
	# plt.colorbar()
	# plt.show()

	# plt.imshow(direct/direct_error, origin='lower')
	# plt.title('Direct S/N')
	# plt.colorbar()
	# plt.show()

	
	# icenter_low = None
	# jcenter_low = None

	# return jnp.array(obs_map), jnp.array(obs_error), jnp.array(direct), jnp.array(direct_error), jnp.array(broad_band_mask), xcenter_detector, ycenter_detector, icenter_prior, jcenter_prior,icenter_low, jcenter_low, wave_space, d_wave, index_min, index_max, wavelength, theta