"""
Put all of the necessary pre-processing functions here so that fit_numpyro is cleaner
Eventually should also add here scripts to automatically create folders for sources, with all of the right cutouts etc
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""

#imports
import utils
import grism
import models

import numpy as np
import math
import matplotlib.pyplot as plt

import jax.numpy as jnp

import yaml

from scipy.ndimage import median_filter, sobel
from scipy import ndimage

#for the masking
from skimage.morphology import  dilation, disk
from skimage.filters import threshold_otsu

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import wcs
from astropy.table import Table

from reproject import reproject_adaptive
from scipy.constants import c

from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground

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
	
	#no ../ because the open() function reads from terminal directory (not module directory)
	if field == 'ALT':
		print('Setting paths to none because everything will be derived from fits files')
		broad_filter = 'F356W'
		med_filter = None
		grism_filter = 'F356W'
		broad_band_mask_path = None
		med_band_path = None
		broad_band_path = None
		grism_spectrum_path = None
	else:
		if line == 'H_alpha':
			broad_filter = 'F444W'
			med_filter = 'F410M'
			grism_filter = broad_filter
		elif line == 'OIII_5007' or 'OIII_4959':
			broad_filter = 'F356W'
			med_filter = 'F335M'
			grism_filter = broad_filter
		elif line == 'H_beta':
			#in this case we use Halpha as prior
			broad_filter = 'F444W'
			med_filter = 'F410M'
			#but still use the correct grim data!
			grism_filter = 'F356W'
		else:
			print('Line not recognized')
			return None
		med_band_path = 'fitting_results/' + output + str(ID) + '_' + med_filter + '.fits'
		broad_band_path = 'fitting_results/' + output + str(ID) + '_' + broad_filter + '.fits'
		if field == 'GOODS-N' or field == 'GOODS-N-CONGRESS':
			grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_GDN_' + grism_filter + '_ID' + str(ID) + '_comb.fits'
		elif field == 'GOODS-S-FRESCO':
			grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_FRESCO_' + grism_filter + '_ID' + str(ID) + '_comb.fits'
		
		broad_band_mask_path = 'fitting_results/' + output + str(ID) + '_' + 'F444W' + '.fits'

	wavelength = master_cat[master_cat['ID'] == ID][str(line) + '_lambda'][0]

	redshift = master_cat[master_cat['ID'] == ID]['zspec'][0]

	PA_morph = master_cat[master_cat['ID'] == ID]['theta_q50'][0]
	if isinstance(PA_morph, float) == True:
		if PA_morph<0:
			PA_morph += jnp.pi
		PA_morph = 180 - PA_morph* (180/jnp.pi) #convert to degrees and the right range

	e = master_cat[master_cat['ID'] == ID]['ellip_q50'][0]
	if isinstance(e, float) == True:
		inclination = math.acos(1 - e)* (180/jnp.pi) #convert to degrees because acos returns radians
	else:
		inclination = None

	y_factor = params['Params']['y_factor']

	if y_factor == 1:
		print('Fitting the low res image')
		res = 'low'
	elif y_factor == 2:
		print('Fitting the high res image')
		res = 'high'
		
	flux_threshold = inference['Inference']['flux_threshold']
	
	delta_wave_cutoff = params['Params']['delta_wave_cutoff']
	factor = params['Params']['factor']
	wave_factor = params['Params']['wave_factor']
	
	x0 = params['Params']['x0']
	y0 = params['Params']['y0']
	

	# x0_vel = np.array(params['Params']['x0_vel'])
	# y0_vel = np.array(params['Params']['y0_vel'])

	#project from 40x40 to 200x200 => they are both in 0.03'' pixel size
	x0_vel = master_cat[master_cat['ID'] == ID]['xc_q50'][0] 
	y0_vel = master_cat[master_cat['ID'] == ID]['yc_q50'][0] 
	print('vel: ', x0_vel)

	if isinstance(x0_vel, float) == True:
		x0_vel += 80
		y0_vel += 80
		#get the RA, DEC of these coordinates
		wcs_f444w = wcs.WCS(fits.open(broad_band_mask_path)[1].header)
		RA_vel,DEC_vel = wcs_f444w.array_index_to_world_values(y0_vel,x0_vel)
		print('RA, DEC of the velocity map: ', RA_vel, DEC_vel)
	else:
		RA_vel = master_cat[master_cat['ID'] == ID]['RA'][0]
		DEC_vel = master_cat[master_cat['ID'] == ID]['DEC'][0]



	r_eff = master_cat[master_cat['ID'] == ID]['r_eff_q50'][0] #already in 0.03'' pixel size 

	model_name = inference['Inference']['model']
	
    #import all of the bounds needed for the priors
	flux_bounds = inference['Inference']['flux_bounds']
	flux_type = inference['Inference']['flux_type']
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
	return data, params, inference, ID, broad_filter, med_filter, grism_filter, med_band_path, broad_band_path, broad_band_mask_path, grism_spectrum_path, field, wavelength, redshift, line, y_factor, res, flux_threshold, factor, wave_factor, x0, y0, x0_vel, y0_vel, model_name, flux_bounds, flux_type, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff, PA_morph, inclination, r_eff, RA_vel, DEC_vel


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

	mask_image_seg, cat_image = utils.find_central_object(direct, 4)
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


def make_EL_map(med_band, broad_band, med_band_err,broad_band_err,z, line):
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

	z_444W_PAalpha = [1.2945, 1.6571] #PAalpha falls in F444W but not 410M
	z_410M_PAalpha = [1.0699, 1.2944] #PAalpha falls in F410M

	#filter bandwidths in um
	width_F410M = 0.436
	width_F444W = 1.024
	width_F335M = 0.347
	width_F356W = 0.787

	#pivot wavelengths of the filters
	l410M = 4.092
	l444W = 4.421
	l335M = 3.365
	l356W = 3.563

	box_size = med_band.shape[0]

	EL_map = np.zeros((box_size, box_size))

	if line == 'H_alpha' or line == 'H_beta': #use Halpha map as prior for Hbeta 
		if 4.916158536585365 < z < 5.557926829268291:
			print('Halpha is in F410M and F444W')
			# Halpha_map = image_cutout_F410M - (image_cutout_F444W-image_cutout_F410M)/(width_F444W-width_F410M)*(width_F410M)
			broad_band = broad_band*c_m/l444W**2
			med_band = med_band*c_m/l410M**2
			broad_band_err = broad_band_err*c_m/l444W**2
			med_band_err = med_band_err*c_m/l410M**2
			EL_map = (med_band*width_F410M*width_F444W - broad_band*width_F444W*width_F410M)/(width_F444W-width_F410M)
			width_sum = width_F410M + width_F444W
			EL_map_err = np.sqrt(med_band_err**2*width_F410M/width_sum + broad_band_err**2*width_F444W/width_sum)

			#see if a source is detected in the map:
			segment_map, cat = utils.find_central_object(EL_map[69:131, 69:131], 4)
			if segment_map == None:
				print('No sources detected in EL map, using the med band map')
				EL_map = med_band
				EL_map_err = med_band_err
			# Halpha_map = image_cutout_F410M - image_cutout_F444W
		#if the galaxy is in z_444W, then estimate Halpha map as F444W-F410M
		elif 5.5579268292682915 < z < 6.594512195121951:
			print('Halpha is only in F444W')
			broad_band = broad_band*c_m/l444W**2
			med_band = med_band*c_m/l410M**2
			broad_band_err = broad_band_err*c_m/l444W**2
			med_band_err = med_band_err*c_m/l410M**2
			EL_map = broad_band*width_F444W- med_band*width_F444W
			width_sum = width_F410M + width_F444W
			EL_map_err = np.sqrt(med_band_err**2*width_F410M/width_sum + broad_band_err**2*width_F444W/width_sum)
			#see if a source is detected in the map:
			segment_map, cat = utils.find_central_object(EL_map[69:131, 69:131], 4)
			if segment_map == None:
				print('No sources detected in EL map, using the broad band map')
				EL_map = broad_band
				EL_map_err = broad_band_err

		elif z_356W_halpha[0] < z < z_356W_halpha[1]:
			print('Halpha is only in F356W') 
			broad_band = broad_band*c_m/l356W**2
			med_band = med_band*c_m/l335M**2
			broad_band_err = broad_band_err*c_m/l356W**2
			med_band_err = med_band_err*c_m/l335M**2
			EL_map = broad_band*width_F356W - med_band*width_F356W 
			width_sum = width_F335M + width_F356W
			EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
		elif z_335M_halpha[0] < z < z_335M_halpha[1]:
			print('Halpha is in F335M and F356W')
			broad_band = broad_band*c_m/l356W**2
			med_band = med_band*c_m/l335M**2
			broad_band_err = broad_band_err*c_m/l356W**2
			med_band_err = med_band_err*c_m/l335M**2
			EL_map = (med_band*width_F335M*width_F356W - broad_band*width_F356W*width_F335M)/(width_F356W-width_F335M)
			width_sum = width_F335M + width_F356W
			EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)

		else:
			print('Halpha is not in the observed range, returning zeros')
	
	elif line == 'OIII_5008':
		if 5.343849840255591 < z < 6.062699680511182 :
			print('OIII_5008 is in F335M and F356W')

			broad_band = broad_band*c_m/l356W**2
			med_band = med_band*c_m/l335M**2
			broad_band_err = broad_band_err*c_m/l356W**2
			med_band_err = med_band_err*c_m/l335M**2
			EL_map = (med_band*width_F335M*width_F356W - broad_band*width_F356W*width_F335M)/(width_F356W-width_F335M)
			width_sum = width_F335M + width_F356W
			EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
			EL_map = med_band
			EL_map_err = med_band_err
		#if the galaxy is in z_356W, then estimate OIII map as F356W-F335M
		elif (5.259984025559104 < z < 5.3438409840255591) or (6.062699680511182< z < 6.949281150159743) or z == 4.480:
			print('OIII_5008 is only in F356W ')

			broad_band = broad_band*c_m/l356W**2
			med_band = med_band*c_m/l335M**2
			broad_band_err = broad_band_err*c_m/l356W**2
			med_band_err = med_band_err*c_m/l335M**2
			EL_map = broad_band*width_F356W - med_band*width_F356W 
			width_sum = width_F335M + width_F356W
			EL_map_err = np.sqrt(med_band_err**2*width_F335M/width_sum + broad_band_err**2*width_F356W/width_sum)
			EL_map = broad_band
			EL_map_err = broad_band_err
		else:
			print('OIII_508 is not in the observed range, returning zeros')

	
	# elif line == 'PAalpha':
	# 	if z_410M_PAalpha[0] < z < z_410M_PAalpha[1]:
	# 		print('PAalpha is in F410M and F444W')
	# 		EL_map = (med_band*width_F410M*width_F444W - broad_band*width_F444W*width_F410M)/(width_F444W-width_F410M)
	# 	#if the galaxy is in z_444W, then estimate PAalpha map as F444W-F410M
	# 	elif z_444W_PAalpha[0] < z < z_444W_PAalpha[1]:
	# 		print('PAalpha is only in F444W')
	# 		EL_map = broad_band*width_F444W- med_band*width_F444W

	
	return EL_map, EL_map_err

def rotate_wcs(med_band_fits, EL_map, EL_error, broad_map, field, cutout_size):
	'''
		rotate the WCS of the image to match the PA of the grism image
	'''
	med_band = med_band_fits[1].data
	wcs_med_band = wcs.WCS(med_band_fits[1].header)

	if field == 'GOODS-S' or field =='GOODS-S-FRESCO':
			print('FRESCO PA is the same in GOODS-S, no correction needed')
			med_band_flip = med_band
			EL_map_flip = EL_map
			EL_err_flip = EL_error
			broad_map_flip = broad_map
			rotated_wcs = wcs_med_band
			theta = None
	else:
		if field == 'GOODS-N':
			print('Correcting image for GOODS-N FRESCO PA')
			theta = 230.5098
		elif field == 'GOODS-N-CONGRESS':
			print('Correcting image for GOODS-N Congress PA')
			theta = 228.22379
		# print(wcs.WCS(wcs_med_band.to_header()).wcs.pc)
		rotated_wcs = wcs.WCS(med_band_fits[1].header)
		theta_rad = np.deg2rad(theta)
		sinq = np.sin(theta_rad)
		cosq = np.cos(theta_rad)
		mrot = np.array([[cosq, -sinq],
							[sinq, cosq]])
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
			EL_err_flip, footprint = reproject_adaptive((EL_error, wcs_med_band), rotated_wcs, shape_out = EL_error.shape)
			#flip this too so the PA can be computed
			med_band_flip, footprint = reproject_adaptive((med_band, wcs_med_band), rotated_wcs, shape_out = med_band.shape)
			broad_map_flip, footprint = reproject_adaptive((broad_map, wcs_med_band), rotated_wcs, shape_out = broad_map.shape)

	return rotated_wcs, med_band_flip, EL_map_flip, EL_err_flip, broad_map_flip, theta


def contiuum_subtraction(grism_spectrum_data, min, max):
	'''
		Subtract the continuum from the EL map
	'''
	grism_spectrum_data = grism_spectrum_data[:,min:max] #NOT cont sub

	grism_spectrum_data = jnp.where(jnp.isnan(grism_spectrum_data), 0.0, jnp.array(grism_spectrum_data))

	plt.imshow(grism_spectrum_data, origin='lower')
	plt.title('EL map before continuum subtraction')
	plt.colorbar()
	plt.show()

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

	mask_seg, cat = utils.find_central_object(grism_spectrum_data[:,140:160], 0.5)
	mask = mask_seg.data

	grism_spectrum_data_crop = grism_spectrum_data[:,140:160]
	grism_spectrum_data_crop= grism_spectrum_data_crop.at[jnp.where(mask == 1)].set(jnp.nan)
	grism_spectrum_data_masked = grism_spectrum_data
	grism_spectrum_data_masked= grism_spectrum_data_masked.at[:,140:160].set(grism_spectrum_data_crop)
	plt.imshow(mask, origin='lower')
	plt.title('mask')
	plt.colorbar()
	plt.show()
	L_box, L_mask = 10, 12
	mf_footprint = np.ones((1, L_box * 2 + 1))
	tmp_grism_img_median = ndimage.generic_filter(grism_spectrum_data_masked, np.nanmedian, footprint=mf_footprint, mode='reflect')
	grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map
	plt.imshow(grism_spectrum_data, origin='lower')
	plt.title('EL map after continuum subtraction')
	plt.colorbar()
	plt.show()


	return grism_spectrum_data

def preprocess_data(med_band_path, broad_band_path, broad_band_mask_path, grism_spectrum_path, redshift, line, wavelength, delta_wave_cutoff = 0.02, field = 'GOODS-S', fitting = None, RA_vel = -1, DEC_vel = -1):
	 
	#load data from fits files
	med_band_fits = fits.open(med_band_path)
	med_band = med_band_fits[1].data
	med_band_err = med_band_fits[2].data #should put SCI and ERR here so can be less hardcoded
	med_band_wcs = wcs.WCS(med_band_fits[1].header)

	broad_band_fits = fits.open(broad_band_path)
	broad_band = broad_band_fits[1].data
	broad_band_err = broad_band_fits[2].data
	broad_band_wcs = wcs.WCS(broad_band_fits[1].header)

	broad_band_mask = fits.open(broad_band_mask_path)[1].data

	grism_spectrum_fits = fits.open(grism_spectrum_path)

	cutout_size = med_band.shape[0]

	RA = grism_spectrum_fits[0].header['RA0']
	DEC = grism_spectrum_fits[0].header['DEC0']

	#compute the EL map prior first
	EL_map, EL_error = make_EL_map(med_band, broad_band, med_band_err,broad_band_err, redshift, line)

	plt.imshow(EL_map, origin='lower')
	plt.title('EL map prior')
	plt.show()

	#if needed, rotate the image WCS and the EL map to match the PA of the grism image
	rotated_wcs, med_band_flip, EL_map_flip, EL_err_flip, broad_band_mask_flip, theta = rotate_wcs(med_band_fits, EL_map, EL_error, broad_band_mask, field, cutout_size)
	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
	velocity_centroid = SkyCoord(ra=RA_vel * u.deg, dec=DEC_vel * u.deg)

	#compute the med band center from the WCS coords:
	icenter_high, jcenter_high = rotated_wcs.world_to_array_index_values(galaxy_position.ra.deg, galaxy_position.dec.deg)
	icenter_vel, jcenter_vel = rotated_wcs.world_to_array_index_values(velocity_centroid.ra.deg, velocity_centroid.dec.deg)
	if icenter_high < 0 or icenter_high > cutout_size or jcenter_high < 0 or jcenter_high > cutout_size:
		print('The galaxy is not in the cutout')
		return None
	else:
		print('Center of the galaxy in the high res image: ', icenter_high, jcenter_high)
		print('Center of the velocity map in the high res image: ', icenter_vel, jcenter_vel)
	# plt.imshow(EL_map_flip, origin='lower')
	# plt.title('Correctly rotated image')
	# plt.show()

	#cut the med band image to 62x62 around the photometric center: icenter_medband, jcenter_medband
	high_res_prior = EL_map_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]
	high_res_error = EL_err_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]
	med_band_cutout = jnp.array(med_band_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31])
	# save the broadband image needed to compute mask - using F444W for all ELs to have save mask
	broad_band_mask = broad_band_mask_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]
	# compute mask from med band image
	# threshold = threshold_otsu(med_band_cutout)/3
	# plt.imshow(jnp.where(med_band_cutout>threshold,med_band_cutout, 0.0) , origin  = 'lower')
	# plt.show()

	plt.imshow(high_res_prior, origin='lower')
	plt.title('Flipped and correct scale image 62x62 high res cutout')
	plt.colorbar()
	plt.show()

	plt.imshow(high_res_error, origin='lower')
	plt.title('Flipped and correct scale image 62x62 high res error cutout')
	plt.colorbar()
	plt.show()

	icenter_vel, jcenter_vel = icenter_vel - icenter_high + 31, jcenter_vel - jcenter_high + 31

	icenter_high, jcenter_high = 31, 31 #this is by defintion of how the cutout was made
	#recompute the velocity centroid in the new 62x62 cutout arounf icenter_high, jcenter_high

	x0_vel = jcenter_vel
	y0_vel = icenter_vel

	factor = 2
	#Rescale the med band cutout to 31x31 cutout by halphing the resolution to LW resolution
	blocks = high_res_prior.reshape((int(high_res_prior.shape[0]/(factor)), factor, int(high_res_prior.shape[1]/factor), factor))
	low_res_prior = np.sum(blocks, axis=(1,3))

	# plt.imshow(low_res_prior, origin='lower')
	# plt.title('Flipped and correct scale image 31x31 low res cutout')
	# plt.show()

	# icenter_LWband, jcenter_LWband = np.unravel_index(np.argmax(LW_band_prior, axis=None), LW_band_prior.shape)
	icenter_low, jcenter_low = 15, 15 #this is by defintion of how the cutout was made
	print('Center of the galaxy in the LW band image: ', icenter_low, jcenter_low)

	#now moving on to the grism data



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

	#subtract continuum and crop image by 200 on each size of EL
	grism_spectrum_data = contiuum_subtraction(jnp.array(grism_spectrum_fits['SPEC2D'].data), index_wave - 150, index_wave + 150)

	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	obs_map = grism_spectrum_data[:,index_min-(index_wave -150) +1:index_max-(index_wave -150)+1+1]

	obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min+1:index_max+1+1], - 0.5)
	#mask bad pixels in obs/error map
	obs_error = jnp.where(jnp.isnan(obs_map)| jnp.isnan(obs_error) | jnp.isinf(obs_error), 1e10, obs_error)


	#load number of module A and module B frames from the header
	if field == 'GOODS-S-FRESCO':
		print('Manually setting the module because the N_A and N_B params are not in the header')
		module_A = 0
		module_B = 1
	else:
		module_A = grism_spectrum_fits[0].header['N_A']
		module_B = grism_spectrum_fits[0].header['N_B']

	if module_A == 0:
		print('Flipping map! (Mod B)')
		obs_error = jnp.flip(obs_error, axis = 1)
		obs_map = jnp.flip(obs_map, axis = 1)

	# plt.imshow(obs_map, origin='lower')
	# plt.title('EL map cutout')
	# plt.show()
	
	# if fitting == 'high':
	direct = high_res_prior
	direct_error = high_res_error
	icenter_prior = icenter_high
	jcenter_prior = jcenter_high
	# elif fitting == 'low':
	# 	direct = low_res_prior
	# 	icenter_prior = icenter_low
	# 	jcenter_prior = jcenter_low
	

	return jnp.array(obs_map), jnp.array(obs_error), jnp.array(direct), jnp.array(direct_error), jnp.array(broad_band_mask), xcenter_detector, ycenter_detector, icenter_prior, jcenter_prior,icenter_low, jcenter_low, wave_space, d_wave, index_min, index_max, x0_vel, y0_vel, wavelength, theta

def preprocess_data_ALT(obj_id, output, line, delta_wave_cutoff = 0.02, field = 'ALT', RA_vel = -1, DEC_vel = -1):

	#no need to rotate this data, and not using EL prior bc no med band imaging
	 
	#load data from fits files
	image_hdu = fits.open(f"fitting_results/" + output + "j001420m3023_%s.full.fits"%(str(obj_id).zfill(5)))

	RA = RA_vel
	DEC = DEC_vel

	print('RA, DEC: ', RA, DEC)
	broad_band = image_hdu['DSCI','F356W-CLEAR'].data
	broad_band_mask = image_hdu['DSCI','F356W-CLEAR'].data

	broad_band_err = jnp.sqrt(1/image_hdu['DWHT','F356W-CLEAR'].data)
	broad_band_wcs = wcs.WCS(image_hdu['DSCI','F356W-CLEAR'].header)

	EL_map = image_hdu['LINE',line].data
	EL_map_error = jnp.sqrt(1/image_hdu['LINEWHT',line].data)

	wavelength = image_hdu['LINE',line].header['WAVELEN']*1e-4

	grism_spectrum_fits = fits.open(f"fitting_results/" + output + "j001420m3023_%s.stack.fits"%(str(obj_id).zfill(5)))

	cutout_size = broad_band.shape[0]

	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)
	if RA_vel == None:
		velocity_centroid = galaxy_position
	else:
		velocity_centroid = SkyCoord(ra=RA_vel * u.deg, dec=DEC_vel * u.deg)

	#compute the med band center from the WCS coords:
	icenter_high, jcenter_high = broad_band_wcs.world_to_array_index_values(galaxy_position.ra.deg, galaxy_position.dec.deg)
	icenter_vel, jcenter_vel = broad_band_wcs.world_to_array_index_values(velocity_centroid.ra.deg, velocity_centroid.dec.deg)
	if icenter_high < 0 or icenter_high > cutout_size or jcenter_high < 0 or jcenter_high > cutout_size:
		print('The galaxy is not in the cutout')
		return None
	else:
		print('Center of the galaxy in the high res image: ', icenter_high, jcenter_high)
		print('Center of the velocity map in the high res image: ', icenter_vel, jcenter_vel)
	# plt.imshow(EL_map_flip, origin='lower')
	# plt.title('Correctly rotated image')
	# plt.show()

	#cut the broad band image to 31x31 around the center
	ny,nx = EL_map.shape
	cx = int(0.5*(nx-1))
	dx = 32
	high_res_prior = EL_map[cx-dx:cx+dx,cx-dx:cx+dx]
	high_res_error = EL_map_error[cx-dx:cx+dx,cx-dx:cx+dx]
	broad_band_cutout = jnp.array(broad_band[cx-dx:cx+dx,cx-dx:cx+dx])
	# save the broadband image needed to compute mask - using F444W for all ELs to have save mask
	broad_band_mask = broad_band_mask[cx-dx:cx+dx,cx-dx:cx+dx]
	# compute mask from med band image
	# threshold = threshold_otsu(med_band_cutout)/3
	# plt.imshow(jnp.where(med_band_cutout>threshold,med_band_cutout, 0.0) , origin  = 'lower')
	# plt.show()

	plt.imshow(high_res_prior, origin='lower')
	plt.title('Flipped and correct scale image 62x62 high res cutout')
	plt.colorbar()
	plt.show()

	plt.imshow(high_res_error, origin='lower')
	plt.title('Flipped and correct scale image 62x62 high res error cutout')
	plt.colorbar()
	plt.show()

	icenter_vel, jcenter_vel = icenter_vel - cx + dx , jcenter_vel - cx + dx
	icenter_high, jcenter_high = icenter_high - cx + dx , jcenter_high - cx + dx#this is by defintion of how the cutout was made
	#recompute the velocity centroid in the new 62x62 cutout arounf icenter_high, jcenter_high

	x0_vel = jcenter_vel
	y0_vel = icenter_vel

	xcenter_detector = 1024
	ycenter_detector = 1024

	#grism stuff
	h_i = grism_spectrum_fits[('SCI','F356W')].header
	wave_space = (np.arange(h_i['NAXIS1']+1)-h_i['CRPIX1']+1)*h_i['CD1_1'] + h_i['CRVAL1']


	#choose which wavelengths will be the cutoff of the EL map and save those
	wave_min = wavelength - delta_wave_cutoff 
	wave_max = wavelength + delta_wave_cutoff 

	d_wave = h_i['CD1_1']
	wave_first = wave_space[0]

	# print(wave_min, wave_max)

	index_min = round((wave_min - wave_first)/d_wave) #+10
	index_max = round((wave_max - wave_first)/d_wave) #-10
	# print(index_min, index_max)
	index_wave = round((wavelength - wave_first)/d_wave)


	#subtract continuum and crop image by 200 on each size of EL
	grism_spectrum_data = jnp.array(grism_spectrum_fits[('SCI','F356W')].data) #contiuum_subtraction(jnp.array(grism_spectrum_fits[('SCI','F356W')].data), index_wave - 150, index_wave + 150)

	#cut EL map by using those wavelengths => saved as obs_map which is an input for Fit_Numpyro class
	# obs_map = grism_spectrum_data[:,index_min-(index_wave -150) +1:index_max-(index_wave -150)+1+1]
	obs_map = grism_spectrum_data[:,index_min:index_max+1]

	obs_error = np.power(grism_spectrum_fits[('WHT','F356W')].data[:,index_min+1:index_max+1+1], - 0.5)
	#mask bad pixels in obs/error map
	obs_error = jnp.where(jnp.isnan(obs_map)| jnp.isnan(obs_error) | jnp.isinf(obs_error), 1e10, obs_error)

	plt.imshow(obs_map, origin='lower')
	plt.title('obs_map')
	plt.show()
	
	direct = high_res_prior
	direct_error = high_res_error
	icenter_prior = icenter_high
	jcenter_prior = jcenter_high

	icenter_low = None
	jcenter_low = None

	return jnp.array(obs_map), jnp.array(obs_error), jnp.array(direct), jnp.array(direct_error), jnp.array(broad_band_mask), xcenter_detector, ycenter_detector, icenter_prior, jcenter_prior,icenter_low, jcenter_low, wave_space, d_wave, index_min, index_max, x0_vel, y0_vel, wavelength

def run_full_preprocessing(output,master_cat, line):
    """
        Main function that automatically post-processes the inference data and saves all of the relevant plots
    """
    
    with open('fitting_results/' + output + '/' + 'config_real.yaml', 'r') as file:
        input = yaml.load(file, Loader=yaml.FullLoader)
        print('Read inputs successfully')

    #load of all the parameters from the configuration file
    data, params, inference, ID, broad_filter, med_filter, grism_filter, med_band_path, broad_band_path,broad_band_mask_path, \
	grism_spectrum_path, field, wavelength, redshift, line, y_factor, res, flux_threshold, factor, \
	wave_factor, x0, y0, x0_vel, y0_vel, model_name, flux_bounds, flux_type, PA_sigma, i_bounds, Va_bounds, \
	r_t_bounds, sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff, PA_morph, inclination, r_eff, RA_vel, DEC_vel = read_config_file(input, output + '/', master_cat,line)

    #preprocess the images and the grism spectrum
    if field == 'ALT':
        obs_map, obs_error, direct, direct_error, broad_band, xcenter_detector, ycenter_detector, icenter, jcenter, icenter_low, jcenter_low, \
		wave_space, delta_wave, index_min, index_max, x0_vel, y0_vel, wavelength = preprocess_data_ALT(ID, output, line, delta_wave_cutoff, field, RA_vel, DEC_vel)
    else:
        obs_map, obs_error, direct, direct_error,broad_band, xcenter_detector, ycenter_detector, icenter, jcenter, icenter_low, jcenter_low, \
		wave_space, delta_wave, index_min, index_max, x0_vel, y0_vel, wavelength, theta= preprocess_data(med_band_path, broad_band_path, broad_band_mask_path, grism_spectrum_path, redshift, line, wavelength, delta_wave_cutoff, field, res, RA_vel, DEC_vel)

	#adapt the PA according the rotation of the galaxy to match PA of grism:
    print('PA_morph before:', PA_morph)
    PA_morph += theta
    PA_morph = PA_morph % 180 #put it back in the 0-180 range
    print('PA_morph after:', PA_morph)
    #renormalizing flux prior to EL map
    direct, direct_error,normalization_factor,mask_grism = renormalize_image(direct, direct_error,obs_map)
    print('renormalizing factor:', normalization_factor)
    factor = params['Params']['factor']
    wave_factor = params['Params']['wave_factor']
    # rescale the wave_space array and the direct image according to factor and wave_factor
    len_wave = int(
        (wave_space[len(wave_space)-1]-wave_space[0])/(delta_wave/wave_factor))
    wave_space = jnp.linspace(
        wave_space[0], wave_space[len(wave_space)-1], len_wave+1)

    # take x0 and y0 from the pre-processing unless specified otherwise in the config file
    if x0 == None:
        x0 = jcenter
    if y0 == None:
        y0 = icenter

    x0_grism = jcenter
    y0_grism = icenter
    
    PSF = 'gdn_mpsf_' + broad_filter + '_small.fits'

    # direct_low = utils.resample(direct, y_factor, y_factor)

    grism_object = grism.Grism(direct=direct, direct_scale=0.0629/y_factor, icenter=y0_grism, jcenter=x0_grism, segmentation=None, factor=factor, y_factor=y_factor,
                            xcenter_detector=xcenter_detector, ycenter_detector=ycenter_detector, wavelength=wavelength, redshift=redshift,
                            wave_space=wave_space, wave_factor=wave_factor, wave_scale=delta_wave/wave_factor, index_min=(index_min)*wave_factor, index_max=(index_max)*wave_factor,
                            grism_filter=grism_filter, grism_module='A', grism_pupil='R', PSF = PSF)
    direct_image_size = direct.shape
    
    _, _, _, _, PA_grism, inc_grism,_ = utils.compute_gal_props(obs_map, threshold_sigma = flux_threshold/2) # /2 bc grism lower s/n than imaging data
    print('PA_grism: ', PA_grism)
    print('inc_grism: ', inc_grism)

    # initialize chosen kinematic model
    if model_name == 'Disk':
        kin_model = models.DiskModel()
    elif model_name == 'Disk_prior':
        kin_model = models.DiskModel_withPrior()
    elif model_name == 'Merger':
        kin_model = models.Merger()
    kin_model.set_bounds(direct, direct_error, broad_band, flux_bounds, flux_type, flux_threshold, PA_sigma,i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph, inclination, r_eff)
    
	# plt.imshow(obs_map, origin='lower')
	# plt.show()

	# plt.imshow(direct, origin='lower')
	# plt.show()
    
    return direct, direct_error, obs_map, obs_error, model_name, kin_model, grism_object, y0_grism,x0_grism, num_samples, num_warmup, step_size, target_accept_prob, wave_space, delta_wave, index_min, index_max, factor