"""
Put all of the necessary pre-processing functions here so that fit_numpyro is cleaner
Eventually should also add here scripts to automatically create folders for sources, with all of the right cutouts etc
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""

#imports
import numpy as np
import math
import matplotlib.pyplot as plt

import jax.numpy as jnp

from scipy.ndimage import median_filter
from scipy import ndimage

#for the masking
from skimage.morphology import  dilation, disk

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import wcs

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing
from skimage.color import label2rgb

from reproject import reproject_adaptive



def read_config_file(input, output):
	"""

        Read the config file for the galaxy and returns all of the relevant parameters

	"""
	data = input[0]
	params = input[1]
	inference = input[2]
	priors = input[3]

	ID = data['Data']['ID']
		
	broad_filter = data['Data']['broad_filter']
		
	if broad_filter == 'F444W':
		med_filter = 'F410M'
	elif broad_filter == 'F356W':
		med_filter = 'F335M'
	
	field = data['Data']['field']
	
	#no ../ because the open() function reads from terminal directory (not module directory)
	med_band_path = 'fitting_results/' + output + str(ID) + '_' + med_filter + '.fits'
	broad_band_path = 'fitting_results/' + output + str(ID) + '_' + broad_filter + '.fits'
	if field == 'GOODS-N' or field == 'GOODS-N-CONGRESS':
		grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_GDN_' + broad_filter + '_ID' + str(ID) + '_comb.fits'
	elif field == 'GOODS-S-FRESCO':
		grism_spectrum_path = 'fitting_results/' + output+ 'spec_2d_FRESCO_' + broad_filter + '_ID' + str(ID) + '_comb.fits'




	wavelength = data['Data']['wavelength']

	redshift = data['Data']['redshift']

	line = data['Data']['line']
								
	y_factor = params['Params']['y_factor']
	if y_factor == 1:
		print('Fitting the low res image')
		res = 'low'
	elif y_factor == 2:
		print('Fitting the high res image')
		res = 'high'
		
	to_mask = params['Params']['masking']
	flux_threshold = inference['Inference']['flux_threshold']
	
	delta_wave_cutoff = params['Params']['delta_wave_cutoff']
	factor = params['Params']['factor']
	wave_factor = params['Params']['wave_factor']
	
	x0 = params['Params']['x0']
	y0 = params['Params']['y0']
	

	x0_vel = np.array(params['Params']['x0_vel'])
	y0_vel = np.array(params['Params']['y0_vel'])

	model_name = inference['Inference']['model']
	
    #import all of the bounds needed for the priors
	flux_bounds = inference['Inference']['flux_bounds']
	flux_type = inference['Inference']['flux_type']
	PA_bounds = inference['Inference']['PA_bounds']
	i_bounds = inference['Inference']['i_bounds']
	Va_bounds = inference['Inference']['Va_bounds']
	r_t_bounds = inference['Inference']['r_t_bounds']
	sigma0_bounds = inference['Inference']['sigma0_bounds']
	sigma0_mean = inference['Inference']['sigma0_mean']
	sigma0_disp = inference['Inference']['sigma0_disp']
	obs_map_bounds = inference['Inference']['obs_map_bounds']

	clump_v_prior = inference['Inference']['clump_v_prior']
	clump_sigma_prior = inference['Inference']['clump_sigma_prior']
	clump_flux_prior = inference['Inference']['clump_flux_prior']
	
	clump_bool = inference['Inference']['clump_bool']
	
	num_samples = inference['Inference']['num_samples']
	num_warmup = inference['Inference']['num_warmup']
	
	step_size = inference['Inference']['step_size']
	target_accept_prob = inference['Inference']['target_accept_prob']
	
	#return all of the parameters
	return data, params, inference, priors, ID, broad_filter, med_filter, med_band_path, broad_band_path, grism_spectrum_path, field, wavelength, redshift, line, y_factor, res, to_mask, flux_threshold, factor, wave_factor, x0, y0, x0_vel, y0_vel, model_name, flux_bounds, flux_type, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, sigma0_mean, sigma0_disp, obs_map_bounds, clump_v_prior, clump_sigma_prior, clump_flux_prior, clump_bool, num_samples, num_warmup, step_size, target_accept_prob, delta_wave_cutoff


def renormalize_image(direct, obs_map, flux_threshold, y_factor):
	"""
		Normalize the image to match the total flux in the EL map
	"""

	threshold = flux_threshold*0.5*direct.max()
	mask = jnp.zeros_like(direct)
	mask = mask.at[jnp.where(direct>threshold)].set(1)
	# mask = dilation(mask, disk(2))

	# plot the direct image found within the mask, the rest set to 0
	plt.imshow(direct*mask, cmap='viridis', origin='lower')
	plt.title('Direct image within the mask')
	plt.show()

	#create a mask for the grism map
	threshold_grism = flux_threshold*0.5*obs_map.max()
	mask_grism = jnp.zeros_like(obs_map)
	mask_grism = mask_grism.at[jnp.where(obs_map>threshold_grism)].set(1)
	# mask_grism = dilation(mask_grism, disk(6))

	# plot the grism image found within the mask, the rest set to 0
	plt.imshow(obs_map*mask_grism, cmap='viridis', origin='lower')
	plt.title('Grism data within the mask')
	plt.show()

	#compute the normalization factor
	normalization_factor = obs_map[jnp.where(mask_grism == 1)].sum()/direct[jnp.where(mask == 1)].sum()
	#normalize the direct image to the grism image
	direct = direct*normalization_factor

	#now recompute the broader mask for the image
	threshold = flux_threshold*direct.max()
	mask = jnp.zeros_like(direct)
	mask = mask.at[jnp.where(direct>threshold)].set(1)
	mask = dilation(mask, disk(6*int(y_factor)))

	return direct, normalization_factor, mask, mask_grism

def mask_bad_pixels(image,errors,tolerance=3.5):
	"""
		Set the hot and dead pixels to zero and set their errors to a high value so they do not
		carry weight in the fit
	"""

	blurred = median_filter(image, size=3)
	difference = image - blurred
	threshold = tolerance*np.std(difference)

	hot_pixels = np.nonzero((np.abs(difference)>threshold) )
	hot_pixels = np.array(hot_pixels)

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

def rotate_wcs(med_band_fits, EL_map, field, cutout_size):
	'''
		rotate the WCS of the image to match the PA of the grism image
	'''
	med_band = med_band_fits[1].data
	wcs_med_band = wcs.WCS(med_band_fits[1].header)

	if field == 'GOODS-S' or field =='GOODS-S-FRESCO':
			print('FRESCO PA is the same in GOODS-S, no correction needed')
			med_band_flip = med_band
			EL_map_flip = EL_map
			rotated_wcs = wcs_med_band
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


	return rotated_wcs, med_band_flip, EL_map_flip

def contiuum_subtraction(grism_spectrum_fits):
	'''
		Subtract the continuum from the EL map
	'''
	grism_spectrum_data = grism_spectrum_fits['SPEC2D'].data #NOT cont sub

	#do the cont subtraction
	L_box, L_mask = 25, 4
	mf_footprint = np.ones((1, L_box * 2 + 1))
	mf_footprint[:, L_box-L_mask:L_box+L_mask+1] = 0
	tmp_grism_img_median = ndimage.median_filter(grism_spectrum_data, footprint=mf_footprint, mode='reflect')
	grism_spectrum_data = grism_spectrum_data - tmp_grism_img_median  # emission line map

	return grism_spectrum_data

def preprocess_data(med_band_path, broad_band_path, grism_spectrum_path, redshift, line, wavelength, delta_wave_cutoff = 0.02, field = 'GOODS-S', fitting = None):
	 
	#load data from fits files
	med_band_fits = fits.open(med_band_path)
	med_band = med_band_fits[1].data

	broad_band_fits = fits.open(broad_band_path)
	broad_band = broad_band_fits[1].data

	grism_spectrum_fits = fits.open(grism_spectrum_path)

	cutout_size = med_band.shape[0]

	RA = grism_spectrum_fits[0].header['RA0']
	DEC = grism_spectrum_fits[0].header['DEC0']

	#compute the EL map prior first
	EL_map = make_EL_map(med_band, broad_band, redshift, line)

	#if needed, rotate the image WCS and the EL map to match the PA of the grism image
	rotated_wcs, med_band_flip, EL_map_flip = rotate_wcs(med_band_fits, EL_map, field, cutout_size)
	
	galaxy_position = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg)

	#compute the med band center from the WCS coords:
	icenter_high, jcenter_high = rotated_wcs.world_to_array_index_values(galaxy_position.ra.deg, galaxy_position.dec.deg)
	if icenter_high < 0 or icenter_high > cutout_size or jcenter_high < 0 or jcenter_high > cutout_size:
		print('The galaxy is not in the cutout')
		return None
	else:
		print('Center of the galaxy in the high res image: ', icenter_high, jcenter_high)

	# plt.imshow(EL_map_flip, origin='lower')
	# plt.title('Correctly rotated image')
	# plt.show()

	#cut the med band image to 62x62 around the photometric center: icenter_medband, jcenter_medband
	high_res_prior = EL_map_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]
	med_band_cutout = med_band_flip[icenter_high-31:icenter_high+31,jcenter_high-31:jcenter_high+31]


	icenter_high, jcenter_high = 31, 31 #this is by defintion of how the cutout was made

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

	#subtract continuum
	grism_spectrum_data = contiuum_subtraction(grism_spectrum_fits)

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

	#save the error array of the EL map cutout
	obs_error = np.power(grism_spectrum_fits['WHT2D'].data[:,index_min+1:index_max+1+1], - 0.5)

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
	
	if fitting == 'high':
		direct = high_res_prior
		icenter_prior = icenter_high
		jcenter_prior = jcenter_high
	elif fitting == 'low':
		direct = low_res_prior
		icenter_prior = icenter_low
		jcenter_prior = jcenter_low
	
	# plt.imshow(low_res_prior, origin='lower')
	# plt.show()

	#compute PA from the cropped med band image (not the EL map)
	PA_truth = compute_PA(med_band_cutout)

	return jnp.array(obs_map), jnp.array(obs_error), jnp.array(direct), PA_truth, xcenter_detector, ycenter_detector, icenter_prior, jcenter_prior,icenter_low, jcenter_low, wave_space, d_wave, index_min, index_max
