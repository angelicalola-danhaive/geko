"""
Put all of the necessary pre-processing functions here so that fit_numpyro is cleaner
Eventually should also add here scripts to automatically create folders for sources, with all of the right cutouts etc
    
    
    Written by A L Danhaive: ald66@cam.ac.uk
"""

__all__ = ['run_full_preprocessing', 'prep_grism']

#imports
from . import  utils
from . import  grism #just for now since in active dev
from . import  models
# import run_pysersic as py

import numpy as np
import math
import matplotlib.pyplot as plt

import jax.numpy as jnp

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

# Handle mocked scipy.constants for documentation builds
try:
	c_m = c*1e-9
except TypeError:
	c_m = 299792458.0*1e-9  # Speed of light in m/s * 1e-9

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
    if field == 'GOODS-S-FRESCO' or field == 'GOODS-N' or field == 'GOODS-N-CONGRESS' or field == 'manual':

        modules = grism_spectrum_fits[5].data['module']
        module_A = int('A' in modules)

        # Read pupil parameter from FITS file
        pupils = grism_spectrum_fits[5].data['pupil']
        pupil = pupils[0]  # Take first element from array


    else:
        module_A = grism_spectrum_fits[0].header['N_A']
        module_B = grism_spectrum_fits[0].header['N_B']

        # Default to 'R' pupil for non-manual fields
        pupil = 'R'


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


    return module, pupil, jnp.array(obs_map), jnp.array(obs_error),  wave_space, d_wave, index_min, index_max, wavelength




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
    target_accept_prob = 0.8

    
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

    # plt.close()

    
    return  obs_map, obs_error, direct, direct_error, broad_band, xcenter_detector, ycenter_detector, icenter, jcenter, icenter_low, jcenter_low, \
                wave_space, delta_wave, index_min, index_max, wavelength, theta, grism_object

def run_full_preprocessing(output, master_cat, line, mock_params=None, priors=None, save_runs_path='fitting_results/',
                            source_id=None, field=None, grism_filter='F444W', delta_wave_cutoff=0.005,
                            factor=5, wave_factor=10, model_name='Disk',
                            manual_psf_name=None, manual_grism_file=None):
    """
        Main function that automatically preprocesses data for geko fitting.

        Loads grism spectroscopy data, performs background subtraction and cropping,
        loads PSF, initializes the Grism dispersion object and kinematic model.

        Parameters
        ----------
        output : str
            Name of output subfolder
        master_cat : str
            Path to master catalog file
        line : int
            Emission line wavelength in Angstroms
        mock_params : dict, optional
            Parameters for mock data (for testing)
        priors : dict, optional
            Custom priors (not used in current implementation)
        save_runs_path : str
            Base directory containing data files
        source_id : int
            Source ID number
        field : str
            Field name: 'GOODS-N', 'GOODS-N-CONGRESS', 'GOODS-S-FRESCO', or 'manual'
        grism_filter : str
            Grism filter name (default: 'F444W')
        delta_wave_cutoff : float
            Wavelength bin size cutoff in microns
        factor : int
            Spatial oversampling factor
        wave_factor : int
            Wavelength oversampling factor
        model_name : str
            Kinematic model type (default: 'Disk')
        manual_psf_name : str, optional
            PSF filename in save_runs_path/psfs/ (required if field='manual')
        manual_grism_file : str, optional
            Grism spectrum filename in save_runs_path/output/ (required if field='manual')

        Returns
        -------
        z_spec : float
            Spectroscopic redshift
        wavelength : float
            Observed wavelength in microns
        wave_space : numpy.ndarray
            Wavelength array for the cropped spectrum
        obs_map : numpy.ndarray
            Observed 2D grism spectrum (spatial x wavelength)
        obs_error : numpy.ndarray
            Error map for observations
        kin_model : KinModels
            Initialized kinematic model object
        grism_object : Grism
            Initialized grism dispersion object
        delta_wave : float
            Wavelength bin size in microns
    """

    if mock_params == None:
        ID = source_id

        # Construct grism spectrum path based on field
        if field == 'manual':
            if manual_grism_file is None:
                raise ValueError("manual_grism_file must be provided when field='manual'")
            grism_spectrum_path = save_runs_path + output + '/' + manual_grism_file
        elif field == 'GOODS-N':
            grism_spectrum_path = save_runs_path + output + '/spec_2d_GDN_' + grism_filter + '_ID' + str(ID) + '_comb.fits'
        elif field == 'GOODS-N-CONGRESS':
            grism_spectrum_path = save_runs_path + output + '/spec_2d_GDN_' + grism_filter + '_ID' + str(ID) + '_comb.fits'
        elif field == 'GOODS-S-FRESCO':
            grism_spectrum_path = save_runs_path + output + '/spec_2d_FRESCO_' + grism_filter + '_ID' + str(ID) + '_comb.fits'
        else:
            raise ValueError(f"Field {field} is not supported. Supported fields are: GOODS-N, GOODS-N-CONGRESS, GOODS-S-FRESCO, manual.")

        # Get wavelength and redshift from master catalog
        master_cat_table = Table.read(master_cat, format="ascii")
        wavelength = master_cat_table[master_cat_table['ID'] == ID][str(line) + '_lambda'][0]
        redshift = master_cat_table[master_cat_table['ID'] == ID]['zspec'][0]
        #preprocess the images and the grism spectrum
        if field == 'ALT':
            #generate an error that says not updated
            raise ValueError("The field ALT is not updated in the preprocessing function. Please use a different field.")
        else:
            module, pupil, obs_map, obs_error,wave_space, delta_wave, index_min, index_max, wavelength = preprocess_data(grism_spectrum_path, wavelength, delta_wave_cutoff, field)
        grism_object = None
    else:
        broad_filter, grism_filter, wavelength, redshift, line, y_factor, flux_threshold, factor, \
        wave_factor, x0, y0, model_name, flux_bounds, flux_type, PA_sigma, i_bounds, Va_bounds, \
        r_t_bounds, sigma0_bounds,num_samples, num_warmup, step_size, target_accept_prob = define_mock_params()
        obs_map, obs_error, direct, direct_error, broad_band, xcenter_detector, ycenter_detector, icenter, jcenter, icenter_low, jcenter_low, \
                wave_space, delta_wave, index_min, index_max, wavelength, theta, grism_object = preprocess_mock_data(mock_params)
        field = 'GOODS-S-FRESCO'

    #load the PSF that corresponds to the grism program
    if field == 'manual':
        if manual_psf_name is None:
            raise ValueError("manual_psf_name must be provided when field='manual'")
        psf_path = save_runs_path + 'psfs/' + manual_psf_name
    elif field == 'GOODS-N':
        psf_path = save_runs_path + 'psfs/mpsf_jw018950.gn.f444w.fits'
    elif field == 'GOODS-N-CONGRESS':
        psf_path = save_runs_path + 'psfs/mpsf_jw035770.f356w.fits'
    elif field == 'GOODS-S-FRESCO':
        psf_path = save_runs_path + 'psfs/mpsf_jw018950.gs.f444w.fits'


    PSF = fits.getdata(psf_path)

    # PSF = utils.load_psf(grism_filter, 1, 9)

    #downsample it down to the grism resolution
    PSF = utils.downsample_psf_centered(PSF, size = 15)
    #run pysersic fit to get morphological parameters
    if mock_params == None:
        path_output = save_runs_path + output
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
        grism_object = grism.Grism(im_shape, im_scale, icenter = icenter, jcenter = jcenter, wavelength = wavelength, wave_space = wave_space_model, index_min = (index_min)*wave_factor, index_max = (index_max+1)*wave_factor,
                       grism_filter = grism_filter, grism_module = module, grism_pupil = pupil, PSF = PSF)



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

    # Set grism-specific configuration
    # All priors (morphological and kinematic) will be set later in fitting.py via
    # set_priors_from_config() or set_parametric_priors()
    kin_model.set_bounds((obs_map.shape[0], obs_map.shape[0]), factor, wave_factor, x0, x0_vel, y0, y0_vel)

    return redshift, wavelength, wave_space, obs_map, obs_error, kin_model, grism_object, delta_wave

