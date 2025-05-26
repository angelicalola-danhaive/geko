import os
import argparse
# current_working_directory = os.getcwd()
# geko_dir = 'geko'
# os.chdir(os.path.join(current_working_directory, geko_dir))
import preprocess as pre
import utils
import matplotlib
import numpy as np
import yaml
from astropy.table import Table
import run_pysersic as py

import traceback

# matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--line', type=str, default='H_alpha',
                    help='line to fit')
parser.add_argument('--master_cat', type=str, default='catalogs/fresco_Ha_cat',
                    help='master catalog file name')
parser.add_argument('--parametric', type=bool, default=True,
                    help='parametric flux model or not')

args = parser.parse_args()
# main_folder = '/Users/lola/ASTRO/JWST/grism_project/FrescoHa/Galaxies' + args.output_list
master_cat_path = args.master_cat
line = args.line
parametric = args.parametric

#open the master catalog
master_cat = Table.read(master_cat_path, format="ascii")

table = Table.read('Gold_Silver_FRESCO_CONGRESS.txt', format='ascii')

#read all of the properties
for col in table.colnames:
    globals()[col] = table[col]

Ids =  [1090054]
mask = np.isin(table['ID'], Ids)

z_spec = table['z_spec'][mask]
RA = table['RA'][mask]
DEC = table['DEC'][mask]
folders = table['ID'][mask]
fields = table['field'][mask]
clumpy = table['clumpy'][mask]

failed_folders = []

# folders = [1120765]

for i, output in enumerate(folders):

    # output = str(1029047)

    print('Preprocessing ' + str(output))

    field = fields[i]
    print(field)
    ID = folders[i]
        
    line = 'H_alpha'

    redshift = z_spec[i]
    print(redshift)
      # need to run run Pysersic on 3 filters
    if redshift<5:
        print('Redshift is less than 5, using F090W, F150W and F277W filters')
        filters = [ 'F150W', 'F277W'] #'F090W',
    else:
        print('Redshift is greater than 5, using F200W and F356W filters')
        filters = ['F090W', 'F200W', 'F356W'] #'F090W', 

    for filter in filters:

        #Define variables
        # define the right variables
        med_band_path = 'fitting_results/' + str(output) + '/' + str(output) + '_' + filter + '.fits'
        # not used for fitting, just need to set it to smth
        broad_band_path = 'fitting_results/' + str(output) + '/' + str(output) + '_' + filter + '.fits'
        broad_band_mask_path = 'fitting_results/' + str(output) + '/' + str(output) + '_' + filter + '.fits'  # idem


        if field == 'GDN':
            grism_spectrum_path = 'fitting_results/' + str(output) + '/spec_2d_GDN_' + 'F444W' + '_ID' + str(output) + '_comb.fits'
            grism_filter = 'F444W'
            field = 'GOODS-N'
            #check that the file exists
            if os.path.exists(grism_spectrum_path):
                print('Grism spectrum file exists')
            else:
                grism_spectrum_path = 'fitting_results/' + str(output) + '/spec_2d_GDN_' + 'F356W' + '_ID' + str(output) + '_comb.fits'
                grism_filter = 'F356W'
                field = 'GOODS-N-CONGRESS'

        elif field == 'GOODS-S-FRESCO':
            grism_spectrum_path = 'fitting_results/' + str(output) + '/spec_2d_FRESCO_' + 'F444W' + '_ID' + str(output) + '_comb.fits'
            grism_filter = 'F444W'



        if True: #no redshift or field cut needed - want to run for all

            wavelength = 0.6563*(1 + redshift)  # H_alpha
            # run the preprocessing
            try:
                module, obs_map, obs_error, direct, direct_error, broad_band_mask, xcenter_detector, ycenter_detector, \
                            icenter_prior, jcenter_prior, icenter_low, jcenter_low, wave_space, d_wave, index_min, index_max, wavelength, \
                            theta, direct_low, direct_error_low = pre.preprocess_data(med_band_path, broad_band_path, broad_band_mask_path, grism_spectrum_path, redshift, line, wavelength, delta_wave_cutoff=0.02, field = field, fitting=None)
            except Exception as e:
                print(e)
                #print the full error
                traceback.print_exc()
                print('Preprocessing failed for ' + str(output) + ' with filter ' + filter)

                #move on to the next galaxy
                failed_folders.append(output)
                continue                                                                                                                                                           

            # Construct the expected file path
            file_path = '/Users/lola/ASTRO/JWST/grism_project/fitting_results/' + str(output) + '/' + str(output) + '_image_' + filter + '_corner_svi.pdf'
            # Check if the file exists
            # if os.path.exists(file_path):
            #     print(f"File {file_path} already exists. Skipping {output}.")
            #     continue
            if True:
            # else:
                # run the pysersic fitting
                path_output = 'fitting_results/' + str(output) + '/'
                PSF_py_image = utils.load_psf(filter=filter, y_factor=2)

                #resample the direct image by a factor of 2
                # direct_low = utils.resample(direct, 2,2)

                # direct_error_low = utils.downsample_error(direct_error)

                # try a few different sigma_rms values from 20 downwards
                if clumpy[i] == 1:
                    mask = False
                else:
                    mask = True
                try:
                    inclination, py_PA, phot_PA, r_eff, x0_vel, y0_vel = py.run_pysersic_fit(
                    filter=filter, id=output, path_output=path_output, im=direct, sig=direct_error, psf=PSF_py_image, type='image', perform_masking = True, sigma_rms=3)

                except Exception as e:
                    print('Trying sigma_rms = 10')
                    try:
                        inclination, py_PA, phot_PA, r_eff, x0_vel, y0_vel = py.run_pysersic_fit(
                            filter=filter, id=output, path_output=path_output, im=direct, sig=direct_error, psf=PSF_py_image, type='image', sigma_rms=10)
                    except Exception as e:
                        print('Trying sigma_rms = 5')
                        try:
                            inclination, py_PA, phot_PA, r_eff, x0_vel, y0_vel = py.run_pysersic_fit(
                                filter=filter, id=output, path_output=path_output, im=direct, sig=direct_error, psf=PSF_py_image, type='image', sigma_rms=5)
                        except Exception as e:
                            print('Trying sigma_rms = 1')
                            try:
                                inclination, py_PA, phot_PA, r_eff, x0_vel, y0_vel = py.run_pysersic_fit(
                                    filter=filter, id=output, path_output=path_output, im=direct, sig=direct_error, psf=PSF_py_image, type='image', sigma_rms=1)
                            except Exception as e:
                                print("Image" + filter + "fit didn't work")
                                traceback.print_exc()
            
    # print('Successfully preprocessed' + output + 'with the filter' + filter)


    # # #now fit the grism spectrum
    # # Construct the expected file path
    # file_path = '/Users/lola/ASTRO/JWST/grism_project/fitting_results/' + str(output) + '/' + str(output) + '_grism_' + grism_filter + '_corner_svi.pdf'

    # # Check if the file exists
    # if os.path.exists(file_path):
    #         print(f"File {file_path} already exists. Skipping {output}.")
    # else:
    #     PSF_py_grism = utils.load_psf(filter=grism_filter, y_factor=1)
    #     size = obs_map.shape[0]
    #     square_obs_map= obs_map[:, obs_map.shape[1]//2 - size//2:obs_map.shape[1]//2 + size//2 + 1]
    #     square_obs_error = obs_error[:, obs_error.shape[1]//2 - size//2 :obs_error.shape[1]//2 + size//2 + 1]
    #     try:
    #         inc_grism, py_PA_grism, PA_grism, r_eff_grism, _, _ = py.run_pysersic_fit(filter = grism_filter, id = output, path_output = 'fitting_results/' + str(output), im = square_obs_map, sig = square_obs_error, psf = PSF_py_grism,perform_masking=True, type = 'grism', sigma_rms =5)
    #     except Exception as e:
    #         print('Trying sigma_rms = 10')
    #         try:
    #             inc_grism, py_PA_grism, PA_grism, r_eff_grism, _, _ = py.run_pysersic_fit(filter = grism_filter, id = output, path_output = 'fitting_results/' + str(output), im = square_obs_map, sig = square_obs_error, psf = PSF_py_grism, type = 'grism', perform_masking=True, sigma_rms = 3)
    #         except Exception as e:
    #             print('Trying sigma_rms = 5')
    #             try:
    #                 inc_grism, py_PA_grism, PA_grism, r_eff_grism, _, _ = py.run_pysersic_fit(filter = grism_filter, id = output, path_output = 'fitting_results/' + str(output), im = square_obs_map, sig = square_obs_error, psf = PSF_py_grism, type = 'grism', perform_masking=True, sigma_rms = 5)
    #             except Exception as e:
    #                 print('Trying sigma_rms = 1')
    #                 try:
    #                     inc_grism, py_PA_grism, PA_grism, r_eff_grism, _, _ = py.run_pysersic_fit(filter = grism_filter, id = output, path_output = 'fitting_results/' + str(output), im = square_obs_map, sig = square_obs_error, psf = PSF_py_grism, type = 'grism', perform_masking=True, sigma_rms = 1)
    #                 except Exception as e:
    #                     print("Grism fit didn't work")
    #                     traceback.print_exc()
    

    

