
# load modules

from geko import utils
from geko import preprocess

import os
import numpy as np 
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve
from astropy.wcs import WCS                                                            


from pysersic import check_input_data, FitSingle, FitMulti, PySersicMultiPrior
from pysersic.priors import autoprior
from pysersic.loss import student_t_loss
from pysersic.results import plot_image, plot_residual

from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D, MedianBackground

from jax import random
import jax.numpy as jnp
from pysersic.rendering import HybridRenderer





import asdf 

import matplotlib

import smplotlib

matplotlib.use('Agg')

# define key functions

def make_mask(im, sigma_rms):
    im_conv = im #convolve(im, make_2dgaussian_kernel(3.0, size=5))
    # print('pre-bckg')
    bkg = Background2D(im_conv, (15, 15), filter_size=(3, 3), exclude_percentile=90.0)
    segment_map = detect_sources(im_conv, sigma_rms*bkg.background_rms, npixels=20)

    #manually mask all pixels outside the central 10x10 pixel region
    # segment_map[segme
    print('post-bckg')
    # fig,ax = plt.subplots(1,1)
    # ax.imshow(segment_map.data, origin='lower')
    # # plt.title('Pysersic mask')
    # plt.show()
    # plt.close()
    return im_conv, segment_map

def prepare_data(im, sig,psf, fit_multi=False, perform_masking=False, plot=False, sigma_rms = 3.0):
    '''
    Prepare the data to fit:
     - im: the image itself
     - mask: a mask indicating pixels which should not be included in the fit
     - sig: a map of pixel uncertainties 
     - psf: a map of the PSF (for now, only one psf can be used).
    '''


    # check how much no data coverage
    frac_0 = np.sum(im == 0.0) / (im.shape[0] * im.shape[1])

    if (frac_0 > 0.5):
        print("data looks bad!")
        return(None, None, None, None, None, None)


    # make segmentation map and identify sources
    im_conv, segment_map = make_mask(im, sigma_rms)
    segm_deblend = deblend_sources(im_conv, segment_map, npixels=10, nlevels = 64,  contrast=0.001, progress_bar=False) #contrast=0.001nlevels=50,
    source_cat = SourceCatalog(im, segm_deblend, convolved_data=im_conv, error=sig)
    source_tbl = source_cat.to_table()

    # plt.imshow(segment_map, origin='lower')
    # plt.title('Pysersic mask')
    # plt.show()


    # identify main label
    main_label = segm_deblend.data[int(0.5*im.shape[0]), int(0.5*im.shape[1])]

    phot_PA = source_cat.orientation[main_label-1].value
    if phot_PA < 0:
        phot_PA += 180.0

    # select sources to fit
    snr = source_tbl['segment_flux']/source_tbl['segment_fluxerr']
    idx_signifcant = (snr > 10.0) | (source_tbl['label']==main_label)
    labels_sources = source_tbl['label'][idx_signifcant].value.tolist()
    to_pysersic = {}
    to_pysersic['flux'] = source_tbl['segment_flux'][idx_signifcant].value.tolist()
    to_pysersic['x'] = source_tbl['xcentroid'][idx_signifcant].value.tolist()
    to_pysersic['y'] = source_tbl['ycentroid'][idx_signifcant].value.tolist()
    to_pysersic['r'] = source_tbl['semimajor_sigma'][idx_signifcant].value.tolist()
    to_pysersic['type'] = ['sersic'] * np.sum(idx_signifcant)

    # construct mask
    if perform_masking:
        mask = segm_deblend.data
        if fit_multi:
            for ii_lab in labels_sources:
                mask[mask == ii_lab] = 0.0
        else:
            mask[mask == main_label] = 0.0
        mask[mask > 0] = 1.0
            #manually mask all pixels outside the central 10x10 pixel region
        # mask[0:25, :] = 1.0
        # mask[:, 0:25] = 1.0
        # mask[35:,:] = 1.0
        # mask[:, 35:] = 1.0

    else:
        mask = np.zeros(im.shape)
    
    # plot data
    if plot:
        fig, ax = plot_image(im, mask, sig, psf)
        for i in range(3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        plt.show()

    # check data
    if check_input_data(data=im, rms=sig, psf=psf, mask=mask):
        print("data looks good!")
        return (im, mask, sig, psf, to_pysersic, phot_PA)

    else:
        print("data looks bad!")
        return (None, None, None, None, None, None)



def fit_data(filter, id, path_output=None, im = None, sig = None, psf = None,
             fit_multi=False, posterior_estimation=False, do_sampling=False, perform_masking=False, plot=False, type = 'image', sigma_rms = None):

    print("=============================================================================")

    # load data
    # print("load data for filter", filter.upper(), " and galaxy", id)
    # print('Printing here: ', prepare_data(im, sig,psf,fit_multi=fit_multi, perform_masking=perform_masking, plot=plot, sigma_rms = sigma_rms))
    im, mask, sig, psf, to_pysersic,phot_PA = prepare_data(im, sig,psf,fit_multi=fit_multi, perform_masking=perform_masking, plot=plot, sigma_rms = sigma_rms)
    #remove null values from the rms = sig 
    median_sig = np.median(sig)
    sig = jnp.where(sig == 0.0,median_sig, sig)
    # abort if no data available
    if im is None:
        return()

    # get priors
    if fit_multi:
        prior = PySersicMultiPrior(catalog=to_pysersic, sky_type='none')
    else:
        prior  = autoprior(image=im, profile_type='sersic', mask=mask, sky_type='flat')
        prior.set_uniform_prior('theta', 0.5, 6.28)
        # prior.set_gaussian_prior('xc', 31, 1.0)
        # prior.set_gaussian_prior('yc', 31, 1.0)
        prior.set_uniform_prior('r_eff', 0.5, 30.0)
        # prior.set_truncated_gaussian_prior('n', 1.0,0.5, 0.5, 8.0)
    print(prior)


    if fit_multi:
        post_name = '_multi'
    else:
        post_name = ''

    mean_error = np.mean(sig)*np.ones(im.shape)

    if posterior_estimation:
        # setup fitter
        if fit_multi:
            fitter = FitMulti(data=im, rms=sig, psf=psf, prior=prior, loss_func=student_t_loss)
        else:
            fitter = FitSingle(data=im, rms=sig ,mask=mask,psf=psf, prior=prior, loss_func=student_t_loss)
        # posterior estimation
        fitter.estimate_posterior(method='svi-flow', rkey = random.PRNGKey(10))
        summary = fitter.svi_results.summary()
        print(summary)
        # # mod_sampling = sampling_res.get_median_model()
        # # print("residual of sampling:", filter.upper())
        # # map_params = fitter.find_MAP(rkey = random.PRNGKey(0))
        # # mod_sampling = map_params['model']
        # # res = fitter.estimate_posterior(rkey = random.PRNGKey(1001), method='laplace')
        summary = fitter.svi_results.summary()
        dict = {}
        for a, b in zip(summary.index, summary["mean"]):
            dict[a] = b

        bf_model = HybridRenderer(im.shape, jnp.array(psf.astype(np.float32))).render_source(dict, profile_type="sersic")
        print("residual of posteriorr estimation:", filter.upper())


        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.1)
        axs[0].imshow(im, cmap = 'PuOr', vmin = -bf_model.max()/2, vmax = bf_model.max()/2)
        axs[1].imshow(bf_model, cmap = 'PuOr', vmin = -bf_model.max()/2, vmax = bf_model.max()/2)
        axs[2].imshow(im - bf_model, cmap = 'PuOr', vmin = -bf_model.max()/2, vmax = bf_model.max()/2)
        for i in range(3):
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].contour(mask, levels=[0.5], colors='crimson', linewidths=1.5)
        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_paperplot' + post_name + '.png'), bbox_inches='tight', dpi=1000)
        plt.show()

        # # save figures
        fig, ax = plot_residual(im, bf_model, mask=None, vmin=-0.45, vmax=0.45)
        for i in range(3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].contour(mask, levels=[0.5], colors='cornflowerblue', linewidths=1.5)
        
        #reduce the separation between the subplots
        fig.tight_layout()

        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_residual_svi' + post_name + '.png'), bbox_inches='tight', dpi=1000)
        plt.show()
        fig = fitter.svi_results.corner(color='C0') 
        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_corner_svi' + post_name + '.png'), bbox_inches='tight',  dpi = 1000)        # save results
        plt.show()
        fitter.svi_results.save_result(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_svi' + post_name + '.asdf'))
        plt.close()

    if do_sampling:
        # setup fitter
        if fit_multi:
            fitter = FitMulti(data=im, rms=sig, psf=psf, prior=prior, loss_func=student_t_loss)
        else:
            fitter = FitSingle(data=im, rms=sig ,mask=mask,psf=psf, prior=prior, loss_func=student_t_loss)
        # sampling
        fitter.sample(rkey = random.PRNGKey(0))
        sampling_res = fitter.sampling_results
        print("sampling results:", sampling_res)
        summary =  sampling_res.summary()
        print(summary)
        # mod_sampling = sampling_res.get_median_model()
        # print("residual of sampling:", filter.upper())
        # map_params = fitter.find_MAP(rkey = random.PRNGKey(0))
        # mod_sampling = map_params['model']
        # res = fitter.estimate_posterior(rkey = random.PRNGKey(1001), method='laplace')
        # summary = fitter.svi_results.summary()

        # commenting next section out bc bugging on GPUs

        dict = {}
        for a, b in zip(summary.index, summary["mean"]):
            dict[a] = b
        bf_model = HybridRenderer(im.shape, jnp.array(psf.astype(np.float32))).render_source(dict, profile_type="sersic")
        print(dict)

        # # mod_sampling = fitter.svi_results.get_median_model()
        # # save figures
        fig, ax = plot_residual(im, bf_model, vmin=-0.02, vmax=0.02)
        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_residual_sampling' + post_name + '.pdf'), bbox_inches='tight')
        fig = fitter.sampling_results.corner(color='C0') 
        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_corner_sampling' + post_name + '.pdf'), bbox_inches='tight')
        # # save results
        fitter.sampling_results.save_result(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_sampling' + post_name + '.asdf'))
        plt.close()

    print("=============================================================================")
    return phot_PA

def get_physical_params(params_q50):
    #params_q50 = ellip, flux, n, r_eff, theta, xc, yc
    #get the inclination, position angle, and effective radius
    ellip = params_q50[0]
    q0 = 0.2
    axis_ratio = float(np.maximum(1 - ellip, q0+0.05))
    # print((ellip,axis_ratio))
    inc = np.arccos(np.sqrt( (axis_ratio**2 - q0**2)/(1 - q0**2) ))*(180/np.pi)
    # print((1-axis_ratio**2)/(1-q0**2))
    # inc = np.arcsin((1-axis_ratio**2)/(1-q0**2))*(180/np.pi)


    theta = params_q50[4]
    print('theta', theta)
    PA = (theta-jnp.pi/2) * (180/jnp.pi) #convert to degrees
    if PA < 0:
        print('Converting pysersic PA from ', PA, ' to ', PA + 180, ' degrees')
        PA += 180
    elif PA > 180:
        print('Converting pysersic PA from ', PA, ' to ', PA - 180, ' degrees')
        PA -= 180

    r_eff = params_q50[3] #already in the right pixel size since measured in pixels on the image

    x_c = params_q50[5]
    y_c = params_q50[6] #already in right ref frame since measured in pixels on the image

    return inc, PA, r_eff, x_c, y_c

def get_params(path_output, id, filter, type = 'image'):
    #type can be either image, grism, or model
    af = asdf.open(os.path.join(path_output, str(id) + "_" + str(type) + "_" +  filter.upper() + "_svi.asdf"))
    params_single = list(af['posterior'].keys())
    print(params_single)
    params_q50 = []
    #get the medians of all the parameters from the fit
    for param in params_single: #in order: ellip, flux, n, r_eff, theta, xc, yc
        params_q50.append(np.percentile(np.concatenate(af['posterior'][param][:]), 50))
    
    #compute the inclination, position angle, and effective radius
    inc, PA, r_eff, x_c, y_c = get_physical_params(params_q50)
    return inc, PA, r_eff, x_c, y_c

def write_catalog(path_output, id, filter, type = 'image'):
    af = asdf.open(os.path.join(path_output, str(id) + "_" + str(type) + "_" + filter.upper() + "_svi.asdf"))
    
    ii =0
    params_single = list(af['posterior'].keys())
    all_params_single = [[i + "_q16", i + "_q50", i + "_q84"] for i in params_single]
    cat_col = np.append(["ID"], np.concatenate(all_params_single))
    t_empty = np.zeros((len(cat_col), 1))
    # t_empty[0, :] = str(id)
    t = Table(t_empty.T, names=cat_col)

    for ii_p in params_single:
        t[ii_p + "_q16"] = np.percentile(np.concatenate(af['posterior'][ii_p][:]), 16)
        t[ii_p + "_q50"] = np.percentile(np.concatenate(af['posterior'][ii_p][:]), 50)
        t[ii_p + "_q84"] = np.percentile(np.concatenate(af['posterior'][ii_p][:]), 84)

        # t['ID'] = t['ID'].astype(int)

        for ii in t.keys()[1:]:
            t[ii].info.format = '.3f'

    t.write(os.path.join(path_output, "summary_" + str(id) + "_" + str(type) + "_" + filter.upper() + "_svi.cat"), format='ascii', overwrite = True)

def run_pysersic_fit(filter, id, path_output, im, sig, psf, fit_multi=False, posterior_estimation=True, do_sampling=False, perform_masking=True, plot=True, type = 'image', sigma_rms = 3.0):
    phot_PA = fit_data(filter, id, path_output=path_output,  im = im, sig =sig, psf = psf,
             fit_multi=fit_multi, posterior_estimation=posterior_estimation, do_sampling=do_sampling, perform_masking=perform_masking, plot=plot, type = type, sigma_rms = sigma_rms)
    inc, PA, r_eff, x_c, y_c = get_params(path_output, id, filter, type = type)
    print('Fitting of ' + str(type) + ' completed. Best fit params: inclination = ' + str(inc) + ', position angle = ' + str(PA) + ', effective radius = ' + str(r_eff) + ', x_c = ' + str(x_c) + ', y_c = ' + str(y_c) + ', phot_PA = ' + str(phot_PA))
    write_catalog(path_output, id, filter, type = type)
    return inc, PA, phot_PA, r_eff, x_c, y_c

def prepare_grism(grism_path,wavelength):
    grism_spectrum_fits = fits.open(grism_path)
    #from 2D spectrum, extract wave_space (and the separate things like WRANGE, w_scale, and size), and aperture radius (to be used above)
    wave_first = grism_spectrum_fits['SPEC2D'].header['WAVE_1']
    d_wave = grism_spectrum_fits['SPEC2D'].header['D_WAVE']
    naxis_x = grism_spectrum_fits['SPEC2D'].header['NAXIS1']
    naxis_y = grism_spectrum_fits['SPEC2D'].header['NAXIS2']
    # print(wave_first, d_wave, naxis_x, naxis_y)

    wave_space = wave_first + jnp.arange(0, naxis_x, 1) * d_wave

    #calculate the delta_wave_cutoff such that the image is square
    delta_wave_cutoff = (naxis_y-1)*d_wave/2 
    #crop and continuum subtract the grism spectrum
    grism_spectrum = grism_spectrum_fits['SPEC2D'].data
    grism_spectrum_error = grism_spectrum_fits['WHT2D'].data
    obs_map, obs_error, index_min, index_max = preprocess.prep_grism(grism_spectrum,grism_spectrum_error, wavelength, delta_wave_cutoff, wave_first, d_wave)

    print('obs_map shape:', obs_map.shape, 'obs_error shape:', obs_error.shape)

    return obs_map, obs_error

def main_callum():

    IDs_run =  ['ZD4' ] 
    #done: 'ALT-41799', 'BULLSEYE2-E','BULLSEYE2-W', 'BULLSEYE3', 'BULLSEYE4','BULLSEYE5','YD1','YD4','YD6','YD7-E','YD7-W','YD8','z8-2','ZD1', 'ZD2','ZD1', 'ZD2','ZD3','ZD6', 'ZD12-W', 's1' ,'ZD4',
    # didn't work: 'PC6','ZD12-E',
    path_output = '/Users/lola/Downloads/CallumCutouts/'
    path_cutouts = '/Users/lola/Downloads/CallumCutouts/'
    
    filter_list = ['F150W']
    sigma_rms = 1

    psf_path = '/Users/lola/Downloads/CallumCutouts/PSF_NIRCam_in_flight_opd_filter_F150W.fits'
    psf = fits.getdata(psf_path)

    #crop the psf to 50x50 size
    psf_crop = psf[psf.shape[0]//2-25:psf.shape[0]//2+25, psf.shape[1]//2-25:psf.shape[1]//2+25]

    #resample the psf down to the pixel scale of the images
    psf_lowres = utils.resample(psf_crop, 5,5)


    for count,id in enumerate(IDs_run):
            
        try:

            #print the number of iterations we are at out of the total ones to run
            print('Running iteration ' + str(count) + ' out of ' + str(len(IDs_run)) + ' with ID ' + str(id))

                
            for filter in filter_list: #

                file = fits.open(path_cutouts + str(id) + '_' + filter + '_flux.fits')
                im = jnp.array(file[0].data)
                file_err = fits.open(path_cutouts + str(id) + '_' + filter + '_error.fits')
                wht = file_err[0].data


                
                # sig = 1/np.sqrt(wht)
                sig = jnp.array(wht)
                sig =jnp.where(np.isnan(im)| np.isnan(sig) | np.isinf(sig), 1e10, sig)	
                #crop the image to 15x15 pixels
                im = im[im.shape[0]//2-7:im.shape[0]//2+8, im.shape[1]//2-7:im.shape[1]//2+8]
                sig = sig[sig.shape[0]//2-7:sig.shape[0]//2+8, sig.shape[1]//2-7:sig.shape[1]//2+8]

                fit_data(filter, id, path_output=path_output,  im = im , sig =sig, psf = psf_lowres,
                        fit_multi=False, posterior_estimation=True, do_sampling=False, perform_masking=False, plot=True, sigma_rms = sigma_rms)
                inc, PA, r_eff, x_c, y_c = get_params(path_output, id, filter, type = 'image')
                # print(inc, PA, r_eff, x_c, y_c)
                write_catalog(path_output, id, filter, type = 'image')

        except Exception as e:
            print('Error in iteration ' + str(count) + ' with ID ' + str(id) + ' and filter ' + str(filter))
            print(e)
            continue


def main_lola_images():

    #load my catalog
    cat = Table.read('/Users/lola/ASTRO/JWST/grism_project/catalogs/Gold_Silver_Unres_FRESCO_CONGRESS.txt', format = 'ascii')

    IDs_run = cat['ID']
    RA = cat['RA']
    DEC = cat['DEC']
    field = cat['field']
    path_output = '/Users/lola/ASTRO/JWST/grism_project/PysersicFits_v1/'
    path_cutouts = '/Users/lola/ASTRO/JWST/grism_project/cutouts_v1/'

    filter_list = ['F150W']
    alternate_filter_list = ['F182M']
    sigma_rms = 1

    psf = None


    for count,id in enumerate(IDs_run):
            
        try:

            #print the number of iterations we are at out of the total ones to run
            print('Running iteration ' + str(count) + ' out of ' + str(len(IDs_run)) + ' with ID ' + str(id))

                
            for i,filter in enumerate(filter_list): 
            
                im_path = path_cutouts + str(id) + '_' + filter + '.fits'
                file = fits.open(im_path)
                im = jnp.array(file['SCI'].data)
                #check if the image file is all zero
                if np.all(im == 0):
                    filter = alternate_filter_list[i]
                    file = fits.open(path_cutouts + str(id) + '_' + filter + '.fits')
                    im = jnp.array(file['SCI'].data)
            

                err = jnp.array(file['ERR'].data)
                sig = err
                sig =jnp.where(np.isnan(im)| np.isnan(sig) | np.isinf(sig), 1e10, sig)	

                #load the psf
                if psf == None:
                    if field[count] == 'GOODS-S-FRESCO':
                        bithash_file = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gs/program_bithash.goodss.v1.0.0.fits'
                        psf_dir = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gs/'
                    else:
                        bithash_file = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gn/program_bithash.goodsn.v1.0.0.fits'
                        psf_dir = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gn/'

                    psf_path = utils.choose_mspf(bithash_file, psf_dir, RA[count], DEC[count], [im_path])[0]
                    psf = fits.getdata(psf_path)

                fit_data(filter, id, path_output=path_output,  im = im , sig =sig, psf = psf,
                        fit_multi=False, posterior_estimation=True, do_sampling=False, perform_masking=True, plot=True, sigma_rms = sigma_rms)
                
                inc, PA, r_eff, x_c, y_c = get_params(path_output, id, filter, type = 'image')
                # print(inc, PA, r_eff, x_c, y_c)
                write_catalog(path_output, id, filter, type = 'image')
        except Exception as e:
            print('Error in iteration ' + str(count) + ' with ID ' + str(id) + ' and filter ' + str(filter))
            print(e)
            continue



def main_lola_grism():

    #load my catalog
    cat = Table.read('/Users/lola/ASTRO/JWST/grism_project/catalogs/Gold_Silver_Unres_FRESCO_CONGRESS.txt', format = 'ascii')

    IDs_run = cat['ID']
    RA = cat['RA']
    DEC = cat['DEC']
    field = cat['field']
    z_spec = cat['z_spec']

    #calculate the Ha observed wavelength for each object
    wavelength = 0.65646 * (1 + z_spec) #in microns

    path_output = '/Users/lola/ASTRO/JWST/grism_project/PysersicFits_v1_grism/'
    path_cutouts = '/Users/lola/ASTRO/JWST/grism_project/fitting_results/'

    sigma_rms = 1


    for count,id in enumerate(IDs_run):
            
        try:

            #print the number of iterations we are at out of the total ones to run
            print('Running iteration ' + str(count) + ' out of ' + str(len(IDs_run)) + ' with ID ' + str(id))

            #load the psf
            if field[count] == 'GOODS-S-FRESCO':
                psf_path = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gs/mpsf_jw018950.gs.f444w.fits' #fresco
                filter = 'F444W'
                grism_str = 'FRESCO'
            elif field[count] == 'GDN-FRESCO':
                psf_path = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gn/mpsf_jw018950.gn.f444w.fits' #fresco
                filter = 'F444W'
                grism_str = 'GDN'
            else:
                psf_path = '/Users/lola/ASTRO/JWST/grism_project/mpsf_v1/gn/mpsf_jw035770.f356w.fits' #congress
                filter = 'F356W'
                grism_str = 'GDN'

            psf = fits.getdata(psf_path)

            #downsample it down to the grism resolution
            psf = utils.downsample_psf_centered(psf, size = 15)
            
            im_path = path_cutouts + str(id) + '/spec_2d_' + grism_str + '_' + filter + '_ID' + str(id) + '_comb.fits'
            im, err = prepare_grism(im_path, wavelength[count])

            sig = err
            sig =jnp.where(np.isnan(im)| np.isnan(sig) | np.isinf(sig), 1e10, sig)	

            fit_data(filter, id, path_output=path_output,  im = im , sig =sig, psf = psf,
                    fit_multi=False, posterior_estimation=True, do_sampling=False, perform_masking=True, plot=True, sigma_rms = sigma_rms, type = 'grism')
            
            inc, PA, r_eff, x_c, y_c = get_params(path_output, id, filter, type = 'grism')
            # print(inc, PA, r_eff, x_c, y_c)
            write_catalog(path_output, id, filter, type = 'grism')
        except Exception as e:
            print('Error in iteration ' + str(count) + ' with ID ' + str(id) + ' and filter ' + str(filter))
            print(e)
            continue


if __name__=="__main__":
    main_lola_grism()

