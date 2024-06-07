
# load modules

import os
import argparse
import numpy as np 
import glob
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve

from pysersic import check_input_data, FitSingle, FitMulti, PySersicMultiPrior
from pysersic.priors import autoprior
from pysersic.loss import student_t_loss
from pysersic.results import plot_image, plot_residual
import numpyro

from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D, MedianBackground

import corner 
from jax import random
import jax.numpy as jnp
from pysersic.rendering import HybridRenderer

import asdf 


# define key functions

def prepare_data(im, sig,psf, fit_multi=False, perform_masking=False, plot=False):
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
        return(None, None, None, None, None)


    # make segmentation map and identify sources
    im_conv = convolve(im, make_2dgaussian_kernel(1.0, size=3))
    bkg = Background2D(im_conv, (15, 15), filter_size=(3, 3), bkg_estimator=MedianBackground(), exclude_percentile=99.0)
    segment_map = detect_sources(im_conv, 3.0*bkg.background_rms, npixels=10)
    segm_deblend = deblend_sources(im_conv, segment_map, npixels=10, nlevels=32, contrast=1, progress_bar=False)
    source_cat = SourceCatalog(im, segm_deblend, convolved_data=im_conv, error=sig)
    source_tbl = source_cat.to_table()

    # identify main label
    main_label = segm_deblend.data[int(0.5*im.shape[0]), int(0.5*im.shape[1])]

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

    else:
        mask = np.zeros(im.shape)
    
    # plot data
    if plot:
        fig, ax = plot_image(im, mask, sig, psf)
        plt.show()

    # check data
    if check_input_data(data=im, rms=sig, psf=psf, mask=mask):
        print("data looks good!")
        return(im, mask, sig, psf, to_pysersic)

    else:
        print("data looks bad!")
        return(None, None, None, None, None)



def fit_data(filter, id, path_output=None, im = None, sig = None, psf = None,
             fit_multi=False, posterior_estimation=False, do_sampling=False, perform_masking=False, plot=False, type = 'image'):

    print("=============================================================================")

    # load data
    # print("load data for filter", filter.upper(), " and galaxy", id)
    im, mask, sig, psf, to_pysersic = prepare_data(im, sig,psf,fit_multi=fit_multi, perform_masking=perform_masking, plot=plot)

    # abort if no data available
    if im is None:
        return()

    # get priors
    if fit_multi:
        prior = PySersicMultiPrior(catalog=to_pysersic, sky_type='none')
    else:
        prior  = autoprior(image=im, profile_type='sersic', mask=mask, sky_type='none')
        # prior.set_uniform_prior('n', 0.5, 4.5)
    print(prior)


    if fit_multi:
        post_name = '_multi'
    else:
        post_name = ''

    if posterior_estimation:
        # setup fitter
        if fit_multi:
            fitter = FitMulti(data=im, rms=sig, psf=psf, prior=prior, loss_func=student_t_loss)
        else:
            fitter = FitSingle(data=im, rms=sig ,mask=mask,psf=psf, prior=prior, loss_func=student_t_loss)
        # posterior estimation
        fitter.estimate_posterior(method='svi-flow')
        mod = fitter.svi_results.get_median_model()
        print("residual of posteriorr estimation:", filter.upper())
        # save figures
        fig, ax = plot_residual(im, mod[1], mask=mask, vmin=-0.02, vmax=0.02)
        plt.savefig(os.path.join(path_output, id + '_' + filter.upper() + '_residual_svi' + post_name + '.pdf'), bbox_inches='tight')
        fig = fitter.svi_results.corner(color='C0') 
        plt.savefig(os.path.join(path_output, id + '_' + filter.upper() + '_corner_svi' + post_name + '.pdf'), bbox_inches='tight')
        # save results
        fitter.svi_results.save_result(os.path.join(path_output, id + '_' + filter.upper() + '_svi' + post_name + '.asdf'))

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
        # mod_sampling = sampling_res.get_median_model()
        # print("residual of sampling:", filter.upper())
        # map_params = fitter.find_MAP(rkey = random.PRNGKey(0))
        # mod_sampling = map_params['model']
        # res = fitter.estimate_posterior(rkey = random.PRNGKey(1001), method='laplace')
        # summary = fitter.svi_results.summary()
        dict = {}
        for a, b in zip(summary.index, summary["mean"]):
            dict[a] = b
        bf_model = HybridRenderer(im.shape, jnp.array(psf.astype(np.float32))).render_source(dict, profile_type="sersic")
        # mod_sampling = fitter.svi_results.get_median_model()
        # save figures
        fig, ax = plot_residual(im, bf_model, mask=mask, vmin=-0.02, vmax=0.02)
        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_residual_sampling' + post_name + '.pdf'), bbox_inches='tight')
        fig = fitter.sampling_results.corner(color='C0') 
        plt.savefig(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_corner_sampling' + post_name + '.pdf'), bbox_inches='tight')
        # save results
        fitter.sampling_results.save_result(os.path.join(path_output, str(id) + '_' + str(type) + '_' + filter.upper() + '_sampling' + post_name + '.asdf'))

    print("=============================================================================")

def get_physical_params(params_q50):
    #params_q50 = ellip, flux, n, r_eff, theta, xc, yc
    #get the inclination, position angle, and effective radius
    ellip = params_q50[0]
    inc = jnp.acos(1 - ellip)* (180/jnp.pi)

    theta = params_q50[4]
    if theta < 0:
        theta += jnp.pi
    PA = theta * (180/jnp.pi) #convert to degrees

    r_eff = params_q50[3] #already in the right pixel size since measured in pixels on the image

    x_c = params_q50[5]
    y_c = params_q50[6] #already in right ref frame since measured in pixels on the image

    return inc, PA, r_eff, x_c, y_c

def get_params(path_output, id, filter, type = 'image'):
    #type can be either image, grism, or model
    af = asdf.open(os.path.join(path_output, str(id) + "_" + str(type) + "_" +  filter.upper() + "_sampling.asdf"))
    params_single = list(af['posterior'].keys())
    params_q50 = []
    #get the medians of all the parameters from the fit
    for param in params_single: #in order: ellip, flux, n, r_eff, theta, xc, yc
        params_q50.append(np.percentile(np.concatenate(af['posterior'][param][:]), 50))
    
    #compute the inclination, position angle, and effective radius
    inc, PA, r_eff, x_c, y_c = get_physical_params(params_q50)
    return inc, PA, r_eff, x_c, y_c

def write_catalog(path_output, id, filter, type = 'image'):
    af = asdf.open(os.path.join(path_output, str(id) + "_" + str(type) + "_" + filter.upper() + "_sampling.asdf"))
    
    ii =0
    params_single = list(af['posterior'].keys())
    all_params_single = [[i + "_q16", i + "_q50", i + "_q84"] for i in params_single]
    cat_col = np.append(["ID"], np.concatenate(all_params_single))
    t_empty = np.zeros((len(cat_col), 1))
    t_empty[0, :] = str(id)
    t = Table(t_empty.T, names=cat_col)

    for ii_p in params_single:
        t[ii_p + "_q16"] = np.percentile(np.concatenate(af['posterior'][ii_p][:]), 16)
        t[ii_p + "_q50"] = np.percentile(np.concatenate(af['posterior'][ii_p][:]), 50)
        t[ii_p + "_q84"] = np.percentile(np.concatenate(af['posterior'][ii_p][:]), 84)

        t['ID'] = t['ID'].astype(int)

        for ii in t.keys()[1:]:
            t[ii].info.format = '.3f'

    t.write(os.path.join(path_output, "summary_" + str(id) + "_" + str(type) + "_" + filter.upper() + "_sampling.cat"), format='ascii', overwrite = True)

def main():

    print('Starting gal test')
    
    filter = 'F356W'
    path_output = '/Users/lola/ASTRO/JWST/grism_project/CONGRESS_FRESCO/'

    im = fits.getdata('/Users/lola/ASTRO/JWST/grism_project/fitting_results/24065_bounds/24065_cutout.fits',0)
    wht = fits.getdata('/Users/lola/ASTRO/JWST/grism_project/fitting_results/24065_bounds/24065_wht_cutout.fits',0)
    sig = 1/np.sqrt(wht)
    sig =np.where(np.isnan(im)| np.isnan(sig) | np.isinf(sig), 1e10, sig)	
    id = 24065
    psf = fits.getdata('/Users/lola/ASTRO/JWST/grism_project/gdn_mpsf_F356W_small.fits',0)
    c = im.shape[0]//2
    fit_data(filter, id, path_output=path_output,  im = im[c-40:c+40, c-40:c+40] , sig =sig[c-40:c+40, c-40:c+40], psf = psf,
             fit_multi=False, posterior_estimation=False, do_sampling=True, perform_masking=True, plot=True)
    inc, PA, r_eff, x_c, y_c = get_params(path_output, id, filter, type = 'image')
    print(inc, PA, r_eff, x_c, y_c)
    write_catalog(path_output, id, filter, type = 'image')



if __name__=="__main__":
    print('hi')
    main()

