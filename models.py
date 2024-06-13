"""

	Module holding all of the kinematic models used in the fitting process.

	Written by A L Danhaive: ald66@cam.ac.uk
"""


# imports
import numpy as np
# geko related imports
import utils
import plotting

# jax and its functions
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.signal import convolve
# from scipy.signal import convolve
from jax import image

from skimage.morphology import dilation, disk

# scipy and its functions
from scipy.constants import pi
from scipy.ndimage import measurements

# numpyro and its functions
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam, CircularReparam

from matplotlib import pyplot as plt

from photutils import centroids

import time

from scipy.constants import c


class KinModels:

    def compute_factors(self, PA, i, x, y):

        self.cos_PA = jnp.cos(PA)
        self.sin_PA = jnp.sin(PA)
        self.cos_i = jnp.cos(i)
        self.xint = self.x_int(x, y, PA, i)
        self.rad_int = self.r_int(x, y, PA, i)


    def __init__(self):
        print('New kinematic model created')

    def x_int(self, x, y, PA, i):
        return x*self.cos_PA - y*self.sin_PA

    def y_int(self, x, y, PA, i):
        return jnp.where(self.cos_i != 0., (x*self.sin_PA + y*self.cos_PA)/(self.cos_i), 0.)

    def r_int(self, x, y, PA, i):

        return jnp.sqrt(self.xint**2 + (self.y_int(x, y, PA, i))**2)

    def phi_int(self, x, y, PA, i):
        return jnp.where(self.rad_int != 0, jnp.arccos(self.xint/self.rad_int), 0.)

    def v(self, x, y, PA, i, Va, r_t):
        return (2/pi)*Va*jnp.arctan(self.rad_int/r_t)*jnp.sin(i)*jnp.cos(self.phi_int(x, y, PA, i))

    def flux(self, x, y, r_eff, I0, PA, i):
        fluxes = I0*jnp.exp(-self.rad_int/r_eff)
        # fluxes = jnp.where(r_int(x, y, PA, i) > r_eff*1.5, 0.0, fluxes)
        return fluxes

    def sigma_const(self, x, y, sigma0, r_eff=None, I0=None, PA=None, i=None):
        return sigma0

    def gaussian(self, sigma0_max, sigma0_scale, velocities, sigma0_const):
        return sigma0_max*(1/(jnp.sqrt(2*pi)*sigma0_scale))*jnp.exp(-0.5*(velocities)/sigma0_scale**2) + sigma0_const
    
    def logistic(self, sigma0_max, sigma0_const, sigma0_scale, fluxes):
        return (sigma0_max - sigma0_const)/(1 + jnp.exp(-sigma0_scale*(fluxes - jnp.median(fluxes))))
    
    def sigma_disk(self, sigma0_max, sigma0_scale, sigma0_const, fluxes):
        # grad_x, grad_y = jnp.gradient(velocities)
        # center = jnp.argmax(jnp.sqrt(grad_y**2 + grad_x**2))
        # center = jnp.unravel_index(center, velocities.shape)
        # velocites_center = velocities[center[0], center[1]]
        dispersions = self.logistic(sigma0_max, sigma0_scale, sigma0_const, fluxes)
        # dispersions = self.gaussian(sigma0_max, sigma0_scale, jnp.abs(velocities - velocites_center), sigma0_const)

        return dispersions


    def set_main_bounds(self, flux_prior, flux_error, broad_band, flux_bounds, flux_type,flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds,
                         sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph,inclination,r_eff, r_eff_grism,
                        delta_V_bounds, clump, clump_v_prior, clump_sigma_prior, clump_flux_prior):
        """
        Set the bounds for the model parameters by reading the ones from the config file.
        The more specific bounds computations for the different models will be done inside their
        class.
        """
        self.Laplace_kernel = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        self.flux_prior = flux_prior
        self.flux_error = flux_error
        # this is the image used to compute pixel mask of all ELs
        self.broad_band = broad_band 
        # these first two are in the form (scale, high)
        self.flux_bounds = flux_bounds
        self.flux_type = flux_type
        self.flux_threshold = flux_threshold
        self.PA_sigma = PA_sigma
        self.PA_grism = PA_grism
        self.PA_morph = PA_morph
        self.inclination = inclination
        self.r_eff = r_eff
        self.r_eff_grism = r_eff_grism
        # these are in the form (low, high)
        self.i_bounds = i_bounds
        self.Va_bounds = Va_bounds
        self.r_t_bounds = r_t_bounds
        self.sigma0_bounds = sigma0_bounds
        self.clump = clump
        self.delta_V_bounds = delta_V_bounds

        self.y_factor = y_factor

        self.clump_v_prior = clump_v_prior
        self.clump_sigma_prior = clump_sigma_prior
        self.clump_flux_prior = clump_flux_prior

        self.x0 = x0
        self.y0 = y0

        self.x0_vel = x0_vel
        self.mu_y0_vel = y0_vel

        if self.mu_y0_vel == None:
            self.x0_vel = x0
            self.mu_y0_vel = y0

    def compute_flux_bounds(self):
        """
            Compute the flux bounds for the model. 
            If the fluxes are negative, set their prior to zero
        """
        # self.mu = jnp.maximum(
        #     jnp.zeros(self.flux_prior.shape), jnp.array(self.flux_prior))
        # self.std = jnp.maximum(0.000001, self.flux_bounds[0]*self.mu)
        # self.high = (jnp.maximum(
        #     0.000002, self.flux_bounds[1] * self.mu) + self.mu - self.mu)/self.std
        # self.low = (jnp.zeros(self.flux_prior.shape)-self.mu)/self.std
        # # self.high = jnp.maximum(0.000002, self.flux_bounds[1] * self.mu) + self.mu
        # # self.low = jnp.zeros(self.flux_prior.shape)

        #take prior and errors fully from image, and no bounds
        #keep errors but set mean to zero if negative
        self.mu = jnp.maximum(jnp.zeros(self.flux_prior.shape), jnp.array(self.flux_prior))
        self.std = self.flux_error
        self.high = None
        self.low = (jnp.zeros(self.flux_prior.shape) - self.mu)/self.std
    def rescale_to_mask(self, array, mask):
        """
            Rescale the bounds to the mask
        """
        rescaled_array = []
        for a in array:
            a = a[jnp.where(mask == 1)]
            rescaled_array.append(a)
        return rescaled_array

@numpyro.handlers.reparam(
    config={"PA_radians": CircularReparam()} #, "i_radians": CircularReparam()} #, "y0_vel": TransformReparam()}
)

class Disk():
    """
        Class for 1 disk object. Combinations of this will be used for the single disk model, 
        then 2 disks for the 2 component ones etc
    """
    def __init__(self, direct_shape, masked_indices, mu, std, low, high, mu_PA, sigma_PA, i_bounds, mu_i, Va_bounds, r_t_bounds, sigma0_bounds, x0_vel, mu_y0_vel, y_high, y_low, y0_std, r_eff, r_eff_obs, number = ''):
        print('Disk object created')

        #initialize all attributes with function parameters
        self.direct_shape = direct_shape
        self.masked_indices = masked_indices

        self.mu = mu
        self.std = std
        self.low = low 
        self.high = high

        self.mu_PA = mu_PA
        self.sigma_PA = sigma_PA

        self.i_bounds = i_bounds
        self.mu_i = mu_i
        self.sigma_i = 10 #set an error of 50% for inclination
        #has to be rescaled for the normal distribution sampling
        self.i_low = (i_bounds[0]-self.mu_i)/self.sigma_i
        self.i_high = (i_bounds[1]-self.mu_i)/self.sigma_i

        self.Va_bounds = Va_bounds
        self.r_t_bounds = r_t_bounds
        self.sigma0_bounds = sigma0_bounds

        self.x0_vel = x0_vel
        self.mu_y0_vel = mu_y0_vel

        self.y_low = y_low
        self.y_high = y_high

        self.y0_std = y0_std

        self.r_eff = r_eff
        self.r_eff_obs = r_eff_obs
        self.r_t_bounds = r_t_bounds  #[0.0, r_eff_obs] #set the r_t bounds to the observed r_eff, we don't expect r_t/r_e to be > 1!
        self.number = number



    def sample_fluxes(self):
        # f = open("timing_total.txt", "a")
        #sample the fluxes within the mask
        fluxes_scaling = numpyro.sample('fluxes_scaling' + self.number, dist.Uniform())*(2-0.05) + 0.05
        # unscaled_fluxes_sample = numpyro.sample('unscaled_fluxes'+ self.number, dist.TruncatedNormal(jnp.zeros(int(len(self.masked_indices[0]))),jnp.ones(int(len(self.masked_indices[0]))),low = low, high = high), sample_shape=(int(len(self.masked_indices[0])),))
        # fluxes_sample = numpyro.deterministic('fluxes', unscaled_fluxes_sample*self.std + self.mu*fluxes_scaling)
        # fluxes_sample = numpyro.sample('unscaled_fluxes'+ self.number, dist.Uniform(), sample_shape=(int(len(self.masked_indices[0])),))
        # fluxes_sample = norm.ppf(norm.cdf(self.low) + fluxes_sample*(norm.cdf(self.high)-norm.cdf(self.low)))*self.std + self.mu*fluxes_scaling
        # reparam_config = {"fluxes": TransformReparam()}
        # with numpyro.handlers.reparam(config=reparam_config):
        #     fluxes_sample = numpyro.sample("fluxes",dist.TransformedDistribution(
        #             dist.TruncatedNormal(0.0, 1.0, low=low, high=high),
        #             AffineTransform(self.mu*fluxes_scaling, self.std*fluxes_scaling),
        #         ),
        #     )
        fluxes_error_scaling = numpyro.sample('fluxes_error_scaling' + self.number, dist.Uniform())*9 + 1
        #just sample from normal distribution
        # unscaled_fluxes = numpyro.sample('unscaled_fluxes'+ self.number, dist.Normal(jnp.zeros(int(len(self.masked_indices[0]))),jnp.ones(int(len(self.masked_indices[0]))))  )
        # fluxes_sample = numpyro.deterministic('fluxes', unscaled_fluxes*self.std*fluxes_error_scaling + self.mu*fluxes_scaling)  
        # fluxes_sample = numpyro.sample('fluxes', dist.TruncatedNormal(self.mu*fluxes_scaling, self.std*fluxes_scaling, low = self.low))

        low = (jnp.zeros(self.mu.shape) - self.mu*fluxes_scaling)/(self.std*fluxes_error_scaling)

        reparam_config = {"fluxes": TransformReparam()}
        with numpyro.handlers.reparam(config=reparam_config):
            fluxes_sample = numpyro.sample("fluxes",dist.TransformedDistribution(
                    dist.TruncatedNormal(0.0, 1.0, low=low),
                    AffineTransform(self.mu*fluxes_scaling, self.std*fluxes_error_scaling),
                ),
            )

        # f.write('fluxes_sample' + str(fluxes_sample[0]) +' \n')
        fluxes = jnp.zeros(self.direct_shape)
        fluxes = fluxes.at[self.masked_indices].set(fluxes_sample)

        return fluxes

    def sample_params(self):
        """
            Sample all of the parameters needed to model a disk velocity field
        """
        # f = open("timing_total.txt", "a")
        # start = time.time()
        # unscaled_Pa = numpyro.sample('unscaled_PA'+ self.number, dist.Normal())
        # sample the mu_PA + 0 or 180 (orientation of velocity field)
        # rotation = numpyro.sample('rotation'+ self.number, dist.Uniform())

        # simulate a bernouilli discrete distribution
        # PA_morph = self.mu_PA + round(rotation)*180
        # Pa = numpyro.deterministic('PA', unscaled_Pa*self.sigma_PA + PA_morph)
        # Pa = numpyro.sample('Pa'+ self.number, dist.Uniform())
        # Pa = norm.ppf(Pa)*self.sigma_PA + PA_morph

        #sample from circular normal distribution (in radians)


        # Pa_rad = numpyro.sample('PA_radians' + self.number, dist.VonMises(self.mu_PA*jnp.pi/180,1/((self.sigma_PA*jnp.pi/180)**2))) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
        # Pa = numpyro.deterministic('PA' + self.number, Pa_rad*180/jnp.pi)
        Pa_rad = numpyro.sample('PA_radians' + self.number, dist.Normal()) #self.mu_PA*jnp.pi/180, 1/((self.sigma_PA*jnp.pi/180)**2)
        Pa = numpyro.deterministic('PA' + self.number, Pa_rad*self.sigma_PA + self.mu_PA)
        # end = /time.time()
        # f.write('PA sampling time: '+ str(end-start)+ '\n')
        # Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA

        #could probably use utils for this too
        # start = time.time()
        unscaled_i = numpyro.sample('unscaled_i' + self.number, dist.TruncatedNormal(low = self.i_low, high = self.i_high))
        # i = norm.ppf(norm.cdf(self.i_low) + i*(norm.cdf(self.i_high)-norm.cdf(self.i_low)))*self.sigma_i + self.mu_i #arboitray sigma_i=10 for now
        i = numpyro.deterministic('i', unscaled_i*self.sigma_i + self.mu_i)
        # end = time.time()
        # f.write('i sampling time: '+ str(end-start)+ '\n')

        #sample from circular normal distribution (in radians)
        # i_rad = numpyro.sample('i_radians' + self.number, dist.VonMises(self.mu_i*jnp.pi/180, 1/((self.sigma_i*jnp.pi/180)**2)))
        # i = numpyro.deterministic('i' + self.number, i_rad*180/jnp.pi)

        # start = time.time()
        Va = numpyro.sample('Va' + self.number, dist.Uniform()) * \
            (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]
        # end = time.time()
        # f.write('Va sampling time: '+ str(end-start)+ '\n')

        # start = time.time()
        # r_t = numpyro.sample('r_t' + self.number, dist.Uniform(
        # ))*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

        r_t_sigma = jnp.log(self.r_t_bounds[1]/2)
        r_t_high = (jnp.log(self.r_t_bounds[2]) - jnp.log(self.r_t_bounds[1]))/r_t_sigma
        reparam_config = {"log_r_t": TransformReparam()}
        with numpyro.handlers.reparam(config=reparam_config):
            log_r_t = numpyro.sample("log_r_t",dist.TransformedDistribution(
                    dist.TruncatedNormal(0.0, 1.0, high=r_t_high),
                    AffineTransform(jnp.log(self.r_t_bounds[1]), r_t_sigma),
                ),
            )
        r_t = jnp.exp(log_r_t)
        # end = time.time()
        # f.write('r_t sampling time: '+ str(end-start)+ '\n')

        # start = time.time()
        sigma0 = numpyro.sample('sigma0'+ self.number, dist.Uniform(
        ))*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
        # end = time.time()
        # f.write('sigma0 sampling time: '+ str(end-start)+ '\n')

        # sigma0_max = numpyro.sample('sigma0_max'+ self.number, dist.Uniform())*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]
        # sigma0_scale =  numpyro.sample('sigma0_scale'+ self.number, dist.Uniform())*10
        # sigma0_const = numpyro.sample('sigma0_const'+ self.number, dist.Uniform())*sigma0_max

        # sampling the y axis velocity centroids
        # x0 = numpyro.sample('x0', dist.Uniform())
        # x0 = norm.ppf(x0)*(2) + self.x0

        # start = time.time()
        # unscaled_y0_vel = numpyro.sample('unscaled_y0_vel'+ self.number, dist.Normal())
        # # y0_vel = norm.ppf(y0_vel)*(2) + self.y0_vel
        # y0_vel = numpyro.deterministic('y0_vel', unscaled_y0_vel*2 + self.y0_vel)

        y0_vel = numpyro.sample("y0_vel",dist.TruncatedNormal(self.mu_y0_vel, self.y0_std, low=self.y_low, high=self.y_high ))


        # end = time.time()
        # f.write('y0_vel sampling time: '+ str(end-start)+ '\n')
        # sample a global velicity shift v0:
        # start = time.time()
        unscaled_v0 = numpyro.sample('unscaled_v0'+ self.number, dist.Normal())
        # v0 = norm.ppf(v0)*100
        v0 = numpyro.deterministic('v0', unscaled_v0*100)
        # end = time.time()
        # f.write('v0 sampling time: '+ str(end-start)+ '\n')

        return Pa, i, Va, r_t,sigma0, y0_vel, v0, 
        # return Pa, i, Va, r_t,sigma0_max, sigma0_scale, sigma0_const, y0_vel, v0, 
    
    def compute_posterior_means(self, inference_data):
        """
            Retreive the best sample from the MCMC chains for the main disk variables
        """
        best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
        ), inference_data['sample_stats']['lp'].shape)

        # rescale all of the posteriors from uniform to the actual parameter space
        # rotation = float(inference_data.posterior['rotation' + self.number].median(dim=["chain", "draw"]))
        #create lists with variables and their scaling parameters 
        # variables = ['PA'+ self.number, 'i'+ self.number, 'Va'+ self.number, 'r_t'+ self.number, 'sigma0'+ self.number, 'y0_vel'+ self.number, 'v0'+ self.number]
        variables = [ 'Va'+ self.number, 'r_t'+ self.number, 'sigma0'+ self.number]
        # variables = ['PA' + self.number, 'i'+ self.number, 'Va'+ self.number, 'r_t'+ self.number, 'sigma0_max'+ self.number,'sigma0_scale'+ self.number , 'sigma0_const'+ self.number, 'y0_vel'+ self.number, 'v0'+ self.number]
        #for variables drawn from uniform dist, the scaling parameters are (low, high) so mu and sigma are set to none
        # mus = [self.mu_PA + round(rotation)*180, self.mu_i, None, None, None, self.y0_vel, 0.0]
        # sigmas = [self.sigma_PA, self.sigma_i, None, None, None,2.0, 100.0]
        #when using numpyro.deterministic, don't need to change posteriors
        mus = [ None, None, None]
        sigmas = [ None, None, None]

        #for variables drawn from normal dist, the scaling parameters are (mu, sigma) so low and high are set to none
        # highs = [None, self.i_high, self.Va_bounds[1], self.r_t_bounds[1], self.sigma0_bounds[1],None, None]
        # lows = [None,self.i_low, self.Va_bounds[0], self.r_t_bounds[0], self.sigma0_bounds[0], None, None]
        highs = [self.Va_bounds[1], self.r_t_bounds[1], self.sigma0_bounds[1]]
        lows = [self.Va_bounds[0], self.r_t_bounds[0], self.sigma0_bounds[0]]

        #find the best sample for each variable in the list of variables
        # best_sample = utils.find_best_sample(inference_data, variables, mus, sigmas, highs, lows, best_indices)
        #taking the median:
        best_sample = utils.find_best_sample(inference_data, variables, mus, sigmas, highs, lows, MLS = None)


        self.Va_mean, self.r_t_mean, self.sigma0_mean_model = best_sample
        # self.sigma0_mean_model*=0.5
        # self.Va_mean*=0.05
                # self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean, self.y0_vel_mean, self.v0_mean = best_sample

        #taking best sample
        # self.PA_mean =  jnp.array(inference_data.posterior['PA'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        # self.i_mean = jnp.array(inference_data.posterior['i'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        # self.y0_vel_mean = jnp.array(inference_data.posterior['y0_vel'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        # self.v0_mean = jnp.array(inference_data.posterior['v0'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))

        #taking the median:
        self.PA_mean = jnp.array(inference_data.posterior['PA'+ self.number].median(dim=["chain", "draw"]))
        self.i_mean = jnp.array(inference_data.posterior['i'+ self.number].median(dim=["chain", "draw"]))
        self.y0_vel_mean = jnp.array(inference_data.posterior['y0_vel'+ self.number].median(dim=["chain", "draw"]))
        self.v0_mean = jnp.array(inference_data.posterior['v0'+ self.number].median(dim=["chain", "draw"]))

        # self.PA_mean = 180 - self.PA_mean
                # self.Va_mean = jnp.array(inference_data.posterior['Va'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        # self.r_t_mean = jnp.array(inference_data.posterior['r_t'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        # self.sigma0_mean_model = jnp.array(inference_data.posterior['sigma0'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))


        return  self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean
        # return  self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean,  self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean, self.y0_vel_mean, self.v0_mean

    def compute_flux_posterior(self, inference_data, flux_type):

        best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
        ), inference_data['sample_stats']['lp'].shape)

        inference_data.posterior['fluxes_scaling'+ self.number].data = inference_data.posterior['fluxes_scaling'+ self.number].data**(2-0.05) + 0.05 #*(1-0.1) + 0.1
        inference_data.prior['fluxes_scaling'+ self.number].data = inference_data.prior['fluxes_scaling'+ self.number].data**(2-0.05) + 0.05 #*(1-0.1) + 0.1

        inference_data.posterior['fluxes_error_scaling'+ self.number].data = inference_data.posterior['fluxes_error_scaling'+ self.number].data*4 + 1
        inference_data.prior['fluxes_error_scaling'+ self.number].data = inference_data.prior['fluxes_error_scaling'+ self.number].data*4 + 1

        inference_data.posterior['regularization_strength'+ self.number].data = inference_data.posterior['regularization_strength'+ self.number].data*(1-0.01) + 0.01 #*(1-0.1) + 0.1
        inference_data.prior['regularization_strength'+ self.number].data = inference_data.prior['regularization_strength'+ self.number].data*(1-0.01) + 0.01  #*(1-0.1) + 0.1
        #if the fluxes are manually rescaled in the prior, then rescale them
        # self.fluxes_scaling_mean = jnp.array(inference_data.posterior['fluxes_scaling'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        self.fluxes_scaling_mean = jnp.array(inference_data.posterior['fluxes_scaling'+ self.number].median(dim=["chain", "draw"]))
        self.fluxes_error_scaling_mean = jnp.array(inference_data.posterior['fluxes_error_scaling'+ self.number].median(dim=["chain", "draw"]))
        print(flux_type)
        if flux_type == 'manual':
            best_flux_sample = utils.find_best_sample(inference_data, ['fluxes'+ self.number], [self.mu*self.fluxes_scaling_mean], [self.std], [self.high], [self.low], best_indices)

        # self.fluxes_sample_mean = jnp.array(inference_data.posterior['fluxes'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        self.fluxes_sample_mean = jnp.array(inference_data.posterior['fluxes'+ self.number].median(dim=["chain", "draw"]))

        self.fluxes_mean = jnp.zeros(self.direct_shape)
        self.fluxes_mean =self.fluxes_mean.at[self.masked_indices].set(self.fluxes_sample_mean)

        # self.alpha_mean = jnp.array(inference_data.posterior['regularization_strength'].isel(chain=best_indices[0], draw=best_indices[1]))
        self.alpha_mean = jnp.array(inference_data.posterior['regularization_strength'+ self.number].median(dim=["chain", "draw"]))
        print('Alpha mean: ', self.alpha_mean)
        return self.fluxes_mean, self.fluxes_scaling_mean
    
    def v_rot(self, fluxes_mean, model_velocities, i_mean,factor):
        """
            Compute the rotational velocity of the disk component

            If called from multiple component model, the 3 attributes of this function should be only from that component
        """
        threshold = 0.4*fluxes_mean.max()
        mask = jnp.zeros_like(fluxes_mean)
        mask = mask.at[jnp.where(fluxes_mean>threshold)].set(1)
        model_velocities_low = jax.image.resize(model_velocities, (int(model_velocities.shape[0]/factor), int(model_velocities.shape[1]/factor)), method='nearest')
        model_v_rot = 0.5*(jnp.nanmax(jnp.where(mask == 1, model_velocities_low, jnp.nan)) - jnp.nanmin(jnp.where(mask == 1, model_velocities_low, jnp.nan)))/ jnp.sin( jnp.radians(i_mean)) 
        plt.imshow(jnp.where(mask ==1, fluxes_mean, np.nan), origin = 'lower')
        plt.title('Mask for v_rot comp')
        plt.show()
        return model_v_rot

    def plot(self):

        """
            Plot the disk model
        """
        
        #plot the fluxes within the mask and the velocity centroid
        fluxes = jnp.zeros(self.direct_shape)
        fluxes = fluxes.at[self.masked_indices].set(self.mu)
        plt.imshow(fluxes, origin='lower')
        plt.colorbar()
        plt.scatter(self.x0_vel, self.mu_y0_vel, color='red')
        plt.title('Disk' + self.number)
        plt.show()


class DiskModel_withPrior(KinModels):
    """
        Class for the one component exponential disk model
    """

    def __init__(self):
        print('Disk model with prior created')

        #declare var and label names for plotting

        self.var_names = ['PA', 'i', 'Va', 'r_t', 'sigma0']
        self.labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$']
        # self.var_names = ['PA', 'i', 'Va', 'r_t', 'sigma0_max', 'sigma0_scale', 'sigma0_const']
        # self.labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_{max}$', r'$\sigma_{scale}$', r'$\sigma_{const}$']

    def set_bounds(self, flux_prior, flux_bounds, flux_type, flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel):
        """

        Compute all of the necessary bounds for the disk model sampling distributions

        """
        # first set all of the main bounds taken from the config file
        self.set_main_bounds(flux_prior, flux_bounds, flux_type, flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel,
                             delta_V_bounds=None, clump=None, clump_v_prior=None, clump_sigma_prior=None, clump_flux_prior=None)

        # now compute the specific bounds for the disk model
        self.sigma_PA = self.PA_sigma*90

        self.compute_flux_bounds()

        #make the mask and compute the PA and inclination
        seg_1comp, seg_none , mask, mask_none, PA, inc = utils.compute_gal_props(self.flux_prior, threshold_sigma = self.flux_threshold)

        self.mask = mask
        self.mu_PA = 25 #PA[0] #hardcoding for now
        self.mu_i = 45
        self.mu_Va = 165
        self.mu_r_t = 5.7
        self.mu_sigma0 = 52

        self.mask = jnp.array(self.mask)
        self.mask_shape = len(jnp.where(self.mask == 1)[0])
        self.masked_indices = jnp.where(self.mask == 1)
        self.mu, self.std, self.high, self.low = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask)
        #initialize the disk object
        self.disk = Disk(self.flux_prior.shape, self.masked_indices, self.mu, self.std, self.low, self.high, 
                    self.mu_PA, self.sigma_PA, self.i_bounds, self.mu_i, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel, self.y0_vel)
        
        self.disk.plot()

    def inference_model(self, grism_object, obs_map, obs_error, low, high):
        """

        Model used to infer the disk parameters from the data => called in fitting.py as the forward
        model used for the inference

        """
        # Pa, i, Va, r_t, sigma0, y0_vel, v0 = self.disk.sample_params()
        unscaled_Pa = numpyro.sample('unscaled_PA', dist.Normal())
        # rotation = numpyro.sample('rotation'+ self.number, dist.Uniform())

        PA_morph = self.mu_PA #no need for a rotation because assuming same as Halpha

        Pa = numpyro.deterministic('PA', unscaled_Pa*self.sigma_PA + PA_morph)

        unscaled_i = numpyro.sample('unscaled_i', dist.TruncatedNormal(low = (0-self.mu_i)/10, high = (90-self.mu_i)/10))
        i = numpyro.deterministic('i', unscaled_i*10 + self.mu_i)


        unscaled_Va = numpyro.sample('unscaled_Va', dist.TruncatedNormal(low = (0-self.mu_Va)/100, high = (1000-self.mu_Va)/100))
        Va = numpyro.deterministic('Va', unscaled_Va*100 + self.mu_Va)

        unscaled_r_t = numpyro.sample('unscaled_r_t', dist.TruncatedNormal(low = (0-self.mu_r_t), high = (10-self.mu_r_t)))
        r_t = numpyro.deterministic('r_t', unscaled_r_t*1 + self.mu_r_t)

        unscaled_sigma0 = numpyro.sample('unscaled_sigma0', dist.TruncatedNormal(low = (0-self.mu_sigma0)/10, high = (300-self.mu_sigma0)/10))
        sigma0 = numpyro.deterministic('sigma0', unscaled_sigma0*10 + self.mu_sigma0)

        unscaled_y0_vel = numpyro.sample('unscaled_y0_vel', dist.Normal())
        # y0_vel = norm.ppf(y0_vel)*(2) + self.y0_vel
        y0_vel = numpyro.deterministic('y0_vel', unscaled_y0_vel*2 + self.y0_vel)
        # end = time.time()
        # f.write('y0_vel sampling time: '+ str(end-start)+ '\n')
        # sample a global velicity shift v0:
        # start = time.time()
        unscaled_v0 = numpyro.sample('unscaled_v0', dist.Normal())
        # v0 = norm.ppf(v0)*100
        v0 = numpyro.deterministic('v0', unscaled_v0*100)
        # end = time.time()
        # f.write('v0 sampling time: '+ str(end-start)+ '\n')

        # start = time.time()
        fluxes = self.disk.sample_fluxes(low, high)
        # end =  time.time()
        # f.write("Time to sample fluxes: " + str(end-start) + "\n")

        # oversample the fluxes to match the grism object
        # start = time.time()
        fluxes_high = utils.oversample(fluxes, grism_object.factor, grism_object.factor)
        # end =  time.time()
        # f.write("Time to oversample fluxes: " + str(end-start) + "\n")

        # create new grid centered on those centroids
        x = jnp.linspace(0 - self.x0_vel, self.flux_prior.shape[1]-1 - self.x0_vel, self.flux_prior.shape[1]*grism_object.factor)
        y = jnp.linspace(0 - y0_vel, self.flux_prior.shape[0]-1 - y0_vel, self.flux_prior.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        # sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)

        # start  = time.time()
        self.compute_factors(jnp.radians(Pa), jnp.radians(i), X, Y)
        velocities = jnp.array(self.v(X, Y, jnp.radians(Pa), jnp.radians(i), Va, r_t))
        # end =  time.time()
        # f.write("Time to compute velocities: " + str(end-start) + "\n")

        # velocities = jnp.array(v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
        # velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')

        velocities_scaled = velocities + v0

        dispersions = sigma0*jnp.ones_like(velocities_scaled)
        # dispersions = self.sigma_disk(sigma0_max, sigma0_scale, sigma0_const, fluxes)

        
        # #make a data cube with the 'spectra' in each pixel
        # broadcast_velocity_space = grism_object.velocity_space
        # cube = fluxes*norm.pdf(broadcast_velocity_space, velocities, dispersions)

        # #the PSF is already oversampled here and in 3D form
        # PSF_kernel = grism_object.PSF

        # convolved_cube = convolve(cube, PSF_kernel, mode='same')

        # convolved_fluxes = jnp.sum(convolved_cube, axis=0)

        # convolved_velocities = jnp.mean(convolved_cube, axis = 2)
        # convolved_dispersions = jnp.std(convolved_cube, axis = 2)

        # start = time.time()
        self.model_map = grism_object.disperse(fluxes_high, velocities_scaled, dispersions)
        # end =  time.time()
        # f.write("Time to disperse: " + str(end-start) + "\n")
        # self.model_map = grism_object.disperse(convolved_fluxes, convolved_velocities, convolved_dispersions)

        # start = time.time()
        self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, grism_object.wave_factor)
        # end = time.time()
        # f.write("Time to resample: " + str(end-start) + "\n")

        self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0, 1))*9 + 1

        #regularize the fluxes
        reg_strength = numpyro.sample('regularization_strength', dist.Uniform())

        laplace_fluxes = jnp.abs(convolve(fluxes, self.Laplace_kernel, mode='same'))
        laplace_fluxes_err = jnp.ones_like(fluxes)*1e-6
        laplace_fluxes_err = laplace_fluxes_err.at[self.disk.masked_indices].set(self.std)

        model_map_r = jnp.reshape(self.model_map, (1,self.model_map.shape[0]*self.model_map.shape[1]))
        obs_error_r = jnp.reshape(obs_error, (1,obs_error.shape[0]*obs_error.shape[1]))
        obs_map_r = jnp.reshape(obs_map, (1,obs_map.shape[0]*obs_map.shape[1]))
        laplace_fluxes_r = jnp.reshape(laplace_fluxes, (1,laplace_fluxes.shape[0]*laplace_fluxes.shape[1]))
        laplace_fluxes_err_r = jnp.reshape(laplace_fluxes_err, (1,laplace_fluxes_err.shape[0]*laplace_fluxes_err.shape[1]))
        # self.error_scaling = 1
        # start = time.time()
        # numpyro.sample('obs', dist.Normal(self.model_map, self.error_scaling*obs_error), obs=obs_map)
        numpyro.sample('obs', dist.Normal(jnp.concatenate((model_map_r,reg_strength*laplace_fluxes_r), axis = 1),
                    jnp.concatenate((self.error_scaling*obs_error_r, laplace_fluxes_err_r), axis = 1)), 
                    obs=jnp.concatenate((obs_map_r, jnp.zeros_like(laplace_fluxes_r)), axis = 1))

        # end = time.time()
        # f.write("Time to sample obs: " + str(end-start) + "\n")
        # end_all = time.time()
        # f.write("Total time: " + str(end_all-start_all) + "\n")

    def compute_model(self, inference_data, grism_object):
        """

        Function used to post-process the MCMC samples and plot results from the model

        """

        self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean = self.disk.compute_posterior_means(inference_data)
        # self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean,self.y0_vel_mean, self.v0_mean = self.disk.compute_posterior_means(inference_data)

        self.fluxes_mean, self.fluxes_scaling_mean = self.disk.compute_flux_posterior(inference_data, self.flux_type)

        self.model_flux = utils.oversample(self.fluxes_mean, grism_object.factor, grism_object.factor)

        self.x0_vel_mean = self.x0_vel #change this later to put the vel offset into the x0_vel_mean

        x = jnp.linspace(
            0 - self.x0_vel_mean, self.flux_prior.shape[1] - 1 - self.x0_vel_mean, self.flux_prior.shape[1]*grism_object.factor)
        y = jnp.linspace(
            0 - self.y0_vel_mean, self.flux_prior.shape[0] - 1 - self.y0_vel_mean, self.flux_prior.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        self.compute_factors(jnp.radians(self.PA_mean), jnp.radians(jnp.radians(self.i_mean)), X, Y)
        self.model_velocities = jnp.array(self.v(X, Y, jnp.radians(
            self.PA_mean), jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
        # self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

        self.model_velocities = self.model_velocities  + self.v0_mean

        self.model_dispersions = self.sigma0_mean_model *jnp.ones_like(self.model_velocities)
        # self.model_dispersions = self.sigma_disk( self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean,self.model_flux)

        # # inference_data.posterior['V_re'].data = self.v(r[0], r[1], inference_data.posterior['PA'].data, inference_data.posterior['i'].data, inference_data.posterior['Va'].data, inference_data.posterior['r_t'].data) + inference_data.posterior['v0'].data
        # #make a data cube with the 'spectra' in each pixel
        # broadcast_velocity_space = grism_object.velocity_space
        # cube = self.model_flux*norm.pdf(broadcast_velocity_space, self.model_velocities, self.model_dispersions)

        # #the PSF is already oversampled here and in 3D form
        # PSF_kernel = grism_object.PSF

        # convolved_cube = convolve(cube, PSF_kernel, mode='same')

        # self.convolved_fluxes = jnp.sum(convolved_cube, axis=0)
        # print(convolved_cube[:,1,1])
        # print(broadcast_velocity_space[:,1,1])
        # print(jnp.multiply(convolved_cube[:,1,1], broadcast_velocity_space[:,1,1]))
        # self.convolved_velocities = jnp.mean(jnp.multiply(convolved_cube/convolved_cube.sum(axis = 0),broadcast_velocity_space), axis = 0)
        # self.convolved_dispersions = jnp.std(jnp.multiply(convolved_cube/convolved_cube.sum(axis = 0),broadcast_velocity_space), axis = 0)

        # plt.imshow(self.model_flux,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.convolved_fluxes,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.model_velocities,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.convolved_velocities,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.model_dispersions,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.convolved_dispersions,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        print(self.model_flux.shape, self.model_velocities.shape, self.model_dispersions.shape)
        self.model_map_high = grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
        # self.model_map_high = grism_object.disperse(self.convolved_fluxes, self.convolved_velocities, self.convolved_dispersions)

        self.model_map = utils.resample(self.model_map_high, grism_object.factor*grism_object.y_factor, grism_object.wave_factor)

        self.model_v_rot = self.disk.v_rot(self.fluxes_mean, self.model_velocities, self.i_mean, grism_object.factor)

        #compute velocity grid in flux image resolution for plotting velocity maps
        self.model_velocities_low = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/grism_object.factor), int(self.model_velocities.shape[1]/grism_object.factor)), method='nearest')
        self.model_dispersions_low = image.resize(self.model_dispersions, (int(self.model_dispersions.shape[0]/grism_object.factor), int(self.model_dispersions.shape[1]/grism_object.factor)), method='nearest')

        return self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions

    def plot_summary(self, obs_map, obs_error, inf_data, wave_space, save_to_folder = None, name = None):

        plotting.plot_disk_summary(obs_map, self.model_map, obs_error, self.model_velocities_low, self.model_dispersions_low, self.model_v_rot, self.fluxes_mean, inf_data, wave_space, self.mask, x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = save_to_folder, name = name)




class DiskModel(KinModels):
    """
        Class for the one component exponential disk model
    """

    def __init__(self):
        print('Disk model created')

        #declare var and label names for plotting

        self.var_names = ['PA', 'i', 'Va', 'r_t', 'sigma0', 'fluxes_scaling']
        self.labels = ['PA', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$', r'$f_{scale}$']
        # self.var_names = ['PA', 'i', 'Va', 'r_t', 'sigma0_max', 'sigma0_scale', 'sigma0_const']
        # self.labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_{max}$', r'$\sigma_{scale}$', r'$\sigma_{const}$']

    def set_bounds(self, flux_prior, flux_error, broad_band, flux_bounds, flux_type, flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph, inclination, r_eff, r_eff_grism, delta_wave, wavelength):
        """

        Compute all of the necessary bounds for the disk model sampling distributions

        """
        # first set all of the main bounds taken from the config file
        self.set_main_bounds(flux_prior,flux_error, broad_band,flux_bounds, flux_type, flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel, PA_grism, PA_morph,inclination,r_eff, r_eff_grism,
                             delta_V_bounds=None, clump=None, clump_v_prior=None, clump_sigma_prior=None, clump_flux_prior=None)

        # now compute the specific bounds for the disk model

        self.compute_flux_bounds()

        #make the mask 
        # this is done using the broad band image
        seg_1comp, seg_none , mask, mask_none, PA, inc, r_eff, center, ratio = utils.compute_gal_props(self.broad_band, threshold_sigma = self.flux_threshold)

        self.mask = mask
        if isinstance(self.PA_morph, float) == False:
            self.PA_morph = PA[0]
            if self.PA_morph<0:
                self.PA_morph = self.PA_morph + 180
            print('Setting PA morph prior to: ', self.PA_morph)
        if isinstance(self.inclination, float) == False:
            self.inclination = inc[0]
            print('Setting i prior to: ', self.inclination)
        if isinstance(self.r_eff, float) == False:
            self.r_eff = r_eff[0]/2 #/2 bc the want the 1/2 light rad not the full rad...
            print('Setting r_eff prior to: ', self.r_eff)

        #only put this for ALT
        # self.x0_vel = center[0][0]
        # self.mu_y0_vel = center[0][1]
   
        #using the grism position angle, determine the right PA for the rotation orientation
        # if self.PA_grism < self.PA_morph:
        #     print('PA grism less than PA morph')
        #     self.mu_PA = self.PA_morph + 180
        # else:  
        #     print('PA grism greater than PA morph')
        #     self.mu_PA = self.PA_morph

            
        self.mu_PA = self.PA_morph

        #correct r_eff for the observed height
        q0=0.2
        axis_ratio = np.sqrt(np.cos(np.radians(self.inclination))**2*(1-q0**2) + q0**2)
        minor_r_eff = axis_ratio*r_eff
        self.r_eff_y = jnp.maximum(self.r_eff * jnp.abs(jnp.sin(jnp.radians(self.PA_morph))), minor_r_eff*jnp.abs(jnp.sin(jnp.radians(90-self.PA_morph))))[0]
        
        #convert to the PA used in kinematics
        self.mu_PA = 180 - self.mu_PA

        # if self.mu_PA == -140.01989118108816: #hard coded for FREKMOS
        #     self.mu_PA = 0.0
        
        #the error has a floor and then increases for how much the gal is circular (how close the axis ratio is to 1 )
        self.sigma_PA = 50/(jnp.abs(ratio[0])) 
        print(self.sigma_PA)
        # old error self.PA_sigma*90 
        print('PA morph: ', self.PA_morph, 'PA grism: ', self.PA_grism, 'final PA: ', self.mu_PA)

        self.mu_i = self.inclination


        self.mask = jnp.array(self.mask)
        self.mask_shape = len(jnp.where(self.mask == 1)[0])
        self.masked_indices = jnp.where(self.mask == 1)

        print('Masked indices len: ', len(self.masked_indices[0]))
        # self.mu, self.std, self.high, self.low = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask)
        self.mu, self.std, self.low= self.rescale_to_mask([self.mu, self.std, self.low], self.mask)
        #manually make the flux prior error bigger to account for uncertainties about the EL map itself
        # self.std = self.std
        #set the velocity center to the brightest center of the gal within the mask
        # fluxes = jnp.zeros(self.mask.shape)
        # fluxes = fluxes.at[self.masked_indices].set(self.mu)
        # self.mu_y0_vel, self.x0_vel = np.unravel_index(np.argmax( fluxes),  fluxes.shape)


        #restricting y0_vel to inside 1/2 light radius
        #centering on y0_vel bc that's taken from pysersic fit centroid
        self.y_high = self.mu_y0_vel + self.r_eff_y
        self.y_low = self.mu_y0_vel - self.r_eff_y

        self.y0_std = 0.5*self.r_eff_y
    
        print('r_eff_y: ', self.r_eff_y) 

        #set the V_a bounds to 2*the observed vmax of the galaxy
        vel_pix_scale = (delta_wave/wavelength)*(c/1000) #put c in km/s
        V_grad = self.r_eff_grism*vel_pix_scale
        self.Va_bounds = [0, 2*V_grad]
        print('Va grad: ', V_grad)
        #set the sigma0 bounds to 2*the observed vmax of the galaxy
        self.sigma0_bounds = [0, 2*V_grad]
        #set the r_t bounds in the form [min, mu, max]:
        r_t_mu = 0.4*self.r_eff/1.676
        r_t_max = 2*r_eff
        self.r_t_bounds = [0, r_t_mu, r_t_max]
        #initialize the disk object
        self.disk = Disk(self.flux_prior.shape, self.masked_indices, self.mu, self.std,self.low, self.high, 
                    self.mu_PA, self.sigma_PA, self.i_bounds, self.mu_i, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel, self.mu_y0_vel, self.y_high, self.y_low, self.y0_std, self.r_eff, self.r_eff_y)
        
        self.disk.plot()

    def inference_model(self, grism_object, obs_map, obs_error):
        """

        Model used to infer the disk parameters from the data => called in fitting.py as the forward
        model used for the inference

        """
        # f = open("timing_total.txt", "a")
        # sample the fluxes within the mask from a truncated normal distribution
        start_all = time.time()
        # start = time.time()
        Pa, i, Va, r_t, sigma0, y0_vel, v0 = self.disk.sample_params()
        # end =  time.time()
        # Pa, i, Va, r_t, sigma0_max, sigma0_scale, sigma0_const, y0_vel, v0 = self.disk.sample_params()
        # f.write("Time to sample params: " + str(end-start) + "\n")

        # start = time.time()
        fluxes = self.disk.sample_fluxes()
        # end =  time.time()
        # f.write("Time to sample fluxes: " + str(end-start) + "\n")

        # oversample the fluxes to match the grism object
        # start = time.time()
        fluxes_high = utils.oversample(fluxes, grism_object.factor, grism_object.factor)
        # end =  time.time()
        # f.write("Time to oversample fluxes: " + str(end-start) + "\n")

        # create new grid centered on those centroids
        x = jnp.linspace(0 - self.x0_vel, self.flux_prior.shape[1]-1 - self.x0_vel, self.flux_prior.shape[1]*grism_object.factor)
        y = jnp.linspace(0 - y0_vel, self.flux_prior.shape[0]-1 - y0_vel, self.flux_prior.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        # sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)

        # start  = time.time()
        self.compute_factors(jnp.radians(Pa), jnp.radians(i),X, Y)
        velocities = jnp.array(self.v(X, Y, jnp.radians(Pa), jnp.radians(i), Va, r_t))
        # end =  time.time()
        # f.write("Time to compute velocities: " + str(end-start) + "\n")

        # velocities = jnp.array(v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
        # velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')

        velocities_scaled = velocities + v0

        dispersions = sigma0*jnp.ones_like(velocities_scaled)
        # dispersions = self.sigma_disk(sigma0_max, sigma0_scale, sigma0_const, fluxes)

        
        # #make a data cube with the 'spectra' in each pixel
        # broadcast_velocity_space = grism_object.velocity_space
        # cube = fluxes*norm.pdf(broadcast_velocity_space, velocities, dispersions)

        # #the PSF is already oversampled here and in 3D form
        # PSF_kernel = grism_object.PSF

        # convolved_cube = convolve(cube, PSF_kernel, mode='same')

        # convolved_fluxes = jnp.sum(convolved_cube, axis=0)

        # convolved_velocities = jnp.mean(convolved_cube, axis = 2)
        # convolved_dispersions = jnp.std(convolved_cube, axis = 2)

        # start = time.time()
        self.model_map = grism_object.disperse(fluxes_high, velocities_scaled, dispersions)
        # end =  time.time()
        # f.write("Time to disperse: " + str(end-start) + "\n")
        # self.model_map = grism_object.disperse(convolved_fluxes, convolved_velocities, convolved_dispersions)

        # start = time.time()
        self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, grism_object.wave_factor)
        # end = time.time()
        # f.write("Time to resample: " + str(end-start) + "\n")

        self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0, 1))*9 + 1

        #regularize the fluxes
        
        reg_strength = numpyro.sample('regularization_strength', dist.Uniform()) #*(1-0.00001) + 0.00001

        laplace_fluxes = convolve(fluxes, self.Laplace_kernel, mode='same')
        # sum and renormalize regularization term
        # threshold_grism = 0.2*obs_map.max()
        # sum_reg = jnp.sum(jnp.abs(laplace_fluxes))
        # #use a cut to be consistent  with grism/direct renormalization + avoid negative pixels
        # sum_reg_norm = sum_reg/jnp.sum(jnp.where(obs_map>threshold_grism,obs_map, 0.0)) #*self.model_map.shape[0]*self.model_map.shape[1]
        # error_reg = jnp.sqrt(jnp.sum(jnp.where(obs_map>threshold_grism,obs_error, 0.0)**2))/jnp.sum(jnp.where(obs_map>threshold_grism,obs_error, 0.0))
        # # error_reg_norm = error_reg*self.model_map.shape[0]*self.model_map.shape[1]/self.model_map.sum()

        #new renormalization of reg term

        sum_reg = jnp.sum(jnp.abs(laplace_fluxes))/jnp.sum(jnp.abs(fluxes))
        sum_reg_norm = sum_reg*jnp.max(obs_map)

        error_reg = obs_error[jnp.unravel_index(jnp.argmax(obs_map), (obs_map.shape[0], obs_map.shape[1]))]


        laplace_fluxes_r = jnp.reshape(sum_reg_norm, (1,1))
        laplace_fluxes_err_r = jnp.reshape(error_reg, (1,1))

        # reshaping the grism items to add the regularization term
        model_map_r = jnp.reshape(self.model_map, (1,self.model_map.shape[0]*self.model_map.shape[1]))
        obs_error_r = jnp.reshape(obs_error, (1,obs_error.shape[0]*obs_error.shape[1]))
        obs_map_r = jnp.reshape(obs_map, (1,obs_map.shape[0]*obs_map.shape[1]))

        # self.error_scaling = 1
        # start = time.time()
        # numpyro.sample('obs', dist.Normal(self.model_map, self.error_scaling*obs_error), obs=obs_map)
        numpyro.sample('obs', dist.Normal(jnp.concatenate((model_map_r,reg_strength*laplace_fluxes_r), axis = 1),
                    jnp.concatenate((self.error_scaling*obs_error_r, laplace_fluxes_err_r), axis = 1)), 
                    obs=jnp.concatenate((obs_map_r, jnp.zeros_like(laplace_fluxes_r)), axis = 1))

        # end = time.time()
        # f.write("Time to sample obs: " + str(end-start) + "\n")
        # end_all = time.time()
        # f.write("Total time: " + str(end_all-start_all) + "\n")

    def compute_model(self, inference_data, grism_object):
        """

        Function used to post-process the MCMC samples and plot results from the model

        """

        self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean = self.disk.compute_posterior_means(inference_data)
        # self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean,self.y0_vel_mean, self.v0_mean = self.disk.compute_posterior_means(inference_data)

        self.fluxes_mean, self.fluxes_scaling_mean = self.disk.compute_flux_posterior(inference_data, self.flux_type)
        # self.fluxes_mean/= self.fluxes_scaling_mean
        # self.fluxes_mean *= 0.6
        self.model_flux = utils.oversample(self.fluxes_mean, grism_object.factor, grism_object.factor)

        self.x0_vel_mean = self.x0_vel #change this later to put the vel offset into the x0_vel_mean

        x = jnp.linspace(
            0 - self.x0_vel_mean, self.flux_prior.shape[1] - 1 - self.x0_vel_mean, self.flux_prior.shape[1]*grism_object.factor)
        y = jnp.linspace(
            0 - self.y0_vel_mean, self.flux_prior.shape[0] - 1 - self.y0_vel_mean, self.flux_prior.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        self.compute_factors(jnp.radians(self.PA_mean), jnp.radians(jnp.radians(self.i_mean)),X, Y)

        self.model_velocities = jnp.array(self.v(X, Y, jnp.radians(
            self.PA_mean), jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
        # self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

        self.model_velocities = self.model_velocities  + self.v0_mean

        self.model_dispersions = self.sigma0_mean_model *jnp.ones_like(self.model_velocities)
        # self.model_dispersions = self.sigma_disk( self.sigma0_max_mean, self.sigma0_scale_mean, self.sigma0_const_mean,self.model_flux)

        # # inference_data.posterior['V_re'].data = self.v(r[0], r[1], inference_data.posterior['PA'].data, inference_data.posterior['i'].data, inference_data.posterior['Va'].data, inference_data.posterior['r_t'].data) + inference_data.posterior['v0'].data
        # #make a data cube with the 'spectra' in each pixel
        # broadcast_velocity_space = grism_object.velocity_space
        # cube = self.model_flux*norm.pdf(broadcast_velocity_space, self.model_velocities, self.model_dispersions)

        # #the PSF is already oversampled here and in 3D form
        # PSF_kernel = grism_object.PSF

        # convolved_cube = convolve(cube, PSF_kernel, mode='same')

        # self.convolved_fluxes = jnp.sum(convolved_cube, axis=0)
        # print(convolved_cube[:,1,1])
        # print(broadcast_velocity_space[:,1,1])
        # print(jnp.multiply(convolved_cube[:,1,1], broadcast_velocity_space[:,1,1]))
        # self.convolved_velocities = jnp.mean(jnp.multiply(convolved_cube/convolved_cube.sum(axis = 0),broadcast_velocity_space), axis = 0)
        # self.convolved_dispersions = jnp.std(jnp.multiply(convolved_cube/convolved_cube.sum(axis = 0),broadcast_velocity_space), axis = 0)

        # plt.imshow(self.model_flux,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.convolved_fluxes,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.model_velocities,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.convolved_velocities,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.model_dispersions,origin = 'lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(self.convolved_dispersions,origin = 'lower')
        # plt.colorbar()
        # plt.show()

        print(self.model_flux.shape, self.model_velocities.shape, self.model_dispersions.shape)
        self.model_map_high = grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
        # self.model_map_high = grism_object.disperse(self.convolved_fluxes, self.convolved_velocities, self.convolved_dispersions)

        self.model_map = utils.resample(self.model_map_high, grism_object.factor*grism_object.y_factor, grism_object.wave_factor)

        self.model_v_rot = self.disk.v_rot(self.fluxes_mean, self.model_velocities, self.i_mean, grism_object.factor)

        #compute velocity grid in flux image resolution for plotting velocity maps
        self.model_velocities_low = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/grism_object.factor), int(self.model_velocities.shape[1]/grism_object.factor)), method='nearest')
        self.model_dispersions_low = image.resize(self.model_dispersions, (int(self.model_dispersions.shape[0]/grism_object.factor), int(self.model_dispersions.shape[1]/grism_object.factor)), method='nearest')

        return self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions

    def plot_summary(self, obs_map, obs_error, inf_data, wave_space, save_to_folder = None, name = None):

        plotting.plot_disk_summary(obs_map, self.model_map, obs_error, self.model_velocities_low, self.model_dispersions_low, self.model_v_rot, self.fluxes_mean, inf_data, wave_space, self.mask, x0 = self.x0, y0 = self.y0, factor = self.y_factor , direct_image_size = self.flux_prior.shape[0], save_to_folder = save_to_folder, name = name)




class TwoComps(KinModels):
    """
        Class for the two component fit (two overlapping disks or two separate disks = merger like scenario)
    """

    def __init__(self):
        #to fill 
        print('fill this')
        

    def set_bounds(self, flux_prior, flux_bounds, flux_type, flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel, dilation_param = 6):
        """

        Compute all of the necessary bounds

        """
        # first set all of the main bounds taken from the config file
        self.set_main_bounds(flux_prior, flux_bounds, flux_type, flux_threshold, PA_sigma, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel,
                             delta_V_bounds=None, clump=None, clump_v_prior=None, clump_sigma_prior=None, clump_flux_prior=None)


        #make the masks and compute the PAs and inclinations
        seg_1comp, seg_2comp , mask, mask_2comp, PA, inc = utils.compute_gal_props(self.flux_prior, n = 2, threshold_sigma=self.flux_threshold)

        self.mask = mask

        self.mask1 = jnp.where(mask_2comp == 1, 1.0, 0.0)
        self.mask2 = jnp.where(mask_2comp == 2, 1.0, 0.0)

        self.mask1 = dilation(self.mask1, disk(3))
        self.mask2 = dilation(self.mask2, disk(1))

        self.mu_PA1= PA[0]
        self.mu_PA2= PA[1]
        self.mu_i1 = inc[0]
        self.mu_i2 = inc[1]

        #plot the main mask
        plt.imshow(self.mask, origin='lower')
        # plt.colorbar()
        plt.title('Main mask')
        plt.show()

        #make two separate flux priors with each mask
        self.flux_prior1 = self.flux_prior*self.mask1
        self.flux_prior2 = self.flux_prior*self.mask2

        plt.imshow(self.flux_prior1, origin = 'lower')
        plt.show()

        plt.imshow(self.flux_prior2, origin = 'lower')
        plt.show()

        self.sigma_PA1 = self.PA_sigma
        self.sigma_PA2 = self.PA_sigma

        #the velocity centroids can be initialized at the center of mass of each mask object
        self.x0_vel1, self.y0_vel1 = centroids.centroid_2dg(np.array(self.flux_prior1))
        self.x0_vel2, self.y0_vel2 = centroids.centroid_2dg(np.array(self.flux_prior2))

        print(self.x0_vel1, self.y0_vel1)
        print(self.x0_vel2, self.y0_vel2)

        #correct for the mask = 0 pixels for each mask
        # only fitting for pixels in the mask
        self.mask1 = jnp.array(self.mask1)
        self.mask2 = jnp.array(self.mask2)

        self.mask_shape1 = len(jnp.where(self.mask1 == 1)[0])
        self.masked_indices1 = jnp.where(self.mask1 == 1)

        self.mask_shape2 = len(jnp.where(self.mask2 == 1)[0])
        self.masked_indices2 = jnp.where(self.mask2 == 1)

        self.mask_shape = len(jnp.where(self.mask == 1)[0])
        self.masked_indices = jnp.where(self.mask == 1)


        self.compute_flux_bounds()

        self.mu1, self.std1, self.high1, self.low1 = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask1)
        self.mu2, self.std2, self.high2, self.low2 = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask2)
        self.mu, self.std, self.high, self.low = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask)
        #initialize the disk objects
        self.disk1 = Disk(self.flux_prior.shape, self.masked_indices1, self.mu1, self.std1, self.low1, self.high1,
                    self.mu_PA1, self.sigma_PA1, self.i_bounds, self.mu_i1, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel1, self.y0_vel1, number = '1')
        self.disk2 = Disk(self.flux_prior.shape, self.masked_indices2, self.mu2, self.std2, self.low2, self.high2,
                    self.mu_PA2, self.sigma_PA2, self.i_bounds, self.mu_i2, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel2, self.y0_vel2, number = '2')
        
        self.disk1.plot()
        self.disk2.plot()

        # plt.imshow(self.flux_prior*self.mask, origin = 'lower')
        # plt.scatter(self.x0_vel1, self.y0_vel1, color='red')
        # plt.scatter(self.x0_vel2, self.y0_vel2, color='red')
        # plt.show()

        #compute the fraction of each component in the mask by scaling with distance from the centroid of comp 1
        self.distances = jnp.sqrt((self.masked_indices[0] - self.y0_vel1)**2 + (self.masked_indices[1] - self.x0_vel1)**2)
        self.distances = self.distances/jnp.max(self.distances)
        self.frac_disk1_prior = 1 - self.distances

        #just for plotting purposes
        fraction = jnp.zeros(self.flux_prior.shape)
        fraction = fraction.at[self.masked_indices].set(self.frac_disk1_prior)
        # plt.imshow(fraction, origin='lower')
        # plt.colorbar()
        # plt.title('Fraction of disk 1 in mask')
        # plt.show()


    # def inference_model(self, grism_object, obs_map, obs_error):
    #     """

    #     Inference model for two merging disks

    #     """



    # def compute_model(self, inference_data, grism_object):
    #     """

    #     Function used to post-process the MCMC samples and plot results from the model

    #     """


class Merger(TwoComps):
    """
        Class for the merger scenario
    """

    #this one shouldn't change much from the top two comp one 
    #set two rotating disks with their own masks and centers
    #in the init need to make two masks?

    def __init__(self):
        print('Merger model created')
        self.var_names = ['PA1', 'i1', 'Va1', 'r_t1', 'sigma01', 'PA2', 'i2', 'Va2', 'r_t2', 'sigma02']
        self.labels =  [r'$PA_1$', r'$i1$', r'$Va1$', r'$r_{t,1}$', r'$\sigma_{0,1}$', r'$PA_2$', r'$i_2$', r'$Va_2$', r'$r_{t,2}$', r'$\sigma_{0,2}$']

    def set_bounds(self, flux_prior, flux_bounds, flux_type, flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel):

        #first use the set_bounds function from parent class
        super().set_bounds(flux_prior, flux_bounds, flux_type, flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel)
        #for now nothing else...

    def inference_model(self, grism_object, obs_map, obs_error):
        """

        Inference model for two merging disks

        """
        #sample the fractional contribution of each pixel to a component
        frac_disk1 = numpyro.sample('frac_disk1', dist.Uniform(), sample_shape=(int(len(self.masked_indices[0])),))
        frac_disk1 = norm.ppf(frac_disk1)*0.2 + self.frac_disk1_prior

        frac_disk2 = 1 - frac_disk1

        #sample the fluxes 
        fluxes_scaling = numpyro.sample('fluxes_scaling', dist.Uniform())*5
        fluxes_sample = numpyro.sample('fluxes', dist.Uniform(), sample_shape=(int(len(self.masked_indices[0])),))
        fluxes_sample = norm.ppf(norm.cdf(self.low) + fluxes_sample*(norm.cdf(self.high)-norm.cdf(self.low)))*self.std + self.mu*fluxes_scaling

        fluxes1 = jnp.zeros(self.flux_prior.shape)
        fluxes2 = jnp.zeros(self.flux_prior.shape)

        fluxes1 = fluxes1.at[self.masked_indices].set(fluxes_sample*frac_disk1)
        fluxes2 = fluxes2.at[self.masked_indices].set(fluxes_sample*frac_disk2)

        # sample the kin params for each disk model
        Pa1, i1, Va1, r_t1, sigma01, y0_vel1, v01 = self.disk1.sample_params()
        Pa2, i2, Va2, r_t2, sigma02, y0_vel2, v02 = self.disk2.sample_params()
    
        # Pa1, i1, Va1, r_t1, sigma0_max1, sigma0_scale1, y0_vel1, v01 = self.disk1.sample_params()
        # Pa2, i2, Va2, r_t2, sigma0_max2, sigma0_scale2, y0_vel2, v02 = self.disk2.sample_params()

        fluxes1 = utils.oversample(fluxes1, grism_object.factor, grism_object.factor)
        fluxes2 = utils.oversample(fluxes2, grism_object.factor, grism_object.factor)
    
        # create new grid centered on those centroids
        x1 = jnp.linspace(0 - self.x0_vel1, self.flux_prior.shape[1]-1 - self.x0_vel1, self.flux_prior.shape[1]*grism_object.factor)
        y1 = jnp.linspace(0 - self.y0_vel1, self.flux_prior.shape[0]-1 - self.y0_vel1, self.flux_prior.shape[0]*grism_object.factor)
        X1, Y1 = jnp.meshgrid(x1, y1)

        x2 = jnp.linspace(0 - self.x0_vel2, self.flux_prior.shape[1]-1 - self.x0_vel2, self.flux_prior.shape[1]*grism_object.factor)
        y2 = jnp.linspace(0 - self.y0_vel2, self.flux_prior.shape[0]-1 - self.y0_vel2, self.flux_prior.shape[0]*grism_object.factor)
        X2, Y2 = jnp.meshgrid(x2, y2)
    
        # sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)
    
        velocities1 = jnp.array(self.v(X1, Y1, jnp.radians(Pa1), jnp.radians(i1), Va1, r_t1))
        velocities2 = jnp.array(self.v(X2, Y2, jnp.radians(Pa2), jnp.radians(i2), Va2, r_t2))
    
        velocities1 = velocities1 + v01
        velocities2 = velocities2 + v02
    
        dispersions1 = sigma01*jnp.ones_like(velocities1)
        dispersions2 = sigma02*jnp.ones_like(velocities2)

        # dispersions1 = self.sigma_disk( sigma0_max1, sigma0_scale1, velocities1)
        # dispersions2 = self.sigma_disk(sigma0_max2, sigma0_scale2, velocities2)
    
        self.model_map1 = grism_object.disperse(fluxes1, velocities1, dispersions1)
        self.model_map2 = grism_object.disperse(fluxes2, velocities2, dispersions2)

        self.model_map = self.model_map1 + self.model_map2

        self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, grism_object.wave_factor)

        self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0, 1))*9 + 1
        # self.error_scaling = 1
        numpyro.sample('obs', dist.Normal(self.model_map, self.error_scaling*obs_error), obs=obs_map)


    def compute_model(self, inference_data, grism_object):
        """

        Function used to post-process the MCMC samples and plot results from the model
        plot results for the Merger model

        """

        self.PA_mean_1,self.i_mean_1, self.Va_mean_1, self.r_t_mean_1, self.sigma0_mean_model_1, self.y0_vel_mean_1, self.v0_mean_1 = self.disk1.compute_posterior_means(inference_data)
        self.PA_mean_2,self.i_mean_2, self.Va_mean_2, self.r_t_mean_2, self.sigma0_mean_model_2, self.y0_vel_mean_2, self.v0_mean_2 = self.disk2.compute_posterior_means(inference_data)

        # self.PA_mean_1,self.i_mean_1, self.Va_mean_1, self.r_t_mean_1, self.sigma0_max_mean_1, self.sigma0_scale_mean_1, self.y0_vel_mean_1, self.v0_mean_1 = self.disk1.compute_posterior_means(inference_data)
        # self.PA_mean_2,self.i_mean_2, self.Va_mean_2, self.r_t_mean_2, self.sigma0_max_mean_2, self.sigma0_scale_mean_2, self.y0_vel_mean_2, self.v0_mean_2 = self.disk2.compute_posterior_means(inference_data)

        best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
        ), inference_data['sample_stats']['lp'].shape)

        inference_data.posterior['fluxes_scaling'].data = inference_data.posterior['fluxes_scaling'].data*5
        #if the fluxes are manually rescaled in the prior, then rescale them
        self.fluxes_scaling_mean = jnp.array(inference_data.posterior['fluxes_scaling'].isel(chain=best_indices[0], draw=best_indices[1]))

        best_flux_sample = utils.find_best_sample(inference_data, ['fluxes'], [self.mu*self.fluxes_scaling_mean], [self.std], [self.high], [self.low], best_indices)

        self.fluxes_sample_mean = jnp.array(inference_data.posterior['fluxes'].isel(chain=best_indices[0], draw=best_indices[1]))

        best_frac_sample = utils.find_best_sample(inference_data, ['frac_disk1'], [self.frac_disk1_prior], [0.2], [None], [None], best_indices) 
        self.frac_disk1_mean = jnp.array(inference_data.posterior['frac_disk1'].isel(chain=best_indices[0], draw=best_indices[1]))
        self.frac_disk2_mean = 1 - self.frac_disk1_mean

        self.fluxes_mean1 = jnp.zeros(self.flux_prior.shape)
        self.fluxes_mean1 = self.fluxes_mean1.at[self.masked_indices].set( self.fluxes_sample_mean*self.frac_disk1_mean)

        self.fluxes_mean2 = jnp.zeros(self.flux_prior.shape)
        self.fluxes_mean2 = self.fluxes_mean2.at[self.masked_indices].set( self.fluxes_sample_mean*self.frac_disk2_mean)

        self.model_flux1 = utils.oversample(self.fluxes_mean1, grism_object.factor, grism_object.factor)
        self.model_flux2 = utils.oversample(self.fluxes_mean2, grism_object.factor, grism_object.factor)

        self.x0_vel_mean1 = self.x0_vel1
        self.x0_vel_mean2 = self.x0_vel2

        x1 = jnp.linspace(
            0 - self.x0_vel_mean1, self.flux_prior.shape[1] - 1 - self.x0_vel_mean1, self.flux_prior.shape[1]*grism_object.factor)
        y1 = jnp.linspace(
            0 - self.y0_vel_mean_1, self.flux_prior.shape[0] - 1 - self.y0_vel_mean_1, self.flux_prior.shape[0]*grism_object.factor)
        X1, Y1 = jnp.meshgrid(x1, y1)

        x2 = jnp.linspace(
            0 - self.x0_vel_mean2, self.flux_prior.shape[1] - 1 - self.x0_vel_mean2, self.flux_prior.shape[1]*grism_object.factor)
        y2 = jnp.linspace(
            0 - self.y0_vel_mean_2, self.flux_prior.shape[0] - 1 - self.y0_vel_mean_2, self.flux_prior.shape[0]*grism_object.factor)
        X2, Y2 = jnp.meshgrid(x2, y2)

        self.model_velocities1 = jnp.array(self.v(X1, Y1, jnp.radians(
            self.PA_mean_1), jnp.radians(self.i_mean_1), self.Va_mean_1, self.r_t_mean_1))
        self.model_velocities2 = jnp.array(self.v(X2, Y2, jnp.radians(
            self.PA_mean_2), jnp.radians(self.i_mean_2), self.Va_mean_2, self.r_t_mean_2))
        
        self.model_velocities1 = self.model_velocities1 + self.v0_mean_1
        self.model_velocities2 = self.model_velocities2 + self.v0_mean_2

        self.model_dispersions1 = self.sigma0_mean_model_1 *jnp.ones_like(self.model_velocities1)
        self.model_dispersions2 = self.sigma0_mean_model_2 *jnp.ones_like(self.model_velocities2)

        # self.model_dispersions1 = self.sigma_disk(X1, Y1, jnp.radians(self.PA_mean_1), jnp.radians(self.i_mean_1), self.Va_mean_1, self.r_t_mean_1, self.sigma0_max_mean_1, self.sigma0_scale_mean_1)
        # self.model_dispersions2 = self.sigma_disk(X2, Y2, jnp.radians(self.PA_mean_2), jnp.radians(self.i_mean_2), self.Va_mean_2, self.r_t_mean_2, self.sigma0_max_mean_2, self.sigma0_scale_mean_2)

        self.model_map_high1 = grism_object.disperse(self.model_flux1, self.model_velocities1, self.model_dispersions1)
        self.model_map_high2 = grism_object.disperse(self.model_flux2, self.model_velocities2, self.model_dispersions2)

        self.model_map_high = self.model_map_high1 + self.model_map_high2
        self.model_map = utils.resample(self.model_map_high, grism_object.factor*grism_object.y_factor, grism_object.wave_factor)

        self.model_v_rot1 = self.disk1.v_rot(self.fluxes_mean1, self.model_velocities1, self.i_mean_1, grism_object.factor)
        self.model_v_rot2 = self.disk2.v_rot(self.fluxes_mean2, self.model_velocities2, self.i_mean_2, grism_object.factor)

        #compute velocity grid in flux image resolution for plotting velocity maps
        self.model_velocities_low1 = image.resize(self.model_velocities1, (int(self.model_velocities1.shape[0]/grism_object.factor), int(self.model_velocities1.shape[1]/grism_object.factor)), method='nearest')
        self.model_velocities_low2 = image.resize(self.model_velocities2, (int(self.model_velocities2.shape[0]/grism_object.factor), int(self.model_velocities2.shape[1]/grism_object.factor)), method='nearest')

        return self.model_map, self.model_flux1+self.model_flux2,  self.fluxes_mean1 + self.fluxes_mean2, self.model_velocities1 + self.model_velocities2, self.model_dispersions1+self.model_dispersions2

    def plot_summary(self, obs_map, obs_error, inf_data, wave_space, save_to_folder = None, name = None):

        plotting.plot_merger_summary(obs_map, self.model_map, obs_error, [self.model_velocities_low1, self.model_velocities_low2], [self.model_v_rot1, self.model_v_rot2], self.fluxes_mean1 + self.fluxes_mean2, inf_data, wave_space, [self.mask1,self.mask2, self.mask] , x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = save_to_folder, name = name)

class TwoDisks(TwoComps): 
    """
        Class for 2 disk-like components within a galaxy
    """

class Clumps(TwoComps):
    """
        Class for 2 components within a galaxy, one disk and one free
    """   



#maybe the two disks can be a 2 component general one and then have the 2 disks cases above but also one with a free 
#vel component => all 3 inherit from the first main one that has the bulk of the code but then small things change between each