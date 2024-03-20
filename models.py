"""

	Module holding all of the kinematic models used in the fitting process.

	Written by A L Danhaive: ald66@cam.ac.uk
"""


# imports
import numpy as np
# geko related imports
import utils

# jax and its functions
import jax.numpy as jnp
from jax.scipy.stats import norm

# scipy and its functions
from scipy.constants import pi

# numpyro and its functions
import numpyro
# from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam


class KinModels:

    def __init__(self):
        print('New kinematic model created')

    def x_int(self, x, y, PA, i):
        return x*jnp.cos(PA) - y*jnp.sin(PA)

    def y_int(self, x, y, PA, i):
        return jnp.where(jnp.cos(i) != 0., (x*jnp.sin(PA) + y*jnp.cos(PA))/(jnp.cos(i)), 0.)

    def r_int(self, x, y, PA, i):

        return jnp.sqrt((self.x_int(x, y, PA, i))**2 + (self.y_int(x, y, PA, i))**2)

    def phi_int(self, x, y, PA, i):
        return jnp.where(self.r_int(x, y, PA, i) != 0, jnp.arccos(self.x_int(x, y, PA, i)/self.r_int(x, y, PA, i)), 0.)

    def v(self, x, y, PA, i, Va, r_t):
        return (2/pi)*Va*jnp.arctan(self.r_int(x, y, PA, i)/r_t)*jnp.sin(i)*jnp.cos(self.phi_int(x, y, PA, i))

    def flux(self, x, y, r_eff, I0, PA, i):
        fluxes = I0*jnp.exp(-self.r_int(x, y, PA, i)/r_eff)
        # fluxes = jnp.where(r_int(x, y, PA, i) > r_eff*1.5, 0.0, fluxes)
        return fluxes

    def sigma(self, x, y, sigma0, r_eff=None, I0=None, PA=None, i=None):
        return sigma0

    def set_main_bounds(self, flux_prior, flux_bounds, flux_type, PA_bounds, PA_normal, i_bounds, Va_bounds, r_t_bounds,
                        sigma0_mean, sigma0_disp, sigma0_bounds, obs_map_bounds,  mask, y_factor, x0, x0_vel, y0, y0_vel,
                        delta_V_bounds, clump, clump_v_prior, clump_sigma_prior, clump_flux_prior):
        """
        Set the bounds for the model parameters by reading the ones from the config file.
        The more specific bounds computations for the different models will be done inside their
        class.
        """

        self.flux_prior = flux_prior
        # these first two are in the form (scale, high)
        self.flux_bounds = flux_bounds
        self.flux_type = flux_type
        self.PA_bounds = PA_bounds
        self.PA_normal = PA_normal
        # these are in the form (low, high)
        self.i_bounds = i_bounds
        self.Va_bounds = Va_bounds
        self.r_t_bounds = r_t_bounds
        self.sigma0_mean = sigma0_mean
        self.sigma0_disp = sigma0_disp
        self.sigma0_bounds = sigma0_bounds
        self.obs_map_bounds = obs_map_bounds
        self.clump = clump
        self.mask = mask
        self.delta_V_bounds = delta_V_bounds

        self.y_factor = y_factor

        self.clump_v_prior = clump_v_prior
        self.clump_sigma_prior = clump_sigma_prior
        self.clump_flux_prior = clump_flux_prior

        if self.PA_bounds[1] != 'const':
            self.mu_PA = self.PA_bounds[0]
            self.sigma_PA = self.PA_bounds[1]*90

        self.x0 = x0
        self.y0 = y0

        self.x0_vel = x0_vel
        self.y0_vel = y0_vel

        if self.y0_vel == None:
            self.x0_vel = x0
            self.y0_vel = y0


class Disk(KinModels):
    """
        Class for the one component exponential disk model
    """

    def __init__(self):
        print('Disk model created')

    def set_bounds(self, flux_prior, flux_bounds, flux_type, PA_bounds, PA_normal, i_bounds, Va_bounds, r_t_bounds,
                   sigma0_mean, sigma0_disp, sigma0_bounds, obs_map_bounds, mask, y_factor, x0, x0_vel, y0, y0_vel):
        """

        Compute all of the necessary bounds for the disk model sampling distributions

        """
        # first set all of the main bounds taken from the config file
        self.set_main_bounds(flux_prior, flux_bounds, flux_type, PA_bounds, PA_normal, i_bounds, Va_bounds, r_t_bounds,
                             sigma0_mean, sigma0_disp, sigma0_bounds, obs_map_bounds, mask, y_factor, x0, x0_vel, y0, y0_vel,
                             delta_V_bounds=None, clump=None, clump_v_prior=None, clump_sigma_prior=None, clump_flux_prior=None)

        # now compute the specific bounds for the disk model

        # if the fluxes are negative, set their prior to zero
        self.mu = jnp.maximum(
            jnp.zeros(self.flux_prior.shape), jnp.array(self.flux_prior))
        # if the flux is now zero, set the sigma to 0.000001 => so that sigma is never 0
        self.std = jnp.maximum(0.000001, self.flux_bounds[0]*self.mu)
        # same here, if the flux is zero, set the high to 0.000002
        self.high = (jnp.maximum(
            0.000002, self.flux_bounds[1] * self.mu) + self.mu - self.mu)/self.std
        self.low = (jnp.zeros(self.flux_prior.shape)-self.mu)/self.std

        # correct for the mask = 0 pixels
        if self.mask is not None:
            # only fitting for pixels in the mask
            self.mask = jnp.array(self.mask)
            self.mask_shape = len(jnp.where(self.mask == 1)[0])
            self.masked_indices = jnp.where(self.mask == 1)
            self.mu = self.mu[jnp.where(self.mask == 1)]
            self.std = self.std[jnp.where(self.mask == 1)]
            self.high = self.high[jnp.where(self.mask == 1)]
            self.low = self.low[jnp.where(self.mask == 1)]

    def inference_model(self, grism_object, obs_map, obs_error):
        """

        Model used to infer the disk parameters from the data => called in fitting.py as the forward
        model used for the inference

        """

        # sample the fluxes within the mask from a truncated normal distribution
        reparam_config = {"fluxes": TransformReparam()}
        with numpyro.handlers.reparam(config=reparam_config):
            # in order to use TransformReparam we have to express the prior
            # over betas as a TransformedDistribution
            fluxes_sample = numpyro.sample("fluxes", dist.TransformedDistribution(dist.TruncatedNormal(jnp.zeros(int(
                self.mask_shape)), jnp.ones_like(int(self.mask_shape)), low=self.low, high=self.high), AffineTransform(self.mu, self.sigma),),)

        fluxes = jnp.zeros_like(self.flux_prior)
        fluxes = fluxes.at[self.masked_indices].set(fluxes_sample)

        # oversample to reach model space resolution
        fluxes = utils.oversample(
            fluxes, grism_object.factor, grism_object.factor)

        Pa = numpyro.sample('PA', dist.Uniform())
        # sample the mu_PA + 0 or 180 (orientation of velocity field)
        rotation = numpyro.sample('rotation', dist.Uniform())

        # simulate a bernouilli discrete distribution
        PA_morph = self.mu_PA + round(rotation)*180

        Pa = norm.ppf(Pa)*self.sigma_PA + PA_morph
        # Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA

        #could probably use utils for this too 
        i = numpyro.sample('i', dist.Uniform()) * \
            (self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

        Va = numpyro.sample('Va', dist.Uniform()) * \
            (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

        r_t = numpyro.sample('r_t', dist.Uniform(
        ))*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

        sigma0 = numpyro.sample('sigma0', dist.Uniform(
        ))*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

        # sampling the y axis velocity centroids
        # x0 = numpyro.sample('x0', dist.Uniform())
        # x0 = norm.ppf(x0)*(2) + self.x0
        y0_vel = numpyro.sample('y0_vel', dist.Uniform())
        y0_vel = norm.ppf(y0_vel)*(2) + self.y0_vel

        # create new grid centered on those centroids
        x = jnp.linspace(0 - self.x0_vel, grism_object.direct.shape[1]-1 - self.x0_vel, grism_object.direct.shape[1]*grism_object.factor)
        y = jnp.linspace(0 - y0_vel, grism_object.direct.shape[0]-1 - y0_vel, grism_object.direct.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        # sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)

        velocities = jnp.array(
            self.v(X, Y, jnp.radians(Pa), jnp.radians(i), Va, r_t))
        # velocities = jnp.array(v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
        # velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')

        # sample a global velicity shift v0:
        v0 = numpyro.sample('v0', dist.Uniform())
        v0 = norm.ppf(v0)*100

        velocities = velocities + v0

        dispersions = sigma0*jnp.ones_like(velocities)

        # sample a shift in the dispersion wavelength
        # corrected_wavelength = numpyro.sample('wavelength', dist.Uniform())
        # corrected_wavelength = norm.ppf(corrected_wavelength)*0.001 + self.wavelength

        # self.grism_object.set_wavelength(corrected_wavelength)

        self.model_map = grism_object.disperse(
            fluxes, velocities, dispersions)

        self.model_map = utils.resample(
            self.model_map, grism_object.y_factor*grism_object.factor, grism_object.wave_factor)

        self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0, 1))*9 + 1
        # self.error_scaling = 1
        numpyro.sample('obs', dist.Normal(
            self.model_map, self.error_scaling*obs_error), obs=obs_map)

    def plot_model(self, inference_data, grism_object):
        """

        Function used to post-process the MCMC samples and plot results from the model

        """

        best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
        ), inference_data['sample_stats']['lp'].shape)

        # rescale all of the posteriors from uniform to the actual parameter space
        rotation = float(inference_data.posterior['rotation'].median(dim=["chain", "draw"]))
        #create lists with variables and their scaling parameters 
        variables = ['PA', 'i', 'Va', 'r_t', 'sigma0', 'y0_vel', 'v0']
        #for variables drawn from uniform dist, the scaling parameters are (low, high) so mu and sigma are set to none
        mus = [self.mu_PA + round(rotation)*180, None, None, None, None, self.y0_vel, 0.0]
        sigmas = [self.sigma_PA, None, None, None, None, 2.0, 100.0]

        #for variables drawn from normal dist, the scaling parameters are (mu, sigma) so low and high are set to none
        highs = [None, self.i_bounds[1], self.Va_bounds[1], self.r_t_bounds[1], self.sigma0_bounds[1], None, None]
        lows = [None,self.i_bounds[0], self.Va_bounds[0], self.r_t_bounds[0], self.sigma0_bounds[0], None, None]

        #if the fluxes are manually rescaled in the prior, then add them to list => otherwise dists are already rescaled

        if self.flux_type == 'manual':
            variables.append('fluxes')
            mus.append(self.mu)
            sigmas.append(self.std)
            highs.append(self.high)
            lows.append(self.low)

        rescaled_posterior, rescaled_prior, best_sample = utils.find_best_sample(
            inference_data, variables, mus, sigmas, highs, lows, best_indices)
        
        for i,var in enumerate(variables):
            inference_data.posterior[var].data = rescaled_posterior[i]
            inference_data.prior[var].data = rescaled_prior[i]
        
        if self.flux_type == 'manual':
            self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean, self.fluxes_sample_mean = best_sample
        else:
            self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean = best_sample

        self.fluxes_mean = jnp.zeros_like(self.flux_prior)
        self.fluxes_mean = self.fluxes_mean.at[self.masked_indices].set(self.fluxes_sample_mean)


        self.x0_vel_mean = self.x0_vel #change this later to put the vel offset into the x0_vel_mean

        x = jnp.linspace(
            0 - self.x0_vel_mean, grism_object.direct.shape[1] - 1 - self.x0_vel_mean, grism_object.direct.shape[1]*grism_object.factor)
        y = jnp.linspace(
            0 - self.y0_vel_mean, grism_object.direct.shape[0] - 1 - self.y0_vel_mean, grism_object.direct.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        self.model_velocities = jnp.array(self.v(X, Y, jnp.radians(
            self.PA_mean), jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
        # self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

        self.model_velocities = self.model_velocities + self.v0_mean
        # self.model_dispersions = jnp.array(sigma(self.x, self.y, self.sigma0_mean_model))
        self.model_dispersions = self.sigma0_mean_model * \
            jnp.ones_like(self.model_velocities)

        #compute posterior and prior for the rotation velocity v_r = v_obs/sin i = 1/2*(vmax - vmin)/sin i where 
        #vmax and vmin are computed within the mask
        inference_data.posterior['v_min'] = self.v(X,Y, inference_data.posterior['PA'].data, inference_data.posterior['i'].data, inference_data.posterior['Va'].data, inference_data.posterior['r_t'].data)
        inference_data.prior['v_min'] = self.v(X,Y, inference_data.prior['PA'].data, inference_data.prior['i'].data, inference_data.prior['Va'].data, inference_data.prior['r_t'].data)

        inference_data.posterior['v_max'] = self.v(X,Y, inference_data.posterior['PA'].data, inference_data.posterior['i'].data, inference_data.posterior['Va'].data, inference_data.posterior['r_t'].data)
        inference_data.prior['v_max'] = self.v(X,Y, inference_data.prior['PA'].data, inference_data.prior['i'].data, inference_data.prior['Va'].data, inference_data.prior['r_t'].data)

        inference_data.posterior['v_r'] = 0.5*(inference_data.posterior['v_max'] - inference_data.posterior['v_min'])/jnp.sin(inference_data.posterior['i'].data)
        inference_data.prior['v_r'] = 0.5*(inference_data.prior['v_max'] - inference_data.prior['v_min'])/jnp.sin(inference_data.prior['i'].data)

        self.model_flux = utils.oversample(
            self.fluxes_mean, grism_object.factor, grism_object.factor)

        self.model_map_high = grism_object.disperse(
            self.model_flux, self.model_velocities, self.model_dispersions)
        self.model_map = utils.resample(
            self.model_map_high, self.y_factor*grism_object.factor, grism_object.wave_factor)
        # self.model_map = resample(self.model_map_high, self.factor, self.wave_factor
        return self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions
