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
from jax import image

# scipy and its functions
from scipy.constants import pi
from scipy.ndimage import measurements

# numpyro and its functions
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam

from matplotlib import pyplot as plt

from photutils import centroids

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

    def set_main_bounds(self, flux_prior, flux_bounds, flux_type,flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds,
                         sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel,
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
        self.flux_threshold = flux_threshold
        self.PA_bounds = PA_bounds
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
        self.y0_vel = y0_vel

        if self.y0_vel == None:
            self.x0_vel = x0
            self.y0_vel = y0

    def compute_flux_bounds(self):
        """
            Compute the flux bounds for the model. 
            If the fluxes are negative, set their prior to zero
        """
        self.mu = jnp.maximum(
            jnp.zeros(self.flux_prior.shape), jnp.array(self.flux_prior))
        self.std = jnp.maximum(0.000001, self.flux_bounds[0]*self.mu)
        self.high = (jnp.maximum(
            0.000002, self.flux_bounds[1] * self.mu) + self.mu - self.mu)/self.std
        self.low = (jnp.zeros(self.flux_prior.shape)-self.mu)/self.std
    
    def rescale_to_mask(self, array, mask):
        """
            Rescale the bounds to the mask
        """
        rescaled_array = []
        for a in array:
            a = a[jnp.where(mask == 1)]
            rescaled_array.append(a)
        return rescaled_array


class Disk():
    """
        Class for 1 disk object. Combinations of this will be used for the single disk model, 
        then 2 disks for the 2 component ones etc
    """
    def __init__(self, direct_shape, masked_indices, mu, std, low, high, mu_PA, sigma_PA, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, x0_vel, y0_vel, number = ''):
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
        self.Va_bounds = Va_bounds
        self.r_t_bounds = r_t_bounds
        self.sigma0_bounds = sigma0_bounds

        self.x0_vel = x0_vel
        self.y0_vel = y0_vel

        self.number = number

    
    def sample_params(self):
        """
            Sample all of the parameters needed to model a disk velocity field
        """
        #sample the fluxes within the mask
        fluxes_scaling = numpyro.sample('fluxes_scaling' + self.number, dist.Uniform())*5
        fluxes_sample = numpyro.sample('fluxes'+ self.number, dist.Uniform(), sample_shape=(int(len(self.masked_indices[0])),))
        fluxes_sample = norm.ppf(norm.cdf(self.low) + fluxes_sample*(norm.cdf(self.high)-norm.cdf(self.low)))*self.std + self.mu*fluxes_scaling

        fluxes = jnp.zeros(self.direct_shape)
        fluxes = fluxes.at[self.masked_indices].set(fluxes_sample)

        Pa = numpyro.sample('PA'+ self.number, dist.Uniform())
        # sample the mu_PA + 0 or 180 (orientation of velocity field)
        rotation = numpyro.sample('rotation'+ self.number, dist.Uniform())

        # simulate a bernouilli discrete distribution
        PA_morph = self.mu_PA + round(rotation)*180

        Pa = norm.ppf(Pa)*self.sigma_PA + PA_morph
        # Pa = norm.ppf(  norm.cdf(self.low_PA) + Pa*(norm.cdf(self.high_PA)-norm.cdf(self.low_PA)) )*self.sigma_PA + self.mu_PA

        #could probably use utils for this too
        i = numpyro.sample('i' + self.number, dist.Uniform()) * \
            (self.i_bounds[1]-self.i_bounds[0]) + self.i_bounds[0]

        Va = numpyro.sample('Va' + self.number, dist.Uniform()) * \
            (self.Va_bounds[1]-self.Va_bounds[0]) + self.Va_bounds[0]

        r_t = numpyro.sample('r_t' + self.number, dist.Uniform(
        ))*(self.r_t_bounds[1]-self.r_t_bounds[0]) + self.r_t_bounds[0]

        sigma0 = numpyro.sample('sigma0'+ self.number, dist.Uniform(
        ))*(self.sigma0_bounds[1]-self.sigma0_bounds[0]) + self.sigma0_bounds[0]

        # sampling the y axis velocity centroids
        # x0 = numpyro.sample('x0', dist.Uniform())
        # x0 = norm.ppf(x0)*(2) + self.x0
        y0_vel = numpyro.sample('y0_vel'+ self.number, dist.Uniform())
        y0_vel = norm.ppf(y0_vel)*(2) + self.y0_vel

        # sample a global velicity shift v0:
        v0 = numpyro.sample('v0'+ self.number, dist.Uniform())
        v0 = norm.ppf(v0)*100


        return fluxes, Pa, i, Va, r_t, sigma0, y0_vel, v0
    
    def compute_posterior_means(self, inference_data, flux_type = 'manual'):
        """
            Retreive the best sample from the MCMC chains for the main disk variables
        """
        best_indices = np.unravel_index(inference_data['sample_stats']['lp'].argmin(
        ), inference_data['sample_stats']['lp'].shape)

        # rescale all of the posteriors from uniform to the actual parameter space
        rotation = float(inference_data.posterior['rotation' + self.number].median(dim=["chain", "draw"]))
        #create lists with variables and their scaling parameters 
        variables = ['PA' + self.number, 'i'+ self.number, 'Va'+ self.number, 'r_t'+ self.number, 'sigma0'+ self.number, 'y0_vel'+ self.number, 'v0'+ self.number, 'fluxes_scaling'+ self.number]
        #for variables drawn from uniform dist, the scaling parameters are (low, high) so mu and sigma are set to none
        mus = [self.mu_PA + round(rotation)*180, None, None, None, None, self.y0_vel, 0.0, None]
        sigmas = [self.sigma_PA, None, None, None, None, 2.0, 100.0, None]

        #for variables drawn from normal dist, the scaling parameters are (mu, sigma) so low and high are set to none
        highs = [None, self.i_bounds[1], self.Va_bounds[1], self.r_t_bounds[1], self.sigma0_bounds[1], None, None, 5.0]
        lows = [None,self.i_bounds[0], self.Va_bounds[0], self.r_t_bounds[0], self.sigma0_bounds[0], None, None, 0.0]

        #find the best sample for each variable in the list of variables
        best_sample = utils.find_best_sample(inference_data, variables, mus, sigmas, highs, lows, best_indices)

        self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean, self.fluxes_scaling_mean = best_sample
        
        #if the fluxes are manually rescaled in the prior, then rescale them

        if flux_type == 'manual':
            best_flux_sample = utils.find_best_sample(inference_data, ['fluxes'+ self.number], [self.mu*self.fluxes_scaling_mean], [self.std], [self.high], [self.low], best_indices)

        self.fluxes_sample_mean = jnp.array(inference_data.posterior['fluxes'+ self.number].isel(chain=best_indices[0], draw=best_indices[1]))
        self.fluxes_mean = jnp.zeros(self.direct_shape)
        self.fluxes_mean =self.fluxes_mean.at[self.masked_indices].set(self.fluxes_sample_mean)

        return  self.fluxes_mean, self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean, self.fluxes_scaling_mean

    
    def v_rot(self, fluxes_mean, model_velocities, i_mean,factor):
        """
            Compute the rotational velocity of the disk component

            If called from multiple component model, the 3 attributes of this function should be only from that component
        """
        threshold = 0.4*fluxes_mean.max()
        mask = jnp.zeros_like(fluxes_mean)
        mask = mask.at[jnp.where(fluxes_mean>threshold)].set(1)
        model_velocities_low = jax.image.resize(model_velocities, (int(model_velocities.shape[0]/factor), int(model_velocities.shape[1]/factor)), method='nearest')
        model_v_rot = 0.5*(jnp.nanmax(jnp.where(mask == 1, model_velocities_low, jnp.nan)) - jnp.nanmin(jnp.where(mask == 1, model_velocities_low, jnp.nan)))/jnp.sin( jnp.radians(i_mean)) 

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
        plt.scatter(self.x0_vel, self.y0_vel, color='red')
        plt.title('Disk' + self.number)
        plt.show()



class DiskModel(KinModels):
    """
        Class for the one component exponential disk model
    """

    def __init__(self):
        print('Disk model created')

        #declare var and label names for plotting

        self.var_names = ['PA', 'i', 'Va', 'r_t', 'sigma0']
        self.labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$', r'$V_r$']

    def set_bounds(self, flux_prior, flux_bounds, flux_type, flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel):
        """

        Compute all of the necessary bounds for the disk model sampling distributions

        """
        # first set all of the main bounds taken from the config file
        self.set_main_bounds(flux_prior, flux_bounds, flux_type, flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel,
                             delta_V_bounds=None, clump=None, clump_v_prior=None, clump_sigma_prior=None, clump_flux_prior=None)

        # now compute the specific bounds for the disk model
        self.mu_PA = self.PA_bounds[0]
        self.sigma_PA = self.PA_bounds[1]*90

        self.compute_flux_bounds()

        #make the mask here!!!
        self.mask = utils.make_mask(self.flux_prior, 1, flux_threshold, 6)

        #compute the PA here too 
        #not sure about this, maybe only need to recompute indiviudal PAs when 2 masks are used

        # correct for the mask = 0 pixels
        # only fitting for pixels in the mask
        self.mask = jnp.array(self.mask)
        self.mask_shape = len(jnp.where(self.mask == 1)[0])
        self.masked_indices = jnp.where(self.mask == 1)
        self.mu, self.std, self.high, self.low = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask)
        #initialize the disk object
        self.disk = Disk(self.flux_prior.shape, self.masked_indices, self.mu, self.std, self.low, self.high, 
                    self.mu_PA, self.sigma_PA, self.i_bounds, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel, self.y0_vel)
        
        self.disk.plot()

    def inference_model(self, grism_object, obs_map, obs_error):
        """

        Model used to infer the disk parameters from the data => called in fitting.py as the forward
        model used for the inference

        """

        # sample the fluxes within the mask from a truncated normal distribution

        fluxes, Pa, i, Va, r_t, sigma0, y0_vel, v0 = self.disk.sample_params()

        fluxes = utils.oversample(fluxes, grism_object.factor, grism_object.factor)

        # create new grid centered on those centroids
        x = jnp.linspace(0 - self.x0_vel, self.flux_prior.shape[1]-1 - self.x0_vel, self.flux_prior.shape[1]*grism_object.factor)
        y = jnp.linspace(0 - y0_vel, self.flux_prior.shape[0]-1 - y0_vel, self.flux_prior.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        # sample for a shift in the y velocity centroid (since the x vel centroid is degenerate with the delta V that is sampled below)

        velocities = jnp.array(self.v(X, Y, jnp.radians(Pa), jnp.radians(i), Va, r_t))
        # velocities = jnp.array(v(self.x, self.y, jnp.radians(Pa),jnp.radians(i), Va, r_t))
        # velocities = image.resize(velocities, (int(velocities.shape[0]/10), int(velocities.shape[1]/10)), method='nearest')

        velocities = velocities + v0

        dispersions = sigma0*jnp.ones_like(velocities)

        self.model_map = grism_object.disperse(fluxes, velocities, dispersions)

        self.model_map = utils.resample(self.model_map, grism_object.y_factor*grism_object.factor, grism_object.wave_factor)

        self.error_scaling = numpyro.sample('error_scaling', dist.Uniform(0, 1))*9 + 1
        # self.error_scaling = 1
        numpyro.sample('obs', dist.Normal(self.model_map, self.error_scaling*obs_error), obs=obs_map)

    def compute_model(self, inference_data, grism_object):
        """

        Function used to post-process the MCMC samples and plot results from the model

        """

        self.fluxes_mean, self.PA_mean,self.i_mean, self.Va_mean, self.r_t_mean, self.sigma0_mean_model, self.y0_vel_mean, self.v0_mean, self.fluxes_scaling_mean = self.disk.compute_posterior_means(inference_data)

        self.model_flux = utils.oversample(self.fluxes_mean, grism_object.factor, grism_object.factor)

        self.x0_vel_mean = self.x0_vel #change this later to put the vel offset into the x0_vel_mean

        x = jnp.linspace(
            0 - self.x0_vel_mean, self.flux_prior.shape[1] - 1 - self.x0_vel_mean, self.flux_prior.shape[1]*grism_object.factor)
        y = jnp.linspace(
            0 - self.y0_vel_mean, self.flux_prior.shape[0] - 1 - self.y0_vel_mean, self.flux_prior.shape[0]*grism_object.factor)
        X, Y = jnp.meshgrid(x, y)

        self.model_velocities = jnp.array(self.v(X, Y, jnp.radians(
            self.PA_mean), jnp.radians(self.i_mean), self.Va_mean, self.r_t_mean))
        # self.model_velocities = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/10), int(self.model_velocities.shape[1]/10)), method='bicubic')

        self.model_velocities = self.model_velocities  + self.v0_mean

        self.model_dispersions = self.sigma0_mean_model *jnp.ones_like(self.model_velocities)


        self.model_map_high = grism_object.disperse(self.model_flux, self.model_velocities, self.model_dispersions)
        self.model_map = utils.resample(self.model_map_high, grism_object.factor*grism_object.y_factor, grism_object.wave_factor)

        self.model_v_rot = self.disk.v_rot(self.fluxes_mean, self.model_velocities, self.i_mean, grism_object.factor)

        #compute velocity grid in flux image resolution for plotting velocity maps
        self.model_velocities_low = image.resize(self.model_velocities, (int(self.model_velocities.shape[0]/grism_object.factor), int(self.model_velocities.shape[1]/grism_object.factor)), method='nearest')

        return self.model_map, self.model_flux, self.fluxes_mean, self.model_velocities, self.model_dispersions

    def plot_summary(self, obs_map, obs_error, inf_data, wave_space):

        plotting.plot_disk_summary(obs_map, self.model_map, obs_error, self.model_velocities_low, self.model_v_rot, self.fluxes_mean, inf_data, wave_space, self.mask, x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = None, name = None)

class TwoComps(KinModels):
    """
        Class for the two component fit (two overlapping disks or two separate disks = merger like scenario)
    """

    def __init__(self):
        #to fill 
        print('fill this')
        

    def set_bounds(self, flux_prior, flux_bounds, flux_type, flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel, dilation_param = 6):
        """

        Compute all of the necessary bounds

        """
        # first set all of the main bounds taken from the config file
        self.set_main_bounds(flux_prior, flux_bounds, flux_type, flux_threshold, PA_bounds, i_bounds, Va_bounds, r_t_bounds, sigma0_bounds, y_factor, x0, x0_vel, y0, y0_vel,
                             delta_V_bounds=None, clump=None, clump_v_prior=None, clump_sigma_prior=None, clump_flux_prior=None)

        #here I need to define 2 Pas, 2 masks, 2 y0_vel + x0_vel

        #start by computing the masks for the 2 components
        #this may be different for clump/merger scenarios but just doing the merger scenario for now
        self.mask1, self.mask2 = utils.make_mask(self.flux_prior, 2, self.flux_threshold, dilation_param)

        #make two separate flux priors with each mask
        self.flux_prior1 = self.flux_prior*self.mask1
        self.flux_prior2 = self.flux_prior*self.mask2

        #compute the PA of each component
        self.mu_PA1 = utils.compute_PA(self.flux_prior1)
        self.mu_PA2 = utils.compute_PA(self.flux_prior2)

        self.sigma_PA1 = self.PA_bounds[1]*90
        self.sigma_PA2 = self.PA_bounds[1]*90

        #the velocity centroids can be initialized at the center of mass of each mask object
        self.x0_vel1, self.y0_vel1 = centroids.centroid_2dg(np.array(self.flux_prior1))
        self.x0_vel2, self.y0_vel2 = centroids.centroid_2dg(np.array(self.flux_prior2))

        #correct for the mask = 0 pixels for each mask
        # only fitting for pixels in the mask
        self.mask1 = jnp.array(self.mask1)
        self.mask2 = jnp.array(self.mask2)

        self.mask_shape1 = len(jnp.where(self.mask1 == 1)[0])
        self.masked_indices1 = jnp.where(self.mask1 == 1)

        self.mask_shape2 = len(jnp.where(self.mask2 == 1)[0])
        self.masked_indices2 = jnp.where(self.mask2 == 1)

        self.compute_flux_bounds()

        self.mu1, self.std1, self.high1, self.low1 = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask1)
        self.mu2, self.std2, self.high2, self.low2 = self.rescale_to_mask([self.mu, self.std, self.high, self.low], self.mask2)

        #initialize the disk objects
        self.disk1 = Disk(self.flux_prior.shape, self.masked_indices1, self.mu1, self.std1, self.low1, self.high1,
                    self.mu_PA1, self.sigma_PA1, self.i_bounds, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel1, self.y0_vel1, number = '1')
        self.disk2 = Disk(self.flux_prior.shape, self.masked_indices2, self.mu2, self.std2, self.low2, self.high2,
                    self.mu_PA2, self.sigma_PA2, self.i_bounds, self.Va_bounds, self.r_t_bounds,
                    self.sigma0_bounds, self.x0_vel2, self.y0_vel2, number = '2')
        
        self.disk1.plot()
        self.disk2.plot()

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
            
        # sample the fluxes within the mask from a truncated normal distribution
        fluxes1, Pa1, i1, Va1, r_t1, sigma01, y0_vel1, v01 = self.disk1.sample_params()
        fluxes2, Pa2, i2, Va2, r_t2, sigma02, y0_vel2, v02 = self.disk2.sample_params()
    
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

        self.fluxes_mean1, self.PA_mean_1,self.i_mean_1, self.Va_mean_1, self.r_t_mean_1, self.sigma0_mean_model_1, self.y0_vel_mean_1, self.v0_mean_1, self.fluxes_scaling_mean_1 = self.disk1.compute_posterior_means(inference_data)
        self.fluxes_mean2, self.PA_mean_2,self.i_mean_2, self.Va_mean_2, self.r_t_mean_2, self.sigma0_mean_model_2, self.y0_vel_mean_2, self.v0_mean_2, self.fluxes_scaling_mean_2 = self.disk2.compute_posterior_means(inference_data)

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

    def plot_summary(self, obs_map, obs_error, inf_data, wave_space):

        plotting.plot_merger_summary(obs_map, self.model_map, obs_error, [self.model_velocities_low1, self.model_velocities_low2], [self.model_v_rot1, self.model_v_rot2], self.fluxes_mean1 + self.fluxes_mean2, inf_data, wave_space, [self.mask1,self.mask2] , x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = None, name = None)

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