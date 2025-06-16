"""
All the plotting related functions for the post-processing
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""
__all__ = ['plot_disk_summary', 'plot_pp_cornerplot']


import matplotlib.pyplot as plt
import numpy as np
import corner
from matplotlib import gridspec
from scipy.constants import c, pi
from jax import image
# import smplotlib
from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog, SegmentationImage
from photutils.background import Background2D
from astropy.convolution import convolve
from astropy.table import Table
from scipy.special import gammainc, gamma
from scipy.optimize import root_scalar




def plot_image(image, x0, y0, direct_size, limits = None, save_to_folder = None, name = None):
	x = np.linspace(0 - x0, direct_size- 1 - x0, image.shape[1])
	y = np.linspace(0 - y0, direct_size- 1 - y0, image.shape[0])
	X, Y = np.meshgrid(x, y)

	if limits == None:
		limits = image

	fig, ax = plt.subplots(figsize = (8,6))
	cp = ax.pcolormesh(X,Y,image,shading= 'nearest', vmax=limits.max(), vmin=limits.min()) #RdBu
	ax.set_xlabel(r'$\Delta$ RA [px]',fontsize = 20)
	ax.set_ylabel(r'$\Delta$ DEC [px]',fontsize = 20)
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp)
	cbar.ax.set_ylabel(r"Flux [a.u.]")
	plt.tight_layout()
	

	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)

	plt.show()
	plt.close()
	
def plot_grism(map, y0, direct_size, wave_space, limits = None, save_to_folder = None, name = None):
	x = wave_space
	y = np.linspace(0 - y0, direct_size- 1 - y0, map.shape[0])
	X, Y = np.meshgrid(x, y)

	if limits == None:
		limits = map

	fig, ax = plt.subplots(figsize = (8,6))
	cp = ax.pcolormesh(X,Y,map,shading= 'nearest', vmax=limits.max(), vmin=limits.min()) #RdBu
	ax.set_xlabel(r'wavelength $[\mu m]$',fontsize = 20)
	ax.set_ylabel(r'$\Delta$ DEC [px]',fontsize = 20)
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp)
	cbar.ax.set_ylabel(r"Flux [a.u.]")
	plt.tight_layout()

	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
	
	plt.show()
	plt.close()

def plot_image_residual(image, model, errors, x0, y0, direct_size,save_to_folder = None, name = None):
	x = np.linspace(0 - x0, direct_size- 1 - x0, image.shape[1])
	y = np.linspace(0 - y0, direct_size- 1 - y0, image.shape[0])
	X, Y = np.meshgrid(x, y)

	fig, ax = plt.subplots(figsize = (8,6))
	cp = ax.pcolormesh(X,Y,(model-image)/image,shading= 'nearest', vmin = -3, vmax  = 3) #RdBu
	ax.set_xlabel(r'$\Delta$ RA [px]',fontsize = 20)
	ax.set_ylabel(r'$\Delta$ DEC [px]',fontsize = 20)
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp)
	cbar.ax.set_ylabel(r"Model-image residuals")
	plt.tight_layout()

	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
	
	plt.show()
	plt.close()

def plot_velocities_nice(kin_model):
	fig, vel_map_ax = plt.subplots(1, 1, figsize=(6, 5))
	#plot the velocity field 
	factor = 1
	x = np.linspace(0 - 15, 31 - 1 - 15, 31)
	y = np.linspace(0 - 15, 31 - 1 - 15, 31)
	X, Y = np.meshgrid(x, y)
	X *= 0.0629/factor#put it in arcseconds
	Y *= 0.0629/factor

	#find the coordinates of the velocity centroid from model_velocities
	# model_velocities = image.resize(kin_model.model_velocities_low, (direct.shape[0]*2, direct.shape[1]*2), method='nearest')
	model_velocities = kin_model.model_velocities_low
	grad_x, grad_y = np.gradient(model_velocities)
	center = np.nanargmax(np.sqrt(grad_y**2 + grad_x**2))
	center = np.unravel_index(center, model_velocities.shape)
	velocites_center = model_velocities[center[0], center[1]]
	# cp = plt.pcolormesh(X[5:26,5:26], Y[5:26,5:26],(model_velocities-velocites_center)[5:26,5:26], shading='nearest', cmap = 'RdBu_r')
	cp = vel_map_ax.pcolormesh(X[5:26,5:26], Y[5:26,5:26],(model_velocities-velocites_center)[5:26,5:26], shading='nearest', cmap = 'RdBu_r')
	vel_map_ax.set_xlabel(r'$\Delta$ RA ["]',fontsize = 5)
	vel_map_ax.set_ylabel(r'$\Delta$ DEC ["]',fontsize = 5)
	vel_map_ax.tick_params(axis='both', which='major', labelsize=5)
	cbar = fig.colorbar(cp, ax = vel_map_ax)
	cbar.ax.set_ylabel('velocity [km/s]', fontsize = 15)
	cbar.ax.tick_params(labelsize = 15)
	vel_map_ax.set_title('Velocity map') #, $v_{rot} = $') # + str(np.round(v_rot)) + ' km/s', fontsize=10)
	# plt.show()

def plot_grism_residual(map, model, errors, y0, direct_size, wave_space,save_to_folder = None, name = None):
	x = wave_space
	y = np.linspace(0 - y0, direct_size- 1 - y0, map.shape[0])
	X, Y = np.meshgrid(x, y)

	fig, ax = plt.subplots(figsize = (8,6))
	cp = ax.pcolormesh(X,Y,(model-map)/errors,shading= 'nearest') #RdBu
	ax.set_xlabel(r'wavelength $[\mu m]$',fontsize = 20)
	ax.set_ylabel(r'$\Delta$ DEC [px]',fontsize = 20)
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp)
	cbar.ax.set_ylabel(r"Model-image")
	plt.tight_layout()

	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)

	plt.show()
	plt.close()


def plot_velocity_profile(image, x0, y0, direct_size, velocities, save_to_folder = None, name = None):
	x = np.linspace(0 - x0, direct_size- 1 - x0, velocities.shape[1])
	y = np.linspace(0 - y0, direct_size- 1 - y0, velocities.shape[0])
	X, Y = np.meshgrid(x, y)

	extent = -y0, y0, -x0, x0
	plt.imshow(image, origin = 'lower', extent = extent, cmap = 'binary')
	CS = plt.contour(X,Y,velocities, 7, cmap = 'RdBu_r', origin = 'lower')
	cbar =plt.colorbar(CS)
	plt.tick_params(axis='both', which='major', labelsize=11)
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 11)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 11)
	cbar.ax.set_ylabel('velocity [km/s]')
	cbar.add_lines(CS)
	plt.tight_layout()

	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)

	plt.show()
	plt.close()

def plot_velocity_map(velocities, mask, x0, y0, direct_size,save_to_folder = None, name = None):
	x = np.linspace(0 - x0, direct_size- 1 - x0, direct_size)
	y = np.linspace(0 - y0, direct_size- 1 - y0, direct_size)
	X, Y = np.meshgrid(x, y)

	if mask.shape[0]!=velocities.shape[0]:
		velocities = image.resize(velocities, (int(velocities.shape[0]/2), int(velocities.shape[1]/2)), method='nearest')

	plt.imshow(np.where(mask ==1, velocities, np.nan), origin = 'lower', cmap = 'RdBu_r')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 11)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 11)
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('velocity [km/s]')
	plt.tight_layout()


	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)

	plt.show()
	plt.close()

def plot_summary(image, image_model, image_error, map, map_model, map_error, x0, y0, direct_size, wave_space, title = None, save_to_folder = None, name = None):
	x = np.linspace(0 - x0, direct_size- 1 - x0, image.shape[1])
	y = np.linspace(0 - y0, direct_size- 1 - y0, image.shape[0])
	X, Y = np.meshgrid(x, y)

	fig, axs = plt.subplots(2, 3, figsize=(50, 30))

	cp = axs[0,0].pcolormesh(X,Y,image,shading= 'nearest', vmax=image.max(), vmin=image.min()) #RdBu
	axs[0,0].set_xlabel(r'$\Delta$ RA [px]',fontsize = 30)
	axs[0,0].set_ylabel(r'$\Delta$ DEC [px]',fontsize = 30)
	axs[0,0].tick_params(axis='both', which='major', labelsize=30)
	axs[0,0].tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=axs[0,0])
	cbar.ax.set_ylabel(r"Flux [a.u.]")
	cbar.ax.tick_params(labelsize=30)
	axs[0,0].set_title('Observed image', fontsize = 50)
	# plt.tight_layout()

	cp = axs[0,1].pcolormesh(X,Y,image_model,shading= 'nearest', vmax=image.max(), vmin=image.min()) #RdBu
	axs[0,1].set_xlabel(r'$\Delta$ RA [px]',fontsize = 30)
	axs[0,1].set_ylabel(r'$\Delta$ DEC [px]',fontsize = 30)
	axs[0,1].tick_params(axis='both', which='major', labelsize=30)
	axs[0,1].tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=axs[0,1])
	cbar.ax.set_ylabel(r"Flux [a.u.]")
	cbar.ax.tick_params(labelsize=30)	
	axs[0,1].set_title('Model image', fontsize = 50)

	cp = axs[0,2].pcolormesh(X,Y,(image_model-image)/image_error,shading= 'nearest') #RdBu
	axs[0,2].set_xlabel(r'$\Delta$ RA [px]',fontsize = 30)
	axs[0,2].set_ylabel(r'$\Delta$ DEC [px]',fontsize = 30)
	axs[0,2].tick_params(axis='both', which='major', labelsize=30)
	axs[0,2].tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=axs[0,2])
	cbar.ax.set_ylabel(r"Chi")
	cbar.ax.tick_params(labelsize=30)
	axs[0,2].set_title('Residuals', fontsize = 50)

	x = wave_space
	y = np.linspace(0 - y0, direct_size- 1 - y0, map.shape[0])
	X, Y = np.meshgrid(x, y)

	cp = axs[1,0].pcolormesh(X,Y,map,shading= 'nearest', vmax=map.max(), vmin=map.min()) #RdBu
	axs[1,0].set_xlabel(r'wavelength $[\mu m]$',fontsize = 30)
	axs[1,0].set_ylabel(r'$\Delta$ DEC [px]',fontsize = 30)
	axs[1,0].tick_params(axis='both', which='major', labelsize=30)
	axs[1,0].tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=axs[1,0])
	cbar.ax.set_ylabel(r"Flux [a.u.]")
	cbar.ax.tick_params(labelsize=30)
	axs[1,0].set_title('Observed grism', fontsize = 50)

	cp = axs[1,1].pcolormesh(X,Y,map_model,shading= 'nearest', vmax=map.max(), vmin=map.min()) #RdBu
	axs[1,1].set_xlabel(r'wavelength $[\mu m]$',fontsize = 30)
	axs[1,1].set_ylabel(r'$\Delta$ DEC [px]',fontsize = 30)
	axs[1,1].tick_params(axis='both', which='major', labelsize=30)
	axs[1,1].tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=axs[1,1])
	cbar.ax.set_ylabel(r"Flux [a.u.]")
	cbar.ax.tick_params(labelsize=30)
	axs[1,1].set_title('Model grism', fontsize = 50)

	cp = axs[1,2].pcolormesh(X,Y,(map_model-map)/map_error,shading= 'nearest') #RdBu
	axs[1,2].set_xlabel(r'wavelength $[\mu m]$',fontsize = 30)
	axs[1,2].set_ylabel(r'$\Delta$ DEC [px]',fontsize = 30)
	axs[1,2].tick_params(axis='both', which='major', labelsize=30)
	axs[1,2].tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=axs[1,2])
	cbar.ax.set_ylabel(r'Chi')
	cbar.ax.tick_params(labelsize=30)
	axs[1,2].set_title('Residuals',	fontsize = 50)

	if title != None:
		fig.suptitle(title, fontsize = 100)
		#add a bigger space between title and rest of figure
		fig.subplots_adjust(top=20)


	plt.tight_layout()

	if save_to_folder != None:
		plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)

	plt.show()
	plt.close()
	
def make_mask(im, sigma_rms, save_to_folder):
	im_conv = convolve(im, make_2dgaussian_kernel(3.0, size=5))
	# print('pre-bckg')
	bkg = Background2D(im_conv, (15, 15), filter_size=(3, 3), exclude_percentile=90.0)
	segment_map = detect_sources(im_conv, sigma_rms*bkg.background_rms, npixels=10)

	ny, nx = segment_map.shape
	yc, xc = ny / 2, nx / 2

	# Get source properties
	catalog = SourceCatalog(im,segment_map)

	# Step 1: Find segment closest to image center
	min_dist = np.inf
	closest_label = None
	closest_obj = None

	for obj in catalog:
		y, x = obj.ycentroid, obj.xcentroid
		dist = np.sqrt((x - xc)**2 + (y - yc)**2)
		if dist < min_dist:
			min_dist = dist
			closest_label = obj.label
			closest_obj = obj

	# Step 2: Create segmentation map with only the closest segment
	closest_segm_array = np.where(segment_map.data == closest_label, closest_label, 0)
	closest_segm = SegmentationImage(closest_segm_array)

	# Step 3: Compute max radius from centroid
	yc_obj, xc_obj = closest_obj.ycentroid, closest_obj.xcentroid
	yy, xx = np.where(closest_segm.data == closest_label)

	# Compute Euclidean distance from centroid
	r_max = np.max(np.sqrt((xx - xc_obj)**2 + (yy - yc_obj)**2))


	# Projected radial extent along y-axis
	y_extent = np.max(np.abs(yy - yc_obj))

	#plot the image and the segmentation map
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	ax[0].imshow(im_conv, origin='lower', cmap='Greys_r', interpolation='nearest')
	ax[0].set_title('Data')
	#plot the r_max
	ax[0].add_patch(plt.Circle((xc_obj, yc_obj), r_max, color='red', fill=False, lw=2, label='r_max'))
	#plot a line of length y_extent along the y axis starting from the center
	ax[0].plot([xc_obj, xc_obj], [yc_obj - y_extent, yc_obj + y_extent], color='red', lw=2, label='y_extent')

	ax[1].imshow(closest_segm, origin='lower', cmap='tab20', interpolation='nearest')
	ax[1].set_title('Segmentation map')

	#plot the r_max
	ax[1].add_patch(plt.Circle((xc_obj, yc_obj), r_max, color='red', fill=False, lw=2, label='r_max'))
	plt.tight_layout()
	plt.savefig('bboxes/' + str(save_to_folder) + '_bbox.png', dpi=300)
	# plt.show()
	plt.close()
	return im_conv, segment_map, r_max, y_extent


def sersic_radius_fraction(r_frac, n, r_eff, frac=0.9):
	"""Solve for r_frac such that it encloses the given light fraction."""
	# Compute b_n, the Sersic coefficient
	b_n = 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2)  # Approximation
	
	# Compute the light fraction enclosed
	enclosed_frac = gammainc(2 * n, b_n * (r_frac / r_eff) ** (1 / n))
	
	return enclosed_frac - frac  # Find root where this is zero

def compute_r90(n, r_eff):
	"""Find the radius that encloses 90% of the total light."""
	result = root_scalar(sersic_radius_fraction, args=(n, r_eff, 0.9), bracket=[r_eff, 10 * r_eff])
	return result.root if result.converged else None


def plot_disk_summary(obs_map, model_map, obs_error, model_velocities, model_dispersions, v_rot, fluxes_mean, inf_data, wave_space, x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = None, name = None,  PA = None, i = None, Va = None, r_t = None, sigma0 = None, obs_radius = None, ellip = None, theta_obs = None, theta_Ha =None, n = None):
	# plt.show()

	fig = plt.figure(constrained_layout=True)
	fig.set_size_inches(7,5)
	# spec = gridspec.GridSpec(ncols=3, nrows=3,
	# 						width_ratios=[2, 4, 3], wspace=0.2,
	# 						hspace=0.5, height_ratios=[1, 1, 1])
	# gs0 = fig.add_gridspec(3, 3, width_ratios=[4,4,4], height_ratios=[1, 1, 3], hspace = 0.05) #, hspace=10) #lines - columns
	gs0 = fig.add_gridspec(2, 3, width_ratios=[4,4,4], height_ratios=[1, 1], hspace = 0.05) #, hspace=10) #lines - columns
	# gs00 = gs0[0:2].subgridspec(nrows = 3, ncols = 2, width_ratios=[1,2])
	# gs01 = gs0[2].subgridspec(3, 1)

	ax_obs =  fig.add_subplot(gs0[0,0] )
	# x = wave_space
	y = np.linspace(0 - y0, direct_image_size- 1 - y0, obs_map.shape[0])
	x = np.linspace(0 - x0, direct_image_size- 1 - x0, obs_map.shape[1])
	X, Y = np.meshgrid(x, y)
	#put Y in arcseconds
	# Y *= 0.063
	cp = ax_obs.pcolormesh(X, Y, obs_map, shading='nearest',cmap = 'BuPu',
						vmax=obs_map.max(), vmin=obs_map.min())  # RdBu




	# cbar = fig.colorbar(cp, ax=ax_obs)
	# cbar.ax.set_ylabel(r"Flux [a.u.]", fontsize=5)
	# cbar.ax.tick_params(labelsize=5)
	#add a bar in the y direction to show the scale
	ax_obs.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=ax_obs.transAxes)
	ax_obs.text(0.18, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=ax_obs.transAxes)
	ax_obs.set_title('Observed grism', fontsize=10)

	# Add axis arrows using plot
	ax_obs.plot([0.05, 0.15], [0.05, 0.05], '-', lw=2, transform=ax_obs.transAxes, clip_on=False, c = 'mediumblue')  # Horizontal arrow
	ax_obs.plot([0.05, 0.05], [0.05, 0.15], '-', lw=2, transform=ax_obs.transAxes, clip_on=False, c = 'forestgreen')  # Vertical arrow

	# Add labels
	ax_obs.text(0.2, 0.05, "Dispersion", fontsize=10, color="mediumblue",
				ha="left", va="center", transform=ax_obs.transAxes)

	ax_obs.text(0.03, 0.3, "Spatial", fontsize=10, color="forestgreen",
				ha="left", va="center", rotation=90, transform=ax_obs.transAxes)


	#fit the observed data with photutils to get its radius
	# make segmentation map and identify sources
	im_conv, segment_map, rmax, rmax_proj = make_mask(obs_map, 5, save_to_folder)

	# get the source properties
	obs_radius = rmax #already computed in the make_mask function 

	obs_radius_proj = rmax_proj
	obs_radius_ax_scale = obs_radius_proj/obs_map.shape[0]

	re_50 = float(inf_data.posterior['r_eff'].quantile(0.5, dim=["chain", "draw"]).values)
	re_50_minor = re_50*(1-ellip)
	#project it on the y axis using the PA
	re_50_proj = np.maximum(re_50*np.abs(np.cos(np.pi/2- theta_Ha)), re_50_minor)
	re_ax_scale = re_50_proj/obs_map.shape[0]
	ax_obs.axis('off')

	#write the obs_radius and re_50 in a table and save it in a file
	params = ['r_eff', 'r_obs']
	t_empty = np.zeros((len(params), 1))
	res = Table(t_empty.T, names=params)
	res['r_eff'] = re_50
	res['r_obs'] = obs_radius


	# res.write('fitting_results/' + save_to_folder + '/radii', format='ascii', overwrite=True)

	ax_obs.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=ax_obs.transAxes)
	ax_obs.text(0.18, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=ax_obs.transAxes)


	#do the same but for the effective radius
	ax_obs.text(0.75, 0.5, r'$2r_{\text{e}}$', color = 'crimson', fontsize = 10, ha='center', va='center', transform=ax_obs.transAxes)
	ax_obs.plot([0.7, 0.7], [0.5 - re_ax_scale, 0.5 + re_ax_scale], c = 'crimson', lw=2, transform=ax_obs.transAxes)

	ax_obs.text(0.9, 0.5, r'$2r_{\text{obs}}$', color = 'orange', fontsize = 10, ha='center', va='center', transform=ax_obs.transAxes)
	ax_obs.plot([0.8, 0.8], [0.5 - obs_radius_ax_scale, 0.5 + obs_radius_ax_scale], c = 'orange', lw=2, transform=ax_obs.transAxes)

	ax_model = fig.add_subplot(gs0[0,1])
	cp = ax_model.pcolormesh(X, Y, model_map, shading='nearest',cmap = 'BuPu', vmax=obs_map.max(), vmin=obs_map.min())  # RdBu
	# ax_model.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	# ax_model.set_ylabel(r'$\Delta$ DEC ["]', fontsize=5)
	# ax_model.tick_params(axis='both', which='major', labelsize=5)
	ax_model.axis('off')
	# cbar = fig.colorbar(cp, ax=ax_model)
	# cbar.ax.set_ylabel(r"Flux [a.u.]", fontsize=5)
	# cbar.ax.tick_params(labelsize=5)

	ax_model.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=ax_model.transAxes)
	ax_model.text(0.18, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=ax_model.transAxes)

	#do the same but for the effective radius
	ax_model.text(0.75, 0.5, r'$2r_{\text{e}}$', color = 'crimson', fontsize = 10, ha='center', va='center', transform=ax_model.transAxes)
	ax_model.plot([0.7, 0.7], [0.5 - re_ax_scale, 0.5 + re_ax_scale], c = 'crimson', lw=2, transform=ax_model.transAxes)

	ax_model.text(0.9, 0.5, r'$2r_{\text{obs}}$', color = 'orange', fontsize = 10, ha='center', va='center', transform=ax_model.transAxes)
	ax_model.plot([0.8, 0.8], [0.5 - obs_radius_ax_scale, 0.5 + obs_radius_ax_scale], c = 'orange', lw=2, transform=ax_model.transAxes)
	
	ax_model.set_title('Model grism', fontsize=10)


	chi = (model_map-obs_map)/obs_error
	chi_icenter,chi_jcenter = np.unravel_index(np.argmax(obs_map), obs_map.shape)
	chi_central_region = chi[chi_icenter-10:chi_icenter+10,chi_jcenter-10:chi_jcenter+10]
	ax_residuals = fig.add_subplot(gs0[0,2])
	cp = ax_residuals.pcolormesh(X, Y, chi, shading='nearest', cmap = 'BuPu' , vmin = -5, vmax = 5)  # RdBu
	# ax_residuals.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	# ax_residuals.set_ylabel(r'$\Delta$ DEC ["]', fontsize=5)
	# ax_residuals.tick_params(axis='both', which='major', labelsize=5)
	ax_residuals.axis('off')
	# cbar = fig.colorbar(cp, ax=ax_residuals)
	# cbar.ax.set_ylabel(r"$\chi$", fontsize=5)
	# cbar.ax.tick_params(labelsize=5)
	ax_residuals.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=ax_residuals.transAxes)
	ax_residuals.text(0.2, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=ax_residuals.transAxes)

	ax_residuals.text(0.75, 0.5, r'$2r_{\text{e}}$', color = 'crimson', fontsize = 10, ha='center', va='center', transform=ax_residuals.transAxes)
	ax_residuals.plot([0.7, 0.7], [0.5 - re_ax_scale, 0.5 + re_ax_scale], c = 'crimson', lw=2, transform=ax_residuals.transAxes)

	ax_residuals.text(0.9, 0.5, r'$2r_{\text{obs}}$', color = 'orange', fontsize = 10, ha='center', va='center', transform=ax_residuals.transAxes)
	ax_residuals.plot([0.8, 0.8], [0.5 - obs_radius_ax_scale, 0.5 + obs_radius_ax_scale], c = 'orange', lw=2, transform=ax_residuals.transAxes)

	# ax_residuals.set_title(r'$\bar\chi_{center} = $' + str(round(np.sum(np.abs(chi_central_region))/(chi_central_region.shape[0]*chi_central_region.shape[1]),3)), fontsize=10)
	ax_residuals.set_title(r'Residual $\chi$ map', fontsize=10)

	x = np.linspace(0 - x0, direct_image_size - 1 - x0, direct_image_size)
	y = np.linspace(0 - y0, direct_image_size - 1 - y0, direct_image_size)
	X, Y = np.meshgrid(x, y)
	# X *= 0.0629/factor#put it in arcseconds
	# Y *= 0.0629/factor

	#compute the maximum velocity within the main label
	#find y-pixel extend of the main label
	ymin = np.min(np.where(segm_deblend.data == main_label)[0])
	ymax = np.max(np.where(segm_deblend.data == main_label)[0])
	#find the max velocity in the direct image within the main label
	v_max = np.nanmax(np.abs(model_velocities)[ymin:ymax+1, :])
	# print('v_max', v_max)
	# #plot the velocity map with the upper and lower limits
	# plt.imshow(model_velocities, origin = 'lower', cmap = 'RdBu_r')
	# #horizonal line
	# plt.plot([0, direct_image_size], [ymin, ymin], 'k-', lw=2)
	# plt.plot([0, direct_image_size], [ymax, ymax], 'k-', lw=2)
	# plt.show()


	#find the coordinates of the velocity centroid from model_velocities
	grad_x, grad_y = np.gradient(model_velocities)
	center = np.nanargmax(np.sqrt(grad_y**2 + grad_x**2))
	center = np.unravel_index(center, model_velocities.shape)
	v0 = inf_data.posterior['v0'].quantile(0.5, dim=["chain", "draw"]).values
	#the center is the pixel whose value is closest to v0
	# center = [15,16] #np.unravel_index(np.nanargmin(np.abs(model_velocities - v0)), model_velocities.shape)
	velocites_center = model_velocities[center[0], center[1]]
	vel_map_ax = fig.add_subplot(gs0[1,0])
	cp = vel_map_ax.pcolormesh(X, Y,(model_velocities - v0), shading='nearest', cmap = 'RdBu_r')
	# plt.xlabel(r'$\Delta$ RA ["]',fontsize = 5)
	# plt.ylabel(r'$\Delta$ DEC ["]',fontsize = 5)
	vel_map_ax.axis('off')
	from mpl_toolkits.axes_grid1.inset_locator import inset_axes

	from matplotlib.patches import Ellipse

	# Define ellipse parameters
	center_x = (center[1]-x0)  # Assuming the ellipse is centered at the median x position
	center_y = (center[0]-y0) # If the galaxy is centered at y=0 in arcsec
	width = 2*obs_radius  # Major axis (r_obs is the semi-major axis)
	height = 2 * (1 - ellip)*obs_radius  # Minor axis, where ellip is the ellipticity (ellip = 1 - b/a)
	angle = -np.degrees(theta_Ha)  # Rotation angle in degrees

	# Create and add the ellipse
	ellipse_robs = Ellipse((center_x, center_y),width,height, angle=angle,
					edgecolor='orange', facecolor='none', linewidth=2, alpha = 0.5)
	ellipse_robs2 = Ellipse((center_x, center_y),width,height, angle=angle,
			 					edgecolor='orange', facecolor='none', linewidth=2, alpha = 0.5)	
	ellipse_robs3 = Ellipse((center_x, center_y),width,height, angle=angle,
			 					edgecolor='orange', facecolor='none', linewidth=2, alpha = 0.5)

	vel_map_ax.add_patch(ellipse_robs)

	# Define ellipse parameters

	width = 2*re_50  # Major axis (r_obs is the semi-major axis)
	height = 2 * (1 - ellip)*re_50  # Minor axis, where ellip is the ellipticity (ellip = 1 - b/a)
	angle = -np.degrees(theta_Ha)  # Rotation angle in degrees

	# Create and add the ellipse
	ellipse_re = Ellipse((center_x, center_y),width,height, angle=angle,
					edgecolor='crimson', facecolor='none', linewidth=2, alpha = 0.5)
	ellipse_re2 = Ellipse((center_x, center_y),width,height, angle=angle,
					edgecolor='crimson', facecolor='none', linewidth=2, alpha = 0.5)
	ellipse_re3 = Ellipse((center_x, center_y),width,height, angle=angle,
					edgecolor='crimson', facecolor='none', linewidth=2, alpha = 0.5)

	vel_map_ax.add_patch(ellipse_re)



	# Define inset axes for the colorbar (position and size)
	cax = inset_axes(vel_map_ax, width="30%", height="5%", loc="lower right", borderpad=0.5)
	cbar = plt.colorbar(cp, cax = cax, orientation='horizontal')
	cbar.ax.tick_params(labelsize = 10)
	vel_map_ax.set_title(r'$v_{\text{obs}}$ map', fontsize=10)

	vel_map_ax.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=vel_map_ax.transAxes)
	vel_map_ax.text(0.2, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=vel_map_ax.transAxes)

	# Add axis arrows using plot
	# vel_map_ax.plot([0.05, 0.15], [0.05, 0.05], '-', lw=2, transform=vel_map_ax.transAxes, clip_on=False, c = 'forestgreen')  # Horizontal arrow
	# vel_map_ax.plot([0.05, 0.05], [0.05, 0.15], '-', lw=2, transform=vel_map_ax.transAxes, clip_on=False, c = 'forestgreen')  # Vertical arrow

	# Add labels
	# vel_map_ax.text(0.2, 0.05, "Spatial", fontsize=10, color="forestgreen",
	# 			ha="left", va="center", transform=vel_map_ax.transAxes)

	# vel_map_ax.text(0.03, 0.3, "Spatial", fontsize=10, color="forestgreen",
	# 			ha="left", va="center", rotation=90, transform=vel_map_ax.transAxes)



	vel_map_ax.plot((center[1]-x0), (center[0]-y0), '+', markersize=10, label = 'velocity centroid', color = 'black') #*0.0629/factor
	# vel_map_ax.legend(fontsize = 10, loc = 'lower right', borderaxespad = 2)

	veldisp_map_ax = fig.add_subplot(gs0[1,1])

	veldisp_map_ax.pcolormesh(X, Y,model_dispersions, shading='nearest', cmap = 'RdBu_r', vmin = np.nanmin(model_velocities - velocites_center), vmax = np.nanmax(model_velocities - velocites_center))
	# plt.xlabel(r'$\Delta$ RA ["]',fontsize = 5)
	# plt.ylabel(r'$\Delta$ DEC ["]',fontsize = 5)
	# veldisp_map_ax.tick_params(axis='both', which='major', labelsize=5)
	# cbar = plt.colorbar(cp, ax = veldisp_map_ax)
	# cbar.ax.set_ylabel(r'$\sigma_v$ [km/s]', fontsize = 5)
	veldisp_map_ax.axis('off')
	veldisp_map_ax.set_title(r'$\sigma_0$ map', fontsize=10)
	veldisp_map_ax.plot((center[1]-x0), (center[0]-y0), '+', markersize=10, label = 'velocity centroid', color = 'black')
	veldisp_map_ax.legend(fontsize = 10, loc = 'lower right',borderaxespad = 2)

	veldisp_map_ax.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=veldisp_map_ax.transAxes)
	veldisp_map_ax.text(0.2, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=veldisp_map_ax.transAxes)

	cax = inset_axes(veldisp_map_ax, width="30%", height="5%", loc="upper right", borderpad=0.5)
	cbar = plt.colorbar(cp, cax = cax, orientation='horizontal')
	cbar.ax.tick_params(labelsize = 10)

	veldisp_map_ax.add_patch(ellipse_re2)
	veldisp_map_ax.add_patch(ellipse_robs2)

	flux_map_ax = fig.add_subplot(gs0[1,2])

	cp = flux_map_ax.pcolormesh(X, Y,fluxes_mean, shading='nearest', cmap = 'BuPu')
	# plt.xlabel(r'$\Delta$ RA ["]',fontsize = 5)
	# plt.ylabel(r'$\Delta$ DEC ["]',fontsize = 5)
	flux_map_ax.axis('off')
	# cbar = plt.colorbar(cp, ax = flux_map_ax)
	# cbar.ax.set_ylabel('flux [Mjy?]', fontsize = 5)
	# cbar.ax.tick_params(labelsize = 5)
	flux_map_ax.set_title(r'H$\alpha$ map', fontsize=10)
	flux_map_ax.plot((center[1]-x0), (center[0]-y0), '+', markersize=10, label = 'velocity centroid', color = 'black')
	# flux_map_ax.legend(fontsize = 10, loc = 'lower right',borderaxespad = 2)

	flux_map_ax.plot([0.1, 0.1], [0.37, 0.63], 'k-', lw=2, transform=flux_map_ax.transAxes)
	flux_map_ax.text(0.2, 0.5, '0.5"', color = 'black', fontsize = 10, ha='center', va='center', rotation = 90, transform=flux_map_ax.transAxes)

	flux_map_ax.add_patch(ellipse_re3)
	flux_map_ax.add_patch(ellipse_robs3)


	# fig.suptitle('Object JADES ID: ' + str(save_to_folder), fontsize=15, fontweight='bold')
	# fig.savefig('FrescoHa/GoldSummaries/' + str(save_to_folder) + '.png', dpi=500)

	if save_to_folder != None:
		if name == 'summary':
			fig.savefig('fitting_results/' + save_to_folder + '/summary.png', dpi=300)
		else:
			fig.savefig('testing/' + save_to_folder + '/' + name + '_summary.png', dpi=500)


	# gs1 = gs0[2,:]

	fig, corner_ax = plt.subplots(figsize=(6, 5))
	# corner_ax = fig.add_subplot(gs1)

 #, 'fluxes_scaling': None}
	
	v_sigma_16 = float(inf_data.posterior['v_sigma'].quantile(0.16, dim=["chain", "draw"]).values)
	v_sigma_84 = float(inf_data.posterior['v_sigma'].quantile(0.84, dim=["chain", "draw"]).values)
	v_sigma_50 = float(inf_data.posterior['v_sigma'].quantile(0.5, dim=["chain", "draw"]).values)

	sigma0_16 = float(inf_data.posterior['sigma0'].quantile(0.16, dim=["chain", "draw"]).values)
	sigma0_84 = float(inf_data.posterior['sigma0'].quantile(0.84, dim=["chain", "draw"]).values)
	sigma0_50 = float(inf_data.posterior['sigma0'].quantile(0.5, dim=["chain", "draw"]).values)

	v_re_50 = float(inf_data.posterior['v_re'].quantile(0.5, dim=["chain", "draw"]).values)
	v_re_16 = float(inf_data.posterior['v_re'].quantile(0.16, dim=["chain", "draw"]).values)
	v_re_84 = float(inf_data.posterior['v_re'].quantile(0.84, dim=["chain", "draw"]).values)
	
	v_sigma_min = 0.5*v_sigma_16
	v_sigma_max = 1.5*v_sigma_84
		
	sigma0_min = 0.5*sigma0_16
	sigma0_max = 1.5*sigma0_84

	# range = [(None,None), (None,None), (None,None), (None,None), (sigma0_min,sigma0_max), (None,None), (None,None), (None,None), (v_sigma_min,v_sigma_max), (None,None)]
	CORNER_KWARGS = dict(
		smooth=2,
		smooth1d = 5,
		label_kwargs=dict(fontsize=20),
		title_kwargs=dict(fontsize=20),
		plot_density=False,
		plot_datapoints=True,
		fill_contours=True,
		plot_contours=True,
		# labels=[r'PA [deg]', r'$i$ [deg]', r'$V_a$ [km/s]', r'$r_t$ [px]', r'$\sigma_0$ [km/s]',r'n', r'$r_e$ [px]', r'$v(r_e)$ [km/s]' ], # r'$x_{0}$ [px]', r'$y_{0}$ [px]'],
		show_titles=False,
		levels = [0.05,0.16,0.5,0.68, 0.95],
		alpha = 0.1,
		max_n_ticks=3)
	bin_factor = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
	figure = corner.corner(inf_data, group='prior', var_names=['PA', 'i', 'Va', 'r_t', 'sigma0',  'PA_morph','amplitude', 'n', 'r_eff', 'xc_morph', 'yc_morph', 'x0_vel', 'y0_vel'],
						color='palevioletred', hist_bin_factor = 4, weights = np.ones_like(inf_data.prior['sigma0'].values[0])*2, **CORNER_KWARGS)
	CORNER_KWARGS = dict(
		smooth=2,
		smooth1d=5,
		label_kwargs=dict(fontsize=20),
		title_kwargs=dict(fontsize=20),
		quantiles=[0.16, 0.5, 0.84],
		plot_density=False,
		plot_datapoints=False,
		fill_contours=True,
		plot_contours=True,
		show_titles=False,
		labels=[r'PA [deg]', r'$i$ [deg]', r'$V_a$ [km/s]', r'$r_t$ [px]', r'$\sigma_0$ [km/s]', r'PA$_{\rm morph}$ [deg]', r'$\text{amplitude}$', r'$n$', r'$r_{\text{e}}$ [px]', r'$x_{0}$ [px]', r'$y_{0}$ [px]', r'$x_{0,v}$ [px]', r'$y_{0,v}$ [px]'],
		titles= [r'PA', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$', r'PA$_{\rm morph}$', r'$\text{amplitude}$', r'$n$', r'$r_{\text{e}}$', r'$x_{0}$', r'$y_{0}$', r'$x_{0, v}$', r'$y_{0,v}$'],
		max_n_ticks=3,
		divergences=False,
		linewidth=2,
		title_fmt = '.1f')
	# truths = { 'PA': PA, 'i': i, 'Va': Va,
	# 					  'r_t': r_t, 'sigma0': sigma0}
	# corner_range = [0.999, 0.999,[-600,0], [0,10], [60,170], 0.999, 0.999, 0.999,[1,6], 0.999, [14,15.5]]
	figure = corner.corner(inf_data, group='posterior', var_names=['PA', 'i', 'Va', 'r_t', 'sigma0', 'PA_morph','amplitude', 'n', 'r_eff', 'xc_morph', 'yc_morph', 'x0_vel', 'y0_vel'],truths = None, truth_color='blue',
						color='royalblue', fig = figure, hist_bin_factor = 4, **CORNER_KWARGS) #range = corner_range, 

	
	#overplot a posteriors colored in green for the last six paramters
	# corner.overplot(figure, inf_data, group='posterior', var_names=['r_eff', 'xc_morph', 'yc_morph', 'PA_morph', 'amplitude', 'n'], color = 'green', range = corner_range, **CORNER_KWARGS)
	#set the axis ratio
	# plt.gca().set(box_aspect=1)
	if save_to_folder != None:
		if name == 'summary':
			plt.savefig('fitting_results/' + str(save_to_folder) + '/cornerplot' + '.png', dpi=300)
			figure_image = plt.imread('fitting_results/' + str(save_to_folder) + '/cornerplot' + '.png')
		elif name == 'pretty':
			plt.savefig('FrescoHa/PrettySummaries/' + save_to_folder + '_corner.png', dpi=500)
			figure_image = plt.imread('FrescoHa/PrettySummaries/' + save_to_folder + '_corner.png')
		else:
			plt.savefig('testing/' + save_to_folder + '/' + name + '_corner.png', dpi=500)
			figure_image = plt.imread('testing/' + save_to_folder + '/' + name + '_corner.png')

	corner_ax.imshow(figure_image, origin = 'upper')
	corner_ax.axis('off')
	corner_ax.text(0.6,0.8, r'$\sigma_0 = $' + str(round(sigma0_50,1)) + r'$^{+' + str(round(sigma0_84- sigma0_50, 1)) + r'}' +  r'_{-' + str(round(sigma0_50- sigma0_16, 1)) + r'}$' + r' km/s', transform=corner_ax.transAxes, fontsize=10, va='top', color='black')
	corner_ax.text(0.6,0.75, r'$v_{re} = $' + str(round(v_re_50,1)) +  r'$^{+' + str(round(v_re_84 - v_re_50,1)) + r'}' +  r'_{-' + str(round(v_re_50 - v_re_16,1)) + r'}$' + r' km/s', transform=corner_ax.transAxes, fontsize=10, va='top', color='black')
	corner_ax.text(0.6,0.7, r'$v/\sigma_0 = $' + str(round(v_sigma_50,1)) +  r'$^{+' + str(round(v_sigma_84 - v_sigma_50,1)) + r'}' +  r'_{-' + str(round(v_sigma_50 - v_sigma_16,1)) + r'}$', transform=corner_ax.transAxes, fontsize=10, va='top', color='black')
# + r'$^{+' + str(round(v_re_84 - v_re_50,2)) + r'$}_{-$' + str(round(v_re_50 - v_re_16,2)) + r'} km/s'
	# corner_ax.set_title('Kinematic posteriors', fontsize=8)

	corner_ax.title.set_position([.5, 0.9])
	fig.suptitle('Object JADES ID: ' + str(save_to_folder), fontsize=10, fontweight='bold')

	if save_to_folder != None:
		if name == 'summary':
				fig.savefig('fitting_results/' + str(save_to_folder) + '/summary_corner' + '.png', dpi=300)
		elif name == 'pretty':
			fig.savefig('FrescoHa/PrettySummaries/' + save_to_folder + '_corner.png', dpi=500)
		else:
			fig.savefig('testing/' + save_to_folder + '/' + name + '_corner.png', dpi=500)
		# fig.savefig('summary_plots/CONGRESS/' + save_to_folder.split('/')[0] + '_' + name + '.png', dpi=500)
	else:
		raise ValueError('Please provide a folder to save the corner plot')
	# plt.show()
	# fig.savefig('fitting_results/' + str(save_to_folder) + '/cornerplot_text' + '.png', dpi=300)
	plt.close()

	return ymin,ymax


def plot_merger_summary(obs_map, model_map, obs_error, model_velocities, v_rot, fluxes_mean, inf_data, wave_space, mask, x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = None, name = None):
	var_names = ['PA1', 'i1', 'Va1', 'r_t1', 'sigma01', 'PA2', 'i2', 'Va2', 'r_t2', 'sigma02']
	labels = [r'$PA_1$', r'$i1$', r'$Va1$', r'$r_{t,1}$', r'$\sigma_{0,1}$', r'$PA_2$', r'$i_2$', r'$Va_2$', r'$r_{t,2}$', r'$\sigma_{0,2}$']
	#here all of the kinematic entries are shaped as [param1, param2] for the two disks
	fig = plt.figure( constrained_layout=True)
	fig.set_size_inches(11, 6)
	# spec = gridspec.GridSpec(ncols=3, nrows=3,
	# 						width_ratios=[2, 4, 3], wspace=0.2,
	# 						hspace=0.5, height_ratios=[1, 1, 1])
	gs0 = fig.add_gridspec(1, 3, width_ratios=[5,5,4], hspace=10)

	gs00 = gs0[0:2].subgridspec(nrows = 3, ncols = 2, width_ratios=[1,2])
	gs01 = gs0[2].subgridspec(3, 1)

	ax_obs =  fig.add_subplot(gs00[0,0])
	x = wave_space
	y = np.linspace(0 - y0, direct_image_size- 1 - y0, obs_map.shape[0])
	X, Y = np.meshgrid(x, y)
	cp = ax_obs.pcolormesh(X, Y, obs_map, shading='nearest',
						vmax=obs_map.max(), vmin=obs_map.min())  # RdBu
	ax_obs.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_obs.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_obs.tick_params(axis='both', which='major', labelsize=5)
	ax_obs.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_obs)
	cbar.ax.set_ylabel(r"Flux [a.u.]", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_obs.set_title('Observed grism', fontsize=10)

	ax_model = fig.add_subplot(gs00[1,0])
	cp = ax_model.pcolormesh(X, Y, model_map, shading='nearest', vmax=obs_map.max(), vmin=obs_map.min())  # RdBu
	ax_model.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_model.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_model.tick_params(axis='both', which='major', labelsize=5)
	ax_model.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_model)
	cbar.ax.set_ylabel(r"Flux [a.u.]", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_model.set_title('Model grism', fontsize=10)


	chi = (model_map-obs_map)/obs_error
	ax_residuals = fig.add_subplot(gs00[2,0])
	cp = ax_residuals.pcolormesh(X, Y, chi, shading='nearest', vmin = -3, vmax = 3)  # RdBu
	ax_residuals.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_residuals.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_residuals.tick_params(axis='both', which='major', labelsize=5)
	ax_residuals.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_residuals)
	cbar.ax.set_ylabel(r"$\chi$", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_residuals.set_title(r'$\bar\chi = $' + str(np.sum(chi)/(chi.shape[0]*chi.shape[1])), fontsize=10)


	x = np.linspace(0 - x0, direct_image_size - 1 - x0, direct_image_size)
	y = np.linspace(0 - y0, direct_image_size - 1 - y0, direct_image_size)
	X, Y = np.meshgrid(x, y)

	vel_map_ax1 = fig.add_subplot(gs01[0])
	cp = vel_map_ax1.pcolormesh(X, Y,np.where(mask[0] ==1, model_velocities[0], np.nan), shading='nearest', cmap = 'RdBu_r')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 5)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 5)
	vel_map_ax1.tick_params(axis='both', which='major', labelsize=5)
	cbar = plt.colorbar(cp, ax = vel_map_ax1)
	cbar.ax.set_ylabel('velocity [km/s]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	vel_map_ax1.set_title(r'Velocity map, $v_{rot} = $' + str(round(v_rot[0])) + ' km/s', fontsize=10)

	#find the coordinates of the velocity centroid from model_velocities
	grad_x, grad_y = np.gradient(model_velocities[0])
	center1 = np.argmax(np.sqrt(grad_y**2 + grad_x**2))
	center1 = np.unravel_index(center1, model_velocities[0].shape)
	# vel_centroid = np.unravel_index(np.argmin(np.abs(model_velocities)), model_velocities.shape)
	# vel_map_ax1.plot(center1[1]-y0, center1[0]-x0, '+', markersize=10, label = 'velocity centroid', color = 'black')
	vel_map_ax1.legend(fontsize = 5, loc = 'lower right', borderaxespad = 2)

	vel_map_ax2 = fig.add_subplot(gs01[1])
	cp = vel_map_ax2.pcolormesh(X, Y,np.where(mask[1] != 0, model_velocities[1], np.nan), shading='nearest', cmap = 'RdBu_r')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 5)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 5)
	vel_map_ax2.tick_params(axis='both', which='major', labelsize=5)
	cbar = plt.colorbar(cp, ax = vel_map_ax2)
	cbar.ax.set_ylabel('velocity [km/s]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	vel_map_ax2.set_title(r'Velocity map, $v_{rot} = $' + str(round(v_rot[1])) + ' km/s', fontsize=10)

	#find the coordinates of the velocity centroid from model_velocities
	grad_x, grad_y = np.gradient(model_velocities[1])
	center2 = np.argmax(np.sqrt(grad_y**2 + grad_x**2))
	center2 = np.unravel_index(center2, model_velocities[1].shape)
	# vel_centroid = np.unravel_index(np.argmin(np.abs(model_velocities)), model_velocities.shape)
	vel_map_ax2.plot(center2[1]-y0, center2[0]-x0, '+', markersize=10, label = 'velocity centroid', color = 'black')
	vel_map_ax2.legend(fontsize = 5, loc = 'lower right', borderaxespad = 2)

	flux_map_ax = fig.add_subplot(gs01[2])

	cp = flux_map_ax.pcolormesh(X, Y,np.where(mask[2] ==1, fluxes_mean, np.nan), shading='nearest', cmap = 'PuRd')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 5)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 5)
	flux_map_ax.tick_params(axis='both', which='major', labelsize=5)
	cbar = plt.colorbar(cp, ax = flux_map_ax)
	cbar.ax.set_ylabel('flux [Mjy?]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	flux_map_ax.set_title('Flux map', fontsize=10)
	flux_map_ax.plot(center1[1]-y0, center1[0]-x0, '+', markersize=10, label = 'velocity centroid 1', color = 'black')
	flux_map_ax.plot(center2[1]-y0, center2[0]-x0, '+', markersize=10, label = 'velocity centroid 2', color = 'black')

	flux_map_ax.legend(fontsize = 5, loc = 'lower right',borderaxespad = 2)


	corner_ax = fig.add_subplot(gs00[:,1])
	CORNER_KWARGS = define_corner_args(divergences = False, var_names =var_names, labels = labels)

	fig = corner.corner(inf_data, group='posterior',color='crimson', **CORNER_KWARGS)
				
	CORNER_KWARGS = define_corner_args(divergences = False, fill_contours = False, plot_contours = False, show_titles = False,var_names = var_names, labels= labels)

	fig = corner.corner(inf_data, group='prior',fig=fig,color='lightgray', **CORNER_KWARGS)

	plt.savefig('cornerplot.png', dpi=300)
	figure_image = plt.imread('cornerplot.png')
	corner_ax.imshow(figure_image)
	corner_ax.axis('off')
	corner_ax.set_title('Kinematic posteriors', fontsize=10)
	plt.close()

	fig.suptitle(str(save_to_folder), fontsize=10, fontweight='bold')

	if save_to_folder != None:
		fig.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=500)
		
	plt.show()
	plt.close()

def define_corner_args(divergences = False, fill_contours = True, plot_contours = True, show_titles = True, quantiles = [0.16,0.5,0.84], var_names = ['PA', 'Va', 'i','r_t','sigma0'], labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$', r'$V_r$'], show_labels = True):
	"""
		Defines the cornerplot arguments
	"""

	CORNER_KWARGS = dict(
		smooth=2,
		label_kwargs=dict(fontsize=30),
		title_kwargs=dict(fontsize=20),
		quantiles=quantiles,
		plot_density=False,
		plot_datapoints=False,
		fill_contours=fill_contours,
		plot_contours=plot_contours,
		show_titles=show_titles,
		var_names = var_names,
		labels=labels,
		show_labels = show_labels,
		max_n_ticks=3,
		divergences=divergences)

	return CORNER_KWARGS

def plot_pp_cornerplot(data, kin_model, choice='real', PA=None, i=None, Va=None, r_t=None, sigma0=None, save=False, div = False, save_to_folder = None, name = None, prior = True):
	"""

			Plots cornerplot with both prior and posterior, only for the 4/5 central pixels in terms of flux (following Price et al 2021)

	"""
	#figure out what to do with this later
	if choice == 'model':

		v_r = Va * (2/pi) * np.arctan(2/r_t)

		truths = { 'PA': PA, 'i': i, 'Va': Va,
						  'r_t': r_t, 'sigma0': sigma0,'v_r': v_r}

		CORNER_KWARGS = define_corner_args(divergences = div)		

		fig = corner.corner(data, group='posterior', var_names=['PA', 'Va', 'i', 'r_t','sigma0','v_r'], truths = truths, truth_color='crimson',
									color='blue', **CORNER_KWARGS)
			
		if prior:
			CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False)

			fig = corner.corner(data , group='prior', var_names=['PA', 'Va', 'i','r_t','sigma0','v_r'], fig=fig, 
										color='lightgray', **CORNER_KWARGS)
			

	if choice == 'real':

		# truths = { 'amplitude '}
		CORNER_KWARGS = define_corner_args(divergences = div, var_names = ['amplitude', 'r_eff', 'n','i','PA_morph'], labels = ['amplitude', r'$r_{\text{eff}}$', 'n','i',r'$PA_{\text{morph}}$'])

		fig = corner.corner(data, group='posterior',color='crimson', **CORNER_KWARGS, truths = [111,4.19,1,60,0 ],truth_color='blue' )
				
		CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False,var_names = ['amplitude', 'r_eff', 'n','i','PA_morph'], labels=  ['amplitude', r'$r_{\text{eff}}$', 'n','i',r'$PA_{\text{morph}}$'], show_labels = False)

		fig = corner.corner(data, group='prior',fig=fig,color='thistle', **CORNER_KWARGS)

		if name == 'cornerplot_morph':
			plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=500)
		else:
			plt.savefig('testing/' + save_to_folder + '/' + name + '.png', dpi=500)
		plt.show()
		plt.close()


def plot_tuning_parameters(data, model = 'one_component_model', rotation = True, v0 = True, errors = True, y0 = True, scaling = True, error_scaling = True, reg = True, div = False, save_to_folder = None, name = None):
		"""

			Plots cornerplot with both prior and posterior, for the tuning parameters

		"""
	
		if model == 'Disk' or 'Disk_prior':
			number = ['']
		elif model == 'Merger':
			number = ['1', '2']

		var_names = []
		labels = []
		for num in number:
			# if rotation:
			# 	var_names.append('rotation' + num)
			# 	labels.append(r'$\theta$' + num)
			if v0:
				var_names.append('v0'+num)
				labels.append(r'$v_0$'+num)
			if y0:
				var_names.append('y0_vel'+ num)
				labels.append(r'$y_{0,vel}$'+ num)
				
		if scaling:
			var_names.append('fluxes_scaling')
			labels.append(r'$f_{scale}$')
		if error_scaling:
			var_names.append('fluxes_error_scaling' )
			labels.append(r'$f_{err, scale}$' )
		if errors:
			var_names.append('error_scaling' )
			labels.append(r'$f_{err}$' )
		if reg:
			var_names.append('regularization_strength' )
			labels.append(r'$\alpha_{reg}$' )

		CORNER_KWARGS = define_corner_args(divergences = div, labels = labels, var_names = var_names)

		fig = corner.corner(data, group='posterior',color='blue', **CORNER_KWARGS)
				
		CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False, labels = labels, var_names = var_names)

		fig = corner.corner(data, group='prior', fig=fig, color='lightgray', **CORNER_KWARGS)
			
		if save_to_folder != None:
			if name == 'summary':
				plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=500)
			else:
				plt.savefig('testing/' + save_to_folder + '/' + name + '.png', dpi=500)
		plt.show()
		plt.close()


def plot_flux_corner(inf_data, index_min, index_max, model = 'Disk', div = False, save_to_folder = None, name = None):
		"""

			Plots cornerplot with both prior and posterior for some of the fluxes

		"""
	
		if model == 'Disk':

			CORNER_KWARGS = define_corner_args(divergences = div, labels = None, var_names=None)

			fig = corner.corner(inf_data['posterior']['fluxes'][:,:,index_min:index_max], group='posterior', color='blue', **CORNER_KWARGS)
				
			CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False, labels = None, var_names=None)

			fig = corner.corner(inf_data['prior']['fluxes'][:,:,index_min:index_max], group='prior', fig=fig, color='lightgray', **CORNER_KWARGS)
			
		if save_to_folder != None:
			plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
		plt.show()
		plt.close()