"""
All the plotting related functions for the post-processing
	
	
	Written by A L Danhaive: ald66@cam.ac.uk
"""



import matplotlib.pyplot as plt
import numpy as np
import corner
from matplotlib import gridspec
from scipy.constants import c, pi
from jax import image



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
	

def plot_disk_summary(obs_map, model_map, obs_error, model_velocities, model_dispersions, v_rot, fluxes_mean, inf_data, wave_space, mask, x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = None, name = None):
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

	ax_residuals = fig.add_subplot(gs00[2,0])
	cp = ax_residuals.pcolormesh(X, Y, (model_map-obs_map)/obs_error, shading='nearest', vmin = -3, vmax =3)  # RdBu
	ax_residuals.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_residuals.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_residuals.tick_params(axis='both', which='major', labelsize=5)
	ax_residuals.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_residuals)
	cbar.ax.set_ylabel(r"$\chi^2$", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_residuals.set_title('Residuals (M-D)', fontsize=10)


	x = np.linspace(0 - x0, direct_image_size - 1 - x0, direct_image_size)
	y = np.linspace(0 - y0, direct_image_size - 1 - y0, direct_image_size)
	X, Y = np.meshgrid(x, y)

	#find the coordinates of the velocity centroid from model_velocities
	grad_x, grad_y = np.gradient(model_velocities)
	center = np.argmax(np.sqrt(grad_y**2 + grad_x**2))
	center = np.unravel_index(center, model_velocities.shape)
	velocites_center = model_velocities[center[0], center[1]]
	vel_map_ax = fig.add_subplot(gs01[0])
	cp = vel_map_ax.pcolormesh(X, Y,np.where(mask ==1, model_velocities - velocites_center, np.nan), shading='nearest', cmap = 'RdBu_r')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 5)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 5)
	vel_map_ax.tick_params(axis='both', which='major', labelsize=5)
	cbar = plt.colorbar(cp, ax = vel_map_ax)
	cbar.ax.set_ylabel('velocity [km/s]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	vel_map_ax.set_title(r'Velocity map, $v_{rot} = $' + str(round(v_rot)) + ' km/s', fontsize=10)


	vel_map_ax.plot(center[1]-y0, center[0]-x0, '+', markersize=10, label = 'velocity centroid', color = 'black')
	vel_map_ax.legend(fontsize = 5, loc = 'lower right', borderaxespad = 2)

	veldisp_map_ax = fig.add_subplot(gs01[1])

	cp = veldisp_map_ax.pcolormesh(X, Y,np.where(mask ==1, model_dispersions, np.nan), shading='nearest', cmap = 'YlGnBu_r')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 5)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 5)
	veldisp_map_ax.tick_params(axis='both', which='major', labelsize=5)
	cbar = plt.colorbar(cp, ax = veldisp_map_ax)
	cbar.ax.set_ylabel(r'$\sigma_v$ [km/s]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	veldisp_map_ax.set_title(r'$\sigma_v$ map', fontsize=10)
	veldisp_map_ax.plot(center[1]-y0, center[0]-x0, '+', markersize=10, label = 'velocity centroid', color = 'black')
	veldisp_map_ax.legend(fontsize = 5, loc = 'lower right',borderaxespad = 2)

	flux_map_ax = fig.add_subplot(gs01[2])

	cp = flux_map_ax.pcolormesh(X, Y,np.where(mask ==1, fluxes_mean, np.nan), shading='nearest', cmap = 'PuRd')
	plt.xlabel(r'$\Delta$ RA [px]',fontsize = 5)
	plt.ylabel(r'$\Delta$ DEC [px]',fontsize = 5)
	flux_map_ax.tick_params(axis='both', which='major', labelsize=5)
	cbar = plt.colorbar(cp, ax = flux_map_ax)
	cbar.ax.set_ylabel('flux [Mjy?]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	flux_map_ax.set_title('Flux map', fontsize=10)
	flux_map_ax.plot(center[1]-y0, center[0]-x0, '+', markersize=10, label = 'velocity centroid', color = 'black')
	flux_map_ax.legend(fontsize = 5, loc = 'lower right',borderaxespad = 2)


	corner_ax = fig.add_subplot(gs00[:,1])
	CORNER_KWARGS = dict(
		smooth=2,
		label_kwargs=dict(fontsize=30),
		title_kwargs=dict(fontsize=20),
		quantiles=[0.16, 0.84],
		plot_density=False,
		plot_datapoints=False,
		fill_contours=True,
		plot_contours=True,
		show_titles=True,
		labels=[r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$'],
		max_n_ticks=3,
		divergences=False)

	figure = corner.corner(inf_data, group='posterior', var_names=['PA', 'i', 'Va', 'r_t', 'sigma0'],
						color='crimson', **CORNER_KWARGS)
	CORNER_KWARGS = dict(
		smooth=2,
		label_kwargs=dict(fontsize=30),
		title_kwargs=dict(fontsize=20),
		plot_density=False,
		plot_datapoints=True,
		fill_contours=False,
		plot_contours=False,
		labels=[r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$'],
		show_titles=False,
		max_n_ticks=3)

	figure = corner.corner(inf_data, group='prior', var_names=['PA', 'i', 'Va', 'r_t', 'sigma0'], fig=figure,
						color='lightgray', **CORNER_KWARGS)

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

	ax_residuals = fig.add_subplot(gs00[2,0])
	cp = ax_residuals.pcolormesh(X, Y, (model_map-obs_map)/obs_error, shading='nearest', vmin = -3, vmax = 3)  # RdBu
	ax_residuals.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_residuals.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_residuals.tick_params(axis='both', which='major', labelsize=5)
	ax_residuals.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_residuals)
	cbar.ax.set_ylabel(r"$\chi^2$", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_residuals.set_title('Residuals (M-D)', fontsize=10)


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
	vel_map_ax1.plot(center1[1]-y0, center1[0]-x0, '+', markersize=10, label = 'velocity centroid', color = 'black')
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

def define_corner_args(divergences = False, fill_contours = True, plot_contours = True, show_titles = True, quantiles = [0.16,0.84], var_names = ['PA', 'Va', 'i','r_t','sigma0'], labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$', r'$V_r$']):
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


		CORNER_KWARGS = define_corner_args(divergences = div, var_names = kin_model.var_names, labels = kin_model.labels)

		fig = corner.corner(data, group='posterior',color='crimson', **CORNER_KWARGS)
				
		CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False,var_names = kin_model.var_names, labels= kin_model.labels)

		fig = corner.corner(data, group='prior',fig=fig,color='lightgray', **CORNER_KWARGS)

		if save_to_folder != None:
				plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
		plt.show()
		plt.close()


def plot_tuning_parameters(data, model = 'one_component_model', rotation = True, v0 = True, errors = True, y0 = True, scaling = True, reg = True, div = False, save_to_folder = None, name = None):
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
			if rotation:
				var_names.append('rotation' + num)
				labels.append(r'$\theta$' + num)
			if v0:
				var_names.append('v0'+num)
				labels.append(r'$v_0$'+num)
			if y0:
				var_names.append('y0_vel'+ num)
				labels.append(r'$y_{0,vel}$'+ num)
				
		if scaling:
			var_names.append('fluxes_scaling')
			labels.append(r'$f_{scale}$')
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
			plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
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