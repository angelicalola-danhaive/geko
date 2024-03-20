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
	cp = ax.pcolormesh(X,Y,(model-image)/image,shading= 'nearest') #RdBu
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
	CS = plt.contour(X,Y,velocities, 7, cmap = 'RdBu', origin = 'lower')
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

	plt.imshow(np.where(mask ==1, velocities, np.nan), origin = 'lower', cmap = 'RdBu')
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
	

def plot_full_summary(obs_map, model_map, obs_error, model_velocities, fluxes_mean, inf_data, wave_space, x0 = 31, y0 = 31, factor = 2 , direct_image_size = 62, save_to_folder = None, name = None):
	fig = plt.figure()
	fig.set_size_inches(10, 5)
	spec = gridspec.GridSpec(ncols=3, nrows=3,
							width_ratios=[2, 4, 2], wspace=0.2,
							hspace=0.5, height_ratios=[1, 1, 1])

	ax_obs = plt.subplot(spec[0, 0])
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

	ax_model = plt.subplot(spec[1, 0])
	cp = ax_model.pcolormesh(
		X, Y, model_map, shading='nearest', vmax=obs_map.max(), vmin=obs_map.min())  # RdBu
	ax_model.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_model.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_model.tick_params(axis='both', which='major', labelsize=5)
	ax_model.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_model)
	cbar.ax.set_ylabel(r"Flux [a.u.]", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_model.set_title('Model grism', fontsize=10)

	ax_residuals = plt.subplot(spec[2, 0])
	cp = ax_residuals.pcolormesh(
		X, Y, (model_map-obs_map)/obs_error, shading='nearest')  # RdBu
	ax_residuals.set_xlabel(r'wavelength $[\mu m]$', fontsize=5)
	ax_residuals.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	ax_residuals.tick_params(axis='both', which='major', labelsize=5)
	ax_residuals.tick_params(axis='both', which='minor')
	cbar = fig.colorbar(cp, ax=ax_residuals)
	cbar.ax.set_ylabel(r"Flux [a.u.]", fontsize=5)
	cbar.ax.tick_params(labelsize=5)
	ax_residuals.set_title('Residuals (M-D)', fontsize=10)


	vel_ax = plt.subplot(spec[:, 2])
	x = np.linspace(0 - x0, direct_image_size - 1 - x0, direct_image_size)
	y = np.linspace(0 - y0, direct_image_size - 1 - y0, direct_image_size)
	X, Y = np.meshgrid(x, y)

	extent = -y0, y0, -x0, x0
	vel_ax.imshow(fluxes_mean, origin='lower', extent=extent, cmap='viridis')
	CS = vel_ax.contour(X, Y, model_velocities, 7, cmap='RdBu', origin='lower', linewidths=1)
	cbar = plt.colorbar(CS, ax = vel_ax, shrink = 0.4)
	vel_ax.tick_params(axis='both', which='major', labelsize=5)
	vel_ax.set_xlabel(r'$\Delta$ RA [px]', fontsize=5)
	vel_ax.set_ylabel(r'$\Delta$ DEC [px]', fontsize=5)
	cbar.ax.set_ylabel('velocity [km/s]', fontsize = 5)
	cbar.ax.tick_params(labelsize = 5)
	cbar.add_lines(CS)
	vel_ax.set_title('Velocity contours', fontsize=10)


	corner_ax = plt.subplot(spec[:, 1])
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
						color='blue', **CORNER_KWARGS)
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

	fig.suptitle(save_to_folder, fontsize=12)

	if save_to_folder != None:
		fig.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=500)

	plt.show()
	plt.close()

def define_corner_args(divergences = False, fill_contours = True, plot_contours = True, show_titles = True, quantiles = [0.16,0.84], labels = [r'$PA$', r'$i$', r'$V_a$', r'$r_t$', r'$\sigma_0$', r'$V_r$']):
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
		labels=labels,
		max_n_ticks=3,
		divergences=divergences)

	return CORNER_KWARGS

def plot_pp_cornerplot(data, choice='model', model = None, PA=None, i=None, Va=None, r_t=None, sigma0=None, save=False, div = False, save_to_folder = None, name = None, prior = True):
	"""

			Plots cornerplot with both prior and posterior, only for the 4/5 central pixels in terms of flux (following Price et al 2021)

	"""
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

		if model == 'Disk':

			CORNER_KWARGS = define_corner_args(divergences = div)

			fig = corner.corner(data, group='posterior', var_names=['PA', 'i', 'Va', 'r_t', 'sigma0'],
										color='blue', **CORNER_KWARGS)
				
			CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False)

			fig = corner.corner(data, group='prior', var_names=['PA', 'i', 'Va', 'r_t', 'sigma0'], fig=fig,
										color='lightgray', **CORNER_KWARGS)

				
	if model == 'two_disc_model':
		CORNER_KWARGS = dict(
					smooth=2,
					label_kwargs=dict(fontsize=16),
					title_kwargs=dict(fontsize=16),
					quantiles=[0.16, 0.84],
					plot_density=False,
					plot_datapoints=False,
					fill_contours=True,
					plot_contours=True,
					show_titles=True,
					max_n_ticks=3,
					divergences=div)

		fig = corner.corner(data, group='posterior', var_names=['PA_1', 'i_1','sigma0_1','v_r_1', 'v_offset','PA_2', 'i_2','sigma0_2','v_r_2'],
									color='blue', **CORNER_KWARGS)
		CORNER_KWARGS = dict(
					smooth=2,
					label_kwargs=dict(fontsize=16),
					title_kwargs=dict(fontsize=16),
					plot_density=False,
					plot_datapoints=True,
					fill_contours=False,
					plot_contours=False,
					show_titles=False,
					max_n_ticks=3)

		fig = corner.corner(data, group='prior', var_names=['PA_1', 'i_1','sigma0_1','v_r_1','v_offset','PA_2', 'i_2','sigma0_2','v_r_2'], fig=fig,
									color='lightgray', **CORNER_KWARGS)
			
			

		if model == 'unified_two_discs_model':
			CORNER_KWARGS = dict(
					smooth=2,
					label_kwargs=dict(fontsize=16),
					title_kwargs=dict(fontsize=16),
					quantiles=[0.16, 0.84],
					plot_density=False,
					plot_datapoints=False,
					fill_contours=True,
					plot_contours=True,
					show_titles=True,
					max_n_ticks=3,
					divergences=div)

			fig = corner.corner(data, group='posterior', var_names=['PA_1', 'i_1','sigma0_1','v_r_1','PA_2', 'i_2','sigma0_2','v_r_2', 'x0_1', 'y0_1', 'x0_2', 'y0_2'],
									color='blue', **CORNER_KWARGS)
			CORNER_KWARGS = dict(
					smooth=2,
					label_kwargs=dict(fontsize=16),
					title_kwargs=dict(fontsize=16),
					plot_density=False,
					plot_datapoints=True,
					fill_contours=False,
					plot_contours=False,
					show_titles=False,
					max_n_ticks=3)

			fig = corner.corner(data, group='prior', var_names=['PA_1', 'i_1','sigma0_1','v_r_1','PA_2', 'i_2','sigma0_2','v_r_2', 'x0_1', 'y0_1', 'x0_2', 'y0_2'], fig=fig,
									color='lightgray', **CORNER_KWARGS)
			
		if save_to_folder != None:
				plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
		plt.show()
		plt.close()


def plot_tuning_parameters(data, model = 'one_component_model', rotation = True, v0 = True, errors = True, y0 = True, div = False, save_to_folder = None, name = None):
		"""

			Plots cornerplot with both prior and posterior, for the tuning parameters

		"""
	
		if model == 'Disk':

			var_names = []
			labels = []
			if rotation:
				var_names.append('rotation')
				labels.append(r'$\theta$')
			if v0:
				var_names.append('v0')
				labels.append(r'$v_0$')
			if errors:
				var_names.append('error_scaling')
				labels.append(r'$f_{err}$')
			if y0:
				var_names.append('y0_vel')
				labels.append(r'$y_{0,vel}$')

			CORNER_KWARGS = define_corner_args(divergences = div, labels = labels)

			fig = corner.corner(data, group='posterior', var_names=var_names,
										color='blue', **CORNER_KWARGS)
				
			CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False, labels = labels)

			fig = corner.corner(data, group='prior', var_names=var_names, fig=fig,
										color='lightgray', **CORNER_KWARGS)
			
		if save_to_folder != None:
			plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
		plt.show()
		plt.close()


def plot_flux_corner(inf_data, index_min, index_max, model = 'one_component_model', div = False, save_to_folder = None, name = None):
		"""

			Plots cornerplot with both prior and posterior for some of the fluxes

		"""
	
		if model == 'one_component_model':

			CORNER_KWARGS = define_corner_args(divergences = div, labels = None)

			fig = corner.corner(inf_data['posterior']['fluxes'][:,:,index_min:index_max], group='posterior', color='blue', **CORNER_KWARGS)
				
			CORNER_KWARGS = define_corner_args(divergences = div, fill_contours = False, plot_contours = False, show_titles = False, labels = None)

			fig = corner.corner(inf_data['prior']['fluxes'][:,:,index_min:index_max], group='prior', fig=fig, color='lightgray', **CORNER_KWARGS)
			
		if save_to_folder != None:
			plt.savefig('fitting_results/' + save_to_folder + '/' + name + '.png', dpi=300)
		plt.show()
		plt.close()