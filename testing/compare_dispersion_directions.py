"""
Compare R and C dispersion directions to understand multi-observation constraints.

Tests whether C dispersion at theta_rot=0° is equivalent to R dispersion at theta_rot=90°.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp

from geko import grism, models, utils
from geko.grism import GrismObservation

# Set up mock galaxy parameters
print("Setting up mock galaxy parameters...")
PA = 45.0  # degrees
inclination = 60.0  # degrees
Va = 200.0  # km/s
r_t = 2.0  # pixels
sigma0 = 50.0  # km/s
z = 3.0
wavelength = 2.8  # microns (approximate for Ha at z~3)

# Image parameters
im_shape = 31
factor = 5  # oversampling
wave_factor = 9
im_shape_highres = im_shape * factor

# Create mock morphology
print("Creating mock morphology...")
amplitude = 200.0
r_eff = 3.0
n = 1.0
ellip = 1 - utils.compute_axis_ratio(inclination, 0.2)
xc_morph = (im_shape - 1) / 2
yc_morph = (im_shape - 1) / 2

# Load PSF
print("Loading PSF...")
psf_path = '/Users/lola/ASTRO/JWST/grism_project/sapphires/runs/psfs/mpsf_f444w.fits'
from astropy.io import fits
PSF = fits.getdata(psf_path)
PSF = utils.downsample_psf_centered(PSF, size=15)

# Create kinematic model
kin_model = models.DiskModel()

def create_mock_observation(dispersion, theta_rot_deg, name):
    """
    Create a mock grism observation.

    Parameters
    ----------
    dispersion : str
        'R' or 'C'
    theta_rot_deg : float
        Rotation angle in degrees
    name : str
        Name for this observation

    Returns
    -------
    GrismObservation
        Mock observation object
    """
    print(f"\nCreating {name}: {dispersion} dispersion, theta_rot={theta_rot_deg}°")

    # Convert theta_rot to radians
    theta_rot_rad = np.radians(theta_rot_deg)

    # Adjust PA for this observation
    PA_obs = PA - theta_rot_deg

    # Rotate centroids
    xc_obs, yc_obs = utils.rotate_coords(
        xc_morph, yc_morph,
        (im_shape - 1) / 2, (im_shape - 1) / 2,
        theta_rot_rad
    )

    # Generate flux map
    disk = models.Disk(
        (im_shape, im_shape),
        factor,
        xc_obs,
        yc_obs,
        r_eff
    )

    fluxes_high = disk.generate_flux_map(
        amplitude, r_eff, n, ellip, PA_obs, xc_obs, yc_obs
    )

    # Create velocity field
    x0_vel = xc_obs
    y0_vel = yc_obs

    x = jnp.linspace(0 - x0_vel, im_shape - x0_vel - 1, im_shape)
    y = jnp.linspace(0 - y0_vel, im_shape - y0_vel - 1, im_shape)
    X, Y = jnp.meshgrid(x, y)

    from jax import image
    X_grid = image.resize(X, (im_shape_highres, im_shape_highres), method='linear')
    Y_grid = image.resize(Y, (im_shape_highres, im_shape_highres), method='linear')

    # Compute velocity field
    velocities = jnp.asarray(kin_model.v(X_grid, Y_grid, PA_obs, inclination, Va, r_t))
    dispersions = sigma0 * jnp.ones_like(velocities)

    # Create wavelength space
    # This should cover the range around the emission line
    # IMPORTANT: n_wave must be divisible by wave_factor for resampling
    delta_wave = 0.001  # microns, high spectral resolution
    n_wave = wave_factor * 30  # 9*30=270, divisible by wave_factor
    wave_space = np.linspace(wavelength - delta_wave*135, wavelength + delta_wave*135, n_wave)
    index_min = 0
    index_max = n_wave

    # Create Grism object with proper initialization
    # IMPORTANT: im_shape should be the HIGH-RESOLUTION shape since we're passing
    # high-res velocities and fluxes to disperse()
    im_scale = 0.031 / factor  # arcsec/pixel for the high-res model
    grism_obj = grism.Grism(
        im_shape=im_shape_highres,  # Use high-res shape!
        im_scale=im_scale,
        wavelength=wavelength,
        wave_space=wave_space,
        index_min=index_min,
        index_max=index_max,
        grism_pupil=dispersion,
        PSF=PSF
    )

    # Set the factor and wave_factor
    grism_obj.factor = factor
    grism_obj.wave_factor = wave_factor

    # Load dispersion coefficients based on pupil
    if dispersion == 'R':
        grism_obj.load_poly_factors(
            a01=-2.588e-04, a02=-1.520e-06, a03=-4.486e-07, a04=-4.018e-07,
            a05=-2.090e-06, a06=-1.048e-06,
            b01=2.513e-01, b02=1.675e-04, b03=7.431e-05, b04=6.683e-05,
            b05=1.789e-04, b06=8.936e-05,
            c01=-1.161e-05, c02=-7.531e-09, c03=-3.446e-09,
            d01=3.462e-09
        )
    else:  # C
        grism_obj.load_poly_factors(
            a01=2.612e-05, a02=1.495e-06, a03=-1.560e-06, a04=-1.048e-06,
            a05=-2.506e-06, a06=-2.090e-06,
            b01=2.515e-01, b02=-1.675e-04, b03=-1.789e-04, b04=-8.936e-05,
            b05=-7.431e-05, b06=-6.683e-05,
            c01=1.162e-05, c02=7.563e-09, c03=3.462e-09,
            d01=-3.476e-09
        )

    grism_obj.load_poly_coefficients()
    grism_obj.set_wave_array()
    grism_obj.compute_lsf()
    grism_obj.compute_PSF(PSF)

    # Disperse through grism
    model_map = grism_obj.disperse(fluxes_high, velocities, dispersions)

    # For high-res visualization, we'll skip resampling and use the full resolution
    # This gives us the highest quality spectra for comparison
    model_map_highres = np.array(model_map)

    # Still create a resampled version for the GrismObservation object
    model_map_resampled = utils.resample(model_map, factor, wave_factor)

    # Add some noise
    noise_level = 0.01
    obs_map = model_map_resampled + np.random.normal(0, noise_level, model_map_resampled.shape)
    obs_error = noise_level * np.ones_like(obs_map)

    # Create GrismObservation
    obs = GrismObservation(
        grism=grism_obj,
        obs_map=obs_map,
        obs_error=obs_error,
        theta_rot=theta_rot_deg,
        dispersion=dispersion,
        name=name
    )

    return obs, model_map_highres


# Create three observations
print("\n" + "="*70)
print("Creating mock observations...")
print("="*70)

obs_R_0, map_R_0 = create_mock_observation('R', 0.0, 'R_theta0')
obs_C_0, map_C_0 = create_mock_observation('C', 0.0, 'C_theta0')
obs_R_90, map_R_90 = create_mock_observation('R', 90.0, 'R_theta90')

# Compare the images
print("\n" + "="*70)
print("Comparing images...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Images
im1 = axes[0, 0].imshow(map_R_0, origin='lower', cmap='viridis', aspect='auto')
axes[0, 0].set_title(f'R dispersion, θ_rot=0°\nShape: {map_R_0.shape}')
axes[0, 0].set_xlabel('Wavelength pixel')
axes[0, 0].set_ylabel('Spatial pixel (y)')
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(map_C_0, origin='lower', cmap='viridis', aspect='auto')
axes[0, 1].set_title(f'C dispersion, θ_rot=0°\nShape: {map_C_0.shape}')
axes[0, 1].set_xlabel('Wavelength pixel')
axes[0, 1].set_ylabel('Spatial pixel (x)')
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(map_R_90, origin='lower', cmap='viridis', aspect='auto')
axes[0, 2].set_title(f'R dispersion, θ_rot=90°\nShape: {map_R_90.shape}')
axes[0, 2].set_xlabel('Wavelength pixel')
axes[0, 2].set_ylabel('Spatial pixel (y)')
plt.colorbar(im3, ax=axes[0, 2])

# Row 2: Differences
diff_C_vs_R0 = map_C_0 - map_R_0
diff_R90_vs_R0 = map_R_90 - map_R_0
diff_C_vs_R90 = map_C_0 - map_R_90

vmax = max(np.abs(diff_C_vs_R0).max(), np.abs(diff_R90_vs_R0).max(), np.abs(diff_C_vs_R90).max())

im4 = axes[1, 0].imshow(diff_C_vs_R0, origin='lower', cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
axes[1, 0].set_title('Difference: C@0° - R@0°')
axes[1, 0].set_xlabel('Wavelength pixel')
axes[1, 0].set_ylabel('Spatial pixel')
plt.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(diff_R90_vs_R0, origin='lower', cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
axes[1, 1].set_title('Difference: R@90° - R@0°')
axes[1, 1].set_xlabel('Wavelength pixel')
axes[1, 1].set_ylabel('Spatial pixel')
plt.colorbar(im5, ax=axes[1, 1])

im6 = axes[1, 2].imshow(diff_C_vs_R90, origin='lower', cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
axes[1, 2].set_title('Difference: C@0° - R@90°')
axes[1, 2].set_xlabel('Wavelength pixel')
axes[1, 2].set_ylabel('Spatial pixel')
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('/Users/lola/geko/testing/dispersion_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved comparison plot to: /Users/lola/geko/testing/dispersion_comparison.png")

# Quantitative comparison
print("\n" + "="*70)
print("Quantitative comparison:")
print("="*70)
print(f"RMS difference (C@0° - R@0°):     {np.sqrt(np.mean(diff_C_vs_R0**2)):.6f}")
print(f"RMS difference (R@90° - R@0°):    {np.sqrt(np.mean(diff_R90_vs_R0**2)):.6f}")
print(f"RMS difference (C@0° - R@90°):    {np.sqrt(np.mean(diff_C_vs_R90**2)):.6f}")
print(f"\nMax absolute difference (C@0° - R@0°):   {np.abs(diff_C_vs_R0).max():.6f}")
print(f"Max absolute difference (R@90° - R@0°):  {np.abs(diff_R90_vs_R0).max():.6f}")
print(f"Max absolute difference (C@0° - R@90°):  {np.abs(diff_C_vs_R90).max():.6f}")

# Summary statistics
print("\n" + "="*70)
print("Image statistics:")
print("="*70)
print(f"R@0°:  mean={np.mean(map_R_0):.4f}, std={np.std(map_R_0):.4f}, max={np.max(map_R_0):.4f}")
print(f"C@0°:  mean={np.mean(map_C_0):.4f}, std={np.std(map_C_0):.4f}, max={np.max(map_C_0):.4f}")
print(f"R@90°: mean={np.mean(map_R_90):.4f}, std={np.std(map_R_90):.4f}, max={np.max(map_R_90):.4f}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
print("\nKey question: Should C@0° ≈ R@90°?")
print("If they're very different, it explains why R+C doesn't constrain sigma0 like R+R_90 does.")
