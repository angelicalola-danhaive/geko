from geko.grism import *
from geko import utils
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@pytest.fixture
def grism_instance():
    #make a blank array for the direct image initialization
    wave = 4.0
    #make a wavespace array centered on wave with separation of 0.001
    wave_space = np.arange(wave - 0.05, wave + 0.05 + 0.0001, 0.0001)
    PSF = np.zeros((3, 3))  # Placeholder for PSF
    PSF[1, 1] = 1.0  # Set the center pixel to 1.0 for simplicity
    #check that the center of the wave_space is equal to wave
    assert np.isclose(wave_space[len(wave_space)//2], wave, atol=1e-4), "Wave space center does not match the specified wave."
    #initialize a 9x9 detector, where the model space is oversampled 5 times
    return Grism(45, im_scale = 0.0629/5, icenter = 4, jcenter = 4, wavelength = wave , wave_space = wave_space, index_min = 0, index_max = wave_space.shape[0], grism_filter = 'F444W', grism_module = 'A', grism_pupil = 'R', PSF = PSF)

@pytest.fixture
def grism_instance_C():
    """Create Grism instance for Column (C) dispersion"""
    wave = 4.0
    wave_space = np.arange(wave - 0.05, wave + 0.05 + 0.0001, 0.0001)
    PSF = np.zeros((3, 3))
    PSF[1, 1] = 1.0
    assert np.isclose(wave_space[len(wave_space)//2], wave, atol=1e-4), "Wave space center does not match the specified wave."
    return Grism(45, im_scale = 0.0629/5, icenter = 4, jcenter = 4, wavelength = wave , wave_space = wave_space, index_min = 0, index_max = wave_space.shape[0], grism_filter = 'F444W', grism_module = 'A', grism_pupil = 'C', PSF = PSF)

def test_compute_lsf(grism_instance):
    R = grism_instance.compute_lsf()
    assert np.isclose(R, 1608, atol=2)  # Check if the computed LSF is close to the expected value

def test_compute_lsf_new(grism_instance):
    R = grism_instance.compute_lsf_new()
    assert np.isclose(R, 1599, atol = 2)

def test_get_trace(grism_instance):
    dxs,disp_space = grism_instance.get_trace()
    assert disp_space[0] == disp_space.min()  # Check if the first element of disp_space is the minimum value
    assert disp_space[-1] == disp_space.max()  # Check if the last element of disp_space is the maximum value
    assert np.isclose(dxs[0], disp_space[0], atol = 1e-6)
    assert np.isclose(dxs[-1], disp_space[-1], atol = 1e-6)
    assert (np.diff(dxs)- np.diff(dxs)[0]).max() < 1e-5 # Check if the differences in dxs are consistent

def test_init_detector(grism_instance):
    '''
        Test the initialization of the detector by checking that the center of the detector is preserved
    '''
    #the detector is automatically initialized in the Grism class, so we can just check that the center is preserved
    assert grism_instance.detector_space_1d[grism_instance.detector_space_1d.shape[0] // 2] == 1024 

def test_disperse(grism_instance):
    '''
        Test wether the grism object is setup correctly by dispersing a cube with no velocity and checking that the middle is consistent?
    '''
    mock_flux = np.zeros((45,45))
    mock_flux[22, 22] = 1.0  # Set a single pixel to a non-zero value
    mock_vel = np.zeros((45, 45))  # No velocity
    mock_disp = np.zeros((45, 45))  # No dispersion
    mock_grism = grism_instance.disperse(mock_flux, mock_vel, mock_disp)

    #plot and save the mock_grism
    # Use Path to get the directory where this test file is located
    test_dir = Path(__file__).parent
    output_path = test_dir / 'mock_grism.png'

    plt.imshow(mock_grism, origin='lower', cmap='viridis')
    plt.colorbar(label='Flux')
    plt.title('Mock Grism Image')
    plt.savefig(str(output_path))


    # Check that the total flux is preserved (with a tolerance)
    assert np.isclose(np.sum(mock_flux), np.sum(mock_grism), atol=1e-6)
    # Check that the center pixel is still at the same position
    max_position = np.unravel_index(np.argmax(mock_grism, axis = None), mock_grism.shape)
    assert max_position == (mock_grism.shape[0] // 2, mock_grism.shape[1] // 2)


# ============================================================================
# COLUMN (C) DISPERSION TESTS
# ============================================================================

def test_grism_C_initialization(grism_instance_C):
    """Test that Column (C) dispersion Grism can be initialized"""
    assert grism_instance_C is not None
    assert grism_instance_C.pupil == 'C'
    assert grism_instance_C.module == 'A'
    assert grism_instance_C.filter == 'F444W'

def test_grism_C_disperse(grism_instance_C):
    """Test that Column (C) dispersion produces correct output shape"""
    mock_flux = np.zeros((45, 45))
    mock_flux[22, 22] = 1.0
    mock_vel = np.zeros((45, 45))
    mock_disp = np.zeros((45, 45))

    mock_grism = grism_instance_C.disperse(mock_flux, mock_vel, mock_disp)

    # Check output shape is correct for C dispersion
    assert mock_grism.shape[0] == 45  # spatial dimension preserved
    assert mock_grism.shape[1] == grism_instance_C.wave_space.shape[0]  # wavelength dimension

    # Check flux conservation
    assert np.isclose(np.sum(mock_flux), np.sum(mock_grism), atol=1e-6)

def test_grism_C_vs_R_different_coefficients(grism_instance, grism_instance_C):
    """Test that R and C dispersion have different coefficients"""
    # b01 coefficient should be different for R vs C
    assert not np.isclose(grism_instance.b01, grism_instance_C.b01), \
        "R and C dispersion should have different b01 coefficients"

def test_grism_C_no_velocity_tilt(grism_instance_C):
    """Test that C dispersion produces no tilt when velocity gradient is zero"""
    # Create elliptical galaxy with zero velocity
    im_shape = 31
    flux = np.ones((im_shape, im_shape)) * 0.1

    # Create elliptical mask
    center = im_shape // 2
    x = np.arange(im_shape)
    y = np.arange(im_shape)
    xx, yy = np.meshgrid(x, y)

    radius = 8
    mask = ((xx - center)**2) / (radius**2) + ((yy - center)**2) / ((radius*0.6)**2) <= 1.0
    flux[mask] = 1.0

    velocity = np.zeros((im_shape, im_shape))
    dispersion = np.ones((im_shape, im_shape)) * 50.0

    # Create temporary grism for this test
    wave = 4.2
    wave_space = np.linspace(3.8, 5.0, 1000)
    psf_size = 15
    center_psf = psf_size // 2
    y_psf, x_psf = np.meshgrid(np.arange(psf_size), np.arange(psf_size), indexing='ij')
    psf = np.exp(-((x_psf - center_psf)**2 + (y_psf - center_psf)**2) / (2 * 2**2))
    psf = psf / np.sum(psf)

    grism_C = Grism(
        im_shape=im_shape, wavelength=wave, wave_space=wave_space,
        index_min=100, index_max=900, grism_filter='F444W',
        grism_module='A', grism_pupil='C', PSF=psf
    )

    grism_output = grism_C.disperse(flux, velocity, dispersion)

    # For vertical spectrum with no velocity, each row should have similar profile
    # Check that the spectrum is roughly symmetric around the center
    center_col_idx = grism_output.shape[1] // 2
    profile_lower = grism_output[im_shape//2 - 5, :]
    profile_upper = grism_output[im_shape//2 + 5, :]

    # Profiles should be similar (no strong tilt)
    # Use correlation as a measure of similarity
    correlation = np.corrcoef(profile_lower, profile_upper)[0, 1]
    assert correlation > 0.95, f"Vertical spectrum should not have tilt, got correlation {correlation}"


def _make_linear_grism(pupil, im_shape=45, icenter=4, jcenter=4):
    """Grism instance with position-independent, linear-only dispersion
    (only b01 nonzero) so an R and a C instance differ *only* in geko's own
    pupil-handling logic (disperse()'s collapse axis, set_wave_array()'s
    offset-term axis) -- not in real, independently-calibrated AR/AC trace
    curvature, which isn't what's being tested here.
    """
    # Same wave_space shape as grism_instance/grism_instance_C above so this
    # reuses their already-JIT-compiled disperse() trace instead of paying
    # for a fresh XLA compilation at a new array shape.
    wave = 4.0
    wave_space = np.arange(wave - 0.05, wave + 0.05 + 0.0001, 0.0001)
    PSF = np.zeros((3, 3))
    PSF[1, 1] = 1.0
    g = Grism(im_shape, im_scale=0.0629 / 5, icenter=icenter, jcenter=jcenter,
              wavelength=wave, wave_space=wave_space, index_min=0,
              index_max=wave_space.shape[0], grism_filter='F444W',
              grism_module='A', grism_pupil=pupil, PSF=PSF)
    g.load_poly_factors(a01=0., a02=0., a03=0., a04=0., a05=0., a06=0.,
                         b01=1000., b02=0., b03=0., b04=0., b05=0., b06=0.,
                         c01=0., c02=0., c03=0., d01=0.)
    g.load_poly_coefficients()
    g.get_trace()
    g.set_wave_array()
    g.use_psf = False
    g.use_lsf = False
    return g


def test_c_dispersion_matches_r_rotated_90_clockwise():
    """C-pupil dispersion of a galaxy equals R-pupil dispersion of the same
    galaxy rotated 90 deg CLOCKWISE (utils.rotate_coords' own sign convention
    -- positive theta is clockwise), with the R output's spatial axis then
    reversed.

    Rotation direction note: physically this is a *clockwise* 90 deg
    rotation, not counterclockwise -- verified numerically here (exact match,
    RMS=0 to machine precision, for theta=+pi/2; a counterclockwise rotation,
    theta=-pi/2, does not match, with or without the spatial-axis reversal).
    The spatial-axis reversal is also required for an exact match and isn't
    optional: it falls out of how disperse() collapses axis=0 (C) vs axis=1
    (R) of the same (spatial_y, spatial_x, wavelength) cube -- summing over
    each axis maps back to the retained axis with opposite index handedness,
    independent of any calibration.

    Uses identical synthetic linear dispersion coefficients for both pupils
    (see _make_linear_grism) so real AR/AC calibration differences can't
    contaminate the comparison -- this isolates geko's own rotation +
    pupil-collapse code, which is what's under test.
    """
    im_shape = 45
    grism_R = _make_linear_grism('R', im_shape=im_shape)
    grism_C = _make_linear_grism('C', im_shape=im_shape)

    center = (im_shape - 1) / 2

    def gaussian_blob(xc, yc, sigma=3.0):
        yy, xx = np.mgrid[0:im_shape, 0:im_shape]
        return np.exp(-(((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * sigma ** 2)))

    # Offset asymmetric in x and y (dx != dy, dx != -dy) so a clockwise vs
    # counterclockwise rotation give genuinely different, distinguishable results.
    xc0, yc0 = center + 4.0, center + 10.0
    F_C = gaussian_blob(xc0, yc0)
    V = np.zeros((im_shape, im_shape))
    D = 50.0 * np.ones((im_shape, im_shape))

    image_C = np.array(grism_C.disperse(F_C, V, D))

    # theta=+pi/2 is clockwise per utils.rotate_coords' documented convention
    xc_rot, yc_rot = utils.rotate_coords(xc0, yc0, center, center, np.pi / 2)
    F_R = gaussian_blob(float(xc_rot), float(yc_rot))
    image_R_rotated = np.array(grism_R.disperse(F_R, V, D))

    np.testing.assert_allclose(image_C, image_R_rotated[::-1, :], atol=1e-10)

    # Counterclockwise (theta=-pi/2) should NOT match -- guards against the
    # rotation-direction convention silently flipping in a future change.
    xc_rot_ccw, yc_rot_ccw = utils.rotate_coords(xc0, yc0, center, center, -np.pi / 2)
    F_R_ccw = gaussian_blob(float(xc_rot_ccw), float(yc_rot_ccw))
    image_R_ccw = np.array(grism_R.disperse(F_R_ccw, V, D))
    assert not np.allclose(image_C, image_R_ccw[::-1, :], atol=1e-6)
    assert not np.allclose(image_C, image_R_ccw, atol=1e-6)


# ============================================================================
# GRISM OBSERVATION CLASS TESTS
# ============================================================================

def test_grism_observation_R_initialization(grism_instance):
    """Test GrismObservation with R dispersion"""
    obs_map = np.random.uniform(0, 10, (45, 1001))
    obs_error = np.random.uniform(0.1, 1, (45, 1001))
    theta_rot = 45.0

    grism_obs = GrismObservation(
        grism=grism_instance,
        obs_map=obs_map,
        obs_error=obs_error,
        theta_rot=theta_rot,
        dispersion='R',
        name='test_R'
    )

    assert grism_obs.grism is grism_instance
    assert np.array_equal(grism_obs.obs_map, obs_map)
    assert np.array_equal(grism_obs.obs_error, obs_error)
    assert grism_obs.theta_rot == theta_rot
    assert grism_obs.dispersion == 'R'
    assert grism_obs.name == 'test_R'

def test_grism_observation_C_initialization(grism_instance_C):
    """Test GrismObservation with C dispersion"""
    obs_map = np.random.uniform(0, 10, (45, 1001))
    obs_error = np.random.uniform(0.1, 1, (45, 1001))
    theta_rot = 90.0

    grism_obs = GrismObservation(
        grism=grism_instance_C,
        obs_map=obs_map,
        obs_error=obs_error,
        theta_rot=theta_rot,
        dispersion='C',
        name='test_C'
    )

    assert grism_obs.dispersion == 'C'
    assert grism_obs.grism.pupil == 'C'

def test_grism_observation_auto_name(grism_instance):
    """Test that GrismObservation auto-generates name if not provided"""
    obs_map = np.random.uniform(0, 10, (45, 1001))
    obs_error = np.random.uniform(0.1, 1, (45, 1001))

    grism_obs = GrismObservation(
        grism=grism_instance,
        obs_map=obs_map,
        obs_error=obs_error,
        theta_rot=45.0,
        dispersion='R'
    )

    assert grism_obs.name is not None
    assert 'R' in grism_obs.name or '45' in grism_obs.name

def test_grism_observation_pupil_mismatch(grism_instance):
    """Test that GrismObservation raises error when pupil doesn't match dispersion"""
    obs_map = np.random.uniform(0, 10, (45, 1001))
    obs_error = np.random.uniform(0.1, 1, (45, 1001))

    with pytest.raises(ValueError, match="pupil.*doesn't match"):
        GrismObservation(
            grism=grism_instance,  # grism_instance has pupil='R'
            obs_map=obs_map,
            obs_error=obs_error,
            theta_rot=45.0,
            dispersion='C'  # But we're saying dispersion='C'
        )

def test_grism_observation_shape_mismatch(grism_instance):
    """Test that GrismObservation raises error when obs_map and obs_error shapes don't match"""
    obs_map = np.random.uniform(0, 10, (45, 1001))
    obs_error = np.random.uniform(0.1, 1, (45, 500))  # Different shape

    with pytest.raises(ValueError, match="same shape"):
        GrismObservation(
            grism=grism_instance,
            obs_map=obs_map,
            obs_error=obs_error,
            theta_rot=45.0,
            dispersion='R'
        )