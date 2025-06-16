from geko.preprocess import *
import pytest
import numpy as np


#make a mock grism spectrum and errors
@pytest.fixture
def mock_grism_spectrum():
    # Create a mock grism spectrum with random values
    mock_grism = np.zeros((31,1301))
    wave_first = 3.8005  # Starting wavelength in microns
    d_wave = 0.000999999999999889  # Wavelength increment in microns
    mock_wave_space = wave_first + np.arange(0, 1301) * d_wave 
    #fill the poxel at waavelength 4.0 with a value of 1.0
    index_wave = np.argmin(np.abs(mock_wave_space - 4.0), axis = 0)  # Find the index closest to 4.0 microns
    mock_grism[15, index_wave] = 1.0  # Set a single pixel to a non-zero value
    # Create a mock error spectrum with small random values
    mock_errors = np.random.normal(0.01, 0.005, mock_grism.shape)  # Small random errors
    return mock_wave_space, mock_grism, mock_errors


def test_prep_grism(mock_grism_spectrum):
    mock_wave_space, grism_spectrum, grism_spectrum_error = mock_grism_spectrum
    wave_first = mock_wave_space[0]
    d_wave = mock_wave_space[1] - mock_wave_space[0]  # Assuming uniform spacing
    wavelength = 4.0
    obs_map, obs_error, index_min, index_max = prep_grism(grism_spectrum,grism_spectrum_error, wavelength, delta_wave_cutoff = 0.02, wave_first = wave_first, d_wave = d_wave)

    #check if the obs_map shape  is the same as obs_error
    assert obs_map.shape[0] == obs_error.shape[0], "obs_map and obs_error should have the same number of rows"
    assert obs_map.shape[1] == obs_error.shape[1], "obs_map and obs_error should have the same number of columns"
    #check if the obs_map shape is odd
    assert obs_map.shape[0] % 2 == 1, "obs_map should have an odd number of rows"
    assert obs_map.shape[1] % 2 == 1, "obs_map should have an odd number of columns"
    #the center of the obs_map should be the same as the wavelength
    center_row = obs_map.shape[0] // 2
    center_col = obs_map.shape[1] // 2
    assert np.isclose(obs_map[center_row, center_col], 1.0, atol=1e-6), "The center of the obs_map should have a value close to 1.0"

    #check that index_max and index_min are correct by oversampling the wave_space and cropping it like is done in the grism module
    wave_factor = 9
    #this is how the high res model space has to be made to pass the test - i.e. to make sure that when the high res model obs_map is resampled down, the wavelengths match the original mock_wave_space
    half_step = (d_wave / wave_factor)*(wave_factor//2)
    wave_space_model = np.arange(mock_wave_space[0]- half_step, mock_wave_space[-1] + d_wave + half_step, d_wave / wave_factor)
    index_min_model = (index_min)*wave_factor
    index_max_model = (index_max +1)*wave_factor
    wave_space_model_crop = wave_space_model[index_min_model:index_max_model]
    #resample wave_space_model_crop to the original wave_space_model shape
    blocks = wave_space_model_crop.reshape(int(wave_space_model_crop.shape[0]/wave_factor), wave_factor)
    wave_space_model_crop_resampled = np.sum(blocks, axis=1) / wave_factor
    assert np.isclose(wave_space_model_crop_resampled, mock_wave_space[index_min:index_max+1], atol=1e-6).all(), "The cropped wave space should match the expected values"
