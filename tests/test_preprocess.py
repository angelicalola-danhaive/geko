from geko.preprocess_dev import *
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
    index_wave = np.where(np.isclose(mock_wave_space, 4.0, atol=1e-6))[0][0]
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
