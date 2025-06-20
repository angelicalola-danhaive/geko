from geko.grism import *
import pytest
import numpy as np
import matplotlib.pyplot as plt

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
    plt.imshow(mock_grism, origin='lower', cmap='viridis')
    plt.colorbar(label='Flux')
    plt.title('Mock Grism Image')
    plt.savefig('tests/mock_grism.png')


    # Check that the total flux is preserved (with a tolerance)
    assert np.isclose(np.sum(mock_flux), np.sum(mock_grism), atol=1e-6)
    # Check that the center pixel is still at the same position
    max_position = np.unravel_index(np.argmax(mock_grism, axis = None), mock_grism.shape)
    assert max_position == (mock_grism.shape[0] // 2, mock_grism.shape[1] // 2)