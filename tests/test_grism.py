from geko.grism_dev import *
import pytest
import numpy as np

@pytest.fixture
def grism_instance():
    #make a blank array for the direct image initialization
    direct_image = np.zeros((10, 10))
    wave = 4.0
    #make a wavespace array centered on wave with separation of 0.001
    wave_space = np.arange(wave - 0.05, wave + 0.05, 0.0001)
    PSF = np.zeros((10, 10))  # Placeholder for PSF
    return Grism(direct = direct_image, wave_space=wave_space, PSF = PSF, wavelength=wave)  #use all of the default parameters

def test_compute_lsf(grism_instance):
    R = grism_instance.compute_lsf()
    assert np.isclose(R, 1608, atol=2)  # Check if the computed LSF is close to the expected value

def test_compute_lsf_new(grism_instance):
    R = grism_instance.compute_lsf_new()
    assert np.isclose(R, 1568, atol = 2)

def test_get_trace(grism_instance):
    dxs,disp_space = grism_instance.get_trace()
    assert disp_space[0] == disp_space.min()  # Check if the first element of disp_space is the minimum value
    assert disp_space[-1] == disp_space.max()  # Check if the last element of disp_space is the maximum value
    assert np.isclose(dxs[0], disp_space[0], atol = 1e-6)
    assert np.isclose(dxs[-1], disp_space[-1], atol = 1e-6)
    assert (np.diff(dxs)- np.diff(dxs)[0]).max() < 1e-5 # Check if the differences in dxs are consistent
