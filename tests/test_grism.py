from geko.grism import *
import pytest
import numpy as np

@pytest.fixture
def grism_instance():
    #make a blank array for the direct image initialization
    direct_image = np.zeros((10, 10))
    wave = 4.0
    #make a wavespace array centered on 4.2 microns with 0.001 micron steps
    wave_space = np.arange(wave, wave + 0.001 * 10, 0.001)  # 10 steps of 0.001 microns
    PSF = np.zeros((10, 10))  # Placeholder for PSF
    return Grism(direct = direct_image, wave_space=wave_space, PSF = PSF, wavelength=wave)  #use all of the default parameters

def test_compute_lsf(grism_instance):
    R = grism_instance.compute_lsf()
    assert np.isclose(R, 1608, atol=2)  # Check if the computed LSF is close to the expected value

def test_compute_lsf_new(grism_instance):
    R = grism_instance.compute_lsf_new()
    assert np.isclose(R, 1568, atol = 2)