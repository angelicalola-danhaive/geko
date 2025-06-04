from geko.utils import *
import pytest
import numpy as np

def test_oversample():
    #generate a sample 30x30 2D array
    arr = np.random.rand(30, 30)
    #oversample the array by a factor of 2
    oversampled_arr = oversample(arr, 2,2)
    #check if the shape of the oversampled array is correct
    assert oversampled_arr.shape == (60, 60)
    #check that the total flux is preserved
    assert np.isclose(np.sum(arr), np.sum(oversampled_arr))

def test_resample():
    #test that if you oversample an array and then resample it back to the original size, you get the same array
    arr = np.random.rand(30, 30)
    oversampled_arr = oversample(arr, 2, 2)
    resampled_arr = resample(oversampled_arr, 2, 2)
    assert np.allclose(arr, resampled_arr, atol=1e-6), "Resampling should return the original array within a small tolerance"

def test_compute_inclination_axis_ratio():
    axis_ratio = 0.5
    inclination = compute_inclination(axis_ratio, q0 = 0)
    assert np.isclose(float(inclination), 60.0)  # cos^-1(0.5) = 60 degrees
    new_axis_ratio = compute_axis_ratio(inclination, q0 = 0)
    assert np.isclose(float(new_axis_ratio), axis_ratio)  # cos(60 degrees) = 0.5


