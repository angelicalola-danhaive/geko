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

def test_compute_inclination_axis_ratio():
    axis_ratio = 0.5
    inclination = compute_inclination(axis_ratio, q0 = 0)
    assert float(inclination) == 60.0  # cos^-1(0.5) = 60 degrees
    new_axis_ratio = compute_axis_ratio(inclination, q0 = 0)
    assert float(axis_ratio) == new_axis_ratio  # should return to original axis ratio


