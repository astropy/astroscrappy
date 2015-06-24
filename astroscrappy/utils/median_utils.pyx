# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: profile=True, boundscheck=False, nonecheck=False, wraparound=False
# cython: cdivision=True
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""
Name : median_utils
Author : Curtis McCully
Date : October 2014
"""
import numpy as np
cimport numpy as np

np.import_array()


cdef extern from "medutils.h":
    float PyMedian(float * a, int n) nogil
    float PyOptMed3(float * a) nogil
    float PyOptMed5(float * a) nogil
    float PyOptMed7(float * a) nogil
    float PyOptMed9(float * a) nogil
    float PyOptMed25(float * a) nogil
    void PyMedFilt3(float * data, float * output, int nx, int ny) nogil
    void PyMedFilt5(float * data, float * output, int nx, int ny) nogil
    void PyMedFilt7(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt3(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt5(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt7(float * data, float * output, int nx, int ny) nogil
    void PySepMedFilt9(float * data, float * output, int nx, int ny) nogil


"""
Wrappers for the C functions in medutils.c
"""


def median(np.ndarray[np.float32_t, mode='c', cast=True] a, int n):
    """median(a, n)\n
    Find the median of the first n elements of an array.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median.

    n : int
        Number of elements of the array to median.

    Returns
    -------
    med : float
        The median value.

    Notes
    -----
    Wrapper for PyMedian in medutils.
    """
    cdef float * aptr = < float * > np.PyArray_DATA(a)
    cdef float med = 0.0
    with nogil:
        med = PyMedian(aptr, n)
    return med

cdef float cymedian(float* a, int n) nogil:
    """cymedian(a, n)\n
    Cython function to calculate the median without requiring the GIL.
    :param a:
    :param n:
    :return:
    """
    cdef float med = 0.0
    med = PyMedian(a, n)
    return med

def optmed3(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed3(a)\n
    Optimized method to find the median value of an array of length 3.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 3.

    Returns
    -------
    med3 : float
        The median of the 3-element array.

    Notes
    -----
    Wrapper for PyOptMed3 in medutils.
    """
    cdef float * aptr3 = < float * > np.PyArray_DATA(a)
    cdef float med3 = 0.0
    with nogil:
        med3 = PyOptMed3(aptr3)
    return med3


def optmed5(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed5(a)\n
    Optimized method to find the median value of an array of length 5.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 5.

    Returns
    -------
    med5 : float
        The median of the 5-element array.

    Notes
    -----
    Wrapper for PyOptMed5 in medutils.
    """
    cdef float * aptr5 = < float * > np.PyArray_DATA(a)
    cdef float med5 = 0.0
    with nogil:
        med5 = PyOptMed5(aptr5)
    return med5


def optmed7(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed7(a)\n
    Optimized method to find the median value of an array of length 7.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 7.

    Returns
    -------
    med7 : float
        The median of the 7-element array.

    Notes
    -----
    Wrapper for PyOptMed7 in medutils.
    """
    cdef float * aptr7 = < float * > np.PyArray_DATA(a)
    cdef float med7 = 0.0
    with nogil:
        med7 = PyOptMed7(aptr7)
    return med7


def optmed9(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed9(a)\n
    Optimized method to find the median value of an array of length 9.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 9.

    Returns
    -------
    med9 : float
        The median of the 9-element array.

    Notes
    -----
    Wrapper for PyOptMed9 in medutils.
    """
    cdef float * aptr9 = < float * > np.PyArray_DATA(a)
    cdef float med9 = 0.0
    with nogil:
        med9 = PyOptMed9(aptr9)
    return med9


def optmed25(np.ndarray[np.float32_t, ndim=1, mode='c', cast=True] a):
    """optmed25(a)\n
    Optimized method to find the median value of an array of length 25.

    Parameters
    ----------
    a : float numpy array
        Input array to find the median. Must be length 25.

    Returns
    -------
    med25 : float
        The median of the 25-element array.

    Notes
    -----
    Wrapper for PyOptMed25 in medutils.
    """
    cdef float * aptr25 = < float * > np.PyArray_DATA(a)
    cdef float med25 = 0.0
    with nogil:
        med25 = PyOptMed25(aptr25)
    return med25


def medfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d3):
    """medfilt3(d3)\n
    Calculate the 3x3 median filter of an array.

    Parameters
    ----------
    d3 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt3 in medutils.
    """
    cdef int nx = d3.shape[1]
    cdef int ny = d3.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)
    cdef float * d3ptr = < float * > np.PyArray_DATA(d3)
    cdef float * outd3ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt3(d3ptr, outd3ptr, nx, ny)

    return output


def medfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d5):
    """medfilt5(d5)\n
    Calculate the 5x5 median filter of an array.

    Parameters
    ----------
    d5 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt5 in medutils.
    """
    cdef int nx = d5.shape[1]
    cdef int ny = d5.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)
    cdef float * d5ptr = < float * > np.PyArray_DATA(d5)
    cdef float * outd5ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt5(d5ptr, outd5ptr, nx, ny)
    return output


def medfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] d7):
    """medfilt7(d7)\n
    Calculate the 7x7 median filter of an array.

    Parameters
    ----------
    d7 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyMedFilt7 in medutils.
    """
    cdef int nx = d7.shape[1]
    cdef int ny = d7.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * d7ptr = < float * > np.PyArray_DATA(d7)
    cdef float * outd7ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyMedFilt7(d7ptr, outd7ptr, nx, ny)
    return output


def sepmedfilt3(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep3):
    """sepmedfilt3(dsep3)\n
    Calculate the 3x3 separable median filter of an array.

    Parameters
    ----------
    dsep3 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 1 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt3 in medutils.
    """
    cdef int nx = dsep3.shape[1]
    cdef int ny = dsep3.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep3ptr = < float * > np.PyArray_DATA(dsep3)
    cdef float * outdsep3ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt3(dsep3ptr, outdsep3ptr, nx, ny)
    return np.asarray(output)


def sepmedfilt5(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep5):
    """sepmedfilt5(dsep5)\n
    Calculate the 5x5 separable median filter of an array.

    Parameters
    ----------
    dsep5 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 2 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt5 in medutils.
    """
    cdef int nx = dsep5.shape[1]
    cdef int ny = dsep5.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep5ptr = < float * > np.PyArray_DATA(dsep5)
    cdef float * outdsep5ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt5(dsep5ptr, outdsep5ptr, nx, ny)

    return output


def sepmedfilt7(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep7):
    """sepmedfilt7(dsep7)\n
    Calculate the 7x7 separable median filter of an array.

    Parameters
    ----------
    dsep7 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 3 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt7 in medutils.
    """
    cdef int nx = dsep7.shape[1]
    cdef int ny = dsep7.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep7ptr = < float * > np.PyArray_DATA(dsep7)
    cdef float * outdsep7ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt7(dsep7ptr, outdsep7ptr, nx, ny)
    return output


def sepmedfilt9(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsep9):
    """sepmedfilt9(dsep9)\n
    Calculate the 9x9 separable median filter of an array.

    Parameters
    ----------
    dsep9 : float numpy array
        Array to median filter.

    Returns
    -------
    output : float numpy array
        Median filtered array.

    Notes
    -----
    The separable median medians the rows followed by the columns instead of
    using a square window. Therefore it is not identical to the full median
    filter but it is approximatly the same, but it is signifcantly faster.
    The median filter is not calculated for a 4 pixel border around the image.
    These pixel values are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PySepMedFilt9 in medutils.
    """

    cdef int nx = dsep9.shape[1]
    cdef int ny = dsep9.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dsep9ptr = < float * > np.PyArray_DATA(dsep9)
    cdef float * outdsep9ptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySepMedFilt9(dsep9ptr, outdsep9ptr, nx, ny)
    return output
