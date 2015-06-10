# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: profile=True, boundscheck=False, nonecheck=False, wraparound=False
# cython: cdivision=True
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""
Name : image_utils
Author : Curtis McCully
Date : October 2014
"""
import numpy as np
cimport numpy as np

np.import_array()

from libc.stdint cimport uint8_t


cdef extern from "imutils.h":
    void PySubsample(float * data, float * output, int nx, int ny) nogil
    void PyRebin(float * data, float * output, int nx, int ny) nogil
    void PyConvolve(float * data, float * kernel, float * output, int nx,
                    int ny, int kernx, int kerny) nogil
    void PyLaplaceConvolve(float * data, float * output, int nx, int ny) nogil
    void PyDilate3(uint8_t * data, uint8_t * output, int nx, int ny) nogil
    void PyDilate5(uint8_t * data, uint8_t * output, int niter, int nx,
                   int ny) nogil


def subsample(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dsub):
    """subsample(dsub)\n
    Subsample an array 2x2 given an input array dsub.

    Parameters
    ----------
    dsub : float numpy array
        Array to be subsampled.

    Returns
    -------
    output : float numpy array
        Subsampled array. Output dimensions will be 2 times the input
        dimensions.

    Notes
    -----
    Each pixel is replicated into 4 pixels; no averaging is performed.
    The array needs to be C-contiguous order. Wrapper for PySubsample in
    imutils.
    """
    cdef int nx = dsub.shape[1]
    cdef int ny = dsub.shape[0]
    cdef int nx2 = 2 * nx
    cdef int ny2 = 2 * ny

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny2, nx2), dtype=np.float32)

    cdef float * dsubptr = < float * > np.PyArray_DATA(dsub)
    cdef float * outdsubptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PySubsample(dsubptr, outdsubptr, nx, ny)
    return output


def rebin(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] drebin):
    """rebin(drebin)\n
    Rebin an array 2x2.

    Rebin the array by block averaging 4 pixels back into 1.

    Parameters
    ----------
    drebin : float numpy array
        Array to be rebinned 2x2.

    Returns
    -------
    output : float numpy array
        Rebinned array. The size of the output array will be 2 times smaller
        than drebin.

    Notes
    -----
    This is effectively the opposite of subsample (although subsample does not
    do an average). The array needs to be C-contiguous order. Wrapper for
    PyRebin in imutils.
    """
    cdef int nx = drebin.shape[1] / 2
    cdef int ny = drebin.shape[0] / 2

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * drebinptr = < float * > np.PyArray_DATA(drebin)
    cdef float * outdrebinptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyRebin(drebinptr, outdrebinptr, nx, ny)
    return output


def convolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dconv,
             np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] kernel):
    """convolve(dconv, kernel)\n
    Convolve an array with a kernel.

    Parameters
    ----------
    dconv : float numpy array
        Array to be convolved.

    kernel : float numpy array
        Kernel to use in the convolution.

    Returns
    -------
    output : float numpy array
        Convolved array.

    Notes
    -----
    Both the data and kernel arrays need to be C-contiguous order. Wrapper for
    PyConvolve in imutils.
    """
    cdef int nx = dconv.shape[1]
    cdef int ny = dconv.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dconvptr = < float * > np.PyArray_DATA(dconv)
    cdef float * outdconvptr = < float * > np.PyArray_DATA(output)

    cdef int knx = kernel.shape[1]
    cdef int kny = kernel.shape[0]
    cdef float * kernptr = < float * > np.PyArray_DATA(kernel)

    with nogil:
        PyConvolve(dconvptr, kernptr, outdconvptr, nx, ny, knx, kny)
    return output


def laplaceconvolve(np.ndarray[np.float32_t, ndim=2, mode='c', cast=True] dl):
    """laplaceconvolve(dl)\n
    Convolve an array with the Laplacian kernel.

    Convolve with the discrete version of the Laplacian operator with kernel:\n
     0 -1  0\n
    -1  4 -1\n
     0 -1  0\n

    Parameters
    ----------
    dl : float numpy array
        Array to be convolved.

    Returns
    -------
    output: float numpy array
        Convolved array.

    Notes
    -----
    The array needs to be C-contiguous order. Wrapper for PyLaplaceConvolve
    in imutils.
    """
    cdef int nx = dl.shape[1]
    cdef int ny = dl.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.float32)

    cdef float * dlapptr = < float * > np.PyArray_DATA(dl)
    cdef float * outdlapptr = < float * > np.PyArray_DATA(output)
    with nogil:
        PyLaplaceConvolve(dlapptr, outdlapptr, nx, ny)
    return output


def dilate3(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] dgrow):
    """dilate3(dgrow)\n
    Perform a boolean dilation on an array.

    Parameters
    ----------
    dgrow : boolean numpy array
        Array to dilate.

    Returns
    -------
    output : boolean numpy array
        Dilated array.

    Notes
    -----
    Dilation is the boolean equivalent of a convolution but using logical ors
    instead of a sum.
    We apply the following kernel:\n
    1 1 1\n
    1 1 1\n
    1 1 1\n
    The binary dilation is not computed for a 1 pixel border around the image.
    These pixels are copied from the input data. The array needs to be
    C-contiguous order. Wrapper for PyDilate3 in imutils.
    """
    cdef int nx = dgrow.shape[1]
    cdef int ny = dgrow.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.bool)

    cdef uint8_t * dgrowptr = < uint8_t * > np.PyArray_DATA(dgrow)
    cdef uint8_t * outdgrowptr = < uint8_t * > np.PyArray_DATA(output)
    with nogil:
        PyDilate3(dgrowptr, outdgrowptr, nx, ny)
    return output


def dilate5(np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True] ddilate,
            int niter):
    """dilate5(data, niter)\n
    Do niter iterations of boolean dilation on an array.

    Parameters
    ----------
    ddilate : boolean numpy array
        Array to dilate.

    niter : int
        Number of iterations.

    Returns
    -------
    output : boolean numpy array
        Dilated array.

    Notes
    -----
    Dilation is the boolean equivalent of a convolution but using logical ors
    instead of a sum.
    We apply the following kernel:\n
    0 1 1 1 0\n
    1 1 1 1 1\n
    1 1 1 1 1\n
    1 1 1 1 1\n
    0 1 1 1 0\n
    The edges are padded with zeros so that the dilation operator is defined
    for all pixels. The array needs to be C-contiguous order. Wrapper for
    PyDilate5 in imutils.
    """
    cdef int nx = ddilate.shape[1]
    cdef int ny = ddilate.shape[0]

    # Allocate the output array here so that Python tracks the memory and will
    # free the memory when we are finished with the output array.
    output = np.zeros((ny, nx), dtype=np.bool)

    cdef uint8_t * ddilateptr = < uint8_t * > np.PyArray_DATA(ddilate)
    cdef uint8_t * outddilateptr = < uint8_t * > np.PyArray_DATA(output)
    with nogil:
        PyDilate5(ddilateptr, outddilateptr, niter, nx, ny)
    return output
