/*
 * imutils.h
 *
 * Author: Curtis McCully
 * October 2014
 *
 * Licensed under a 3-clause BSD style license - see LICENSE.rst
 */

#ifndef IMUTILS_H_
#define IMUTILS_H_

/* Including definitions of the standard int types is necesssary for Windows,
 * and does no harm on other platforms. 
 */
#include <stdint.h> 

/* Define a bool type because there isn't one built in ANSI C */
typedef uint8_t bool;
#define true 1
#define false 0

/* Subsample an array 2x2 given an input array data with size nx x ny. Each
 * pixel is replicated into 4 pixels; no averaging is performed. The results
 * are saved in the output array. The output array should already be allocated
 * as we work on it in place. Data should be striped in the x direction such
 * that the memory location of pixel i,j is data[nx *j + i].
 */
void
PySubsample(float* data, float* output, int nx, int ny);

/* Rebin an array 2x2, with size (2 * nx) x (2 * ny). Rebin the array by block
 * averaging 4 pixels back into 1. This is effectively the opposite of
 * subsample (although subsample does not do an average). The results are saved
 * in the output array. The output array should already be allocated as we work
 * on it in place. Data should be striped in the x direction such that the
 * memory location of pixel i,j is data[nx *j + i].
 */
void
PyRebin(float* data, float* output, int nx, int ny);

/* Convolve an image of size nx x ny with a kernel of size  kernx x kerny. The
 * results are saved in the output array. The output array should already be
 * allocated as we work on it in place. Data and kernel should both be striped
 * in the x direction such that the memory location of pixel i,j is
 * data[nx *j + i].
 */
void
PyConvolve(float* data, float* kernel, float* output, int nx, int ny,
           int kernx, int kerny);

/* Convolve an image of size nx x ny the following kernel:
 *  0 -1  0
 * -1  4 -1
 *  0 -1  0
 * The results are saved in the output array. The output array should
 * already be allocated as we work on it in place.
 * This is a discrete version of the Laplacian operator.
 * Data should be striped in the x direction such that the memory location of
 * pixel i,j is data[nx *j + i].
 */
void
PyLaplaceConvolve(float* data, float* output, int nx, int ny);

/* Perform a boolean dilation on an array of size nx x ny. The results are
 * saved in the output array. The output array should already be allocated as
 * we work on it in place.
 * Dilation is the boolean equivalent of a convolution but using logical ors
 * instead of a sum.
 * We apply the following kernel:
 * 1 1 1
 * 1 1 1
 * 1 1 1
 * The binary dilation is not computed for a 1 pixel border around the image.
 * These pixels are copied from the input data. Data should be striped along
 * the x direction such that the memory location of pixel i,j is
 * data[i + nx * j].
 */
void
PyDilate3(bool* data, bool* output, int nx, int ny);

/* Do niter iterations of boolean dilation on an array of size nx x ny. The
 * results are saved in the output array. The output array should already be
 * allocated as we work on it in place.
 * Dilation is the boolean equivalent of a convolution but using logical ors
 * instead of a sum.
 * We apply the following kernel:
 * 0 1 1 1 0
 * 1 1 1 1 1
 * 1 1 1 1 1
 * 1 1 1 1 1
 * 0 1 1 1 0
 * The edges are padded with zeros so that the dilation operator is defined for
 * all pixels. Data should be striped along the x direction such that the
 * memory location of pixel i,j is data[i + nx * j].
 */
void
PyDilate5(bool* data, bool* output, int iter, int nx, int ny);

#endif /* IMUTILS_H_ */
