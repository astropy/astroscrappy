/*
 * medutils.h
 *
 * Author: Curtis McCully
 * October 2014
 *
 * Licensed under a 3-clause BSD style license - see LICENSE.rst
 */

#ifndef MEDUTILS_H_
#define MEDUTILS_H_

/* Including definitions of the standard int types is necesssary for Windows,
 * and does no harm on other platforms. 
 */
#include <stdint.h> 

/* Define a bool type because there isn't one built in ANSI C */
typedef uint8_t bool;
#define true 1
#define false 0

/*Find the median value of an array "a" of length n. */
float
PyMedian(float* a, int n);

/*Optimized method to find the median value of an array "a" of length 3. */
float
PyOptMed3(float* a);

/*Optimized method to find the median value of an array "a" of length 5. */
float
PyOptMed5(float* a);

/*Optimized method to find the median value of an array "a" of length 7. */
float
PyOptMed7(float* a);

/*Optimized method to find the median value of an array "a" of length 9. */
float
PyOptMed9(float* a);

/*Optimized method to find the median value of an array "a" of length 25. */
float
PyOptMed25(float* a);

/* Calculate the 3x3 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 1 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the x
 * direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j].
 */
void
PyMedFilt3(float* data, float* output, int nx, int ny);

/* Calculate the 5x5 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 2 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j].
 */
void
PyMedFilt5(float* data, float* output, int nx, int ny);

/* Calculate the 7x7 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 3 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j].
 */
void
PyMedFilt7(float* data, float* output, int nx, int ny);

/* Calculate the 3x3 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place. The median
 * filter is not calculated for a 1 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along
 * the x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j]. Note that the rows are median filtered first,
 * followed by the columns.
 */
void
PySepMedFilt3(float* data, float* output, int nx, int ny);

/* Calculate the 5x5 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place.The median
 * filter is not calculated for a 2 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j]. Note that the rows are median filtered first, followed by
 * the columns.
 */
void
PySepMedFilt5(float* data, float* output, int nx, int ny);

/* Calculate the 7x7 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place. The median
 * filter is not calculated for a 3 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j]. Note that the rows are median filtered first, followed by
 * the columns.
 */
void
PySepMedFilt7(float* data, float* output, int nx, int ny);

/* Calculate the 9x9 separable median filter of an array data that has
 * dimensions nx x ny. The results are saved in the output array. The output
 * array should already be allocated as we work on it in place. The median
 * filter is not calculated for a 4 pixel border around the image. These pixel
 * values are copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j]. Note that the rows are median filtered first, followed by
 * the columns.
 */
void
PySepMedFilt9(float* data, float* output, int nx, int ny);

#endif /* MEDUTILS_H_ */
