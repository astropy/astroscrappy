/*
 * Author: Curtis McCully
 * October 2014
 * Licensed under a 3-clause BSD style license - see LICENSE.rst
 *
 * Originally written in C++ in 2011
 * See also https://github.com/cmccully/lacosmicx
 *
 * This file contains median utility functions for Astro-SCRAPPY. These are the most
 * computationally expensive pieces of the calculation so they have been ported
 * to C.
 *
 * Many thanks to Nicolas Devillard who wrote the optimized methods for finding
 * the median and placed them in the public domain. I have noted in the
 * comments places that use Nicolas Devillard's code.
 *
 * Parallelization has been achieved using OpenMP. Using a compiler that does
 * not support OpenMP, e.g. clang currently, the code should still compile and
 * run serially without issue. I have tried to be explicit as possible about
 * specifying which variables are private and which should be shared, although
 * we never actually have any shared variables. We use firstprivate instead.
 * This does mean that it is important that we never have two threads write to
 * the same memory position at the same time.
 *
 * All calculations are done with 32 bit floats to keep the memory footprint
 * small.
 */
#include<Python.h>
#include "medutils.h"
#define ELEM_SWAP(a,b) { float t=(a);(a)=(b);(b)=t; }

float
PyMedian(float* a, int n)
{
    /* Get the median of an array "a" with length "n"
     * using the Quickselect algorithm. Returns a float.
     * This Quickselect routine is based on the algorithm described in
     * "Numerical recipes in C", Second Edition, Cambridge University Press,
     * 1992, Section 8.5, ISBN 0-521-43108-5
     * This code by Nicolas Devillard - 1998. Public domain.
     */

    /* Make a copy of the array so that we don't alter the input array */
    float* arr = (float *) malloc(n * sizeof(float));
    /* Indices of median, low, and high values we are considering */
    int low = 0;
    int high = n - 1;
    int median = (low + high) / 2;
    /* Running indices for the quick select algorithm */
    int middle, ll, hh;
    /* The median to return */
    float med;

    /* running index i */
    int i;
    /* Copy the input data into the array we work with */
    for (i = 0; i < n; i++) {
        arr[i] = a[i];
    }

    /* Start an infinite loop */
    while (true) {

        /* Only One or two elements left */
        if (high <= low + 1) {
            /* Check if we need to swap the two elements */
            if ((high == low + 1) && (arr[low] > arr[high]))
                ELEM_SWAP(arr[low], arr[high]);
            med = arr[median];
            free(arr);
            return med;
        }

        /* Find median of low, middle and high items;
         * swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle,
         * swap items when stuck */
        ll = low + 1;
        hh = high;
        while (true) {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position low) back into
         * the correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }

}

#undef ELEM_SWAP

/* All of the optimized median methods below were written by
 * Nicolas Devillard and are in the public domain.
 */

#define PIX_SORT(a,b) { if (a>b) PIX_SWAP(a,b); }
#define PIX_SWAP(a,b) { float temp=a; a=b; b=temp; }

/* ----------------------------------------------------------------------------
 Function :   PyOptMed3()
 In       :   pointer to array of 3 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 3 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made on the nature of the input
 signal.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed3(float* p)
{
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[0], p[1]);
    return p[1];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed5()
 In       :   pointer to array of 5 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 5 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made on the nature of the input
 signal.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed5(float* p)
{
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[4]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[2], p[3]);
    PIX_SORT(p[1], p[2]);
    return p[2];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed7()
 In       :   pointer to array of 7 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 7 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made on the nature of the input
 signal.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed7(float* p)
{
    PIX_SORT(p[0], p[5]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[1], p[6]);
    PIX_SORT(p[2], p[4]);
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[5]);
    PIX_SORT(p[2], p[6]);
    PIX_SORT(p[2], p[3]);
    PIX_SORT(p[3], p[6]);
    PIX_SORT(p[4], p[5]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[1], p[3]);
    PIX_SORT(p[3], p[4]);
    return p[3];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed9()
 In       :   pointer to an array of 9 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 9 pixel values
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Formula from:
 XILINX XCELL magazine, vol. 23 by John L. Smith

 The input array is modified in the process
 The result array is guaranteed to contain the median
 value in middle position, but other elements are NOT sorted.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed9(float* p)
{
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[4], p[5]);
    PIX_SORT(p[7], p[8]);
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[4]);
    PIX_SORT(p[6], p[7]);
    PIX_SORT(p[1], p[2]);
    PIX_SORT(p[4], p[5]);
    PIX_SORT(p[7], p[8]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[5], p[8]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[3], p[6]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[2], p[5]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[4], p[2]);
    PIX_SORT(p[6], p[4]);
    PIX_SORT(p[4], p[2]);
    return p[4];
}

/* ----------------------------------------------------------------------------
 Function :   PyOptMed25()
 In       :   pointer to an array of 25 pixel values
 Out      :   a pixel value
 Job      :   optimized search of the median of 25 pixel values
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Code taken from Graphic Gems.
 Code adapted from Nicolas Devillard.
 --------------------------------------------------------------------------- */
float
PyOptMed25(float* p)
{
    PIX_SORT(p[0], p[1]);
    PIX_SORT(p[3], p[4]);
    PIX_SORT(p[2], p[4]);
    PIX_SORT(p[2], p[3]);
    PIX_SORT(p[6], p[7]);
    PIX_SORT(p[5], p[7]);
    PIX_SORT(p[5], p[6]);
    PIX_SORT(p[9], p[10]);
    PIX_SORT(p[8], p[10]);
    PIX_SORT(p[8], p[9]);
    PIX_SORT(p[12], p[13]);
    PIX_SORT(p[11], p[13]);
    PIX_SORT(p[11], p[12]);
    PIX_SORT(p[15], p[16]);
    PIX_SORT(p[14], p[16]);
    PIX_SORT(p[14], p[15]);
    PIX_SORT(p[18], p[19]);
    PIX_SORT(p[17], p[19]);
    PIX_SORT(p[17], p[18]);
    PIX_SORT(p[21], p[22]);
    PIX_SORT(p[20], p[22]);
    PIX_SORT(p[20], p[21]);
    PIX_SORT(p[23], p[24]);
    PIX_SORT(p[2], p[5]);
    PIX_SORT(p[3], p[6]);
    PIX_SORT(p[0], p[6]);
    PIX_SORT(p[0], p[3]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[1], p[7]);
    PIX_SORT(p[1], p[4]);
    PIX_SORT(p[11], p[14]);
    PIX_SORT(p[8], p[14]);
    PIX_SORT(p[8], p[11]);
    PIX_SORT(p[12], p[15]);
    PIX_SORT(p[9], p[15]);
    PIX_SORT(p[9], p[12]);
    PIX_SORT(p[13], p[16]);
    PIX_SORT(p[10], p[16]);
    PIX_SORT(p[10], p[13]);
    PIX_SORT(p[20], p[23]);
    PIX_SORT(p[17], p[23]);
    PIX_SORT(p[17], p[20]);
    PIX_SORT(p[21], p[24]);
    PIX_SORT(p[18], p[24]);
    PIX_SORT(p[18], p[21]);
    PIX_SORT(p[19], p[22]);
    PIX_SORT(p[8], p[17]);
    PIX_SORT(p[9], p[18]);
    PIX_SORT(p[0], p[18]);
    PIX_SORT(p[0], p[9]);
    PIX_SORT(p[10], p[19]);
    PIX_SORT(p[1], p[19]);
    PIX_SORT(p[1], p[10]);
    PIX_SORT(p[11], p[20]);
    PIX_SORT(p[2], p[20]);
    PIX_SORT(p[2], p[11]);
    PIX_SORT(p[12], p[21]);
    PIX_SORT(p[3], p[21]);
    PIX_SORT(p[3], p[12]);
    PIX_SORT(p[13], p[22]);
    PIX_SORT(p[4], p[22]);
    PIX_SORT(p[4], p[13]);
    PIX_SORT(p[14], p[23]);
    PIX_SORT(p[5], p[23]);
    PIX_SORT(p[5], p[14]);
    PIX_SORT(p[15], p[24]);
    PIX_SORT(p[6], p[24]);
    PIX_SORT(p[6], p[15]);
    PIX_SORT(p[7], p[16]);
    PIX_SORT(p[7], p[19]);
    PIX_SORT(p[13], p[21]);
    PIX_SORT(p[15], p[23]);
    PIX_SORT(p[7], p[13]);
    PIX_SORT(p[7], p[15]);
    PIX_SORT(p[1], p[9]);
    PIX_SORT(p[3], p[11]);
    PIX_SORT(p[5], p[17]);
    PIX_SORT(p[11], p[17]);
    PIX_SORT(p[9], p[17]);
    PIX_SORT(p[4], p[10]);
    PIX_SORT(p[6], p[12]);
    PIX_SORT(p[7], p[14]);
    PIX_SORT(p[4], p[6]);
    PIX_SORT(p[4], p[7]);
    PIX_SORT(p[12], p[14]);
    PIX_SORT(p[10], p[14]);
    PIX_SORT(p[6], p[7]);
    PIX_SORT(p[10], p[12]);
    PIX_SORT(p[6], p[10]);
    PIX_SORT(p[6], p[17]);
    PIX_SORT(p[12], p[17]);
    PIX_SORT(p[7], p[17]);
    PIX_SORT(p[7], p[10]);
    PIX_SORT(p[12], p[18]);
    PIX_SORT(p[7], p[12]);
    PIX_SORT(p[10], p[18]);
    PIX_SORT(p[12], p[20]);
    PIX_SORT(p[10], p[20]);
    PIX_SORT(p[10], p[12]);

    return p[12];
}

#undef PIX_SORT
#undef PIX_SWAP

float PyOptMed49(float* p){
    return PyMedian(p, 49);
}

#define MEDIAN_INNER_LOOP(half_width) \
int medcounter = 0; \
int nxk; \
/* The compiler should optimize away these loops */ \
for (int k = -half_width; k < half_width + 1; k++) {\
    nxk = nx * k; \
    for (int l = -half_width; l < half_width + 1; l++) { \
        medarr[medcounter] = data[nxj + i + nxk + l]; \
        medcounter++; \
    } \
}\

static inline void populate_median_array_1(float* medarr, float* data, int half_width, int nxj, int i, int nx) {
    MEDIAN_INNER_LOOP(1);
}

static inline void populate_median_array_2(float* medarr, float* data, int half_width, int nxj, int i, int nx) {
    MEDIAN_INNER_LOOP(2);
}
static inline void populate_median_array_3(float* medarr, float* data, int half_width, int nxj, int i, int nx) {
    MEDIAN_INNER_LOOP(3);
}

#undef MEDIAN_INNER_LOOP

#define EDGE_ROW_0 \
output[i] = data[i];\
output[nxny - nx + i] = data[nxny - nx + i]

#define EDGE_ROW_1 \
output[i + nx] = data[i + nx]; \
output[nxny - nx - nx + i] = data[nxny - nx - nx + i]

#define EDGE_ROW_2 \
output[i + nx + nx] = data[i + nx + nx]; \
output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i]

#define EDGE_ROW_3 \
output[i + nx + nx + nx] = data[i + nx + nx + nx]; \
output[nxny - nx - nx - nx - nx + i] = data[nxny - nx - nx - nx - nx + i]

static inline void edge_rows_1(float* data, float* output, int i, int nx, int nxny) {
    EDGE_ROW_0;
}

static inline void edge_rows_2(float* data, float* output, int i, int nx, int nxny) {
    EDGE_ROW_0;
    EDGE_ROW_1;
}

static inline void edge_rows_3(float* data, float* output, int i, int nx, int nxny) {
    EDGE_ROW_0;
    EDGE_ROW_1;
    EDGE_ROW_2;
}

static inline void edge_rows_4(float* data, float* output, int i, int nx, int nxny) {
    EDGE_ROW_0;
    EDGE_ROW_1;
    EDGE_ROW_2;
    EDGE_ROW_3;
}

#undef EDGE_ROW_0
#undef EDGE_ROW_1
#undef EDGE_ROW_2
#undef EDGE_ROW_3

#define EDGE_COLUMN_0 \
int nxj = nx * j;\
output[nxj] = data[nxj]; \
output[nxj + nx - 1] = data[nxj + nx - 1]

#define EDGE_COLUMN_1 \
output[nxj + 1] = data[nxj + 1];\
output[nxj + nx - 2] = data[nxj + nx - 2]

#define EDGE_COLUMN_2 \
output[nxj + 2] = data[nxj + 2]; \
output[nxj + nx - 3] = data[nxj + nx - 3]

#define EDGE_COLUMN_3 \
output[nxj + 3] = data[nxj + 3]; \
output[nxj + nx - 4] = data[nxj + nx - 4]

static inline void edge_columns_1(float* data, float* output, int j, int nx) {
    EDGE_COLUMN_0;
}

static inline void edge_columns_2(float* data, float* output, int j, int nx) {
    EDGE_COLUMN_0;
    EDGE_COLUMN_1;
}

static inline void edge_columns_3(float* data, float* output, int j, int nx) {
    EDGE_COLUMN_0;
    EDGE_COLUMN_1;
    EDGE_COLUMN_2;
}

static inline void edge_columns_4(float* data, float* output, int j, int nx) {
    EDGE_COLUMN_0;
    EDGE_COLUMN_1;
    EDGE_COLUMN_2;
    EDGE_COLUMN_3;
}

#undef EDGE_COLUMN_0
#undef EDGE_COLUMN_1
#undef EDGE_COLUMN_2
#undef EDGE_COLUMN_3

static inline void median_filter(float* data, float* output, int nx, int ny,
  int filter_size, float median_function(float*),
  void populate_median_array_function(float*, float*, int, int, int, int),
  void edge_column_function(float*, float*, int, int), void edge_row_function(float*, float*, int, int, int),
  int half_width)
{
    /*Total size of the array */
    int nxny = nx * ny;

    /* Loop indices */
    int i, j, nxj;

    /* 25 element array to calculate the median and a counter index. Note that
     * these both need to be unique for each thread so they both need to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;

    /* Each thread needs to access the data and the output so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(output, data, nx, ny, median_function, half_width) \
    private(i, j, medarr, nxj)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(filter_size * filter_size * sizeof(float));

        /* Go through each pixel excluding the border.*/
#pragma omp for nowait
        for (j = half_width; j < ny - half_width; j++) {
            /* Precalculate the multiplication nx * j, minor optimization */
            nxj = nx * j;
            for (i = half_width; i < nx - half_width; i++) {
                populate_median_array_function(medarr, data, half_width, nxj, i, nx);
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = median_function(medarr);
            }
        }
        /* Each thread needs to free its own copy of medarr */
        free(medarr);
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* Copy the border pixels from the original data into the output array */
    for (i = 0; i < nx; i++) {
        edge_row_function(data, output, i, nx, nxny);
    }

#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        edge_column_function(data, output, j, nx);
    }

    return;
}

/* We have slightly unusual boundary conditions for all of the median filters
 * below. Rather than padding the data, we just don't calculate the median
 * filter for pixels around the border of the output image (n - 1) / 2 from
 * the edge, where we are using an n x n median filter. Edge effects often
 * look like cosmic rays and the edges are often blank so this shouldn't
 * matter. We fill the border with the original data values.
 */

/* Calculate the 3x3 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 1 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the x
 * direction, such that pixel i,j in the 2D image should have memory location
 * data[i + nx *j].
 */
void
PyMedFilt3(float* data, float* output, int nx, int ny)
{
    median_filter(data, output, nx, ny, 3, PyOptMed9, populate_median_array_1, edge_columns_1, edge_rows_1, 1);
}

/* Calculate the 5x5 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 2 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j].
 */

void
PyMedFilt5(float* data, float* output, int nx, int ny){
    median_filter(data, output, nx, ny, 5, PyOptMed25, populate_median_array_2, edge_columns_2, edge_rows_2, 2);
}

/* Calculate the 7x7 median filter of an array data that has dimensions
 * nx x ny. The results are saved in the output array. The output array should
 * already be allocated as we work on it in place. The median filter is not
 * calculated for a 3 pixel border around the image. These pixel values are
 * copied from the input data. The data should be striped along the
 * x direction, such that pixel i,j in the 2D image should have memory
 * location data[i + nx *j].
 */
void
PyMedFilt7(float* data, float* output, int nx, int ny)
{
    median_filter(data, output, nx, ny, 7, PyOptMed49, populate_median_array_3, edge_columns_3, edge_rows_3, 3);
}

#define MEDIAN_ROW_3 \
medarr[0] = data[nxj + i]; \
medarr[1] = data[nxj + i - 1]; \
medarr[2] = data[nxj + i + 1]

#define MEDIAN_ROW_5 \
medarr[3] = data[nxj + i - 2]; \
medarr[4] = data[nxj + i + 2]

#define MEDIAN_ROW_7 \
medarr[5] = data[nxj + i - 3]; \
medarr[6] = data[nxj + i + 3]

#define MEDIAN_ROW_9 \
medarr[7] = data[nxj + i - 4]; \
medarr[8] = data[nxj + i + 4]

static inline void populate_row_median_array_3(float* medarr, float* data, int nxj, int i) {
    MEDIAN_ROW_3;
}

static inline void populate_row_median_array_5(float* medarr, float* data, int nxj, int i) {
    MEDIAN_ROW_3;
    MEDIAN_ROW_5;
}

static inline void populate_row_median_array_7(float* medarr, float* data, int nxj, int i) {
    MEDIAN_ROW_3;
    MEDIAN_ROW_5;
    MEDIAN_ROW_7;
}

static inline void populate_row_median_array_9(float* medarr, float* data, int nxj, int i) {
    MEDIAN_ROW_3;
    MEDIAN_ROW_5;
    MEDIAN_ROW_7;
    MEDIAN_ROW_9;
}

#undef MEDIAN_ROW_3
#undef MEDIAN_ROW_5
#undef MEDIAN_ROW_7
#undef MEDIAN_ROW_9

#define MEDIAN_COLUMN_3 \
medarr[0] = data[i + nxj]; \
medarr[1] = data[i + nxj - nx]; \
medarr[2] = data[i + nxj + nx]

#define MEDIAN_COLUMN_5 \
medarr[3] = data[i + nxj + nx + nx]; \
medarr[4] = data[i + nxj - nx - nx]

#define MEDIAN_COLUMN_7 \
medarr[5] = data[i + nxj + nx + nx + nx]; \
medarr[6] = data[i + nxj - nx - nx - nx]

#define MEDIAN_COLUMN_9 \
medarr[7] = data[i + nxj + nx + nx + nx + nx]; \
medarr[8] = data[i + nxj - nx - nx - nx - nx]

static inline void populate_column_median_array_3(float* medarr, float* data, int nxj, int i, int nx) {
    MEDIAN_COLUMN_3;
}

static inline void populate_column_median_array_5(float* medarr, float* data, int nxj, int i, int nx) {
    MEDIAN_COLUMN_3;
    MEDIAN_COLUMN_5;
}

static inline void populate_column_median_array_7(float* medarr, float* data, int nxj, int i, int nx) {
    MEDIAN_COLUMN_3;
    MEDIAN_COLUMN_5;
    MEDIAN_COLUMN_7;
}

static inline void populate_column_median_array_9(float* medarr, float* data, int nxj, int i, int nx) {
    MEDIAN_COLUMN_3;
    MEDIAN_COLUMN_5;
    MEDIAN_COLUMN_7;
    MEDIAN_COLUMN_9;
}

#undef MEDIAN_COLUMN_3
#undef MEDIAN_COLUMN_5
#undef MEDIAN_COLUMN_7
#undef MEDIAN_COLUMN_9

static inline void separable_median_filter(float* data, float* output, int nx, int ny,
    int filter_size, float median_function(float*),
    void populate_row_median_array_function(float*, float*, int, int),
    void populate_column_median_array_function(float*, float*, int, int, int),
    void edge_column_function(float*, float*, int, int), void edge_row_function(float*, float*, int, int, int),
    int half_width)
{
     /* Total number of pixels */
    int nxny = nx * ny;

    /* Output array for the median filter of the rows. We later median filter
     * the columns of this array. */
    float* rowmed = (float *) malloc(nxny * sizeof(float));

    /* Loop indices */
    int i, j, nxj;

    /* 5 element array to calculate the median and a counter index. Note that
     * this array needs to be unique for each thread so it needs to be
     * private and we wait to allocate memory until the pragma below. */
    float* medarr;

    /* Median filter the rows first */

    /* Each thread needs to access the data and rowmed so we make them
     * firstprivate. We make sure that our algorithm doesn't have multiple
     * threads read or write the same piece of memory. */
#pragma omp parallel firstprivate(data, rowmed, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /*Each thread allocates its own array. */
        medarr = (float *) malloc(filter_size * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            for (i = half_width; i < nx - half_width; i++) {
                populate_row_median_array_function(medarr, data, nxj, i);
                /* Calculate the median in the fastest way possible */
                rowmed[nxj + i] = median_function(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }

    /* Fill in the borders of rowmed with the original data values */
#pragma omp parallel for firstprivate(rowmed, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        edge_column_function(data, rowmed, j, nx);
    }

    /* Median filter the columns */
#pragma omp parallel firstprivate(rowmed, output, nx, ny) \
    private(i, j, nxj, medarr)
    {
        /* Each thread needs to reallocate a new medarr */
        medarr = (float *) malloc(filter_size * sizeof(float));

        /* For each pixel excluding the border */
#pragma omp for nowait
        for (j = half_width; j < ny - half_width; j++) {
            nxj = nx * j;
            for (i = half_width; i < nx - half_width; i++) {
                populate_column_median_array_function(medarr, rowmed, nxj, i, nx);
                /* Calculate the median in the fastest way possible */
                output[nxj + i] = median_function(medarr);
            }
        }
        /* Each thread needs to free its own medarr */
        free(medarr);
    }
    /* Clean up rowmed */
    free(rowmed);

    /* Copy the border pixels from the original data into the output array */
#pragma omp parallel for firstprivate(output, data, nx, nxny) private(i)
    for (i = 0; i < nx; i++) {
        edge_row_function(data, output, i, nx, nxny);
    }
#pragma omp parallel for firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        edge_column_function(data, output, j, nx);
    }

    return;
}

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
PySepMedFilt3(float* data, float* output, int nx, int ny)
{
    separable_median_filter(data, output, nx, ny, 3, PyOptMed3, populate_row_median_array_3,
        populate_column_median_array_3, edge_columns_1, edge_rows_1, 1);
}

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
PySepMedFilt5(float* data, float* output, int nx, int ny)
{
    separable_median_filter(data, output, nx, ny, 5, PyOptMed5, populate_row_median_array_5,
        populate_column_median_array_5, edge_columns_2, edge_rows_2, 2);
}

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
PySepMedFilt7(float* data, float* output, int nx, int ny)
{
    separable_median_filter(data, output, nx, ny, 7, PyOptMed7, populate_row_median_array_7,
        populate_column_median_array_7, edge_columns_3, edge_rows_3, 3);
}

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
PySepMedFilt9(float* data, float* output, int nx, int ny)
{
    separable_median_filter(data, output, nx, ny, 9, PyOptMed9, populate_row_median_array_9,
        populate_column_median_array_9, edge_columns_4, edge_rows_4, 4);
}
