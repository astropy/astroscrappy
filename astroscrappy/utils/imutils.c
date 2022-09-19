/*
 * Author: Curtis McCully
 * October 2014
 * Licensed under a 3-clause BSD style license - see LICENSE.rst
 *
 * Originally written in C++ in 2011
 * See also https://github.com/cmccully/lacosmicx
 *
 * This file contains image utility functions for SCRAPPY. These are the most
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
#include "imutils.h"

/* Subsample an array 2x2 given an input array data with size nx x ny. Each
 * pixel is replicated into 4 pixels; no averaging is performed. The results
 * are saved in the output array. The output array should already be allocated
 * as we work on it in place. Data should be striped in the x direction such
 * that the memory location of pixel i,j is data[nx *j + i].
 */
void
PySubsample(float* data, float* output, int nx, int ny)
{
    /* Precalculate the new length; minor optimization */
    int padnx = 2 * nx;

    /* Loop indices */
    int i, j, nxj, padnxj;

    /* Loop over all pixels */
#pragma omp parallel for firstprivate(data, output, nx, ny, padnx) \
    private(i, j, nxj, padnxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        padnxj = 2 * padnx * j;
        for (i = 0; i < nx; i++) {
            /* Copy the pixel value into a 2x2 grid on the output image */
            output[2 * i + padnxj] = data[i + nxj];
            output[2 * i + padnxj + padnx] = data[i + nxj];
            output[2 * i + 1 + padnxj + padnx] = data[i + nxj];
            output[2 * i + 1 + padnxj] = data[i + nxj];
        }
    }

    return;
}

/* Rebin an array 2x2, with size (2 * nx) x (2 * ny). Rebin the array by block
 * averaging 4 pixels back into 1. This is effectively the opposite of
 * subsample (although subsample does not do an average). The results are saved
 * in the output array. The output array should already be allocated as we work
 * on it in place. Data should be striped in the x direction such that the
 * memory location of pixel i,j is data[nx *j + i].
 */
void
PyRebin(float* data, float* output, int nx, int ny)
{
    /* Size of original array */
    int padnx = nx * 2;

    /* Loop variables */
    int i, j, nxj, padnxj;

    /* Pixel value p. Each thread needs its own copy of this variable so we
     * wait to initialize it until the pragma below */
    float p;
#pragma omp parallel for firstprivate(output, data, nx, ny, padnx) \
    private(i, j, nxj, padnxj, p)
    /*Loop over all of the pixels */
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        padnxj = 2 * padnx * j;
        for (i = 0; i < nx; i++) {
            p = data[2 * i + padnxj];
            p += data[2 * i + padnxj + padnx];
            p += data[2 * i + 1 + padnxj + padnx];
            p += data[2 * i + 1 + padnxj];
            p /= 4.0;
            output[i + nxj] = p;
        }
    }
    return;
}

/* Convolve an image of size nx x ny with a kernel of size  kernx x kerny. The
 * results are saved in the output array. The output array should already be
 * allocated as we work on it in place. Data and kernel should both be striped
 * in the x direction such that the memory location of pixel i,j is
 * data[nx *j + i].
 */
void
PyConvolve(float* data, float* kernel, float* output, int nx, int ny,
           int kernx, int kerny)
{
    /* Get the width of the borders that we will pad with zeros */
    int bnx = (kernx - 1) / 2;
    int bny = (kerny - 1) / 2;

    /* Calculate the dimensions of the array including padded border */
    int padnx = nx + kernx - 1;
    int padny = ny + kerny - 1;
    /* Get the total number of pixels in the padded array */
    int padnxny = padnx * padny;

    /*Allocate the padded array */
    float* padarr = (float *) malloc(padnxny * sizeof(float));

    /* Loop variables. These should all be thread private. */
    int i, j;
    int nxj;
    int padnxj;
    /* Inner loop variables. Again thread private. */
    int k, l;
    int kernxl, padnxl;

    /* Define a sum variable to use in the convolution calculation. Each
     * thread needs its own copy of this so it should be thread private. */
    float sum;

    /* Precompute maximum good index in each dimension */
    int xmaxgood = nx + bnx;
    int ymaxgood = ny + bny;

    /* Set the borders of padarr = 0.0
     * Fill the rest of the padded array with the input data. */
#pragma omp parallel for \
    firstprivate(padarr, data, nx, padnx, padny, bnx, bny, xmaxgood, ymaxgood)\
    private(nxj, padnxj, i, j)
    for (j = 0; j < padny; j++) {
        padnxj = padnx * j;
        nxj = nx * (j - bny);
        for (i = 0; i < padnx; i++) {
            if (i < bnx || j < bny || j >= ymaxgood || i >= xmaxgood) {
                padarr[padnxj + i] = 0.0;
            }
            else {
                padarr[padnxj + i] = data[nxj + i - bnx];
            }
        }

    }

    /* Calculate the convolution  */
    /* Loop over all pixels */
#pragma omp parallel for \
    firstprivate(padarr, output, nx, ny, padnx, bnx, bny, kernx) \
    private(nxj, padnxj, kernxl, padnxl, i, j, k, l, sum)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        /* Note the + bvy in padnxj */
        padnxj = padnx * (j + bny);
        for (i = 0; i < nx; i++) {
            sum = 0.0;
            /* Note that the sums in the definition of the convolution go from
             * -border width to + border width */
            for (l = -bny; l <= bny; l++) {
                padnxl = padnx * (l + j + bny);
                kernxl = kernx * (-l + bny);
                for (k = -bnx; k <= bnx; k++) {
                    sum += kernel[bnx - k + kernxl]
                        * padarr[padnxl + k + i + bnx];
                }
            }
            output[nxj + i] = sum;
        }
    }

    free(padarr);

    return;
}

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
PyLaplaceConvolve(float* data, float* output, int nx, int ny)
{
    /* Precompute the total number of pixels in the image */
    int nxny = nx * ny;

    /* Loop variables */
    int i, j, nxj;

    /* Pixel value p. Each thread will need its own copy of this so we need to
     * make it private*/
    float p;
    /* Because we know the form of the kernel, we can short circuit the
     * convolution and calculate the results with inner nest for loops. */

    /*Loop over all of the pixels except the edges which we will do explicitly
     * below */
#pragma omp parallel for firstprivate(nx, ny, output, data) \
    private(i, j, nxj, p)
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        for (i = 1; i < nx - 1; i++) {
            p = 4.0 * data[nxj + i];
            p -= data[i + 1 + nxj];
            p -= data[i - 1 + nxj];
            p -= data[i + nxj + nx];
            p -= data[i + nxj - nx];

            output[nxj + i] = p;
        }
    }

    /* Leave the corners until the very end */

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* Top and Bottom Rows */
    for (i = 1; i < nx - 1; i++) {
        output[i] = 4.0 * data[i] - data[i + 1] - data[i - 1] - data[i + nx];

        p = 4.0 * data[i + nxny - nx];
        p -= data[i + 1 + nxny - nx];
        p -= data[i + nxny - nx - 1];
        p -= data[i - nx + nxny - nx];
        output[i + nxny - nx] = p;
    }

#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    /* First and Last Column */
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        p = 4.0 * data[nxj];
        p -= data[nxj + 1];
        p -= data[nxj + nx];
        p -= data[nxj - nx];
        output[nxj] = p;

        p = 4.0 * data[nxj + nx - 1];
        p -= data[nxj + nx - 2];
        p -= data[nxj + nx + nx - 1];
        p -= data[nxj - 1];
        output[nxj + nx - 1] = p;
    }

    /* Bottom Left Corner */
    output[0] = 4.0 * data[0] - data[1] - data[nx];
    /* Bottom Right Corner */
    output[nx - 1] = 4.0 * data[nx - 1] - data[nx - 2] - data[nx + nx - 1];
    /* Top Left Corner */
    p = 4.0 * data[nxny - nx];
    p -= data[nxny - nx + 1];
    p -= data[nxny - nx - nx];
    output[nxny - nx] = p;
    /* Top Right Corner */
    p = 4.0 * data[nxny - 1];
    p -= data[nxny - 2];
    p -= data[nxny - 1 - nx];
    output[nxny - 1] = p;

    return;
}

static inline uint8_t dilate_3(uint8_t* data, int i, int nxj, int nx){
    /* Start in the middle and work out */
    uint8_t p = data[i + nxj];
    /* Right 1 */
    p = p || data[i + 1 + nxj];
    /* Left 1 */
    p = p || data[i - 1 + nxj];
    /* Up 1 */
    p = p || data[i + nx + nxj];
    /* Down 1 */
    p = p || data[i - nx + nxj];
    /* Up 1 Right 1 */
    p = p || data[i + 1 + nx + nxj];
    /* Up 1 Left 1 */
    p = p || data[i - 1 + nx + nxj];
    /* Down 1 Right 1 */
    p = p || data[i + 1 - nx + nxj];
    /* Down 1 Left 1 */
    p = p || data[i - 1 - nx + nxj];
    return p;
}

static inline uint8_t dilate_5(uint8_t* data, int i, int nxj, int nx){
    uint8_t p = dilate_3(data, i, nxj, nx);
    /* Right 2 */
    p = p || data[i + 4 + nxj];
    /* Left 2 */
    p = p || data[i + nxj];
    /* Up 2 */
    p = p || data[i + 2 + nx + nx + nxj];
    /* Down 2 */
    p = p || data[i + 2 - nx - nx + nxj];
    /* Right 2 Up 1 */
    p = p || data[i + 4 + nx + nxj];
    /* Right 2 Down 1 */
    p = p || data[i + 4 - nx + nxj];
    /* Left 2 Up 1 */
    p = p || data[i + nx + nxj];
    /* Left 2 Down 1 */
    p = p || data[i - nx + nxj];
    /* Up 2 Right 1 */
    p = p || data[i + 3 + nx + nx + nxj];
    /* Up 2 Left 1 */
    p = p || data[i + 1 + nx + nx + nxj];
    /* Down 2 Right 1 */
    p = p || data[i + 3 - nx - nx + nxj];
    /* Down 2 Left 1 */
    p = p || data[i + 1 - nx - nx + nxj];

    return p;
}

#define EDGE_ROW_3 \
output[i] = data[i];\
output[nxny - nx + i] = data[nxny - nx + i]

#define EDGE_ROW_5 \
output[i + nx] = data[i + nx];\
output[nxny - nx - nx + i] = data[nxny - nx - nx + i]

static inline void dilate_edge_rows_3(uint8_t* data, uint8_t* output, int i, int nx, int nxny){
    EDGE_ROW_3;
}

static inline void dilate_edge_rows_5(uint8_t* data, uint8_t* output, int i, int nx, int nxny){
    EDGE_ROW_3;
    EDGE_ROW_5;
}

#undef EDGE_ROW_3
#undef EDGE_ROW_5

#define EDGE_COLUMN_3 \
output[nxj] = data[nxj];\
output[nxj + nx - 1] = data[nxj + nx - 1]

#define EDGE_COLUMN_5 \
output[nxj + 1] = data[nxj + 1];\
output[nxj + nx - 2] = data[nxj + nx - 2]

static inline void dilate_edge_columns_3(uint8_t* data, uint8_t* output, int nx, int nxj){
    EDGE_COLUMN_3;
}

static inline void dilate_edge_columns_5(uint8_t* data, uint8_t* output, int nx, int nxj){
    EDGE_COLUMN_3;
    EDGE_COLUMN_5;
}

#undef EDGE_COLUMN_3
#undef EDGE_COLUMN_5

static inline void dilate(uint8_t* data, uint8_t* output, int nx, int ny, uint8_t dilate_function(uint8_t*, int, int, int),
    void dilate_edge_rows(uint8_t*, uint8_t*, int, int, int), void dilate_edge_columns(uint8_t*, uint8_t*, int, int))
{
    /* Precompute the total number of pixels; minor optimization */
    int nxny = nx * ny;

    /* Loop variables */
    int i, j, nxj;

    /* Pixel value p. Each thread needs its own unique copy of this so we don't
     initialize this until the pragma below. */

#pragma omp parallel for firstprivate(output, data, nxny, nx, ny) private(i, j, nxj)

    /* Loop through all of the pixels excluding the border */
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        for (i = 1; i < nx - 1; i++) {
            output[i + nxj] = dilate_function(data, i, nxj, nx);
        }
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* For the borders, copy the data from the input array */
    for (i = 0; i < nx; i++) {
        dilate_edge_rows(data, output, i, nx, nxny);
    }
#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        dilate_edge_columns(data, output, nx, nxj);
    }
    return;
}


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
PyDilate3(uint8_t* data, uint8_t* output, int nx, int ny)
{
    dilate(data, output, nx, ny, dilate_3, dilate_edge_rows_3, dilate_edge_columns_3);
}

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
PyDilate5(uint8_t* data, uint8_t* output, int niter, int nx, int ny)
{
    dilate(data, output, nx, ny, dilate_5, dilate_edge_rows_5, dilate_edge_columns_5);
    if (niter == 1) {
    // Short circuit if we are only doing one iteration
    return;
    }
    uint8_t* intermediate = malloc(nx * ny * sizeof(uint8_t));
    int nxny = nx * ny;
    for(int i = 1; i < niter; i++){
        for(int j = 0; j < nxny; j++) {
            // Copy the last run output into the intermediate array
            intermediate[j] = output[j];
        }
        dilate(intermediate, output, nx, ny, dilate_5, dilate_edge_rows_5, dilate_edge_columns_5);
    }
}
