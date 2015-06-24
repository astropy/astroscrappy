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
    PyDoc_STRVAR(PySubsample__doc__,
        "PySubample(data, output, nx, ny) -> void\n\n"
            "Subsample an array 2x2 given an input array data with size "
            "nx x ny.The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. Each"
            " pixel is replicated into 4 pixels; no averaging is performed. "
            "Data should be striped in the x direction such that the memory "
            "location of pixel i,j is data[nx *j + i].");

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
    PyDoc_STRVAR(PyRebin__doc__,
        "PyRebin(data, output, nx, ny) -> void\n    \n"
            "Rebin an array 2x2, with size (2 * nx) x (2 * ny). Rebin the "
            "array by block averaging 4 pixels back into 1. This is "
            "effectively the opposite of subsample (although subsample does "
            "not do an average). The results are saved in the output array. "
            "The output array should already be allocated as we work on it in "
            "place. Data should be striped in the x direction such that the "
            "memory location of pixel i,j is data[nx *j + i].");

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
    PyDoc_STRVAR(PyConvolve__doc__,
        "PyConvolve(data, kernel, output, nx, ny, kernx, kerny) -> void\n\n"
            "Convolve an image of size nx x ny with a a kernel of size "
            "kernx x kerny. The results are saved in the output array. The "
            "output array should already be allocated as we work on it in "
            "place. Data and kernel should both be striped along the x "
            "direction such that the memory location of pixel i,j is "
            "data[nx *j + i].");

    /* Get the width of the borders that we will pad with zeros */
    int bnx = (kernx - 1) / 2;
    int bny = (kerny - 1) / 2;

    /* Calculate the dimensions of the array including padded border */
    int padnx = nx + kernx - 1;
    int padny = ny + kerny - 1;
    /* Get the total number of pixels in the padded array */
    int padnxny = padnx * padny;
    /*Get the total number of pixels in the output image */
    int nxny = nx * ny;

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
    PyDoc_STRVAR(PyLaplaceConvolve__doc__,
        "PyLaplaceConvolve(data, output, nx, ny) -> void\n\n"
            "Convolve an image of size nx x ny the following kernel:\n"
            " 0 -1  0\n"
            "-1  4 -1\n"
            " 0 -1  0\n"
            "This is a discrete version of the Laplacian operator. The results"
            " are saved in the output array. The output array should already "
            "be allocated as we work on it in place.Data should be striped in "
            "the x direction such that the memory location of pixel i,j is "
            "data[nx *j + i].");

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
PyDilate3(bool* data, bool* output, int nx, int ny)
{
    PyDoc_STRVAR(PyDilate3__doc__,
        "PyDilate3(data, output, nx, ny) -> void\n\n"
            "Perform a boolean dilation on an array of size nx x ny. The "
            "results are saved in the output array which should already be "
            "allocated as we work on it in place. "
            "Dilation is the boolean equivalent of a convolution but using "
            "logical or instead of a sum. We apply a 3x3 kernel of all ones. "
            "Dilation is not computed for a 1 pixel border which is copied "
            "from the input data. Data should be striped along the x-axis "
            "such that the location of pixel i,j is data[i + nx * j].");

    /* Precompute the total number of pixels; minor optimization */
    int nxny = nx * ny;

    /* Loop variables */
    int i, j, nxj;

    /* Pixel value p. Each thread needs its own unique copy of this so we don't
     initialize this until the pragma below. */
    bool p;

#pragma omp parallel for firstprivate(output, data, nxny, nx, ny) \
    private(i, j, nxj, p)

    /* Loop through all of the pixels excluding the border */
    for (j = 1; j < ny - 1; j++) {
        nxj = nx * j;
        for (i = 1; i < nx - 1; i++) {
            /*Start in the middle and work out */
            p = data[i + nxj];
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

            output[i + nxj] = p;
        }
    }

#pragma omp parallel firstprivate(output, data, nx, nxny) private(i)
    /* For the borders, copy the data from the input array */
    for (i = 0; i < nx; i++) {
        output[i] = data[i];
        output[nxny - nx + i] = data[nxny - nx + i];
    }
#pragma omp parallel firstprivate(output, data, nx, ny) private(j, nxj)
    for (j = 0; j < ny; j++) {
        nxj = nx * j;
        output[nxj] = data[nxj];
        output[nxj - 1 + nx] = data[nxj - 1 + nx];
    }

    return;
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
PyDilate5(bool* data, bool* output, int niter, int nx, int ny)
{
    PyDoc_STRVAR(PyDilate5__doc__,
        "PyDilate5(data, output, nx, ny) -> void\n\n"
            "Do niter iterations of boolean dilation on an array of size "
            "nx x ny. The results are saved in the output array. The output "
            "array should already be allocated as we work on it in place. "
            "Dilation is the boolean equivalent of a convolution but using "
            "logical ors instead of a sum. We apply the following kernel:\n"
            "0 1 1 1 0\n"
            "1 1 1 1 1\n"
            "1 1 1 1 1\n"
            "1 1 1 1 1\n"
            "0 1 1 1 0\n"
            "Data should be striped along the x direction such that the "
            "location of pixel i,j is data[i + nx * j].");

    /* Pad the array with a border of zeros */
    int padnx = nx + 4;
    int padny = ny + 4;

    /* Precompute the total number of pixels; minor optimization */
    int padnxny = padnx * padny;
    int nxny = nx * ny;

    /* The padded array to work on */
    bool* padarr = (bool *) malloc(padnxny * sizeof(bool));

    /*Loop indices */
    int i, j, nxj, padnxj;
    int iter;

    /* Pixel value p. This needs to be unique for each thread so we initialize
     * it below inside the pragma. */
    bool p;

#pragma omp parallel firstprivate(padarr, padnx, padnxny) private(i)
    /* Initialize the borders of the padded array to zero */
    for (i = 0; i < padnx; i++) {
        padarr[i] = false;
        padarr[i + padnx] = false;
        padarr[padnxny - padnx + i] = false;
        padarr[padnxny - padnx - padnx + i] = false;
    }

#pragma omp parallel firstprivate(padarr, padnx, padny) private(j, padnxj)
    for (j = 0; j < padny; j++) {
        padnxj = padnx * j;
        padarr[padnxj] = false;
        padarr[padnxj + 1] = false;
        padarr[padnxj + padnx - 1] = false;
        padarr[padnxj + padnx - 2] = false;
    }

#pragma omp parallel firstprivate(output, data, nxny) private(i)
    /* Initialize the output array to the input data */
    for (i = 0; i < nxny; i++) {
        output[i] = data[i];
    }

    /* Outer iteration loop */
    for (iter = 0; iter < niter; iter++) {
#pragma omp parallel for firstprivate(padarr, output, nx, ny, padnx, iter) \
    private(nxj, padnxj, i, j)
        /* Initialize the padded array to the output from the latest
         * iteration*/
        for (j = 0; j < ny; j++) {
            padnxj = padnx * j;
            nxj = nx * j;
            for (i = 0; i < nx; i++) {
                padarr[i + 2 + padnx + padnx + padnxj] = output[i + nxj];
            }
        }

        /* Loop over all pixels */
#pragma omp parallel for firstprivate(padarr, output, nx, ny, padnx, iter) \
    private(nxj, padnxj, i, j, p)
        for (j = 0; j < ny; j++) {
            nxj = nx * j;
            /* Note the + 2 padding in padnxj */
            padnxj = padnx * (j + 2);
            for (i = 0; i < nx; i++) {
                /* Start with the middle pixel and work out */
                p = padarr[i + 2 + padnxj];
                /* Right 1 */
                p = p || padarr[i + 3 + padnxj];
                /* Left 1 */
                p = p || padarr[i + 1 + padnxj];
                /* Up 1 */
                p = p || padarr[i + 2 + padnx + padnxj];
                /* Down 1 */
                p = p || padarr[i + 2 - padnx + padnxj];
                /* Up 1 Right 1 */
                p = p || padarr[i + 3 + padnx + padnxj];
                /* Up 1 Left 1 */
                p = p || padarr[i + 1 + padnx + padnxj];
                /* Down 1 Right 1 */
                p = p || padarr[i + 3 - padnx + padnxj];
                /* Down 1 Left 1 */
                p = p || padarr[i + 1 - padnx + padnxj];
                /* Right 2 */
                p = p || padarr[i + 4 + padnxj];
                /* Left 2 */
                p = p || padarr[i + padnxj];
                /* Up 2 */
                p = p || padarr[i + 2 + padnx + padnx + padnxj];
                /* Down 2 */
                p = p || padarr[i + 2 - padnx - padnx + padnxj];
                /* Right 2 Up 1 */
                p = p || padarr[i + 4 + padnx + padnxj];
                /* Right 2 Down 1 */
                p = p || padarr[i + 4 - padnx + padnxj];
                /* Left 2 Up 1 */
                p = p || padarr[i + padnx + padnxj];
                /* Left 2 Down 1 */
                p = p || padarr[i - padnx + padnxj];
                /* Up 2 Right 1 */
                p = p || padarr[i + 3 + padnx + padnx + padnxj];
                /* Up 2 Left 1 */
                p = p || padarr[i + 1 + padnx + padnx + padnxj];
                /* Down 2 Right 1 */
                p = p || padarr[i + 3 - padnx - padnx + padnxj];
                /* Down 2 Left 1 */
                p = p || padarr[i + 1 - padnx - padnx + padnxj];

                output[i + nxj] = p;

            }
        }

    }
    free(padarr);

    return;
}
