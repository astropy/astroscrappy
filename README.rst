Astro-SCRAPPY: The Speedy Cosmic Ray Annihilation Package in Python
===================================================================

.. image:: https://readthedocs.org/projects/astroscrappy/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://astroscrappy.readthedocs.io/en/latest/?badge=latest
.. image:: https://travis-ci.org/astropy/astroscrappy.png
    :target: https://travis-ci.org/astropy/astroscrappy
.. image:: https://coveralls.io/repos/astropy/astroscrappy/badge.png
    :target: https://coveralls.io/r/astropy/astroscrappy
    :alt: Travis Status
.. image:: https://zenodo.org/badge/36837126.svg
    :target: https://zenodo.org/badge/latestdoi/36837126
    

An optimized cosmic ray detector.

:Author: Curtis McCully


Astro-SCRAPPY is designed to detect cosmic rays in images (numpy arrays),
based on Pieter van Dokkum's L.A.Cosmic algorithm.

Much of this was originally adapted from cosmics.py written by Malte Tewes.
I have ported all of the slow functions to Cython/C, and optimized
where I can. This is designed to be as fast as possible so some of the
readability has been sacrificed, specifically in the C code.

If you use this code, please cite the Zendo DOI: https://zenodo.org/record/1482019

Please cite the original paper which can be found at:
http://www.astro.yale.edu/dokkum/lacosmic/

van Dokkum 2001, PASP, 113, 789, 1420
(article : http://adsabs.harvard.edu/abs/2001PASP..113.1420V)

This code requires Cython, preferably version >= 0.21.

Parallelization is achieved using OpenMP. This code should compile (although
the Cython files may have issues) using a compiler that does not support OMP,
e.g. clang.

Notes
-----
There are some differences from original LA Cosmic:

- Automatic recognition of saturated stars.
  This avoids treating such stars as large cosmic rays.

- I have tried to optimize all of the code as much as possible while
  maintaining the integrity of the algorithm. One of the key speedups is to
  use a separable median filter instead of the true median filter. While these
  are not identical, they produce comparable results and the separable version
  is much faster.

- This implementation is much faster than the Python by as much as a factor of
  ~17 depending on the given parameters, even without running multiple threads.
  With multiple threads, this can be increased easily by another factor of 2.
  This implementation is much faster than the original IRAF version, improvment
  by a factor of ~90.

The arrays always must be C-contiguous, thus all loops are y outer, x inner.
This follows the astropy.io.fits (pyfits) convention.

scipy is required for certain tests to pass, but the code itself does not depend on
scipy.


License
-------

This project is Copyright (c) Astropy Developers and licensed under the terms of the BSD 3-Clause license. See the licenses folder for more information.
