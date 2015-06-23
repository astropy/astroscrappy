"""
Header file for Cython functions in the utils package.

This allows the Cython code to call these routines directly
without requiring the GIL.
"""

"""
Calculate the median on the first n elements of C float array
without requiring the GIL.
"""
cdef float cymedian(float* aptr, int n) nogil