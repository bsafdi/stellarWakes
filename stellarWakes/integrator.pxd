###############################################################################
# integrator.pyd
###############################################################################
#
# Class to solve N-dimensional integrals recursively
# using the gsl library
#
###############################################################################

# Import basic functions
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free

# Requires cython_gsl
from cython_gsl cimport *

# Structure that carries information about the integrals
cdef struct Integral:
    int D
    int Nparams
    double * params
    double * bound_low
    double * bound_high
    double * rel_err
    double * abs_err
    double (*fct)(double *,double *) nogil

# Internal memory structure to keep track of parameters and variables 
cdef struct _memory:
    int D_cur
    double * val
    Integral intgr

# Internal functions (implemented in integrator.pyx)
cdef double _RecursiveIntegrate(double x, void * p_mem) nogil

# External functions (implemented in integrator.pyx)
cdef double Integrate(Integral intgr) nogil
