###############################################################################
# integrator.pyx
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


#########################################################
# Internal functions									#
#########################################################

cdef double _RecursiveIntegrate(double x, void * p_mem) nogil:
    """ Recursively solves an N-integral 
    
    ---arguments---
    x : numerical value of the variable requested by the level above
    p_mem : void pointer to the _memory structure
    
    ----output---
    numerical value of the given level for the given fixed set of variables and parameters
    """

    # create local copy of _memory structure	
    cdef _memory mem
    mem.D_cur = (<_memory *>p_mem).D_cur
    mem.intgr = (<_memory *>p_mem).intgr
    mem.val = <double *>malloc(mem.intgr.D * sizeof(double))
    for i in range(0,mem.intgr.D):
        mem.val[i] = (<_memory *>p_mem).val[i]

    # prepare memory strucutre for lower level
    if mem.D_cur < mem.intgr.D:
        mem.val[mem.D_cur] = x
    mem.D_cur = mem.D_cur - 1

    # determine whether a deeper level needs to be imposed
    cdef gsl_function F
    cdef double result, error
    if mem.D_cur >= 0:
        F.function = &_RecursiveIntegrate
        F.params = &mem;
    else:
        result = mem.intgr.fct(mem.val,mem.intgr.params)
        free(mem.val)
        return result

    # call GSL to go a level deeper or solve deepest level
    cdef gsl_integration_workspace * W
    W = gsl_integration_workspace_alloc(1000)
    gsl_integration_qag( &F,mem.intgr.bound_low[mem.D_cur],
                            mem.intgr.bound_high[mem.D_cur],
                            mem.intgr.abs_err[mem.D_cur],
                            mem.intgr.rel_err[mem.D_cur],
                            1000,GSL_INTEG_GAUSS31, W, &result, &error )
    gsl_integration_workspace_free(W)
    free(mem.val)
    return result


#########################################################
# External functions									#
#########################################################

cdef double Integrate(Integral intgr) nogil:
    """ Front-end function to solves an N-integral 
    
    ---arguments---
    Integral : object carries all information about the integrals to solve
    
    ----output---
    numerical value of the integral in intgr
    """

    # prepare _memory structure
    cdef _memory mem
    mem.D_cur = intgr.D
    mem.val = <double *>malloc(intgr.D * sizeof(double))
    mem.intgr = intgr
    cdef double result

    # start solving integral
    result =  _RecursiveIntegrate(-1,&mem)
    free(mem.val)
    return result
