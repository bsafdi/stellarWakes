###############################################################################
# LL.pyx
###############################################################################
#
# Return the log likelihood for the Plummer sphere model
#
# Data is presented in lists kinematic data of the form [x,y,z,v_x,v_y,v_z]
#
###############################################################################

# Import basic functions
import numpy as np
cimport numpy as np
cimport cython

# Requires cython_gsl
from cython_gsl cimport gsl_sf_dawson

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
	double log(double x) nogil
	double pow(double x, double y) nogil
	double sqrt(double x) nogil
	double asinh(double x) nogil


# Define basic physics quantities
cdef double c = 2.99e+05 ##km/s
cdef double MpctoGyr = 0.0032616334 # in Gyr
cdef double GMrr=1.4675919e-05  # GM/r^2 in 1/[Gyr], for r = 1 Mpc, M=1e12 Msun
cdef double GMr=GMrr*MpctoGyr*c**2 # GM/r (km/s)**2, for r = 1 Mpc, M=1e12 Msun
cdef double GMr_kpc=1.0e3*GMrr*MpctoGyr*c**2 # GM/r (km/s)**2, for r = 1 Mpc, M=1e12 Msun
cdef double pi = np.pi


#########################################################
# Internal functions									#
#########################################################


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double _phase_space_plummer(double x0, double y0, double z0, double vx0, double vy0, double vz0, double Halo_x, double Halo_y, double Halo_z,double v_Halo_x, double v_Halo_y, double v_Halo_z, double M_halo,double v0, double rs) nogil:
	""" Returns the phase-space distribution at phase-space point as function of model parameters
    
    ---arguments---
    x0 : star x-position in kpc
    y0 : star y-position in kpc
    z0 : star z-position in kpc
    vx0 : star x-velocity in km/s
    vy0 : star y-velocity in km/s
    vz0 : star z-velocity in km/s

    Halo_x : Halo x-position in kpc
    Halo_y : Halo y-position in kpc
    Halo_z : Halo z-position in kpc
    v_Halo_x : Halo x-velocity in km/s
    v_Halo_y : Halo y-velocity in km/s
    v_Halo_z : Halo z-velocity in km/s

    M_halo : Halo mass in M_sun
    v0 : background velocity dispersion in km/s
    rs : Halo Plummer radius in kpc
    
    ----output---
    phase-space distribution (kpc^{-3} (km/s)^{-3})  
    """

	cdef double x0t = x0 - Halo_x
	cdef double y0t = y0 - Halo_y
	cdef double z0t = z0 - Halo_z

	cdef double r = sqrt( x0t**2 + y0t**2 + z0t**2 )
	cdef double Gf = GMr_kpc*(M_halo/1.0e12)*(1.0/r)

	cdef double vx0_HF = vx0 - v_Halo_x  ##Halo frame quantities
	cdef double vy0_HF = vy0 - v_Halo_y
	cdef double vz0_HF = vz0 - v_Halo_z

	cdef double v_lab_2 = vx0**2 + vy0**2 + vz0**2
	cdef double v_halo = sqrt(vx0_HF**2 + vy0_HF**2 + vz0_HF**2)

	cdef double norm_factor = 2*Gf/pow(v0,2)/v_halo

	cdef double vlab_dot_xhat = (vx0*x0t + vy0*y0t + vz0*z0t)/r
	cdef double vlab_dot_vhalo_hat = (vx0*vx0_HF + vy0*vy0_HF + vz0*vz0_HF)/v_halo
	cdef double xh_dot_vhalo_hat = (x0t*vx0_HF + y0t*vy0_HF + z0t*vz0_HF)/r/v_halo

	cdef double phase_factor 
	
	phase_factor = ((xh_dot_vhalo_hat/sqrt(1+rs**2/r**2)+1)*vlab_dot_xhat-(sqrt(1+rs**2/r**2)+xh_dot_vhalo_hat)*vlab_dot_vhalo_hat)/(1+rs**2/r**2-(xh_dot_vhalo_hat)**2)		 
	return norm_factor*phase_factor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double _volume_fact(double R, double v0, double vsh, double Msh, double rs) nogil:
	""" Returns the predicted number of stars within the ROI as functions of the model parameters
    
    ---arguments---
    R : radius of ROI in kpc
    v0 : background velocity dispersion in km/s
    vsh : speed of subhalo in km/s
    Msh : Halo mass in M_sun
    rs : Halo Plummer radius in kpc
    
    ----output---
    Number of stars within ROI 
    """

	cdef double volume = 4./3.* pi * R**3+4.*pi*GMr_kpc * (Msh/1.0e12)*gsl_sf_dawson(vsh/v0) / (vsh * v0) *(R*sqrt(R**2+rs**2)-rs**2*asinh(R/rs))		
		
	return volume



#########################################################
# External functions									#
#########################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double Plummer(double[:,::1] star_array, double[::1] phase, double Msh, double rs,double v0, double R,double ns) nogil:
	""" Returns the log likelihood for the Plummer sphere model as a function of the stellar kinematic data and the subhalo model pararameters
    
    ---arguments---
    star_array : array of stellar kinematic data.  Each element is of the form [x,y,z,vx,vy,vz], where [x,y,z] are star positions in kpc and [vx,vy,vz] are star velocities in km/s
    phase : position and velocity of the DM subhalo, of the form [x_Halo,y_Halo,z_Halo,v_Halo_x, v_Halo_y, v_Halo_z], with the same units as star_array
    Msh : mass of the subhalo in M_sun
    rs : Halo Plummer radius in kpc
    v0 : background velocity dispersion in km/s
    R : radius of ROI in kpc
    ns : number density (normalization) model parameter for the phase-space distribution, in units of stars/kpc^3. 
    
    ----output---
    log likelihood for the Plummer sphere model 
    """

	cdef double x0 = phase[0]
	cdef double y0 = phase[1]
	cdef double z0 = phase[2]
	cdef double vx0 = phase[3]
	cdef double vy0 = phase[4]
	cdef double vz0 = phase[5]

	cdef double s = 0.0
	cdef double t = 0.0
	cdef int i

	cdef int Nstar = star_array.shape[0]

	for i in range(Nstar):
		t = _phase_space_plummer(star_array[i,0],star_array[i,1],star_array[i,2],star_array[i,3],star_array[i,4],star_array[i,5],x0,y0,z0,vx0,vy0,vz0,Msh,v0,rs)
		if 1. - t > 0:
			s += log(1. - t)   - ((star_array[i,3])**2 + (star_array[i,4])**2 + (star_array[i,5])**2) / pow(v0,2)  - 3./2.*log(pi) - 3*log(v0) + log(ns) 
	s += -  ns * _volume_fact(R, v0, sqrt( vx0**2 + vy0**2 + vz0**2  ),Msh,rs) 
	return s 


