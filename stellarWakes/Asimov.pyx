###############################################################################
# Asimov.pyx
###############################################################################
#
# Performs the Asimov test for a given subhalo phase space
#
###############################################################################

# Import basic functions
import numpy as np
cimport numpy as np
cimport cython

# Requires cython_gsl
from cython_gsl cimport gsl_sf_dawson

# Import own integrator class
cimport integrator

cdef extern from "math.h":
	double log(double x) nogil
	double exp(double x) nogil
	double pow(double x, double y) nogil
	double sqrt(double x) nogil
	double asinh(double x) nogil
	double atan(double x) nogil  
	double cos(double x) nogil
	double sin(double x) nogil
cdef extern from "complex.h":
	complex csqrt(double x) nogil
	complex clog(complex x) nogil   

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
cdef double _Lfac_integrand(double *val, double * params) nogil: 
	""" Returns the integrand of equation (S8), after integrating out tilde{r}
    
    ---arguments---
    val[0]: tilde{v} in [0,infinity]
    val[1]: phi_v in [0,2 Pi]
    val[2]: theta_x in [-1,1]
    val[3]: theta_v in [-1,1]
    
    params[0]: epsilon_v = v0/vsh
    params[1]: Halo Plummer radius r_s in kpc
    params[2]: radius R of ROI in kpc

    ----output---
    value of integrand
    """

	cdef double eps = params[0]
	cdef double rs = params[1]
	cdef double R = params[2]

	cdef double v = val[0]
	cdef double phix = val[1]
	cdef double thetax = val[2]
	cdef double thetav = val[3]
    
	cdef double xvp,vx,vvp,d
	xvp = (eps*v*(cos(thetav)*cos(thetax)+sin(thetav)*sin(thetax)*cos(phix))-cos(thetax))/(sqrt(1+eps**2*v**2-2*eps*v*cos(thetav))) 
	vx = cos(thetav)*cos(thetax)+sin(thetav)*sin(thetax)*cos(phix)
	vvp = (eps*v-cos(thetav))/sqrt(1+eps**2*v**2-2*eps*v*cos(thetav))  
	d = sqrt( R**2 + rs**2 )  

	return <double> (sin(thetav)*sin(thetax)*v**4*exp(-v**2)/(1+eps**2*v**2-2*eps*v*cos(thetav))*(csqrt(-1)*pi*rs*(d-R*xvp)*(-vx+vvp*xvp)*(vx+vvp*xvp-4.*vx*xvp**2+2.*vvp*xvp**3)-xvp*(rs**2*(vx-vvp*xvp)*csqrt(-1+xvp**2)*(vx-3.*vvp*xvp+2.*vx*xvp**2)+rs*(vx-vvp*xvp)*(vx-3.*vvp*xvp+2.*vx*xvp**2)*(R*xvp*csqrt(-1+xvp**2)-csqrt(d**2*(-1+xvp**2)))-R*xvp*(-1+xvp**2)*(R*(-2.*vvp*vx+(vvp**2+vx**2)*xvp)*csqrt(-1+xvp**2)+(vvp**2+vx**2-2.*vvp*vx*xvp)*csqrt(d**2*(-1+xvp**2))))+rs*vx**2*(-1+xvp**2)**2*(-R*xvp*csqrt(-1+xvp**2)+csqrt(d**2*(-1+xvp**2)))*atan(R/rs)-rs*(d-R*xvp)*(-vx+vvp*xvp)*(vx+vvp*xvp-4.*vx*xvp**2+2.*vvp*xvp**3)*(clog(-d+R*xvp)+clog(-xvp-csqrt(-1+xvp**2))-clog(R-d*xvp-rs*csqrt(-1+xvp**2))))/(xvp**2*(-d+R*xvp)*csqrt(-1+xvp**2)**5))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double _Lfac(double eps, double Msh,double rs,double R) nogil:
	""" Returns the numerical integral defined in (S8)
    
    ---arguments---
    eps : epsilon_v = v0/vsh
    Msh : Halo mass in M_sun
    rs : Halo Plummer radius in kpc
    R : radius of ROI in kpc
    
    ----output---
    value of integral
    """

	cdef double result
	cdef integrator.Integral integr 
	cdef double epsilon = 1e-2 # omit edges of phase space
	integr.D = 4  # number of integrals
	integr.Nparams = 3 # number of fixed parameters
	integr.params = [eps,rs,R] # values of parameters
	integr.bound_low = [epsilon,epsilon,epsilon,epsilon] # lower integration bounds
	integr.bound_high = [5,2*pi-epsilon,pi-epsilon,pi-epsilon] # upper integration bounds
	integr.rel_err = [1e-1,1e-1,1e-1,1e-1] # relative target error of each integral
	integr.abs_err = [1e-2,1e-2,1e-2,1e-2] # absolute target error of each integral
	integr.fct = &_Lfac_integrand # pointer to integrand (must be "double (*)(double *, double *) nogil" )
	result = integrator.Integrate(integr)
	return result/8./pi          
 

#########################################################
# External functions									#
#########################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double TS_Asimov(double R,double ns,double v0,double Msh,double [::1] phase,double rs):
	""" Returns the likelihood factor for the Asimov test statistic
    
    ---arguments---
    R : radius of ROI in kpc
    ns : number density (normalization) model parameter for the phase-space distribution, in units of stars/kpc^3. 
    v0 : background velocity dispersion in km/s
    Msh : Halo mass in M_sun
    phase : position and velocity of the DM subhalo, of the form [x_Halo,y_Halo,z_Halo,v_Halo_x, v_Halo_y, v_Halo_z], with the same units as star_array
    rs : Halo Plummer radius in kpc
    
    ----output---
    Asimov value
    """

	cdef double vsh = sqrt(phase[3]**2 + phase[4]**2 + phase[5]**2)
	cdef double eps = v0/vsh
	cdef double Lfactor = np.real(_Lfac(eps,Msh,rs,R))
	cdef double result = - 64.*pi**(1/2.)*ns * GMr_kpc**2 * (Msh/1e12)**2 / (v0**2 * vsh**2 ) * Lfactor
	return result
