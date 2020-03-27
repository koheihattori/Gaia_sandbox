#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os, os.path #
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import math
import numpy
import scipy
import scipy.interpolate

import matplotlib
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator

import pandas as pd
import datetime
#import time

#PyGaia
from pygaia.astrometry.coordinates import CoordinateTransformation, Transformations
from pygaia.astrometry.vectorastrometry import cartesianToSpherical, astrometryToPhaseSpace,\
    phaseSpaceToAstrometry
from pygaia.errors.astrometric import properMotionErrorSkyAvg, properMotionMinError, properMotionMaxError, parallaxErrorSkyAvg

#orbit
import agama

pyplot.rcParams['ps.useafm'] = True
pyplot.rcParams['pdf.use14corefonts'] = True
pyplot.rcParams['text.usetex'] = True

# unit (just for normalization)
_R0=8.0  # 8 kpc
_V0=220. # 220 km/s
"""
#position and velocity of the Sun
_xGC_sun_kpc = -8.0  #you can set to (-_R0) if you want
_yGC_sun_kpc =  0.
_zGC_sun_kpc =  0.
_vxGC_sun    =  11.1
_vyGC_sun    =  232.24 #you can set to (_V0) if you want
_vzGC_sun    =  7.25

#MWPotential2014-----
agama.setUnits( mass=1., length=_R0, velocity=_V0)
p_bulge = dict(type='Spheroid', densityNorm=2.70033e9, gamma=1.8, beta=1.8, scaleRadius=8./_R0, outerCutoffRadius=1.9/_R0)
p_disk  = dict(type='MiyamotoNagai', mass=6.819386e10, scaleradius=3./_R0, scaleheight=0.28/_R0)
p_halo  = dict(type='Spheroid', densityNorm=4.34527e9, axisRatioZ=1.0, gamma=1.0, beta=3.0, scaleRadius=2.)
#--------------------
# Agama potential
potparams_halo_local = p_halo
potparams_disc_local = p_disk
potparams_bulge_local = p_bulge

densityBulge_local   = agama.Density(**potparams_bulge_local)
potentialBulge_local = agama.Potential(type='Multipole', lmax=6, mmax=0, density=densityBulge_local, gridsizer=50)

potentialDisc_local = agama.Potential(**potparams_disc_local)

densityHalo_local   = agama.Density(**potparams_halo_local)
potentialHalo_local = agama.Potential(type='Multipole', lmax=6, mmax=0, density=densityHalo_local, gridsizer=50)
potentialHalo_local = agama.Potential(**potparams_halo_local)

potentialTotal_local = agama.Potential( potentialBulge_local, potentialDisc_local, potentialHalo_local )

c_pot = potentialTotal_local

print(p_halo)
# create ActionFinder
actf= agama.ActionFinder(potentialTotal_local, interp=False)
"""
#ddd
_brown_id = 99
_inttime_back_Myr = 5000.


def v_circ_at_Rkpc(R):
    list_radii = numpy.array([R/_R0,])
    list_vcirc      = numpy.sqrt(-potentialTotal_local.force     (numpy.vstack((list_radii, list_radii*0, list_radii*0)).T)[:,0] * list_radii)*_V0
    return list_vcirc[0]

def cost_function_Zej(x_,y_,z_,vx_,vy_,vz_):
    
    #normalize position
    #normalize flip velocity
    vx_ = -vx_
    vy_ = -vy_
    vz_ = -vz_
    
    inttime_back_Myr = _inttime_back_Myr
    #numpy.sqrt(x_**2 + y_**2 + z_**2)*_R0_norm*2.0/(numpy.sqrt(vx_**2 + vy_**2 + vz_**2)*_V0_norm/1000.)
    inttime_back_    = inttime_back_Myr*(numpy.pi/(102.396349*(240./_V0)*(_R0/8.))) #Roughly: 102Myr = 3.14 if _R0_norm=8 and _V0_norm=240
    times_c, c_orb_car = agama.orbit(ic=[x_,y_,z_,vx_,vy_,vz_], potential=c_pot, time=inttime_back_, trajsize=5000)
    
    
    #xGC_kpc  = _R0 * c_orb_car[:,0]
    #yGC_kpc  = _R0 * c_orb_car[:,1]
    zGC_kpc_tmp  = _R0 * c_orb_car[:,2]
    #vxGC_kms = _V0 * c_orb_car[:,3]
    #vyGC_kms = _V0 * c_orb_car[:,4]
    #vzGC_kms = _V0 * c_orb_car[:,5]
    
    # most recent ejection
    flag_dc = 0
    i_dc = 0
    time_dc = 0.
    z_i      = zGC_kpc_tmp[0]
    z_iplus1 = zGC_kpc_tmp[1]
    for i in range(len(zGC_kpc_tmp)-1):
        z_i      = zGC_kpc_tmp[i]
        z_iplus1 = zGC_kpc_tmp[i+1]
        i_dc = i
        time_dc = times_c[i]
        if (z_i*z_iplus1<0.):
            flag_dc = 1
            break

    print(times_c[i_dc  ], z_i)
    print(times_c[i_dc+1], z_iplus1)
    #assert((z_i*z_iplus1<=0.) or (z_i>10.))
    inttime_back_ = 1.05*times_c[i_dc+1]

    # backward orbit until the most recent disc crossing
    times_c, c_orb_car = agama.orbit(ic=[x_,y_,z_,vx_,vy_,vz_], potential=c_pot, time=inttime_back_, trajsize=3000)

    xGC_kpc  = _R0 * c_orb_car[:,0]
    yGC_kpc  = _R0 * c_orb_car[:,1]
    zGC_kpc  = _R0 * c_orb_car[:,2]
    vxGC_kms = _V0 * c_orb_car[:,3]
    vyGC_kms = _V0 * c_orb_car[:,4]
    vzGC_kms = _V0 * c_orb_car[:,5]

    xGC_kpc_func  = scipy.interpolate.interp1d(times_c,xGC_kpc,kind='cubic')
    yGC_kpc_func  = scipy.interpolate.interp1d(times_c,yGC_kpc,kind='cubic')
    zGC_kpc_func  = scipy.interpolate.interp1d(times_c,zGC_kpc,kind='cubic')
    vxGC_kpc_func = scipy.interpolate.interp1d(times_c,vxGC_kms,kind='cubic')
    vyGC_kpc_func = scipy.interpolate.interp1d(times_c,vyGC_kms,kind='cubic')
    vzGC_kpc_func = scipy.interpolate.interp1d(times_c,vzGC_kms,kind='cubic')
    
    # distance from ejection location (x_ej,y_ej)
    distance2_from_disc_kpc_func = lambda x : ( zGC_kpc_func(x)**2 )
    
    result2 = scipy.optimize.minimize_scalar(distance2_from_disc_kpc_func, bounds = (times_c[0], times_c[-1]), method = 'bounded')
    
    #print ('%lf %lf %lf' % (muellstar_test, mubee_test, rGC_kpc_func_2(result2.x)))
    
    x_ej  = xGC_kpc_func(result2.x)
    y_ej  = yGC_kpc_func(result2.x)
    z_ej  = zGC_kpc_func(result2.x)
    vx_ej = -vxGC_kpc_func(result2.x)
    vy_ej = -vyGC_kpc_func(result2.x)
    vz_ej = -vzGC_kpc_func(result2.x)
    
    flightTime_Myr = (result2.x) / (numpy.pi/(102.396349*(240./_V0)*(_R0/8.)))

    return x_ej, y_ej, z_ej, vx_ej, vy_ej, vz_ej, flightTime_Myr


def calculate_J_agama(xGC_norm, yGC_norm, zGC_norm, vxGC_norm, vyGC_norm, vzGC_norm):
    # 6D info
    xv6d = numpy.array( [xGC_norm, yGC_norm, zGC_norm, vxGC_norm, vyGC_norm, vzGC_norm] ).T
    
    actions = actf(xv6d)
    #print('#actions:')
    #print(actions)
    #J_r   = actions[:,0]  #radial action
    #J_z   = actions[:,1]  #vertcical action
    #J_phi = actions[:,2]  #azimutahal action (Lz = R*v_phi)
    #print(actions[:,0],actions[1],actions[2])
    
    return actions

#calculate orbital parameters
def calculate_orbital_parameters(xGC_norm, yGC_norm, zGC_norm, vxGC_norm, vyGC_norm, vzGC_norm, inttime, numsteps):
    # agama convention
    x_agama  =   xGC_norm
    y_agama  =   yGC_norm
    z_agama  =   zGC_norm
    vx_agama =   vxGC_norm
    vy_agama =   vyGC_norm
    vz_agama =   vzGC_norm
    
    R_agama   =  numpy.sqrt(x_agama**2 + y_agama**2)
    vR_agama  =  ( x_agama*vx_agama + y_agama*vy_agama )/R_agama
    vT_agama  = -( x_agama*vy_agama - y_agama*vx_agama )/R_agama
    phi_agama =  numpy.arctan2(y_agama,x_agama)
    

    inttime=1.
    numsteps=300
    times = numpy.linspace(0, inttime, numsteps)
    times_c, c_orb_car = agama.orbit(ic=[x_agama,y_agama,z_agama,vx_agama,vy_agama,vz_agama], potential=potentialTotal_local, time=inttime, trajsize=numsteps)
    #print(c_orb_car[:,0])
    
    
    x = c_orb_car[:,0]
    y = c_orb_car[:,1]
    z = c_orb_car[:,2]
    vx= c_orb_car[:,3]
    vy= c_orb_car[:,4]
    vz= c_orb_car[:,5]
    R = numpy.sqrt(x**2 + y**2)
    r = numpy.sqrt(x**2 + y**2 + z**2)
    
    rmin = numpy.min(r)
    rmax = numpy.max(r)
    zmax = numpy.max(numpy.fabs(z))
    ecc  = (rmax-rmin)/(rmax+rmin)
    return rmin, rmax, zmax, ecc

#calculate orbital parameters
def calculate_orbital_shape(xGC_norm, yGC_norm, zGC_norm, vxGC_norm, vyGC_norm, vzGC_norm, inttime, numsteps):
    # agama convention
    x_agama  =   xGC_norm
    y_agama  =   yGC_norm
    z_agama  =   zGC_norm
    vx_agama =   vxGC_norm
    vy_agama =   vyGC_norm
    vz_agama =   vzGC_norm
    
    R_agama   =  numpy.sqrt(x_agama**2 + y_agama**2)
    vR_agama  =  ( x_agama*vx_agama + y_agama*vy_agama )/R_agama
    vT_agama  = -( x_agama*vy_agama - y_agama*vx_agama )/R_agama
    phi_agama =  numpy.arctan2(y_agama,x_agama)

    inttime_back_Myr = _inttime_back_Myr
    #numpy.sqrt(x_**2 + y_**2 + z_**2)*_R0_norm*2.0/(numpy.sqrt(vx_**2 + vy_**2 + vz_**2)*_V0_norm/1000.)
    inttime_back_    = inttime_back_Myr*(numpy.pi/(102.396349*(240./_V0)*(_R0/8.))) #Roughly: 102Myr = 3.14 if _R0_norm=8 and _V0_norm=240
    numsteps=3000
    times = numpy.linspace(0, inttime, numsteps)
    times_c, c_orb_car = agama.orbit(ic=[x_agama,y_agama,z_agama,vx_agama,vy_agama,vz_agama], potential=potentialTotal_local, time=inttime, trajsize=numsteps)
    #print(c_orb_car[:,0])
    
    
    x = c_orb_car[:,0]
    y = c_orb_car[:,1]
    z = c_orb_car[:,2]
    vx= c_orb_car[:,3]
    vy= c_orb_car[:,4]
    vz= c_orb_car[:,5]
    R = numpy.sqrt(x**2 + y**2)
    r = numpy.sqrt(x**2 + y**2 + z**2)
    
    
    out_filename = 'x_y_z_vx_vy_vz_vR_vTHETA_vPHI__Orbit__AsObserved_LAMOSThvs%d.txt' % (LAMOSTHVSid)
    outfile      = open(out_filename,'a')
    outfile.write('# x y z vx vy vz vR vTHETA vPHI [all normalized by (x,v)=(1 kpc, 1 km/s)]\n')
    for i in range(len(x)):
        x_  =  x[i]*_R0
        y_  =  y[i]*_R0
        z_  =  z[i]*_R0
        vx_ = vx[i]*_V0
        vy_ = vy[i]*_V0
        vz_ = vz[i]*_V0
        
        R_, phi_, vR_, vTHETA_, vPHI_ = cartesian_2_cylindrical(x_,y_,z_,vx_,vy_,vz_)
        
        printline='%lf %lf %lf %lf %lf %lf %lf %lf %lf\n' % (x_, y_, z_, vx_, vy_, vz_, vR_, vTHETA_, vPHI_)
        outfile.write(printline)
    
    return None


#calculate orbital shape in the past given the current 6D info
def calculate_orbital_shape_PAST(xGC_norm, yGC_norm, zGC_norm, vxGC_norm, vyGC_norm, vzGC_norm, inttime, numsteps):
    # agama convention
    # already normalized position
    x_agama  =   xGC_norm
    y_agama  =   yGC_norm
    z_agama  =   zGC_norm
    # FLIP already normalized velocity
    vx_agama =  -vxGC_norm
    vy_agama =  -vyGC_norm
    vz_agama =  -vzGC_norm
    
    
    inttime_back_Myr = _inttime_back_Myr
    inttime_back_    = inttime_back_Myr*(numpy.pi/(102.396349*(240./_V0)*(_R0/8.))) #Roughly: 102Myr = 3.14 if _R0_norm=8 and _V0_norm=240
    
    inttime=inttime_back_
    numsteps=3000
    times = numpy.linspace(0, inttime, numsteps)
    times_c, c_orb_car = agama.orbit(ic=[x_agama,y_agama,z_agama,vx_agama,vy_agama,vz_agama], potential=potentialTotal_local, time=inttime, trajsize=numsteps)
    #print(c_orb_car[:,0])
    
    
    x = c_orb_car[:,0]
    y = c_orb_car[:,1]
    z = c_orb_car[:,2]
    vx= c_orb_car[:,3]
    vy= c_orb_car[:,4]
    vz= c_orb_car[:,5]
    R = numpy.sqrt(x**2 + y**2)
    r = numpy.sqrt(x**2 + y**2 + z**2)
    
    
    out_filename = 'timeMyr_x_y_z_vx_vy_vz_vR_vTHETA_vPHI__Orbit__AsObserved_LAMOSThvs%d.txt' % (LAMOSTHVSid)
    outfile      = open(out_filename,'a')
    outfile.write('#(orbit in the past)\n#time_Myr x y z vx vy vz vR vTHETA vPHI [all normalized by (x,v)=(1 kpc, 1 km/s)]\n')
    for i in range(len(x)):
        #flip the time to get the correct time
        Time_Myr = (-1.) * times_c[i] / (numpy.pi/(102.396349*(240./_V0)*(_R0/8.)))
        #position
        x_  =   x[i]*_R0
        y_  =   y[i]*_R0
        z_  =   z[i]*_R0
        #re-flip the velocity to get the velocity in the past
        vx_ = -vx[i]*_V0
        vy_ = -vy[i]*_V0
        vz_ = -vz[i]*_V0
        
        R_, phi_, vR_, vTHETA_, vPHI_ = cartesian_2_cylindrical(x_,y_,z_,vx_,vy_,vz_)
    
        printline='%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n' % (Time_Myr, x_, y_, z_, vx_, vy_, vz_, vR_, vTHETA_, vPHI_)
        outfile.write(printline)

    return None


#calculate orbits of LMC and target star in the past given the current 6D info
def calculate_star_and_LMC_orbits_PAST(xGC_norm, yGC_norm, zGC_norm, vxGC_norm, vyGC_norm, vzGC_norm):#, inttime, numsteps):
    # agama convention
    # already normalized position
    x_agama  =   xGC_norm
    y_agama  =   yGC_norm
    z_agama  =   zGC_norm
    # FLIP already normalized velocity
    vx_agama =  -vxGC_norm
    vy_agama =  -vyGC_norm
    vz_agama =  -vzGC_norm
    
    
    inttime_back_Myr = _inttime_back_Myr
    inttime_back_    = inttime_back_Myr*(numpy.pi/(102.396349*(240./_V0)*(_R0/8.))) #Roughly: 102Myr = 3.14 if _R0_norm=8 and _V0_norm=240
    
    inttime=inttime_back_
    numsteps=3000
    times = numpy.linspace(0, inttime, numsteps)
    times_c, c_orb_car = agama.orbit(ic=[x_agama,y_agama,z_agama,vx_agama,vy_agama,vz_agama], potential=potentialTotal_local, time=inttime, trajsize=numsteps)
    #print(c_orb_car[:,0])
    
    x = c_orb_car[:,0]
    y = c_orb_car[:,1]
    z = c_orb_car[:,2]
    vx= c_orb_car[:,3]
    vy= c_orb_car[:,4]
    vz= c_orb_car[:,5]
    R = numpy.sqrt(x**2 + y**2)
    r = numpy.sqrt(x**2 + y**2 + z**2)
    
    #LMC position in the MW-center coordinate system.
    xLMC_norm, yLMC_norm, zLMC_norm, vxLMC_norm, vyLMC_norm, vzLMC_norm = get_LMC_6D()
    # already normalized position
    xLMC_agama  =   xLMC_norm
    yLMC_agama  =   yLMC_norm
    zLMC_agama  =   zLMC_norm
    # FLIP already normalized velocity
    vxLMC_agama =  -vxLMC_norm
    vyLMC_agama =  -vyLMC_norm
    vzLMC_agama =  -vzLMC_norm
    
    #The same orbit integration as the target star
    times_cLMC, c_orb_carLMC = agama.orbit(ic=[xLMC_agama,yLMC_agama,zLMC_agama,vxLMC_agama,vyLMC_agama,vzLMC_agama], potential=potentialTotal_local, time=inttime, trajsize=numsteps)
    
    xLMC = c_orb_carLMC[:,0]
    yLMC = c_orb_carLMC[:,1]
    zLMC = c_orb_carLMC[:,2]
    vxLMC= c_orb_carLMC[:,3]
    vyLMC= c_orb_carLMC[:,4]
    vzLMC= c_orb_carLMC[:,5]

    t_dmin_Myr = 99999.
    dmin_kpc   = 99999.

    #out_filename = 'timeMyr_x_y_z_vx_vy_vz__timeMyr_xLMC_yLMC_zLMC_vxLMC_vyLMC_vzLMC__timeMyr_dx_dy_dz_dvx_dvy_dvz__AsObserved.txt'
    #outfile      = open(out_filename,'a')
    #outfile.write('#(orbit in the past)\n#[time_Myr x y z vx vy vz]  [time_Myr x y z vx vy vz]LMC  [time_Myr x y z vx vy vz]diff\n')
    for i in range(len(x)):
        #flip the time to get the correct time
        Time_Myr = (-1.) * times_c[i] / (numpy.pi/(102.396349*(240./_V0)*(_R0/8.)))
        #position
        x_  =   x[i]*_R0
        y_  =   y[i]*_R0
        z_  =   z[i]*_R0
        #re-flip the velocity to get the velocity in the past
        vx_ = -vx[i]*_V0
        vy_ = -vy[i]*_V0
        vz_ = -vz[i]*_V0
        
        #flip the time to get the correct time
        Time_Myr_LMC = (-1.) * times_c[i] / (numpy.pi/(102.396349*(240./_V0)*(_R0/8.)))
        #position
        xL_  =   xLMC[i]*_R0
        yL_  =   yLMC[i]*_R0
        zL_  =   zLMC[i]*_R0
        #re-flip the velocity to get the velocity in the past
        vxL_ = -vxLMC[i]*_V0
        vyL_ = -vyLMC[i]*_V0
        vzL_ = -vzLMC[i]*_V0
        
        #flip the time to get the correct time
        Time_Myr_LMC = (-1.) * times_c[i] / (numpy.pi/(102.396349*(240./_V0)*(_R0/8.)))
        #relative position
        dx_  =  x_ -  xL_
        dy_  =  y_ -  yL_
        dz_  =  z_ -  zL_
        dvx_ = vx_ - vxL_
        dvy_ = vy_ - vxL_
        dvz_ = vz_ - vxL_
        
        d_kpc  = numpy.sqrt( dx_**2 +  dy_**2 +  dz_**2)*_R0
        dv_kms = numpy.sqrt(dvx_**2 + dvy_**2 + dvz_**2)*_V0
        if (d_kpc<dmin_kpc):
            t_dmin_Myr = Time_Myr
        #print(t_dmin_Myr, d_kpc)
        
        """
        printline='%lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf   %lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf   %lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n' % (Time_Myr,      x_,  y_,  z_,  vx_,  vy_,  vz_,
                        Time_Myr_LMC,  xL_, yL_, zL_, vxL_, vyL_, vzL_,
                        Time_Myr,     dx_, dy_, dz_, dvx_, dvy_, dvz_ )
        """
        #outfile.write(printline)


    x_func  = scipy.interpolate.interp1d(times_c,x,kind='cubic')
    y_func  = scipy.interpolate.interp1d(times_c,y,kind='cubic')
    z_func  = scipy.interpolate.interp1d(times_c,z,kind='cubic')
    vx_func = scipy.interpolate.interp1d(times_c,vx,kind='cubic')
    vy_func = scipy.interpolate.interp1d(times_c,vy,kind='cubic')
    vz_func = scipy.interpolate.interp1d(times_c,vz,kind='cubic')

    xLMC_func  = scipy.interpolate.interp1d(times_c,xLMC,kind='cubic')
    yLMC_func  = scipy.interpolate.interp1d(times_c,yLMC,kind='cubic')
    zLMC_func  = scipy.interpolate.interp1d(times_c,zLMC,kind='cubic')
    vxLMC_func = scipy.interpolate.interp1d(times_c,vxLMC,kind='cubic')
    vyLMC_func = scipy.interpolate.interp1d(times_c,vyLMC,kind='cubic')
    vzLMC_func = scipy.interpolate.interp1d(times_c,vzLMC,kind='cubic')

    # distance from ejection location (x_ej,y_ej)
    distance2_from_LMC_kpc_func = lambda x : ( _R0**2 * ((x_func(x)-xLMC_func(x))**2 + (y_func(x)-yLMC_func(x))**2 + (z_func(x)-zLMC_func(x))**2) )
        
    result2 = scipy.optimize.minimize_scalar(distance2_from_LMC_kpc_func, bounds = (times_c[0], times_c[-1]), method = 'bounded')
        
        
    x_ej  =   x_func(result2.x)*_R0
    y_ej  =   y_func(result2.x)*_R0
    z_ej  =   z_func(result2.x)*_R0
    vx_ej = -vx_func(result2.x)*_V0
    vy_ej = -vy_func(result2.x)*_V0
    vz_ej = -vz_func(result2.x)*_V0
    
    xLMC_ej  =   xLMC_func(result2.x)*_R0
    yLMC_ej  =   yLMC_func(result2.x)*_R0
    zLMC_ej  =   zLMC_func(result2.x)*_R0
    vxLMC_ej = -vxLMC_func(result2.x)*_V0
    vyLMC_ej = -vyLMC_func(result2.x)*_V0
    vzLMC_ej = -vzLMC_func(result2.x)*_V0
    
    dx_ej  =  x_ej -  xLMC_ej
    dy_ej  =  y_ej -  yLMC_ej
    dz_ej  =  z_ej -  zLMC_ej
    dvx_ej = vx_ej - vxLMC_ej
    dvy_ej = vy_ej - vyLMC_ej
    dvz_ej = vz_ej - vzLMC_ej
    
    dmin_kpc = numpy.sqrt( distance2_from_LMC_kpc_func(result2.x) )
    dv_kms   = numpy.sqrt( dvx_ej**2 + dvy_ej**2 + dvz_ej**2 )
        
    Time_Myr_ej = (result2.x) / (numpy.pi/(102.396349*(240./_V0)*(_R0/8.)))
    

    out_filename = 'timeMyr_dmin_dv__timeMyr_x_y_z_vx_vy_vz__timeMyr_xLMC_yLMC_zLMC_vxLMC_vyLMC_vzLMC__timeMyr_dx_dy_dz_dvx_dvy_dvz__AsObserved_LAMOSThvs%d.txt' % (LAMOSTHVSid)
    outfile      = open(out_filename,'a')
    printline='%.3lf %.3lf %.3lf   %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf   %lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf   %lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n' % (Time_Myr_ej, dmin_kpc, dv_kms,     Time_Myr_ej, x_ej,  y_ej,  z_ej,  vx_ej,  vy_ej,  vz_ej,
    Time_Myr_ej,      xLMC_ej,  yLMC_ej,  zLMC_ej,  vxLMC_ej,  vyLMC_ej,  vzLMC_ej,
    Time_Myr_ej,     dx_, dy_, dz_, dvx_, dvy_, dvz_)
    outfile.write(printline)
    outfile.close()
    
    print('---')
    print(Time_Myr_ej, dmin_kpc, dv_kms)
    print('===')
    
    return Time_Myr_ej, dmin_kpc, dv_kms


def get_LMC_6D():#ddd
    DM_mag  =  18.50 + numpy.random.randn()*0.10
    RA_deg  =  78.77
    DEC_deg = -69.01
    HRV_kms = 262.2 + numpy.random.randn()*3.4
    pmra    = 1.850 + numpy.random.randn()*0.03
    pmdec   = 0.234 + numpy.random.randn()*0.03
    
    RA_radian  = deg2radian(RA_deg)
    DEC_radian = deg2radian(DEC_deg)
    
    ell_radian, bee_radian, muellstar_masyr, mubee_masyr = ICRS_to_GAL(RA_radian, DEC_radian, pmra, pmdec)
        
    xHelio_kpc, yHelio_kpc, zHelio_kpc, vxHelio, vyHelio, vzHelio = astrometric_2_cartesian(DM_mag, ell_radian, bee_radian, HRV_kms, muellstar_masyr, mubee_masyr)


    #normalize position
    xLMC_norm  =  (xHelio_kpc + _xGC_sun_kpc)/_R0
    yLMC_norm  =  (yHelio_kpc + _yGC_sun_kpc)/_R0
    zLMC_norm  =  (zHelio_kpc + _zGC_sun_kpc)/_R0
    #normalize  velocity
    vxLMC_norm = (vxHelio + _vxGC_sun)/_V0
    vyLMC_norm = (vyHelio + _vyGC_sun)/_V0
    vzLMC_norm = (vzHelio + _vzGC_sun)/_V0
    
    return xLMC_norm, yLMC_norm, zLMC_norm, vxLMC_norm, vyLMC_norm, vzLMC_norm



#conversion
def deg2radian(deg):
    return deg/180.*numpy.pi

#conversion
def radian2deg(radian):
    return radian/numpy.pi*180.


def GAL_to_ICRS(phi, theta, muphistar, mutheta):
    """
        phi       - The longitude-like angle of the position of the source (radians).
        theta     - The latitude-like angle of the position of the source (radians).
        muphistar - Value of the proper motion in the longitude-like angle, multiplied by cos(latitude).
        mutheta   - Value of the proper motion in the latitude-like angle.
        """
    # transformation ICRS to GAL
    ctGAL2ICRS =CoordinateTransformation(Transformations.GAL2ICRS)
    #
    ra,         dec = ctGAL2ICRS.transformSkyCoordinates(phi,theta)
    murastar, mudec = ctGAL2ICRS.transformProperMotions (phi,theta, muphistar, mutheta)
    
    return ra,dec,murastar,mudec


def ICRS_to_GAL(phi, theta, muphistar, mutheta):
    """
        phi       - The longitude-like angle of the position of the source (radians).
        theta     - The latitude-like angle of the position of the source (radians).
        muphistar - Value of the proper motion in the longitude-like angle, multiplied by cos(latitude).
        mutheta   - Value of the proper motion in the latitude-like angle.
        """
    # transformation ICRS to GAL
    ctICRS2GAL =CoordinateTransformation(Transformations.ICRS2GAL)
    #
    ell,         bee = ctICRS2GAL.transformSkyCoordinates(phi,theta)
    muellstar, mubee = ctICRS2GAL.transformProperMotions (phi,theta, muphistar, mutheta)
    
    return ell,bee,muellstar,mubee



def ProperMotionError__GAL_to_ICRS(phi, theta, sigMuPhiStar, sigMuTheta, rhoMuPhiMuTheta):
    """
        ----------
        phi             - The longitude-like angle of the position of the source (radians).
        theta           - The latitude-like angle of the position of the source (radians).
        sigMuPhiStar    - Standard error in the proper motion in the longitude-like direction (including cos(latitude) factor).
        sigMuTheta      - Standard error in the proper motion in the latitude-like direction.
        
        Keywords (optional)
        -------------------
        rhoMuPhiMuTheta - Correlation coefficient of the proper motion errors. Set to zero if this keyword is not provided.
        
        Retuns
        ------
        sigMuPhiRotStar    - The transformed standard error in the proper motion in the longitude direction
        (including cos(latitude) factor).
        sigMuThetaRot      - The transformed standard error in the proper motion in the longitude direction.
        rhoMuPhiMuThetaRot - The transformed correlation coefficient.
        """
    # transformation GAL to ICRS
    ctGAL2ICRS =CoordinateTransformation(Transformations.GAL2ICRS)
    #
    sigMuPhiRotStar,sigMuThetaRot,rhoMuPhiMuThetaRot = ctGAL2ICRS.transformProperMotionErrors (phi,theta, sigMuPhiStar, sigMuTheta, rhoMuPhiMuTheta)
    
    return sigMuPhiRotStar,sigMuThetaRot,rhoMuPhiMuThetaRot


def ProperMotionError__ICRS_to_GAL(phi, theta, sigMuPhiStar, sigMuTheta, rhoMuPhiMuTheta):
    """
        ----------
        phi             - The longitude-like angle of the position of the source (radians).
        theta           - The latitude-like angle of the position of the source (radians).
        sigMuPhiStar    - Standard error in the proper motion in the longitude-like direction (including cos(latitude) factor).
        sigMuTheta      - Standard error in the proper motion in the latitude-like direction.
        
        Keywords (optional)
        -------------------
        rhoMuPhiMuTheta - Correlation coefficient of the proper motion errors. Set to zero if this keyword is not provided.
        
        Retuns
        ------
        sigMuPhiRotStar    - The transformed standard error in the proper motion in the longitude direction
        (including cos(latitude) factor).
        sigMuThetaRot      - The transformed standard error in the proper motion in the longitude direction.
        rhoMuPhiMuThetaRot - The transformed correlation coefficient.
        """
    # transformation ICRS to GAL
    ctICRS2GAL =CoordinateTransformation(Transformations.ICRS2GAL)
    #
    sigMuPhiRotStar,sigMuThetaRot,rhoMuPhiMuThetaRot = ctICRS2GAL.transformProperMotionErrors (phi,theta, sigMuPhiStar, sigMuTheta, rhoMuPhiMuTheta)
    
    return sigMuPhiRotStar,sigMuThetaRot,rhoMuPhiMuThetaRot



#conversion
def parallax_mas_2_DM(parallax_mas):
    DM = 5.*( numpy.log10(1./parallax_mas) + 2. )
    return DM

#conversion
def DM_2_parallax_mas(DM):
    parallax_mas = numpy.power(10., 2.-DM/5.)
    return parallax_mas

#conversion
def cartesian_2_astrometric(x_pc,y_pc,z_pc,vx,vy,vz):
    ell_radian, bee_radian, parallax_mas, muellstar, mubee, HRV = phaseSpaceToAstrometry(x_pc, y_pc, z_pc, vx, vy, vz)
    DM = 5.*( numpy.log10(1./parallax_mas) + 2. )
    return DM, ell_radian, bee_radian, HRV, muellstar, mubee


#conversion
def astrometric_2_cartesian(DM, ell_radian, bee_radian, HRV, muellstar, mubee):
    parallax_mas = numpy.power(10., 2.-DM/5.)
    xHelio_pc, yHelio_pc, zHelio_pc, vxHelio, vyHelio, vzHelio = astrometryToPhaseSpace(ell_radian, bee_radian, parallax_mas, muellstar, mubee, HRV)
    xHelio_kpc = xHelio_pc/1000.
    yHelio_kpc = yHelio_pc/1000.
    zHelio_kpc = zHelio_pc/1000.
    
    return xHelio_kpc, yHelio_kpc, zHelio_kpc, vxHelio, vyHelio, vzHelio

#conversion
def cartesian_2_cylindrical(x,y,z,vx,vy,vz):
    R =numpy.sqrt(x**2 + y**2)
    r =numpy.sqrt(x**2 + y**2 + z**2)
    
    phi=numpy.arctan2(y,x)
    vR     = (x*vx + y*vy)/R
    vTHETA = (-z*vR+ R*vz)/r
    vPHI   = (x*vy - y*vx)/R
    
    return R, phi, vR, vTHETA, vPHI


def count_csv():
    df = pd.read_csv(Gaia_csv_file)
    parallax              = df.iloc[:]['parallax']
    return len(parallax)

def load_from_csv(id):
    #
    df_all = pd.read_csv(Gaia_csv_file)
    
    df = df_all.loc[df_all['als_1']==id]
    #target
    parallax              = df['parallax'].values[0]
    ra                    = df['ra'].values[0]
    dec                   = df['dec'].values[0]
    #
    # ALMA catalog (Reed et al.2003)
    #radial_velocity       = df['rv'].values[0]
    #radial_velocity_error = df['e_rv'].values[0]

    # SIMBAD
    radial_velocity       = df['rvz_radvel'].values[0]
    radial_velocity_error = df['rvz_err'].values[0]
    
    # GaiaRV
    #
    #radial_velocity       = df['radial_velocity'].values[0]
    #radial_velocity_error = df['radial_velocity_error'].values[0]
    
    pmra                  = df['pmra'].values[0]
    pmdec                 = df['pmdec'].values[0]
    parallax_error        = df['parallax_error'].values[0]
    pmra_error            = df['pmra_error'].values[0]
    pmdec_error           = df['pmdec_error'].values[0]
    parallax_pmra_corr    = df['parallax_pmra_corr'].values[0]
    parallax_pmdec_corr   = df['parallax_pmdec_corr'].values[0]
    pmra_pmdec_corr       = df['pmra_pmdec_corr'].values[0]
    phot_g_mean_mag       = df['phot_g_mean_mag'].values[0]
    
    source_id             = df['source_id'].values[0]
    
    
    return parallax, ra, dec, \
            radial_velocity, pmra, pmdec, \
            parallax_error, radial_velocity_error, \
            pmra_error, pmdec_error, \
            parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr, \
            phot_g_mean_mag, source_id

