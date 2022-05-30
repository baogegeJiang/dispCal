import cyDisp as cyDisp
import numpy as np

def calDisp(thickness, vp, vs, rho, periods,dc0=0.005,domega=0.0001,wave='rayleigh', mode=1, velocity='phase', flat_earth=True,ar=6370.0,parameter='vp',smoothN=1):
    '''
    velocity: phase group kernel kernelGroup
    wave: love rayleigh
    parameter: vp vs rho thickness
    domega: for cal group and delta_omega=omega*domega
    dc0: is for search phase velocity and cal sensitive kernel
    mode: from 1
    '''
    if velocity == 'phase':
        return np.array(cyDisp.calDisp(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),dc0=dc0,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar))
    if velocity == 'group':
        return np.array(cyDisp.group(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),dc0=dc0,domega=domega,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar))
    if velocity =='kernel':
        return np.array(cyDisp.kernel(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),dc0=dc0,domega=domega,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar,isP=(parameter=='vp'),isS=(parameter=='vs'),isRho=(parameter=='rho'),isD=(parameter=='thickness')))
    if velocity =='kernelGroup':
        return np.array(cyDisp.kernelGroup(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),dc0=dc0,domega=domega,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar,isP=(parameter=='vp'),isS=(parameter=='vs'),isRho=(parameter=='rho'),isD=(parameter=='thickness'),smoothN=smoothN))