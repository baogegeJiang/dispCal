# dispCal

This is a python package based on cython to calculate the dispersion curves and their kernels with respect to the layers velocities, density and thickness. It is modified from Herrmann's seismic computation programs. To reduce the calculation cost, I use the local gradient to estimate the group velocity and sensitive kernels. Using the cyhon to speed up, it  have similar efficiency with the original fortran ones.

you can install the program via

```
pip install dispCal
```

I give some examples:

```
from dispCal.disp import calDisp
import numpy as np
thickness = np.array([10, 22, 12., 0])
vs = np.array([3, 3.5, 4, 4.3])
vp = vs * 1.7
rho = vp /3
periods = np.arange(1,30,1).astype(np.float64)#better from short periods to long ones


'''
    calDisp(thickness, vp, vs, rho, periods,dc0=0.005,domega=0.0001,wave='rayleigh', mode=1, velocity='phase', flat_earth=True,ar=6370.0,parameter='vp',smoothN=1)
    velocity: phase group kernel kernelGroup
    wave: love rayleigh
    parameter: vp vs rho thickness
    domega: for cal group and delta_omega=omega*domega
    dc0: is for search phase velocity and cal sensitive kernel and group velocity. For sensitive kernel dcUsed = dc0/10;for group velocity, dcUsed=dc0/100
    domega: for group velocitu and its kernel; the real step is given by domega*omega0;for calculating group velocity's' kernels, domegaUsed =domega/100,dcUsed=dc0/5
    mode: from 1
    ar : the radius for flatting
    smoothN: a new feature under construction
'''
#phase velocity
velocities = calDisp(thickness, vp, vs, rho, periods,wave='love', mode=1, velocity='phase', flat_earth=True,ar=6370,dc0=0.005)
#group velocity
velocities = calDisp(thickness, vp, vs, rho, periods,wave='love', mode=1, velocity='group', flat_earth=True,ar=6370,dc0=0.005)
#phase velocity kernel
KS = calDisp(thickness, vp, vs, rho, periods,wave='rayleigh', mode=1, velocity='kernel', flat_earth=True,ar=6370,dc0=0.005,domega=0.0001,parameter='vs')
KP = calDisp(thickness, vp, vs, rho, periods,wave='rayleigh', mode=1, velocity='kernel', flat_earth=True,ar=6370,dc0=0.005,domega=0.0001,parameter='vp')
#group velocity kernel
KSG = calDisp(thickness, vp, vs, rho, periods,wave='rayleigh', mode=1, velocity='kernelGroup', flat_earth=True,ar=6370,dc0=0.005,domega=0.0001,parameter='vs')
KPG = calDisp(thickness, vp, vs, rho, periods,wave='rayleigh', mode=1, velocity='kernelGroup', flat_earth=True,ar=6370,dc0=0.005,domega=0.0001,parameter='vp')
```

One need to notice that the kernel is just like  K = dc/dm. The m are the paramters of the model. If you want to plot the kernel, it's better to divide it by the thickness (if the parameter is not the thickness it self).

---

# cite

[![DOI](https://zenodo.org/badge/496537113.svg)](https://zenodo.org/badge/latestdoi/496537113)

## References

the original fortran is from Herrmann:

> Herrmann, R. B. (2013) Computer programs in seismology: An evolving tool for instruction and research, Seism. Res. Lettr. 84, 1081-1088, doi:10.1785/0220110096
