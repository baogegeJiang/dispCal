from dispCal.disp import calDisp
import numpy as np
thickness = np.array([1.,1.,1.,1.])
periods=np.arange(1,10.)/10
vp = np.array([1.5,3.,4.,5.])
vs = vp/1.71
vs[0]=0
rho = vp*0+1
c0=calDisp(thickness,vp,vs,rho,periods)
res0 = calDisp(thickness,vp,vs,rho,periods,velocity='kernel',parameter='vs')
print(res0[:1])
thickness = np.array([1/3,1/3,1/3,1.,1.,1.])
periods=np.arange(1,10.)/10
vp = np.array([1.5,1.5,1.5,3.,4.,5.])
vs = vp/1.71
vs[:3]=0
rho = vp*0+1
c1=(calDisp(thickness,vp,vs,rho,periods))
res = calDisp(thickness,vp,vs,rho,periods,velocity='kernel',parameter='vs')
print(res[:4])
print(c1-c0)