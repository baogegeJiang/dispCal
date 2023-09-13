import cyDisp as cyDisp
import numpy as np

def calDisp(thickness, vp, vs, rho, periods,dc0=0.0025,domega=0.0001,wave='rayleigh', mode=1, velocity='phase', flat_earth=True,ar=6370.0,parameter='vp',smoothN=1,Qp=[],Qs=[],fRef=1.0):
    '''
    velocity: phase group kernel kernelGroup
    wave: love rayleigh
    parameter: vp vs rho thickness
    domega: for cal group and delta_omega=omega*domega
    dc0: is for search phase velocity and cal sensitive kernel
    mode: from 1
    '''
    if (vs<=0.01).sum()<=1:
        if len(Qp)==0:
            Qp=np.zeros_like(vp)
        if len(Qs)==0:
            Qs=np.zeros_like(vp)
        if velocity == 'phase':
            return np.array(cyDisp.calDisp(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),Qp,Qs,fRef,dc0=dc0,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar))
        if velocity == 'group':
            return np.array(cyDisp.group(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),Qp,Qs,fRef,dc0=dc0,domega=domega,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar))
        if velocity =='kernel':
            return np.array(cyDisp.kernel(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),Qp,Qs,fRef,dc0=dc0,domega=domega,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar,isP=(parameter=='vp'),isS=(parameter=='vs'),isRho=(parameter=='rho'),isD=(parameter=='thickness')))
        if velocity =='kernelGroup':
            return np.array(cyDisp.kernelGroup(thickness.astype(np.float64), vp.astype(np.float64), vs.astype(np.float64),rho.astype(np.float64),periods.astype(np.float64),Qp,Qs,fRef,dc0=dc0,domega=domega,isR=(wave =='rayleigh'),isFlat=flat_earth,mode=mode,ar=ar,isP=(parameter=='vp'),isS=(parameter=='vs'),isRho=(parameter=='rho'),isD=(parameter=='thickness'),smoothN=smoothN))
    else:
        waterCount = (vs<=0.01).sum()
        thicknessN = thickness[waterCount-1:].copy()
        thicknessN[0] = thickness[:waterCount].sum()
        res = calDisp(thicknessN,vp[waterCount-1:],vs[waterCount-1:],rho[waterCount-1:],periods,dc0,domega,wave,mode,velocity,flat_earth,ar,parameter,smoothN)
        if 'kernel' in velocity:
            return np.concatenate([res[:1]*thickness[:waterCount].reshape([waterCount,1])/thicknessN[0]]+[res[1:]],axis=0)
        else:
            return res

def toQa(vp,vs,Qk,Qmu):
    return 1/( \
        (1-4/3*vs**2/vp**2)/Qk+4/3*(vs**2/vp**2)/Qmu\
        )
if __name__ =='__main__':
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