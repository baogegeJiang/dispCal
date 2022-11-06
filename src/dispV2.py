# cython: initializedcheck=False
# cython: cdivision=False
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cdef extern from "math.h":
    cpdef double sin(double x)
    cpdef double cos(double x)
    cpdef double log(double x)
    #cpdef double sign(double x )
    cpdef double pow(double x,double y)
    cpdef double exp(double x)
    cpdef double sqrt(double x)

# cython


cdef double sign(double x):
    if x>0.0:
        return 1
    else:
        return -1
cdef double del1st
cdef int jsol
cdef int index
cdef double a[500],b[500],d[500],rho[200],X[20],Y[20]
#cdef double xka,xkb,wvnop,wvnom,rb,ra,t,gammk,gamm1,rho1,e1,e2,xmu,q,y,z,cosq,sinq,fac,e01,e20,xnor,ynor,wvno2,dpth,p,znul,plmn
#cdef int   i0
cdef double E0,E1,E2,E3,E4
cdef double E01,E11,E21,E31,E41
cdef double ca01,ca02,ca03,ca04,ca00
cdef double ca11,ca12,ca13,ca14,ca10
cdef double ca21,ca22,ca23,ca24,ca20
cdef double ca31,ca32,ca33,ca34,ca30
cdef double ca41,ca42,ca43,ca44,ca40
cdef double w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
cdef double max( double[:] x,int N):
    cdef double tmp = -999999999
    #iTmp=0
    for i in xrange(N):
        if x[i]>=tmp:
            tmp = x[i]
            #iTmp = i
    return tmp
cdef double min( double[:] x,int i0,int N):
    cdef double tmp = 9999999999
    #iTmp=0
    for i in xrange(i0,N):
        if x[i]<tmp:
            tmp < x[i]
            #iTmp = i
    return tmp
cdef int amin( double[:] x,int N):
    cdef double tmp = -999999999
    cdef int iTmp=0
    for i in xrange(N):
        if x[i]<=tmp:
            tmp = x[i]
            iTmp = i
    return iTmp
cpdef double[:] calDisp(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.005,bint isFlat=False, double ar=6370):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    #C = np.array(TN, dtype=np.float64)
    cdef double[:] c
    c=np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    return c

cpdef double[:] group__(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.001,bint isFlat=False, double ar=6370,double domega = 0.001,double dh=0.01):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    cdef int i,j
    cdef double twopi=2*3.141592653589793
    cdef double omega,wvno,det0,dd,DOmega,DK,dOmega,dK
    cdef double[:] vg ,c
    global d,a,b,rho
    vg =np.zeros(TN, dtype=np.float64) 
    c=np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    dc0 = dc0/100
    for i in xrange(TN):
        DOmega =twopi/T[i]*(domega)
        omega = twopi/T[i]-DOmega
        wvno =  twopi/T[i]/c[i]
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]+DOmega
        wvno  =  twopi/T[i]/c[i]
        if isR:
            dOmega = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        else:
            dOmega = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        #print(dltarR(d,a,b,rho,wvno,omega,LN),det0,DOmega,dOmegaL[i])
        DK =twopi/T[i]/c[i]*(domega)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/c[i]-DK
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/c[i]+DK
        if isR:
            dK = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
        else:
            dK = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
        vg[i]=-dK/dOmega
    return vg
cpdef double[:] group(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.001,bint isFlat=False, double ar=6370,double domega = 0.001,double dh=0.01):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    cdef int i,j
    cdef double twopi=2*3.141592653589793
    cdef double omega,wvno,det0,dd,DOmega,dOmega,dC
    cdef double[:] vg ,c
    global d,a,b,rho
    vg =np.zeros(TN, dtype=np.float64) 
    c=np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    dc0 = dc0/100
    for i in xrange(TN):
        if c[i]<=0:
            continue
        DOmega =twopi/T[i]*(domega)
        omega = twopi/T[i]-DOmega
        wvno =  omega/c[i]
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]+DOmega
        wvno  =  omega/c[i]
        if isR:
            dOmega = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        else:
            dOmega = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        #print(dltarR(d,a,b,rho,wvno,omega,LN),det0,DOmega)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i]-dc0)
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i]+dc0)
        if isR:
            dC = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/dc0/2
        else:
            dC = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/dc0/2
        vg[i]=c[i]/(1+dOmega/dC/c[i]*twopi/T[i])
    return vg
cpdef double[:,:] kernel(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.001,bint isFlat=False, double ar=6370,double domega = 0.001,double dh=0.001,bint isP=False,bint isS=False,bint isD=False,bint isRho=False):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    cdef int i,j
    cdef double twopi=2*3.141592653589793
    cdef double omega,wvno,ddc
    cdef double[:,:] K
    cdef double[:]c,det0L,ddcL,P,p
    #cdef int aoundK
    #cdef double aroundKD,aroundND
    c=np.zeros(TN, dtype=np.float64)
    K=np.zeros([LN,TN], dtype=np.float64)
    det0L = np.zeros(TN, dtype=np.float64)
    ddcL  = np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    dc0 = dc0/10
    for i in xrange(TN):
        omega = twopi/T[i]
        wvno =  omega/c[i]
        if isR:
            det0L[i] = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0L[i] = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  omega/(c[i]+dc0)
        if isR:
            ddcL[i] = -(dltarR(d,a,b,rho,wvno,omega,LN)-det0L[i])/dc0
        else:
            ddcL[i] = -(dltarL(d,a,b,rho,wvno,omega,LN)-det0L[i])/dc0
    if isP:
        P = A
        p = a
    if isS:
        P = B
        p = b
    if isRho:
        P = Rho
        p = rho
    if isD:
        P = D
        p = d
        dc0 = dh
    for i in xrange(LN):
        if isFlat:
            P[i]=P[i]+dc0
            sphere(D, A, B, Rho, LN,isR, ar)
        else:
            p[i]=p[i]+dc0
        for j in xrange(TN):
            omega = twopi/T[j]
            wvno =  omega/c[j]
            if isR:
                K[i,j] = (dltarR(d,a,b,rho,wvno,omega,LN)-det0L[j])/dc0/ddcL[j]
            else:
                K[i,j] = (dltarL(d,a,b,rho,wvno,omega,LN)-det0L[j])/dc0/ddcL[j]
        if isFlat:
            P[i]=P[i]-dc0
            sphere(D, A, B, Rho, LN,isR, ar)
        else:
            p[i]=p[i]-dc0
    return K

cpdef double[:,:] kernelGroup_(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.001,bint isFlat=False, double ar=6370,double domega = 0.001,double dh=0.001,bint isP=False,bint isS=False,bint isD=False,bint isRho=False):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    cdef int i,j
    cdef double twopi=2*3.141592653589793
    cdef double omega,wvno,ddc,det,dKdV,dOmegadV,det0,DOmega,DK
    cdef double[:,:] K
    cdef double[:]c,det0L,ddcL,dOmegaL,dkL
    #cdef int aoundK
    #cdef double aroundKD,aroundND
    c=np.zeros(TN, dtype=np.float64)
    K=np.zeros([LN,TN], dtype=np.float64)
    det0L = np.zeros(TN, dtype=np.float64)
    dOmegaL  = np.zeros(TN, dtype=np.float64)
    dKL  = np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    dc0 = dc0/500
    #print(domega)
    for i in xrange(TN):
        if c[i]<=0:
            continue
        DOmega =twopi/T[i]*(domega)
        omega = twopi/T[i]-DOmega
        wvno =  twopi/T[i]/c[i]
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]+DOmega
        wvno  =  twopi/T[i]/c[i]
        if isR:
            dOmegaL[i] = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        else:
            dOmegaL[i] = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        #print(dltarR(d,a,b,rho,wvno,omega,LN),det0,DOmega,dOmegaL[i])
        DK =twopi/T[i]/c[i]*(domega)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/c[i]-DK
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/c[i]+DK
        if isR:
            dKL[i] = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
        else:
            dKL[i] = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
        #print(-dKL[i]/dOmegaL[i],det0)
        #print(DOmega,dOmegaL[i])
    for i in xrange(LN):
        if isS:
            if isFlat:
                B[i]=B[i]+dc0
                sphere(D, A, B, Rho, LN,isR, ar)
            else:
                b[i]=b[i]+dc0
        for j in xrange(TN):
            if c[j]<=0:
                continue
            DOmega =twopi/T[j]*(domega)
            omega = twopi/T[j]-DOmega
            wvno =  twopi/T[j]/c[j]
            if isR:
                det0 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                det0 = dltarL(d,a,b,rho,wvno,omega,LN)
            omega = twopi/T[j]+DOmega
            wvno  =  twopi/T[j]/c[j]
            if isR:
                dOmegadV = ((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2-dOmegaL[j])/dc0
            else:
                dOmegadV =((dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2-dOmegaL[j])/dc0
            #print((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2,dOmegaL[j],dc0)
            DK =twopi/T[j]/c[j]*(domega)
            omega = twopi/T[j]
            wvno  =  twopi/T[j]/c[j]-DK
            if isR:
                det0 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                det0 = dltarL(d,a,b,rho,wvno,omega,LN)
            omega = twopi/T[j]
            wvno  =  twopi/T[j]/c[j]+DK
            if isR:
                dKdV = ((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2-dKL[j])/dc0
            else:
                dKdV = ((dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DK/2-dKL[j])/dc0
            #
            K[i,j]=-dKdV/dOmegaL[j]+dKL[j]/dOmegaL[j]**2*dOmegadV
            #print(dKdV,dOmegadV,dltarR(d,a,b,rho,wvno,omega,LN),det0,omega,wvno,DK,dc0)
            #print((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2-dKL[j],dc0)
        if isS:
            if isFlat:
                B[i]=B[i]-dc0
                sphere(D, A, B, Rho, LN,isR, ar)
            else:
                b[i]=b[i]-dc0
    return K

cpdef double[:,:] kernelGroup(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.001,bint isFlat=False, double ar=6370,double domega = 0.001,double dh=0.001,bint isP=False,bint isS=False,bint isD=False,bint isRho=False,smoothN=1):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    cdef int i,j
    cdef double twopi=2*3.141592653589793
    cdef double omega,wvno,ddc,det,dKdV,dOmegadV,det0,DOmega,DK,dK,dOmega,C,DW,DC,dv0
    cdef double[:,:] K
    cdef double[:]c,det0L,ddcL,gL,P,p
    #cdef int aoundK
    #cdef double aroundKD,aroundND
    c=np.zeros(TN, dtype=np.float64)
    K=np.zeros([LN,TN], dtype=np.float64)
    det0L = np.zeros(TN, dtype=np.float64)
    ddcL  = np.zeros(TN, dtype=np.float64)
    gL  = np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    dv0 = dc0/5
    dc0 = dc0/100.0
    #print(domega)
    for i in xrange(TN):
        if c[i]<=0:
            continue
        omega = twopi/T[i]
        wvno =  twopi/T[i]/c[i]
        if isR:
            det0L[i] = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0L[i] = dltarL(d,a,b,rho,wvno,omega,LN)
        DOmega =twopi/T[i]*(domega)
        omega = twopi/T[i]-DOmega
        wvno =  omega/c[i]
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]+DOmega
        wvno  =  omega/c[i]
        if isR:
            dOmega = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        else:
            dOmega = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        #print(dltarR(d,a,b,rho,wvno,omega,LN),det0,DOmega)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i]-dc0)
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i]+dc0)
        if isR:
            dC = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/dc0/2
        else:
            dC = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/dc0/2
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i])
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i]+dc0)
        if isR:
            ddcL[i] = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/dc0
        else:
            ddcL[i] = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/dc0
        gL[i]=dOmega/dC/c[i]*twopi/T[i]
        #print(gL[i])
        #print(ddcL[i],gL[i])
        #print(-dKL[i]/dOmegaL[i],det0)
        #print(DOmega,dOmegaL[i])
    if isP:
        P = A
        p = a
    if isS:
        P = B
        p = b
    if isRho:
        P = Rho
        p = rho
    if isD:
        P = D
        p = d
        dv0 = dh/10
    for i in xrange(LN):
        if smoothN==1:
            if isFlat:
                P[i]=P[i]+dv0
                sphere(D, A, B, Rho, LN,isR, ar)
            else:
                p[i]=p[i]+dv0
        else:
            DW = 0
            for j in xrange(-smoothN+1,smoothN):
                if (i+j)>=0 and (i+j)<LN:
                    DW += D[i+j]/D[i]*(1-<double>j/<double>smoothN)
            if isFlat:
                for j in xrange(-smoothN+1,smoothN):
                    P[i+j]=P[i+j]+dv0*(1-<double>j/<double>smoothN)/DW
                sphere(D, A, B, Rho, LN,isR, ar)
            else:
                for j in xrange(-smoothN+1,smoothN):
                    p[i+j]=p[i+j]+dv0*(1-<double>j/<double>smoothN)/DW
        for j in xrange(TN):
            if c[j]<=0:
                continue
            omega = twopi/T[j]
            wvno =  twopi/T[j]/c[j]
            if isR:
                DC = -(dltarR(d,a,b,rho,wvno,omega,LN)-det0L[j])/ddcL[j]
            else:
                DC = -(dltarL(d,a,b,rho,wvno,omega,LN)-det0L[j])/ddcL[j]
            C = c[j]+DC
            #print('#1',det0L[j],dltarR(d,a,b,rho,wvno,omega,LN))
            omega = twopi/T[j]
            wvno =  twopi/T[j]/C
            #print('#2',det0L[j],dltarR(d,a,b,rho,wvno,omega,LN))
            DOmega =twopi/T[j]*(domega)
            omega = twopi/T[j]-DOmega
            wvno =  omega/C
            if isR:
                det0 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                det0 = dltarL(d,a,b,rho,wvno,omega,LN)
            omega = twopi/T[j]+DOmega
            wvno  =  omega/C
            if isR:
                dOmega = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
            else:
                dOmega =(dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
            #print((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2,dOmegaL[j],dc0)
            omega = twopi/T[j]
            wvno  =  omega/(C-dc0)
            if isR:
                det0 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                det0 = dltarL(d,a,b,rho,wvno,omega,LN)
            omega = twopi/T[j]
            wvno  =  omega/(C+dc0)
            if isR:
                dC = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/dc0/2
            else:
                dC = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/dc0/2
            #
            K[i,j]= (C/(1+dOmega/dC/C*omega)-c[j]/(1+gL[j]))/dv0
            if i==20:
                print((C/(1+dOmega/dC/C*omega)-c[j]/(1+gL[j]))/dv0)
                #print(DC,c[j]/(1+gL[j]),dOmega/dc/C*omega,gL[j],(dOmega/dc/C*omega-gL[j]),(DC/(1+gL[j])-c[j]/(1+gL[j])**2*(dOmega/dC/C*omega-gL[j])),dc0)
            #print(dK/dOmega,gL[j],dK,dOmega)
            #print(dKdV,dOmegadV,dltarR(d,a,b,rho,wvno,omega,LN),det0,omega,wvno,DK,dc0)
            #print((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2-dKL[j],dc0)
        if smoothN==1:
            if isFlat:
                P[i]=P[i]-dv0
                sphere(D, A, B, Rho, LN,isR, ar)
            else:
                p[i]=p[i]-dv0
        else:
            if isFlat:
                for j in xrange(-smoothN+1,smoothN):
                    P[i+j]=P[i+j]-dv0*(1-<double>j/<double>smoothN)/DW
                sphere(D, A, B, Rho, LN,isR, ar)
            else:
                for j in xrange(-smoothN+1,smoothN):
                    p[i+j]=p[i+j]-dv0*(1-<double>j/<double>smoothN)/DW
    return K
cpdef double[:,:] kernelGroup__(double[:] D, double[:]A,double[:] B,double[:] Rho,double[:]T, int mode=1,bint isR=True, double sone0=1.5, double dc0=0.001,bint isFlat=False, double ar=6370,double domega = 0.001,double dh=0.01,bint isP=False,bint isS=False,bint isD=False,bint isRho=False,smoothN=1):
    #print(A[0],B[0])
    cdef int LN,TN
    LN = len(A)
    TN = len(T)
    cdef int i,j
    cdef double twopi=2*3.141592653589793
    cdef double omega,wvno,ddc,det,dKdV,dOmegadV,det0,DOmega,DK,dK,dOmega,C,DW
    cdef double[:,:] K
    cdef double[:]c,det0L,ddcL,gL
    #cdef int aoundK
    #cdef double aroundKD,aroundND
    c=np.zeros(TN, dtype=np.float64)
    K=np.zeros([LN,TN], dtype=np.float64)
    det0L = np.zeros(TN, dtype=np.float64)
    ddcL  = np.zeros(TN, dtype=np.float64)
    gL  = np.zeros(TN, dtype=np.float64)
    CalDisp(D, A, B, Rho,T,c,LN, TN , mode,isR, sone0, dc0,isFlat, ar)
    dc0 = dc0/500.0
    #print(domega)
    for i in xrange(TN):
        if c[i]<=0:
            continue
        omega = twopi/T[i]
        wvno =  twopi/T[i]/c[i]
        if isR:
            det0L[i] = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0L[i] = dltarL(d,a,b,rho,wvno,omega,LN)
        DOmega =twopi/T[i]*(domega)
        omega = twopi/T[i]-DOmega
        wvno =  twopi/T[i]/c[i]
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]+DOmega
        wvno  =  twopi/T[i]/c[i]
        if isR:
            dOmega = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        else:
            dOmega = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
        #print(dltarR(d,a,b,rho,wvno,omega,LN),det0,DOmega)
        DK =twopi/T[i]/c[i]*(domega)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/c[i]-DK
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/c[i]+DK
        if isR:
            dK = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
        else:
            dK = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i])
        if isR:
            det0 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            det0 = dltarL(d,a,b,rho,wvno,omega,LN)
        omega = twopi/T[i]
        wvno  =  twopi/T[i]/(c[i]+dc0)
        if isR:
            ddcL[i] = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/dc0
        else:
            ddcL[i] = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/dc0
        gL[i]=-dK/dOmega
        #print(ddcL[i],gL[i])
        #print(-dKL[i]/dOmegaL[i],det0)
        #print(DOmega,dOmegaL[i])
    
    
    for i in xrange(LN):
        if isS:
            if smoothN==1:
                if isFlat:
                    B[i]=B[i]+dc0
                    sphere(D, A, B, Rho, LN,isR, ar)
                else:
                    b[i]=b[i]+dc0
            else:
                DW = 0
                for j in xrange(-smoothN+1,smoothN):
                    if (i+j)>=0 and (i+j)<LN:
                        DW += D[i+j]/D[i]*(1-<double>j/<double>smoothN)
                if isFlat:
                    for j in xrange(-smoothN+1,smoothN):
                        B[i+j]=B[i+j]+dc0*(1-<double>j/<double>smoothN)/DW
                    sphere(D, A, B, Rho, LN,isR, ar)
                else:
                    for j in xrange(-smoothN+1,smoothN):
                        b[i+j]=b[i+j]+dc0*(1-<double>j/<double>smoothN)/DW
        for j in xrange(TN):
            if c[j]<=0:
                continue
            omega = twopi/T[j]
            wvno =  twopi/T[j]/c[j]
            if isR:
                C = c[j]-(dltarR(d,a,b,rho,wvno,omega,LN)-det0L[j])/ddcL[j]
            else:
                C = c[j]-(dltarL(d,a,b,rho,wvno,omega,LN)-det0L[j])/ddcL[j]
            #print('#1',det0L[j],dltarR(d,a,b,rho,wvno,omega,LN))
            omega = twopi/T[j]
            wvno =  twopi/T[j]/C
            #print('#2',det0L[j],dltarR(d,a,b,rho,wvno,omega,LN))
            DOmega =twopi/T[j]*(domega)
            omega = twopi/T[j]-DOmega
            wvno =  twopi/T[j]/C
            if isR:
                det0 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                det0 = dltarL(d,a,b,rho,wvno,omega,LN)
            omega = twopi/T[j]+DOmega
            wvno  =  twopi/T[j]/C
            if isR:
                dOmega = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
            else:
                dOmega =(dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2
            #print((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DOmega/2,dOmegaL[j],dc0)
            DK =twopi/T[j]/c[j]*(domega)
            omega = twopi/T[j]
            wvno  =  twopi/T[j]/C-DK
            if isR:
                det0 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                det0 = dltarL(d,a,b,rho,wvno,omega,LN)
            omega = twopi/T[j]
            wvno  =  twopi/T[j]/C+DK
            if isR:
                dK = (dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
            else:
                dK = (dltarL(d,a,b,rho,wvno,omega,LN)-det0)/DK/2
            #
            K[i,j]=(-dK/dOmega-gL[j])/dc0
            print(dK/dOmega,gL[j],(-dK/dOmega-gL[j]),dOmega*dc0,dOmega)
            #print(dK/dOmega,gL[j],dK,dOmega)
            #print(dKdV,dOmegadV,dltarR(d,a,b,rho,wvno,omega,LN),det0,omega,wvno,DK,dc0)
            #print((dltarR(d,a,b,rho,wvno,omega,LN)-det0)/DK/2-dKL[j],dc0)
        if isS:
            if smoothN==1:
                if isFlat:
                    B[i]=B[i]-dc0
                    sphere(D, A, B, Rho, LN,isR, ar)
                else:
                    b[i]=b[i]-dc0
            else:
                if isFlat:
                    for j in xrange(-smoothN+1,smoothN):
                        B[i+j]=B[i+j]-dc0*(1-<double>j/<double>smoothN)/DW
                    sphere(D, A, B, Rho, LN,isR, ar)
                else:
                    for j in xrange(-smoothN+1,smoothN):
                        b[i+j]=b[i+j]-dc0*(1-<double>j/<double>smoothN)/DW
    return K
cdef int CalDisp(double[:] D,double[:] A,double[:] B,double[:] Rho,double[:] T,double[:] c,int LN, int TN=1 ,int mode=1,bint isR=True,double sone0=1.5,double dc0=0.005,bint isFlat=False,double ar=6370):
    global del1st#,a,b,d,rho
    cdef int index,jsol,m,k,loop,i
    cdef double betmn,betmx,cc1,cc,c1,dc,cm,onea,t1,clow,
    cdef bint ifirst,
    if isFlat:
        sphere(D,A,B,Rho,LN,isR=isR,ar=ar)
    else:
        for i in xrange(LN):
            d[i] = D[i]
            a[i] = A[i]
            b[i] = B[i]
            rho[i] = Rho[i]
        #d[LN-1]=0.0
    #print(b)
    betmn=1e20
    for i in xrange(LN):
        if b[i]>0.01 and b[i]<betmn:
            betmn = b[i]
            index = i
            jsol = 1
        elif b[i]<= 0.01 and a[i]<betmn:
            betmn = a[i]
            index = i
            jsol = 0
    betmx = max(b,LN)
    if jsol==0:
        cc1 = betmn
    else:
        cc1 = gtsolh(a[index],b[index])
    cc1=0.99*cc1
    cc= cc1
    dc = dc0
    c1=cc
    cm=cc
    #c  = zeros(len(T))
    onea = sone0
    del1st=1.0
    #return 0
    for m in xrange(mode):
        loop =TN
        if False:
            if m==mode-1:
                loop = TN
            else:
                loop = 1
        for k in xrange(loop):
            t1=T[k]
            '''
            c-----
            c     get initial phase velocity estimate to begin search
            c
            c     in the notation here, c() is an array of phase velocities
            c     c(k-1) is the velocity estimate of the present mode
            c     at the k-1 period, while c(k) is the phase velocity of the
            c     previous mode at the k period. Since there must be no mode
            c     crossing, we make use of these values. The only complexity
            c     is that the dispersion may be reversed. 
            c
            c     The subroutine getsol determines the zero crossing and refines
            c     the root.
            c-----
            '''
            if True:
                if(k == 0 and m==0):
                    c1 = cc
                    clow = cc
                    ifirst = True
                    #print(t1,c1,clow,dc,cm,betmx,isR,ifirst)
                elif (k == 0 and m > 0):
                    c1 = c[0] + dc
                    clow = c1
                    ifirst = True
                elif(k>0 and m>0):
                    ifirst = False
                    clow = c[k] + dc
                    c1 = c[k-1]
                    if(c1 < clow):
                        c1 = clow
                elif(k > 0 and m == 0):
                    ifirst = False
                    c1 = c[k-1] - onea*dc
                    clow = cm
            else:
                if(k == 0 and m==0):
                    c1 = cc
                    clow = cc
                    ifirst = True
                    #print(t1,c1,clow,dc,cm,betmx,isR,ifirst)
                elif (k == 0 and m > 0):
                    c1 = c[0] + dc
                    clow = c1
                    ifirst = True
                else:
                    ifirst = False
                    c1 = c[k-1] - onea*dc
                    clow = cm
                #print(t1,c1,clow,dc,cm,betmx,isR,ifirst)
            #print(k,m,t1,cc,clow,dc,cm,betmx,isR,ifirst)
            #print(t1,c1,clow,dc,cm,betmx,isR,ifirst)
            #return 0
            #print(d,a,b,rho,t1,c1,clow,dc,cm,betmx,isR,ifirst,LN)
            #return 0
            c[k]=getsol(d,a,b,rho,t1,c1,clow,dc,cm,betmx,isR,ifirst,LN)
            if c[k]<0:
                print(c1,clow,dc,cm,betmx,t1)
            #print(del1st)
            #print(c[0])
    return 0
#@jit
cdef int sphere(double[:] D,double[:]A,double[:]B,double[:]Rho,int LN,bint isR, double ar=6370):
    '''
    c-----
    c     Transform spherical earth to flat earth
    c
    c     Schwab, F. A., and L. Knopoff (1972). Fast surface wave and free
    c     mode computations, in  Methods in Computational Physics, 
    c         Volume 11,
    c     Seismology: Surface Waves and Earth Oscillations,  
    c         B. A. Bolt (ed),
    c     Academic Press, New York
    c
    c     Love Wave Equations  44, 45 , 41 pp 112-113
    c     Rayleigh Wave Equations 102, 108, 109 pp 142, 144
    c
    c     Revised 28 DEC 2007 to use mid-point, assume linear variation in
    c     slowness instead of using average velocity for the layer
    c     Use the Biswas (1972:PAGEOPH 96, 61-74, 1972) density mapping
    c
    c     ifunc   I*4 1 - Love Wave
    c                 2 - Rayleigh Wave
    c     iflag   I*4 0 - Initialize
    c                 1 - Make model  for Love or Rayleigh Wave
    c-----
    '''
    global a,b,d,rho
    cdef double r0,r1,tmp,D0
    cdef int i
    r0=ar
    r1 = r0
    D0 = D[LN-1]
    D[LN-1]=0.0
    for i in xrange(LN):
        r1 = r0-D[i]
        d[i] = ar*log(ar/r1)-ar*log(ar/r0)
        tmp=(ar+ar)/(r1+r0)
        a[i]=A[i]*tmp
        b[i]=B[i]*tmp
        if isR:
            rho[i]=Rho[i]*tmp**(-2.275)
        else:
            rho[i]=Rho[i]*tmp**(-5)
        r0 =r1
    D[LN-1]=D0
    return 0 

#@jit
cdef double gtsolh(double a, double b):
    '''
    c-----
    c     starting solution
    c-----
    '''
    cdef double c, gamma,kappa,k2,gk2,fac1,fac2,fr,frp
    c = 0.95*b
    for i in xrange(1,5):
        gamma = b/a
        kappa = c/b
        k2 = kappa*kappa
        gk2 = (gamma*kappa)*(gamma*kappa)
        fac1 = sqrt(1.0 - gk2)
        fac2 = sqrt(1.0 - k2)
        fr = (2.0 - k2)**2 - 4.0*fac1*fac2
        frp = -4.0*(2.0-k2) *kappa+4.0*fac2*gamma*gamma*kappa/fac1+4.0*fac1*kappa/fac2
        frp = frp/b
        c = c - fr/frp
    return c

#@jit
cdef double dltarL(double[:]d,double[:]a,double[:]b,double[:]rho,double wvno,double omega,int LN):
    '''
    c   find SH dispersion values.
    c
        parameter (NL=100,NP=60)
        implicit double precision (a-h,o-z)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        integer llw,LN
    c        common/modl/ d,a,b,rho,rtp,dtp,btp
    c        common/para/ LN,llw,twopi
    c
    c   Haskell-Thompson love wave formulation from halfspace
    c   to surface.
    c
    '''
    cdef double xkb, wvnop,wvnom,rb,e1,e2,xmu,q,y,z,cosq,fac,sinq,e10,e20,xnor,ynor
    cdef int i0
    xkb=omega/b[LN-1]
    wvnop=wvno+xkb
    wvnom=abs(wvno-xkb)
    rb=sqrt(wvnop*wvnom)
    e1=rho[LN-1]*rb
    e2=1/(b[LN-1]*b[LN-1])
    i0=0
    if b[0]<0.001:
        i0=1
    for m in xrange(LN-2,i0-1,-1):
        xmu=rho[m]*b[m]*b[m]
        xkb=omega/b[m]
        wvnop=wvno+xkb
        wvnom=abs(wvno-xkb)
        rb=sqrt(wvnop*wvnom)
        q = d[m]*rb
        if(wvno<xkb):
            sinq = sin(q)
            y = sinq/rb
            z = -rb*sinq
            cosq = cos(q)
        elif(wvno==xkb):
            cosq=1.000
            y=d[m]
            z=0.00
        else:
            fac = 0.000
            if(q<16):
                fac = exp(-2.0*q)
            cosq = ( 1.000 + fac ) * 0.500
            sinq = ( 1.000 - fac ) * 0.500
            y = sinq/rb
            z = rb*sinq
        e10=e1*cosq+e2*xmu*z
        e20=e1*y/xmu+e2*cosq
        xnor=abs(e10)
        ynor=abs(e20)
        if(ynor>xnor):
            xnor=ynor
        if(xnor<1e-10): 
            xnor=1.000
        e1=e10/xnor
        e2=e20/xnor
    return e1
#@jit
cdef double dltarR(double[:]d,double[:]a,double[:]b,double[:]rho,double wvno,double omega,int LN):
    #print(LN)
    #cdef double a0, w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    global E0,E1,E2,E3,E4
    global E01,E11,E21,E31,E41,cosp,w0
    cdef double wvno2,xka,xkb,wvnop,wvnom,ra,rb,gammk,gam,gamm1,rho1,t,p,q,tmp,dpth,znul
    cdef int i0=1
    wvno2=wvno*wvno
    xka=omega/a[LN-1]
    xkb=omega/b[LN-1]
    wvnop=wvno+xka
    wvnom=abs(wvno-xka)
    ra=sqrt(wvnop*wvnom)
    wvnop=wvno+xkb
    wvnom=abs(wvno-xkb)
    rb=sqrt(wvnop*wvnom)
    t = b[LN-1]/omega
    #E matrix for the bottom half-space.
    gammk = 2.0*t*t
    gam   = gammk*wvno2
    gamm1 = gam - 1
    rho1  = rho[LN-1]
    E0=rho1*rho1*(gamm1*gamm1-gam*gammk*ra*rb)
    E1=(-rho1*ra)
    E2=(rho1*(gamm1-gammk*ra*rb))
    E3=(rho1*rb)
    E4=(wvno2-ra*rb)
    #matrix multiplication from bottom layer upward
    i0=0
    if b[0]<0.001:
        i0=1
    for m in xrange(LN-2,i0-1,-1):
        xka = omega/a[m]
        xkb = omega/b[m]
        t = b[m]/omega
        gammk = 2*t*t
        gam = gammk*wvno2
        wvnop=wvno+xka
        wvnom=abs(wvno-xka)
        ra= sqrt(wvnop*wvnom)
        wvnop=wvno+xkb
        wvnom=abs(wvno-xkb)
        rb=sqrt(wvnop*wvnom)
        dpth=d[m]
        rho1=rho[m]
        p=ra*dpth
        q=rb*dpth
        #evaluate cosP, cosQ,.... in var.#
        # evaluate Dunkin's matrix in dnka.
        var(p,q,ra,rb,wvno,xka,xkb,dpth)#,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
        dnka(wvno2,gam,gammk,rho1)#,,a0)cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
        #print(ca) 
        E01 = E0*ca00+E1*ca10+E2*ca20+E3*ca30+E4*ca40
        E11 = E0*ca01+E1*ca11+E2*ca21+E3*ca31+E4*ca41
        E21 = E0*ca02+E1*ca12+E2*ca22+E3*ca32+E4*ca42
        E31 = E0*ca03+E1*ca13+E2*ca23+E3*ca33+E4*ca43
        E41 = E0*ca04+E1*ca14+E2*ca24+E3*ca34+E4*ca44
        tmp = 0.0000000
        if E01>tmp:
            tmp = E01
        if E11>tmp:
            tmp = E11
        if E21>tmp:
            tmp = E21
        if E31>tmp:
            tmp = E31
        if E41>tmp:
            tmp = E41
        if tmp<0.00000000001:
            tmp=1
        E0 = E01/tmp
        E1 = E11/tmp
        E2 = E21/tmp
        E3 = E31/tmp
        E4 = E41/tmp
        #e[:]=ee[:]
        #print(e)
        #print(E0,E1, E2, E3, E4)
        #print(ca01,ca02,ca03,ca04)
    if i0==1:
        #include water layer.
        xka = omega/a[0]
        wvnop=wvno+xka
        wvnom=abs(wvno-xka)
        ra=sqrt(wvnop*wvnom)
        dpth=d[0]
        rho1=rho[0]
        p = ra*dpth
        znul = 1e-5
        var(p,znul,ra,znul,wvno,xka,znul,dpth)
        w0=-rho1*w
        return cosp*E0 + w0*E1
    else:
        return E0


cdef double dltarL_(double[:]d,double[:]a,double[:]b,double[:]rho,double wvno,double omega,int LN):
    '''
    c   find SH dispersion values.
    c
        parameter (NL=100,NP=60)
        implicit double precision (a-h,o-z)
        real*4 d(NL),a(NL),b(NL),rho(NL),rtp(NL),dtp(NL),btp(NL)
        integer llw,LN
    c        common/modl/ d,a,b,rho,rtp,dtp,btp
    c        common/para/ LN,llw,twopi
    c
    c   Haskell-Thompson love wave formulation from halfspace
    c   to surface.
    c
    '''
    cdef double xkb, wvnop,wvnom,rb,e1,e2,xmu,q,y,z,cosq,fac,sinq,e10,e20,xnor,ynor
    cdef int i0
    xkb=omega/b[LN-1]
    wvnop=wvno+xkb
    wvnom=abs(wvno-xkb)
    rb=sqrt(wvnop*wvnom)
    e1=rho[LN-1]*rb
    e2=1/(b[LN-1]*b[LN-1])
    i0=0
    if b[0]<0.001:
        i0=1
    for m in xrange(LN-2,i0-1,-1):
        xmu=rho[m]*b[m]*b[m]
        xkb=omega/b[m]
        wvnop=wvno+xkb
        wvnom=abs(wvno-xkb)
        rb=sqrt(wvnop*wvnom)
        q = d[m]*rb
        if(wvno<xkb):
            sinq = sin(q)
            y = sinq/rb
            z = -rb*sinq
            cosq = cos(q)
        elif(wvno==xkb):
            cosq=1.000
            y=d[m]
            z=0.00
        else:
            fac = 0.000
            if(q<16):
                fac = exp(-2.0*q)
            cosq = ( 1.000 + fac ) * 0.500
            sinq = ( 1.000 - fac ) * 0.500
            y = sinq/rb
            z = rb*sinq
        e10=e1*cosq+e2*xmu*z
        e20=e1*y/xmu+e2*cosq
        xnor=1.000
        e1=e10/xnor
        e2=e20/xnor
    return e1
#@jit
cdef double dltarR_(double[:]d,double[:]a,double[:]b,double[:]rho,double wvno,double omega,int LN):
    #print(LN)
    #cdef double a0, w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    global E0,E1,E2,E3,E4
    global E01,E11,E21,E31,E41,cosp,w0
    cdef double wvno2,xka,xkb,wvnop,wvnom,ra,rb,gammk,gam,gamm1,rho1,t,p,q,tmp,dpth,znul
    cdef int i0=1
    wvno2=wvno*wvno
    xka=omega/a[LN-1]
    xkb=omega/b[LN-1]
    wvnop=wvno+xka
    wvnom=abs(wvno-xka)
    ra=sqrt(wvnop*wvnom)
    wvnop=wvno+xkb
    wvnom=abs(wvno-xkb)
    rb=sqrt(wvnop*wvnom)
    t = b[LN-1]/omega
    #E matrix for the bottom half-space.
    gammk = 2.0*t*t
    gam   = gammk*wvno2
    gamm1 = gam - 1
    rho1  = rho[LN-1]
    E0=rho1*rho1*(gamm1*gamm1-gam*gammk*ra*rb)
    E1=(-rho1*ra)
    E2=(rho1*(gamm1-gammk*ra*rb))
    E3=(rho1*rb)
    E4=(wvno2-ra*rb)
    #matrix multiplication from bottom layer upward
    i0=0
    if b[0]<0.001:
        i0=1
    for m in xrange(LN-2,i0-1,-1):
        xka = omega/a[m]
        xkb = omega/b[m]
        t = b[m]/omega
        gammk = 2*t*t
        gam = gammk*wvno2
        wvnop=wvno+xka
        wvnom=abs(wvno-xka)
        ra= sqrt(wvnop*wvnom)
        wvnop=wvno+xkb
        wvnom=abs(wvno-xkb)
        rb=sqrt(wvnop*wvnom)
        dpth=d[m]
        rho1=rho[m]
        p=ra*dpth
        q=rb*dpth
        #evaluate cosP, cosQ,.... in var.#
        # evaluate Dunkin's matrix in dnka.
        var(p,q,ra,rb,wvno,xka,xkb,dpth)#,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
        dnka(wvno2,gam,gammk,rho1)#,,a0)cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
        #print(ca) 
        E01 = E0*ca00+E1*ca10+E2*ca20+E3*ca30+E4*ca40
        E11 = E0*ca01+E1*ca11+E2*ca21+E3*ca31+E4*ca41
        E21 = E0*ca02+E1*ca12+E2*ca22+E3*ca32+E4*ca42
        E31 = E0*ca03+E1*ca13+E2*ca23+E3*ca33+E4*ca43
        E41 = E0*ca04+E1*ca14+E2*ca24+E3*ca34+E4*ca44
        tmp=1
        E0 = E01/tmp
        E1 = E11/tmp
        E2 = E21/tmp
        E3 = E31/tmp
        E4 = E41/tmp
        #e[:]=ee[:]
        #print(e)
        #print(E0,E1, E2, E3, E4)
        #print(ca01,ca02,ca03,ca04)
    if i0==1:
        #include water layer.
        xka = omega/a[0]
        wvnop=wvno+xka
        wvnom=abs(wvno-xka)
        ra=sqrt(wvnop*wvnom)
        dpth=d[0]
        rho1=rho[0]
        p = ra*dpth
        znul = 1e-5
        var(p,znul,ra,znul,wvno,xka,znul,dpth)
        w0=-rho1*w
        return cosp*E0 + w0*E1
    else:
        return E0
#@jit
cdef double getsol(double[:] d,double[:]a,double[:]b,double[:]rho,double t1,double c1,double clow,double dc,double cm,double betmx,bint isR,bint ifirst,int LN):
    global del1st

    '''
    c-----
    c     subroutine to bracket dispersion curve
    c     and then refine it
    c-----
    c     t1  - period
    c     c1  - initial guess on low side of mode
    c     clow    - lowest possible value for present mode in a
    c           reversed direction search
    c     dc  - phase velocity search increment
    c     cm  - minimum possible solution
    c     betmx   - maximum shear velocity
    c     iret    - 1 = successful
    c         - -1= unsuccessful
    c     ifunc   - 1 - Love
    c         - 2 - Rayleigh
    c     ifirst  - 1 this is first period for a particular mode
    c         - 0 this is not the first period
    c             (this is to define period equation sign
    c              for mode jumping test)
    c-----
    c-----
    c     to avoid problems in mode jumping with reversed dispersion
    c     we note what the polarity of period equation is for phase
    c     velocities just beneath the zero crossing at the 
    c         first period computed.
    c-----
    c     bracket solution
    c-----
    '''
    cdef double twopi=2*3.141592653589793,omega,wvno,del1,del2,idir,plmn,c2,cn
    omega=twopi/t1
    wvno=omega/c1
    #print(d[0],a[0],rho[0],wvno,omega,LN)
    #return 0
    if isR :
        del1 = dltarR(d,a,b,rho,wvno,omega,LN)
    else:
        del1 = dltarL(d,a,b,rho,wvno,omega,LN)
    #return 0
    if ifirst:
        del1st = del1
    plmn = sign(del1st)*sign(del1)
    if ifirst:
        idir = +1
    elif plmn >=0:
        idir = +1
    else:
        idir = -1
    '''
    c-----
    c     idir indicates the direction of the search for the
    c     true phase velocity from the initial estimate.
    c     Usually phase velocity increases with period and
    c     we always underestimate, so phase velocity should increase
    c     (idir = +1). For reversed dispersion, we should look
    c     downward from the present estimate. However, we never
    c     go below the floor of clow, when the direction is reversed
    c-----
    '''
    for i in xrange(3000):
        if(idir>0):
            c2 = c1 + dc
        else:
            c2 = c1 - dc
        if ifirst:
            pass
            #print(c1,c2)
        if(c2<=clow):
            idir = +1
            c1 = clow
        if(c2<=clow):
            continue
        #if ifirst:
        wvno=omega/c2
        if isR :
            del2 = dltarR(d,a,b,rho,wvno,omega,LN)
        else:
            del2 = dltarL(d,a,b,rho,wvno,omega,LN)
        if not (sign(del1)!=sign(del2)):
            c1=c2
            del1=del2
            if(c1<=cm):
                return -1
            if(c1>=(betmx+dc)):
                return -1
            continue
        else:
            cn=nevill(d,a,b,rho,t1,c1,c2,del1,del2,isR,twopi,LN)
            c1 = cn
            if(c1>(betmx)):
                return -1.0
            return c1
    return c1
#@jit
cdef double nevill(double[:] d,double[:] a,double[:] b,double[:] rho,double t,double c1,double c2,double del1,double del2,bint isR,double twopi,int LN):
    '''
    c-----
    c   hybrid method for refining root once it has been bracketted
    c   between c1 and c2.  interval halving is used where other schemes
    c   would be inefficient.  once suitable region is found neville s
    c   iteration method is used to find root.
    c   the procedure alternates between the interval halving and neville
    c   techniques using whichever is most efficient
    c-----
    c     the control integer nev means the following:
    c
    c     nev = 0 force interval halving
    c     nev = 1 permit neville iteration if conditions are proper
    c     nev = 2 neville iteration is being used
    c-----
    
    c        common/modl/ d,a,b,rho,rtp,dtp,btp
    c        common/para/ LN,llw,twopi
    c-----
    c     initial guess
    c-----
    '''
    cdef double c3,wvno,del3,s13,s32,s1,s2,denom
    cdef int nev,m,kk,j
    cdef bint isBreak
    omega = twopi/t
    c3 = 0.5*(c1 + c2)
    wvno=omega/c3
    if isR :
        del3 = dltarR(d,a,b,rho,wvno,omega ,LN)
    else:
        del3 = dltarL(d,a,b,rho,wvno,omega,LN)
    nev = 1
    #100 continue
    for nctrl in xrange(100):
        '''
        c-----
        c     make sure new estimate is inside the previous values. If not
        c     perform interval halving
        c-----
        '''
        if(c3 -c2)*(c3-c1)>=0:
            nev = 0
            c3 = 0.5*(c1 + c2)
            wvno=omega/c3
            if isR :
                del3 = dltarR(d,a,b,rho,wvno,omega ,LN)
            else:
                del3 = dltarL(d,a,b,rho,wvno,omega,LN)
        s13 = del1 - del3
        s32 = del3 - del2
        '''
        c-----
        c     define new bounds according to the sign of the period equation
        c-----
        '''
        if(sign(del3)*sign(del1) < 0): 
            c2 = c3
            del2 = del3
        else:
            c1 = c3
            del1 = del3
        #check for convergence. A relative error criteria is used
        if abs(c1-c2)<= 1.e-6*c1:
            break
        #if the slopes are not the same between c1, c3 and c3
        if(sign(s13)!=sign (s32)): 
            nev = 0
        ss1=abs(del1)
        s1=0.01*ss1
        ss2=abs(del2)
        s2=0.01*ss2
        if(s1>=ss2 or s2>=ss1 or nev==0):
            c3 = 0.5*(c1 + c2)
            wvno=omega/c3
            if isR :
                del3 = dltarR(d,a,b,rho,wvno,omega ,LN)
            else:
                del3 = dltarL(d,a,b,rho,wvno,omega,LN)
            nev = 1
            m = 0
        else:
            if(nev==2):
                X[m+1] = c3
                Y[m+1] = del3
            else:
                X[0] = c1
                Y[0] = del1
                X[1] = c2
                Y[1] = del2
                m = 0
        '''
        c-----
        c     perform Neville iteration. Note instead of generating y(x)
        c     we interchange the x and y of formula to solve for x(y) when
        c     y = 0
        c-----
        '''
        isBreak =False
        for kk in xrange(0,m+1):
            j = m-kk+1
            denom = Y[m+1] - Y[j]
            if not (abs(denom)<1e-10*abs(Y[m+1])):
                X[j]=(-Y[j]*X[j+1]+Y[m+1]*X[j])/denom
            else:
                c3 = 0.5*(c1 + c2)
                wvno=omega/c3
                if isR :
                    del3 = dltarR(d,a,b,rho,wvno,omega ,LN)
                else:
                    del3 = dltarL(d,a,b,rho,wvno,omega,LN)
                nev = 1
                m = 1
                isBreak=True
                break
        if not isBreak:
            c3 = X[0]
            wvno = omega/c3
            if isR :
                del3 = dltarR(d,a,b,rho,wvno,omega,LN)
            else:
                del3 = dltarL(d,a,b,rho,wvno,omega,LN)
            nev = 2
            m = m + 1
            if(m>9):
                m = 9
    return c3

cdef int var(double p, double q,double ra,double rb,double wvno,double xka,double xkb,double dpth):#,double&a0,double&cpcq,double&cpy,double&cpz,double&cqw,double&cqx,double&xy,double&xz,double&wy,double&wz):
    #var(p,q,ra,rb,wvno,xka,xkb,dpth)
    global a0, w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    '''
    c-----
    c   find variables cosP, cosQ, sinP, sinQ, etc.
    c   as well as cross products required for compound matrix
    c-----
    c   To handle the hyperbolic functions correctly for large
    c   arguments, we use an extended precision procedure,
    c   keeping in mind that the maximum precision in double
    c   precision is on the order of 16 decimal places.
    c
    c   So  cosp = 0.5 ( exp(+p) + exp(-p))
    c            = exp(p) * 0.5 * ( 1.0 + exp(-2p) )
    c   becomes
    c       cosp = 0.5 * (1.0 + exp(-2p) ) with an exponent p
    c   In performing matrix multiplication, we multiply the modified
    c   cosp terms and add the exponents. At the last step
    c   when it is necessary to obtain a true amplitude,
    c   we then form exp(p). For normalized amplitudes at any depth,
    c   we carry an exponent for the numerator and the denominator, and
    c   scale the resulting ratio by exp(NUMexp - DENexp)
    c
    c   The propagator matrices have three basic terms
    c
    c   HSKA        cosp  cosq
    c   DUNKIN      cosp*cosq     1.0
    c
    c   When the extended floating point is used, we use the
    c   largest exponent for each, which is  the following:
    c
    c   Let pex = p exponent > 0 for evanescent waves = 0 otherwise
    c   Let sex = s exponent > 0 for evanescent waves = 0 otherwise
    c   Let exa = pex + sex
    c
    c   Then the modified matrix elements are as follow:
    c
    c   Haskell:  cosp -> 0.5 ( 1 + exp(-2p) ) exponent = pex
    c             cosq -> 0.5 ( 1 + exp(-2q) ) * exp(q-p)
    c                                          exponent = pex
    c          (this is because we are normalizing all elements in the
    c           Haskell matrix )
    c    Compound:
    c            cosp * cosq -> normalized cosp * cosq exponent = pex + qex
    c             1.0  ->    exp(-exa)
    c-----
    '''
    global a0, w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    global ca01,ca02,ca03,ca04,ca00
    global ca11,ca12,ca13,ca14,ca10
    global ca21,ca22,ca23,ca24,ca20
    global ca31,ca32,ca33,ca34,ca30
    global ca41,ca42,ca43,ca44,ca40
    cdef double sinp,x,pex,fac,sex,y,z,cosq,sinq,exa
    a0=1.0
    #examine P-wave eigenfunctions
    #checking whether c> vp c=vp or c < vp
    pex =0 #zeros(1)
    sex= 0#zeros(1)
    if wvno < xka:
        sinp = sin(p)
        w=sinp/ra
        x=-ra*sinp
        cosp=cos(p)
    elif wvno==xka:
        cosp =1# ones(1)
        w = dpth
        x = 0.0
    elif(wvno>xka):
        pex = p
        #fac = zeros(1)
        #if p < 16:
        fac = exp(-2*p)
        cosp = ( 1.00 + fac) * 0.500
        sinp = ( 1.000 - fac) * 0.500
        w=sinp/ra
        x=ra*sinp
    #examine S-wave eigenfunctions
    # #checking whether c > vs, c = vs, c < vs
    if(wvno < xkb):
        sinq=sin(q)
        y=sinq/rb
        z=-rb*sinq
        cosq=cos(q)
    elif(wvno==xkb):
        cosq=1
        y=dpth
        z=0
    elif(wvno > xkb):
        sex = q
        #fac = 0.000
        #if q < 16:
        fac = exp(-2.0*q)
        cosq = ( 1.000 + fac ) * 0.500
        sinq = ( 1.000 - fac ) * 0.500
        y = sinq/rb
        z = rb*sinq
    #form eigenfunction products for use with compound matrices
    exa = pex + sex
    #a0=0.000
    #if exa<60.000: 
    a0=exp(-exa)
    cpcq=cosp*cosq
    cpy=cosp*y
    cpz=cosp*z
    cqw=cosq*w
    cqx=cosq*x
    xy=x*y
    xz=x*z
    wy=w*y
    wz=w*z
    qmp = sex - pex
    #fac = zeros(1)
    #if qmp>-40.000:
    fac = exp(qmp)
    cosq = cosq*fac
    y=fac*y
    z=fac*z
    #print(a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
    return 0

cdef int dnka(double wvno2,double gam,double gammk,double rho):#double&a0,double&cpcq,double&cpy,double&cpz,double&cqw,double&cqx,double&xy,double&xz,double&wy,double&wz):
    #(wvno2,gam,gammk,rho1,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
    #ca=dnka(wvno2,gam,gammk,rho1,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz)
    #print(rho,a0,cpcq)
    global a0, w,cosp,a0,cpcq,cpy,cpz,cqw,cqx,xy,xz,wy,wz
    global ca01,ca02,ca03,ca04,ca00
    global ca11,ca12,ca13,ca14,ca10
    global ca21,ca22,ca23,ca24,ca20
    global ca31,ca32,ca33,ca34,ca30
    global ca41,ca42,ca43,ca44,ca40
    cdef double one,two,gamm1,twgm1,gmgmk,gmgm1,gm1s1,rho2,a0pq,t
    one=1.0
    two=2.0
    gamm1 = gam-one
    twgm1=gam+gamm1
    gmgmk=gam*gammk
    gmgm1=gam*gamm1
    gm1sq=gamm1*gamm1
    rho2=rho*rho
    a0pq=a0-cpcq
    #
    ca00=cpcq-two*gmgm1*a0pq-gmgmk*xz-wvno2*gm1sq*wy
    ca01=(wvno2*cpy-cqx)/rho
    ca02=-(twgm1*a0pq+gammk*xz+wvno2*gamm1*wy)/rho
    ca03=(cpz-wvno2*cqw)/rho
    ca04=-(two*wvno2*a0pq+xz+wvno2*wvno2*wy)/rho2
    ca10=(gmgmk*cpz-gm1sq*cqw)*rho
    ca11=cpcq
    ca12=gammk*cpz-gamm1*cqw
    ca13=-wz
    ca14=ca03
    ca30=(gm1sq*cpy-gmgmk*cqx)*rho
    ca31=-xy
    ca32=gamm1*cpy-gammk*cqx
    ca33=ca11
    ca34=ca01
    ca40=-(two*gmgmk*gm1sq*a0pq+gmgmk*gmgmk*xz+gm1sq*gm1sq*wy)*rho2
    ca41=ca30
    ca42=-(gammk*gamm1*twgm1*a0pq+gam*gammk*gammk*xz+gamm1*gm1sq*wy)*rho
    ca43=ca10
    ca44=ca00
    t=-two*wvno2
    ca20=t*ca42
    ca21=t*ca32
    ca22=a0+two*(cpcq-ca00)
    ca23=t*ca12
    ca24=t*ca02
    #print(type(ca10),type(ca11),type(ca12),type(ca13),type(ca14))
    '''
    ca = cat(\
        [
    torch.cat([ca00.reshape([-1]), ca01.reshape([-1]), ca02.reshape([-1]), ca03.reshape([-1]), ca04.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca10.reshape([-1]), ca11.reshape([-1]), ca12.reshape([-1]), ca13.reshape([-1]), ca14.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca20.reshape([-1]), ca21.reshape([-1]), ca22.reshape([-1]), ca23.reshape([-1]), ca24.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca30.reshape([-1]), ca31.reshape([-1]), ca32.reshape([-1]), ca33.reshape([-1]), ca34.reshape([-1])]).reshape([1,-1]),
    torch.cat([ca40.reshape([-1]), ca41.reshape([-1]), ca42.reshape([-1]), ca43.reshape([-1]), ca44.reshape([-1])]).reshape([1,-1]),
    ]
    )
    '''
    '''
    ca = Data(\
        [
    [ca00, ca01, ca02, ca03, ca04],
    [ca10, ca11, ca12, ca13, ca14],
    [ca20, ca21, ca22, ca23, ca24],
    [ca30, ca31, ca32, ca33, ca34],
    [ca40, ca41, ca42, ca43, ca44],
    ]
    )
    '''
    return 0

