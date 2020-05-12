import numpy as np
from math import factorial
from scipy.misc import *
from scipy.special import factorial2
import math
from scipy.special import eval_hermite as H
from scipy.special import binom
import matplotlib.pyplot as plt
import functools
def X_chang(n,n_p,beta=1):
    alpha=1.
    alphap=1.*beta*beta
    # print(alpha,alphap)
    d=0
    A=2*np.sqrt(float(alpha*alphap))/(alpha+alphap)
    F=A/(float(factorial(n)*factorial(n_p))*(2**(n+n_p)))
    def I(K):
        if K % 2 != 0:
            return 0.
        else:
            return factorial2(int(K-1))/np.sqrt((alpha+alphap)**(K))
    S=0.
    for k in range(n+1):
        for kp in range(n_p+1):
            O1=binom(n_p, kp)*binom(n,k)
            O2=H(n_p-kp,0)*H(n-k,0)
            # print(k,kp,O2)
            O3=((2*np.sqrt(alphap))**kp)*((2*np.sqrt(alpha))**k)
            S=S+O1*O2*O3*I(kp+k)
    return np.sqrt(F)*S

def X(l,k,beta=10):
    b1=np.sqrt(2.*beta/(1.+(beta**2.)))
        # print((1-pow(beta,2.))/(2.*(1.+pow(beta,2.))))
        # b2=pow((1-pow(beta,2.))/(2.*(1.+pow(beta,2.))),((l+k)/2))
    b2=((1-beta**2.)/(2.*(1.+beta**2.)))**int(((l+k)/2))
        # print((1-beta**2.)/(2.*(1.+beta**2.)),(l+k)/2,b2)
        # print((l+k)/2)

    b3=np.sqrt(float(factorial(k)*factorial(l)))
    def fun(j):
        s1=(4.*beta/(1.-beta**2.))**j
        s2=(-1.j)**(l-j)
        s3=factorial(k-j)
        s4=factorial(l-j)
        s5=factorial(j)
        return s1*s2*H(l-j,0)*H(k-j,0)/(s3*s4*s5)
    out=list(map(functools.partial(fun), (range(min(l,k)+1))))
    return b1*b2*b3*np.sum(out)
n=0
n_p=1
print('FC ('+str(n)+str(n_p)+') : ', X_chang(n,n_p))

plt.plot(range(10),list(map(lambda x: X_chang(n,x,beta=0.5),range(10))),label='chang')
plt.plot(range(10),list(map(lambda x: X(n,x,beta=0.5),range(10))),label='palma')
plt.legend()
plt.show()
