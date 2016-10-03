import pygsti
import numpy as _np
from collections import OrderedDict

def H_WF(epsilon,nu):#Taken from Wallman and Flammia- Eq. 9
    return (1./(1-epsilon))**((1.-epsilon)/(nu+1.)) * (float(nu)/(nu+epsilon))**((float(nu)+epsilon)/(nu+1.))

def sigma_m_squared_base_WF(m,r):#Taken from Wallman and Flammia- Eq. 6 (ignoring higher order terms)
    return m**2 * r**2 + 7./4 * m * r**2

def K_WF(epsilon,delta,m,r,sigma_m_squared_func=sigma_m_squared_base_WF):#Taken from Wallman and Flammia- Eq. 10; we round up.
    sigma_m_squared = sigma_m_squared_func(m,r)
    return int(_np.ceil(-_np.log(2./delta) / _np.log(H_WF(epsilon,sigma_m_squared))))

