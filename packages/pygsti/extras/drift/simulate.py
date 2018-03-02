from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np
import numpy.random as _rnd

from . import signal as _sig

def generate_drifting_data(probs,sample_times,coursegrain=1):
    """
    TODO: docstring
    """
    
    params = _np.shape(sample_times)
    S = params[0]
    times = params[1]
    Q = len(list(probs.keys()))//S
    T = times//coursegrain
    
    raw_simdata = _np.zeros((S,Q,times),float)
    coursegrained_simdata = _np.zeros((S,Q,T),float)
    coursegrained_times = _np.zeros((S,T),float)
    
    prob_array = _np.zeros((S,Q,times),float)
    
    for s in range(0,S):
        for q in range(0,Q):
            for t in range(0,times):
                prob_array[s,q,t] = probs[s,q](sample_times[s,t])
                
    raw_simdata = _rnd.binomial(1,prob_array)
     
    for t in range(0,T):
        x = coursegrain*t
        coursegrained_simdata[:,:,t] = _np.sum(raw_simdata[:,:,x:x+coursegrain],axis=2)
        coursegrained_times[:,t] =  _np.mean(sample_times[:,x:x+coursegrain],axis=1)
                
    return coursegrained_simdata, coursegrained_times, prob_array

#def generate_drifting_data(prob,counts):
#    """
#    TODO: docstring
#    """
#    pshape = _np.shape(prob)
#    S = pshape[0]
#    Q = pshape[1]
#    T = pshape[2]
#    
#    simdata = np.zeros(pshape,float)
#    
#    for s in range(0,S):
#        for q in range(0,Q):
#            for t in range(0,T):
#                simdata[s,q,t] = rnd.binomial(N,prob[s,q,t])
#    
#    return simdata

def generate_minsparsity_signal(power, num_modes, n, max_sigreq=None, base = 0.5, 
                                      renormalizer_method='sharp'):   
    """
    TODO: docstring
    """    
    if max_sigreq is None:
        max_sigreq = n-1
        
    amplitude_per_mode = _np.sqrt(power/num_modes)
    possible_sigrequencies = _np.arange(1,max_sigreq+1)
    sampled_sigrequencies = _np.random.choice(possible_sigrequencies, size=num_modes, replace=False, p=None)
    
    modes = _np.zeros(n,float)
    random_phases = _np.random.binomial(1,0.5,size=num_modes)
    for i in range (0,num_modes):
        modes[sampled_sigrequencies[i]] = amplitude_per_mode*(-1)**random_phases[i]
        
    p =  _sig.IDCT(modes,base*_np.ones(n))    

    if renormalizer_method is not None:
        p = _sig.renormalizer(p,method=renormalizer_method)
        
    return p


def generate_gaussianpower_signal(power,center,spread,N,base=0.5,renormalizer_method='sharp'):
    """
    TODO: docstring
    """
    modes = _np.zeros(N)
    modes[0] = 0.
    modes[1:] = _np.exp(-1*(_np.arange(1,N)-center)**2/(2*spread**2))
    modes = modes*(-1)**_np.random.binomial(1,0.5,size=N)
    modes = _np.sqrt(power)*modes/_np.sqrt(sum(modes**2))
    
    p = _sig.IDCT(modes,base*_np.ones(N))
    
    if renormalizer_method is not None:
        p = _sig.renormalizer(p,method=renormalizer_method)
    
    return p