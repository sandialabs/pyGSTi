from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct
from scipy.stats import chi2 as _chi2
from scipy.optimize import leastsq as _leastsq
from scipy import convolve

# -------------------------------------------------------- #
# ---------- Spectrum and Fourier transform tools -------- #
# -------------------------------------------------------- #

def DCT_multicount_data(x,number_of_counts):
    """
    This function is a horrible hack, and the normalization doesn't have 
    any statistical justification.

    """

    
    if _np.mean(x) == 0 or _np.mean(x) == number_of_counts:
        return _np.zeros(len(x))
    
    mean_number_of_ones = _np.mean(x)
    estimated_coin_bias = mean_number_of_ones / number_of_counts
    return _dct((x - mean_number_of_ones)/_np.sqrt(number_of_counts*estimated_coin_bias*(1-estimated_coin_bias)),norm='ortho')

def DCT(x,null_hypothesis=None):
    """
    Returns the Type-II orthogonal discrete cosine transform of y where 
    
    y[k] = (x[k] - null_hypothesis[k])/sqrt(null_hypothesis[k]*(1-null_hypothesis[k])).
    
    If null_hypothesis is None, then null_hypothesis[k] is mean(x), for all k. This is
    with the exception that when mean(x) = 0 or 1 (when the above y[k] is ill-defined),
    in which case the zero vector is returned.
    
    Parameters
    ----------
    x : array
        Bit string, on which the normalization and discrete cosine transformation is performed.
        
    null_hypothesis : array, optional
        If not None, an array to use in the normalization before the DCT. If None, it is
        taken to be an array in which every element is the mean of x.
                
    Returns
    -------
    array
        The DCT modes described above.


    """
    if null_hypothesis is None:
        null_hypothesis = _np.mean(x)
        if null_hypothesis<=0 or null_hypothesis>=1:
            return _np.zeros(len(x))
        
    else:
        if min(null_hypothesis)<=0 or max(null_hypothesis)>=1:
            raise ValueError("All element of null_hypothesis should be in (0,1)")

    return _dct((x - null_hypothesis)/_np.sqrt(null_hypothesis * (1 - null_hypothesis)),norm='ortho')

def IDCT(modes,null_hypothesis):
    """
    Inverts the DCT function.
    
    Parameters
    ----------
    modes : array
        The fourier modes to be transformed to time-domain.
        
    null_hypothesis : array
        The null_hypothesis vector. For the IDCT it is not optional, and all
        elements of this array must be in (0,1).
        
    Returns
    -------
    array
        Inverse of the DCT function
        
    """
    if min(null_hypothesis)<=0 or max(null_hypothesis)>=1:
            raise ValueError("All element of null_hypothesis should be in (0,1)")
    
    return  _idct(modes,norm='ortho')*_np.sqrt(null_hypothesis * (1 - null_hypothesis)) + null_hypothesis


def bartlett_DCT_spectrum(x,N,num_spectra=10,null_hypothesis=None):
    """
    If N/num_spectra is not an integer, then 
    not all of the data points are used.
    """
    
    length = int(_np.floor(N/num_spectra))
    
    if null_hypothesis is None:
        null_hypothesis = _np.mean(x)*_np.ones(N)
    
    spectra = _np.zeros((num_spectra,length))
    bartlett_spectrum = _np.zeros(length)
    
    for i in range(0,num_spectra):
        spectra[i,:] = DCT(x[i*length:((i+1)*length)],null_hypothesis=null_hypothesis[i*length:((i+1)*length)])**2
        
    bartlett_spectrum = _np.mean(spectra,axis=0)
                
    return bartlett_spectrum

# -------------------------------- #
# ---------- Signal tools -------- #
# -------------------------------- #


def hoyer_sparsity_measure(p):
    n = len(p)
    return (_np.sqrt(n) - _np.linalg.norm(p,1)/_np.linalg.norm(p,2))/(_np.sqrt(n)-1)



def renormalizer(p,method='logistic'):
    
    if method == 'logistic':
    
        mean = _np.mean(p)
        nu = min([1-mean ,mean ]) 
        p = mean - nu + (2*nu)/(1 + _np.exp(-2*(p - mean)/nu))
     
    elif method == 'sharp':
        p[p>1] = 1.
        p[p<0] = 0.
    
    else:
        raise ValueError("method should be 'logistic' or 'sharp'")
        
    return p


def low_pass_filter(data,max_freq = None):
    
    n = len(data) 
    
    if max_freq is None:
        max_freq = min(int(np.ceil(n/10)),50)
        
    modes = _dct(data,norm='ortho')
    
    if max_freq < n - 1:
        modes[max_freq + 1:] = _np.zeros(len(data)-max_freq-1)

    return _idct(modes,norm='ortho')

def moving_average(sequence, width=100):
    seq_length = len(sequence)
    base = convolve(_np.ones(seq_length), _np.ones((int(width),))/float(width), mode='same')
    signal = convolve(sequence, _np.ones((int(width),))/float(width), mode='same')
    return signal/base 

# -------------------------------- #
# ------- Signal generators ------ #
# -------------------------------- #

def signal_with_mininum_fourier_sparsity(power, num_modes, n, max_freq=None, base = 0.5, 
                                      renormalizer_method='sharp'):
    
    if max_freq is None:
        max_freq = n-1
        
    amplitude_per_mode = _np.sqrt(power/num_modes)
    possible_frequencies = _np.arange(1,max_freq+1)
    sampled_frequencies = _np.random.choice(possible_frequencies, size=num_modes, replace=False, p=None)
    
    modes = _np.zeros(n,float)
    random_phases = _np.random.binomial(1,0.5,size=num_modes)
    for i in range (0,num_modes):
        modes[sampled_frequencies[i]] = amplitude_per_mode*(-1)**random_phases[i]
        
    p =  IDCT(modes,base*_np.ones(n))    

    if renormalizer_method is not None:
        p = renormalizer(p,method=renormalizer_method)
        
    return p


def signal_with_gaussian_power_distribution(power,center,spread,N,base=0.5,renormalizer_method='sharp'):

    modes = _np.zeros(N)
    modes[0] = 0.
    modes[1:] = _np.exp(-1*(_np.arange(1,N)-center)**2/(2*spread**2))
    modes = modes*(-1)**_np.random.binomial(1,0.5,size=N)
    modes = _np.sqrt(power)*modes/_np.sqrt(sum(modes**2))
    
    p = IDCT(modes,base*_np.ones(N))
    
    if renormalizer_method is not None:
        p = renormalizer(p,method=renormalizer_method)
    
    return p