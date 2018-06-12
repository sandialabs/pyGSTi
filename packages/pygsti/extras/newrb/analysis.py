from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
from scipy.optimize import minimize as _minimize
from scipy.optimize import curve_fit as _curve_fit

def crb_rescaling_factor(lengths,quantity):
    
    rescaling_factor = []
    
    for i in range(len(lengths)):
        
        rescaling_factor.append(quantity[i]/(lengths[i]+1))
        
    rescaling_factor = _np.mean(_np.array(rescaling_factor))
    
    return rescaling_factor 

def p_to_r(p,n): 
    
    return (4**n-1)*(1-p)/4**n

def r_to_p(r,n):
    
    return 1 - (4**n)*r/(4**n-1)   


def custom_fit_data(lengths, ASPs, n, fixed_A=False, fixed_B=False, seed=None):
    
    # The fit to do if a fixed value for A is given    
    if fixed_A is not False:
        
        A = fixed_A
        
        if fixed_B is not False:
            
            B = fixed_B

            def curve_to_fit(m,p):
                return A + B*p**m
            
            if seed is None:
                seed = 0.9
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.],[1.]))
            p = fitout 
            
        else:
            
            def curve_to_fit(m,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1.-A,0.9]
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([-_np.inf,0.],[+_np.inf,1.]))
            B = fitout[0]
            p = fitout[1]
    
    # The fit to do if a fixed value for A is not given       
    else:
        
        if fixed_B is not False:
            
            B = fixed_B
            
            def curve_to_fit(m,A,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,0.9]
                
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,0.],[1.,1.]))
            A = fitout[0]
            p = fitout[1]
        
        else:
            
            def curve_to_fit(m,A,B,p):
                return A + B*p**m
            
            if seed is None:
                seed = [1/2**n,1-1/2**n,0.9]
                    
            fitout, junk = _curve_fit(curve_to_fit,lengths,ASPs,p0=seed,bounds=([0.,-_np.inf,0.],[1.,+_np.inf,1.]))
            A = fitout[0]
            B = fitout[1]
            p = fitout[2]
   
    results = {}
    results['A'] = A
    results['B'] = B
    results['p'] = p
    results['r'] = p_to_r(p,n)
    
    return results

#def obj_func(params,lengths,ASPs):
#    A,Bs,p = params
#    
#    if p > 1. or p < 0. or A > 1. or A < 0.:
#        return 1e10
#    else:
#        return _np.sum((A+(Bs-A)*p**lengths-ASPs)**2)
#
#def obj_func_fixed_asymptote(params,lengths,ASPs,n=None,asymptote=None):
#    
#    assert(not (asymptote == None and n == None)), "asymptote or n must be specified!"
#    
##    if asymptote is None:
#        A = 1/2**n
#    else:
#        A = asymptote
#    
#    Bs,p = params
#    
#    if p > 1. or p < 0.:
#        return 1e10
#    else:
#        return _np.sum((A+(Bs-A)*p**lengths-ASPs)**2)
#
#def obj_func_fixed_decay_rate(params,lengths,ASPs,p):
#    
#    A,Bs = params
#    if A > 1. or A < 0.:
#        return 1e10
#    else:
#        return _np.sum((A+(Bs-A)*p**lengths-ASPs)**2)
#
#def obj_func_fixed_decay_rate_and_asymptote(params,lengths,ASPs,A,p):    
#    Bs = params
#    return _np.sum((A+(Bs-A)*p**lengths-ASPs)**2)


def std_practice_analysis(lengths, SPs, counts, n, ASPs=None, seed=[0.8,0.95], 
                 bootstrap_samples=500, asymptote='std', finite_sample_error=True):
    
    if ASPs == None:       
        ASPs = _np.mean(_np.array(SPs),axis=1)
    
    if asymptote == 'std':
        asymptote = 1/2**n
    
    full_fit, fixed_asymptote_fit = std_fit_data(lengths,ASPs,n,seed=seed,asymptote=asymptote)
    
    full_fit['r_bootstraps'] = bootstrap(lengths, SPs, n, counts, seed=seed, samples=bootstrap_samples, 
              fixed_asymptote=False,  asymptote=None, finite_sample_error=finite_sample_error)
    fixed_asymptote_fit['r_bootstraps'] = bootstrap(lengths, SPs, n, counts, seed=seed, samples=bootstrap_samples, 
              fixed_asymptote=True,  asymptote=asymptote, finite_sample_error=finite_sample_error)
    
    full_fit['r_std'] = _np.std(_np.array(full_fit['r_bootstraps']))
    fixed_asymptote_fit['r_std'] = _np.std(_np.array(fixed_asymptote_fit['r_bootstraps']))

    return full_fit, fixed_asymptote_fit
    
    
def std_fit_data(lengths,ASPs,n,seed=None,asymptote=None):
    
    # Bounds commented out and hard bounds put into the objective function
    
    #fixed_asymptote_fit_out = _minimize(obj_func_fixed_asymptote, seed, args=(lengths,ASPs,n,asymptote), 
    #                                   method='L-BFGS-B')#,bounds=[[None,None],[0.,1.]]) 
    
    if asymptote is not None:
        A = asymptote
    else:
        A = 1/2**n
    
    fixed_asymptote_fit = custom_fit_data(lengths, ASPs, n, fixed_A=A, fixed_B=False, seed=seed)
   
    seed_full = [fixed_asymptote_fit['A'], fixed_asymptote_fit['B'], fixed_asymptote_fit['p']]        
    
    # Bounds commented out and hard bounds put into the objective function
    
    #full_fit_out = _minimize(obj_func, seed_full, args=(lengths,ASPs),method='L-BFGS-B' ) 
    full_fit =  custom_fit_data(lengths, ASPs, n, fixed_A=False, fixed_B=False, seed=seed_full)
    
    return full_fit, fixed_asymptote_fit

def bootstrap(lengths, SPs, n, counts, seed=None, samples=500, 
              fixed_asymptote=False,  asymptote=None, finite_sample_error=True):
    
    failcount = 0
    r = _np.zeros(samples)    
    
    for i in range(samples): 
        
        # A new set of bootstrapped survival probabilities.
        sampled_SPs = []
        
        try:
            for j in range(len(lengths)):

                sampled_SPs.append([])
                k_at_length = len(SPs[j])

                for k in range(k_at_length):
                    sampled_SPs[j].append(SPs[j][_np.random.randint(k_at_length)])
                if finite_sample_error:   
                    sampled_SPs[j] = _np.random.binomial(counts,sampled_SPs[j])/counts                

            sampled_ASPs = [_np.mean(_np.array(sampled_SPs[k])) for k in range(len(lengths))]

            fit_full, fit_fixed_A = std_fit_data(lengths,sampled_ASPs,n,seed=seed,asymptote=asymptote)

            if fixed_asymptote:
                r[i] = fit_fixed_A['r']
            else:
                r[i] = fit_full['r']
        except:
            failcount += 1
            i = i - 1
            if failcount > samples:
                assert(False),"Bootstrap is failing too often!"
            
    if failcount > 0:
        print("Warning: bootstrap failed {} times".format(failcount))
    return r