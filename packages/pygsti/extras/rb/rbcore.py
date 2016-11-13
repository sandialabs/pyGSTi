from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Randomized Benhmarking Core Routines """

from ... import construction as _cnst
from ... import objects as _objs
from ... import io as _io
from ... import tools as _tools
from . import rbutils as _rbutils
from . import rbobjs as _rbobjs

import itertools as _itertools
import numpy as _np
from numpy import random as _rndm
from scipy.optimize import minimize as _minimize
from collections import OrderedDict as _OrderedDict


def create_random_rb_clifford_string(m, clifford_group, 
                                     seed=None, randState=None):
    """
    Generate a random RB clifford sequence.
    
    Parameters
    ----------
    m : int
        Sequence length is m+1 (because m Cliffords are chosen at random,
        then one additional Clifford is selected to invert the sequence).

    clifford_group : MatrixGroup
        Which Clifford group to use.
    
    seed : int, optional
        Seed for the random number generator.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    
    Returns
    -------
    clifford_string : list
        Random Clifford sequence of length m+1.  For ideal Cliffords, the
        sequence implements the identity operation.
    """
    if randState is None:
        rndm = _rndm.RandomState(seed) # ok if seed is None
    else:
        rndm = randState

    rndm_indices = rndm.randint(0,len(clifford_group),m)
    cliff_lbl_string = [ clifford_group.labels[i] for i in rndm_indices ]    
    effective_cliff_lbl = clifford_group.product(cliff_lbl_string)
    cliff_inv = clifford_group.get_inv(effective_cliff_lbl)
    cliff_lbl_string.append( cliff_inv )
    return _objs.GateString(cliff_lbl_string)


def list_random_rb_clifford_strings(m_min, m_max, Delta_m, clifford_group,
                                    K_m_sched, alias_maps=None, seed=None,
                                    randState=None):
    """
    Makes a list of random RB sequences.
    
    Parameters
    ----------
    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.

    clifford_group : MatrixGroup
        Which Clifford group to use.

    K_m_sched : int or dict
        If an integer, the fixed number of Clifford sequences to be sampled at
        each length m.  If a dictionary, then a mapping from Clifford
        sequence length m to number of Cliffords to be sampled at that length.
    
    alias_maps : dict of dicts, optional
        If not None, a dictionary whose keys name other gate-label-sets, e.g.
        "primitive" or "canonical", and whose values are "alias" dictionaries 
        which map the clifford labels (defined by `clifford_group`) to those
        of the corresponding gate-label-set.  For example, the key "canonical"
        might correspond to a dictionary "clifford_to_canonical" for which 
        (as one example) clifford_to_canonical['Gc1'] == ('Gy_pi2','Gy_pi2').
            
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    
    Returns
    -------
    dict or list
        If `alias_maps` is not None, a dictionary of lists-of-gatestring-lists
        whose keys are 'clifford' and all of the keys of `alias_maps` (if any).
        Values are lists of `GateString` lists, one for each K_m value.  If
        `alias_maps` is None, then just the list-of-lists corresponding to the 
        clifford gate labels is returned.
    """

    if randState is None:
        rndm = _rndm.RandomState(seed) # ok if seed is None
    else:
        rndm = randState

    if isinstance(K_m_sched,int):
        K_m_sched_dict = {m : K_m_sched 
                          for m in range(m_min, m_max+1,Delta_m) }
    else: K_m_sched_dict = K_m_sched
    assert hasattr(K_m_sched_dict, 'keys'),'K_m_sched must be a dict or int!'

    string_lists = {'clifford': []} # GateStrings with Clifford-group labels
    if alias_maps is not None:
        for gstyp in alias_maps.keys(): string_lists[gstyp] = []

    for m in range(m_min,m_max+1,Delta_m):
        K_m = K_m_sched_dict[m]
        strs_for_this_m = [ create_random_rb_clifford_string(
            m,clifford_group,randState=rndm) for i in range(K_m) ]
        string_lists['clifford'].append(strs_for_this_m)
        if alias_maps is not None:
            for gstyp,alias_map in alias_maps.items(): 
                string_lists[gstyp].append(
                    _cnst.translate_gatestring_list(strs_for_this_m,alias_map))

    if alias_maps is None:
        return string_lists['clifford'] #only list of lists is clifford one
    else:
        return string_lists #note we also return this if alias_maps == {}


def write_empty_rb_files(filename, m_min, m_max, Delta_m, clifford_group, K_m,
                         alias_maps=None, seed=None, randState=None):
    """
    A wrapper for list_random_rb_clifford_strings which also writes output
    to disk.

    This function returns the same value as list_random_rb_clifford_strings,
    and also:

    - saves the clifford strings in an empty data set file by adding ".txt"
      to `filename`.
    - saves each set of strings to a gatestring list text file by adding
      "_<gate-label-set-name>.txt" to `filename`.  
      
    For example, if "primitive" is the only key of `alias_maps`, and 
    `filename` is set to "test", then the following files are created:

    - "test.txt" (empty dataset with clifford-labelled strings)
    - "test_clifford.txt" (gate string list with clifford-label strings)
    - "test_primitive.txt" (gate string list with primitive-label strings)

    Parameters
    ----------
    filename : str
        The base name of the files to create (see above).

    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.

    clifford_group : MatrixGroup
        Which Clifford group to use.

    K_m_sched : int or dict
        If an integer, the fixed number of Clifford sequences to be sampled at
        each length m.  If a dictionary, then a mapping from Clifford
        sequence length m to number of Cliffords to be sampled at that length.
    
    alias_maps : dict of dicts, optional
        If not None, a dictionary whose keys name other gate-label-sets, e.g.
        "primitive" or "canonical", and whose values are "alias" dictionaries 
        which map the clifford labels (defined by `clifford_group`) to those
        of the corresponding gate-label-set.  For example, the key "canonical"
        might correspond to a dictionary "clifford_to_canonical" for which 
        (as one example) clifford_to_canonical['Gc1'] == ('Gy_pi2','Gy_pi2').
            
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    

    Returns
    -------
    dict or list
        If `alias_maps` is not None, a dictionary of lists-of-gatestring-lists
        whose keys are 'clifford' and all of the keys of `alias_maps` (if any).
        Values are lists of `GateString` lists, one for each K_m value.  If
        `alias_maps` is None, then just the list-of-lists corresponding to the 
        clifford gate labels is returned.
    """
    # line below ensures random_string_lists is *always* a dictionary
    alias_maps_mod = {} if (alias_maps is None) else alias_maps      
    random_string_lists = \
        list_random_rb_clifford_strings(m_min, m_max, Delta_m, clifford_group,
                                        K_m, alias_maps_mod, seed, randState)
    #always write cliffords to empty dataset (in future have this be an arg?)
    _io.write_empty_dataset(
        filename+'.txt', list(
            _itertools.chain(*random_string_lists['clifford'])))
    for gstyp,strLists in random_string_lists.items():
        _io.write_gatestring_list(filename +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
    if alias_maps is None: 
        return random_string_lists['clifford'] 
          #mimic list_random_rb_clifford_strings return value
    else: return random_string_lists


def do_randomized_benchmarking(dataset, clifford_gatestrings,
                               success_spamlabel = 'plus',
                               weight_data = False,
                               infinite_data = False,
                               dim = 2, pre_avg=True, 
                               clifford_to_primitive = None,
                               clifford_to_canonical = None, 
                               canonical_to_primitive = None,
                               f0 = 0.98, A0 = 0.5, ApB0 = 1.0, 
                               C0= 0.0, f_bnd=[0.,1.], 
                               A_bnd=[0.,1.], ApB_bnd=[0.,1.],
                               C_bnd=[-1.,1.]):
    """
    Computes randomized benchmarking (RB) parameters (but not error bars).

    Parameters
    ----------
    dataset : DataSet
        The data to extract counts from.

    clifford_gatestrings : list of GateStrings
        The complete list of RB sequences in terms of Clifford operations,
        i.e., the labels in each `GateString` denote a Clifford operation.

    weight_data : bool, optional
        Whether or not to compute and use weights for each data point for the fit 
        procedures.  Default is False; only works when pre_avg = True.

    infinite_data : bool, optional
        Whether or not the dataset is generated using no sampling error.  Default is
        False; only works when weight_data = True.

    success_spamlabel : str, optional
        The spam label which denotes the *expected* outcome of preparing,
        doing nothing (or the identity), and measuring.  In the ideal case
        of perfect gates, the probability of seeing this outcome when just
        preparing and measuring (no intervening gates) is 100%.
        
    dim : int, optional
        Hilbert space dimension.  Default is 2, corresponding to a single
        qubit.

    pre_avg : bool, optional
        Whether or not survival probabilities for different sequences of
        the same length are to be averaged together before curve fitting
        is performed.  Some information is lost when performing
        pre-averaging, but it follows the literature.

    clifford_to_primitive : dict, optional
        A dictionary mapping clifford labels to tuples of "primitive" labels.
        If not None, the returned result object contains data for the
        'primitive' gate-label-set.  Cannot be specified along with 
        `canonical_to_primitive`.


    clifford_to_canonical : dict, optional
        A dictionary mapping clifford labels to tuples of "canonical Clifford"
        labels (typically {I, X(pi/2),  X(-pi/2), X(pi), Y(pi/2), Y(-pi/2),
        Y(pi)}).  If not None, the returned result object contains data for the
        'canonical' gate-label-set.

    canonical_to_primitive : dict, optional
        A dictionary mapping "canonical-Clifford" labels, defined by the keys
        of `clifford_to_canonical`  to tuples of "primitive" labels. If not 
        None, the returned result object contains data for the 'primitive'
        gate-label-set. Cannot be specified along with `clifford_to_primitive`.

    f0 : float, optional
        A single floating point number, to be used as the starting
        'f' value for the fitting procedure.  The default value is almost
        always fine, and one should only modifiy this parameter in special
        cases.
        
    A0 : float, optional
        A single floating point number, to be used as the starting
        'A' value for the fitting procedure.  The default value is almost
        always fine, and one should only modifiy this parameter in special
        cases. 
        
    ApB0 : float, optional
        A single floating point number, to be used as the starting
        'A'+'B' value for the fitting procedure.  The default value is almost
        always fine, and one should only modifiy this parameter in special
        cases. 
        
    C0 : float, optional
        A single floating point number, to be used as the starting
        'C' value for the first order fitting procedure.  The default value 
        is almost always fine, and one should only modifiy this parameter in 
        special cases.
        
    f_bnd, A_bnd, ApB_bnd, C_bnd : list, optional
        A 2-element list of floating point numbers. Each list gives the upper
        and lower bounds over which the relevant parameter is minimized. The
        default values are well-motivated and should be almost always fine
        with sufficient data.
        
    Returns
    -------
    RBResults
       A results object containing all of the computed RB values as well as
       information about the inputs to the analysis.  This object can be used
       to compute error bars on the RB values.
    """
    alias_maps = {}
    if clifford_to_canonical is not None:
        alias_maps['canonical'] = clifford_to_canonical
        if canonical_to_primitive is not None:
            alias_maps['primitive'] = _cnst.compose_alias_dicts(
                clifford_to_canonical, canonical_to_primitive)
    
    if clifford_to_primitive is not None:
        assert (canonical_to_primitive is None), \
            "primitive gates specified via clifford_to_primitive AND " + \
            "canonical_to_primitive!"
        alias_maps['primitive'] = clifford_to_primitive

    return do_rb_base(dataset, clifford_gatestrings, "clifford", weight_data, infinite_data,
                      alias_maps, success_spamlabel, dim, pre_avg, f0, A0, ApB0, C0, 
                      f_bnd, A_bnd, ApB_bnd, C_bnd)


def do_rb_base(dataset, base_gatestrings, basename, weight_data=False, infinite_data=False,
               alias_maps=None,
               success_spamlabel = 'plus', dim = 2, pre_avg=True,
               f0 = 0.98, A0 = 0.5, ApB0 = 1.0, C0= 0.0, f_bnd=[0.,1.], 
               A_bnd=[0.,1.], ApB_bnd=[0.,1.], C_bnd=[-1.,1.]):
    """
    Core Randomized Benchmarking compute function.

    This function is more general than `do_randomized_benchmarking` and
    may be useful in atypical situations on its own, or as a building
    block of custom-RB methods.

    Parameters
    ----------
    dataset : DataSet
        The data to extract counts from.

    base_gatestrings : list of GateStrings
        The complete list of RB sequences in terms of "base" operations,
        defined as being the ones that occur in the keys of `dataset`.

    basename : str
        A name given to the "base" gate-label-set, which ultimately determines
        under what key they are stored in the returned results object.

    weight_data : bool, optional
        Whether or not to compute and use weights for each data point for the fit 
        procedures.  Default is False; only works when pre_avg = True.

    infinite_data : bool, optional
        Whether or not the dataset is generated using no sampling error.  Default is
        False; only works when weight_data = True.

    alias_maps : dict of dicts, optional
        If not None, a dictionary whose keys name other (non-"base") 
        gate-label-sets, and whose values are "alias" dictionaries 
        which map the "base" labels to those of the named gate-label-set.
        RB values for each of these gate-label-sets will be present in the 
        returned results object.

    success_spamlabel : str, optional
        The spam label which denotes the *expected* outcome of preparing,
        doing nothing (or the identity), and measuring.  In the ideal case
        of perfect gates, the probability of seeing this outcome when just
        preparing and measuring (no intervening gates) is 100%.
        
    dim : int, optional
        Hilbert space dimension.  Default is 2, corresponding to a single
        qubit.

    pre_avg : bool, optional
        Whether or not survival probabilities for different sequences of
        the same length are to be averaged together before curve fitting
        is performed.  Some information is lost when performing
        pre-averaging, but it follows the literature.

    f0 : float, optional
        A single floating point number, to be used as the starting
        'f' value for the fitting procedure.  The default value is almost
        always fine, and one should only modifiy this parameter in special
        cases.
        
    A0 : float, optional
        A single floating point number, to be used as the starting
        'A' value for the fitting procedure.  The default value is almost
        always fine, and one should only modifiy this parameter in special
        cases. 
        
    ApB0 : float, optional
        A single floating point number, to be used as the starting
        'A'+'B' value for the fitting procedure.  The default value is almost
        always fine, and one should only modifiy this parameter in special
        cases. 
        
    C0 : float, optional
        A single floating point number, to be used as the starting
        'C' value for the first order fitting procedure.  The default value 
        is almost always fine, and one should only modifiy this parameter in 
        special cases.
        
    f_bnd, A_bnd, ApB_bnd, C_bnd : list, optional
        A 2-element list of floating point numbers. Each list gives the upper
        and lower bounds over which the relevant parameter is minimized. The
        default values are well-motivated and should be almost always fine
        with sufficient data.
        
    Returns
    -------
    RBResults
       A results object containing all of the computed RB values as well as
       information about the inputs to the analysis.  This object can be used
       to compute error bars on the RB values.
    """

    def preavg_by_length(lengths, success_probs, Ns):
        bins = {}
        for L,p,N in zip(lengths, success_probs, Ns):
            if L not in bins:
                bins[L] = _np.array([L,p,N,1],'d')
            else:
                bins[L] += _np.array([L,p,N,1],'d')
        avgs = { L: ar/ar[3] for L,ar in bins.items() }
        Ls = sorted(avgs.keys())
        preavg_lengths =  [ avgs[L][0] for L in Ls ] 
        preavg_psuccess = [ avgs[L][1] for L in Ls ] 
        preavg_Ns =       [ avgs[L][2] for L in Ls ] 
        return preavg_lengths, preavg_psuccess, preavg_Ns

    def fit(xdata,ydata,weights=None):
        xdata = _np.asarray(xdata) - 1 #discount Clifford-inverse
        ydata = _np.asarray(ydata)
        if weights is None:
            weights = _np.array([1.] * len(xdata))
        def obj_func_full(params):
            A,Bs,f = params
            return _np.sum((A+(Bs-A)*f**xdata-ydata)**2 / weights)

        def obj_func_1d(f):
            A = 0.5
            Bs = 1.
            return obj_func_full([A,Bs,f])
        
        def obj_func_1st_order_full(params):
            A1, B1s, C1, f1 = params
            return _np.sum((A1+(B1s-A1+C1*xdata)*f1**xdata-ydata)**2 / weights)

        initial_soln = _minimize(obj_func_1d,f0, method='L-BFGS-B',
                                 bounds=[(0.,1.)])
        f0b = initial_soln.x[0]
        p0 = [A0,ApB0,f0b] 
        final_soln = _minimize(obj_func_full,p0, method='L-BFGS-B',
                               bounds=[A_bnd,ApB_bnd,f_bnd])      
        A,Bs,f = final_soln.x
        
        p0 = [A,Bs,C0,f]        
        final_soln_1storder = _minimize(obj_func_1st_order_full, p0, 
                               method='L-BFGS-B',
                               bounds=[A_bnd,ApB_bnd,C_bnd,f_bnd])
        
        A1,B1s,C1,f1 = final_soln_1storder.x
        
        if A > 0.6 or A1 > 0.6 or A < 0.4 or A1 < 0.4:
            print("Warning: Asymptotic fit parameter is not within [0.4,0.6].")
                       
        if C1 > 0.1 or C1 < -0.1:
            print("Warning: Additional parameter in first order fit is significantly non-zero")
            
        return {'A': A,'B': Bs-A,'f': f, 'F_avg': _rbutils.f_to_F_avg(f,dim),
                'r': _rbutils.f_to_r(f,dim), 'A1': A1, 'B1': B1s-A1, 'C1': C1,
                'f1': f1, 'F_avg1': _rbutils.f_to_F_avg(f1,dim),
                'r1': _rbutils.f_to_r(f1,dim)}

    result_dicts = {}

    #Note: assumes dataset contains gate strings which use *base* labels
    base_lengths = list(map(len,base_gatestrings))
    occ_indices = _tools.compute_occurance_indices(base_gatestrings)
    Ns = [ dataset.get_row(seq,k).total() 
           for seq,k in zip(base_gatestrings,occ_indices) ]
    successes = [ dataset.get_row(seq,k).fraction(success_spamlabel) 
                  for seq,k in zip(base_gatestrings,occ_indices) ] 

    if pre_avg:
        base_lengths,base_successes,base_Ns = \
            preavg_by_length(base_lengths,successes,Ns)
        if weight_data : 
#            mkn_dict = _rbutils.dataset_to_mkn_dict(dataset,base_gatestrings,success_spamlabel)
#            weight_dict = _rbutils.mkn_dict_to_weighted_delta_f1_hat_dict(mkn_dict)
            if infinite_data == True:
                use_frequencies = True
            else:
                use_frequencies = False
            summary_dict = _rbutils.dataset_to_summary_dict(dataset,base_gatestrings,success_spamlabel,use_frequencies)
#            weight_dict = _rbutils.mkn_dict_to_delta_f1_squared_dict(mkn_dict)
            weight_dict = _rbutils.summary_dict_to_delta_f1_squared_dict(summary_dict, infinite_data)
            weights = _np.array(map(lambda x : weight_dict[x],base_lengths))
            if _np.count_nonzero(weights) != len(weights):
                print("Warning: Zero weight detected!")
        else : 
            weights = None
    else:
        base_lengths,base_successes,base_Ns = base_lengths,successes,Ns
    base_results = fit(base_lengths, base_successes,weights)
    base_results.update({'gatestrings': base_gatestrings,
                          'lengths': base_lengths,
                          'successes': base_successes,
                          'counts': base_Ns })
    result_dicts[basename] = base_results
    
    for gstyp,alias_map in alias_maps.items():
        if alias_map is None: continue #skip when map is None

        gstyp_gatestrings = _cnst.translate_gatestring_list(
            base_gatestrings, alias_map)
        gstyp_lengths = list(map(len,gstyp_gatestrings))

        if pre_avg:
            gstyp_lengths,gstyp_successes,gstyp_Ns = \
                preavg_by_length(gstyp_lengths,successes,Ns)
        else:
            gstyp_lengths,gstyp_successes,gstyp_Ns = gstyp_lengths,successes,Ns
        gstyp_results = fit(gstyp_lengths, gstyp_successes)
        gstyp_results.update({'gatestrings': gstyp_gatestrings,
                              'lengths': gstyp_lengths,
                              'successes': gstyp_successes,
                              'counts': gstyp_Ns })
        result_dicts[gstyp] = gstyp_results

    results = _rbobjs.RBResults(dataset, result_dicts, basename, alias_maps,
                                success_spamlabel, dim, pre_avg, f0, A0, ApB0, C0)
    return results


def generate_sim_rb_data(gateset, expRBdataset, seed=None):
    """
    Creates a DataSet using the gate strings from a given experimental RB
    DataSet and probabilities generated from a given GateSet.

    Parameters
    ----------
    gateset : GateSet
       The gate set used to generate probabilities

    expRBdataset : DataSet
      The data set used to specify which gate strings to compute counts for.
      Usually this is an experimental RB data set.

    seed : int, optional
       Seed for numpy's random number generator.

    Returns
    -------
    DataSet
    """
    gateStrings = list(expRBdataset.keys(stripOccuranceTags=True))
    Ns = [ expRBdataset[s].total() for s in gateStrings ]
    return _cnst.generate_fake_data(gateset,gateStrings,Ns,sampleError='multinomial',
                                    collisionAction=expRBdataset.collisionAction)


def generate_sim_rb_data_perfect(gateset,expRBdataset,N=1e6):
    """
    Creates a "perfect" DataSet using the gate strings from a given
    experimental RB DataSet and probabilities generated from a given GateSet.
    "Perfect" here means the generated counts have no sampling error.

    Parameters
    ----------
    gateset : GateSet
       The gate set used to generate probabilities

    expRBdataset : DataSet
      The data set used to specify which gate strings to compute counts for.
      Usually this is an experimental RB data set.

    N : int, optional
       The (uniform) number of samples to use.

    Returns
    -------
    DataSet
    """
    gateStrings = list(expRBdataset.keys(stripOccuranceTags=True))
    return _cnst.generate_fake_data(gateset,gateStrings,N,sampleError='none',
                                    collisionAction=expRBdataset.collisionAction)

