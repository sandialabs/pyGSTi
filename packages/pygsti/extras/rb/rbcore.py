""" Randomized Benhmarking Core Routines """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

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

def create_random_gatestring(m, group_or_gateset, inverse = True,
                             interleaved = None, seed=None,
                             group_inverse_only = False,
                             compilation = None,
                             generated_group = None,
                             gateset_to_group_labels = None,
                             randState=None):
    # For "generator RB" need to add a subset sampling option. This would create
    # random sequences of only a sub-set of the gates/elements, but with the inverse
    # whatever it needs to be. Can write a wrapper around this to then compile the inverse
    # into another gateset. Could also add a sub-set sampling option which picks sequences
    # of length m+1 that compile to the identity. The easiest way to do this would be to
    # just reject sequences that don't compose to I, but there are possibly more efficient
    # ways.
    """
    Makes a random RB sequence.
    
    Parameters
    ----------
    m : int
        The number of random gates in the sequence.

    group_or_gateset : GateSet or MatrixGroup
        Which GateSet of MatrixGroup to create the random sequence for. If
        inverse is true and this is a GateSet, the GateSet gates must form
        a group (so in this case it requires the *target gateset* rather than 
        a noisy gateset). When inverse is true, the MatrixGroup for the gateset 
        is generated. Therefore, if inverse is true and the function is called 
        multiple times, it will be much faster if the MatrixGroup is provided.
        
    inverse: Bool, optional
        If true, the random sequence is followed by its inverse gate. The gateset
        must form a group if this is true. If it is true then the sequence
        returned is length m+1 (2m+1) if interleaved is False (True).
        
    interleaved: Str, optional
        If not None, then a gatelabel string. When a gatelabel string is provided,
        every random gate is followed by this gate. So the returned sequence is of
        length 2m+1 (2m) if inverse is True (False).
            
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    
    Returns
    -------
    Gatestring
        The random gate string of length:
        m if inverse = False, interleaved = None
        m + 1 if inverse = True, interleaved = None
        2m if inverse = False, interleaved not None
        2m + 1 if inverse = True, interleaved not None

    """   
    assert hasattr(group_or_gateset, 'gates') or hasattr(group_or_gateset, 
                   'product'), 'group_or_gateset must be a MatrixGroup of Gateset'    
    group = None
    gateset = None
    if hasattr(group_or_gateset, 'gates'):
        gateset = group_or_gateset
    if hasattr(group_or_gateset, 'product'):
        group = group_or_gateset
        
    if randState is None:
        rndm = _rndm.RandomState(seed) # ok if seed is None
    else:
        rndm = randState
        
    if (inverse) and (not group_inverse_only):
        if gateset:
            group = _rbobjs.MatrixGroup(group_or_gateset.gates.values(),
                                  group_or_gateset.gates.keys() )
                      
        rndm_indices = rndm.randint(0,len(group),m)
        if interleaved:
            interleaved_index = group.label_indices[interleaved]
            interleaved_indices = interleaved_index*_np.ones((m,2),int)
            interleaved_indices[:,0] = rndm_indices
            rndm_indices = interleaved_indices.flatten()
        
        random_string = [ group.labels[i] for i in rndm_indices ]    
        effective_gate = group.product(random_string)
        inv = group.get_inv(effective_gate)
        random_string.append( inv )
        
    if (inverse) and (group_inverse_only):
        assert (gateset is not None), "gateset_or_group should be a GateSet!"
        assert (compilation is not None), "Compilation of group elements to gateset needs to be specified!"
        assert (generated_group is not None), "Generated group needs to be specified!"        
        if gateset_to_group_labels is None:
            gateset_to_group_labels = {}
            for gate in gateset.gates.keys():
                assert(gate in generated_group.labels), "gateset labels are not in \
                the generated group! Specify a gateset_to_group_labels dictionary." 
                gateset_to_group_labels = {'gate':'gate'}
        else:
            for gate in gateset.gates.keys():
                assert(gate in gateset_to_group_labels.keys()), "gateset to group labels \
                are invalid!"              
                assert(gateset_to_group_labels[gate] in generated_group.labels), "gateset to group labels \
                are invalid!"              
                
        rndm_indices = rndm.randint(0,len(gateset.gates.keys()),m)
        if interleaved:
                interleaved_index = gateset.gates.keys().index(interleaved)
                interleaved_indices = interleaved_index*_np.ones((m,2),int)
                interleaved_indices[:,0] = rndm_indices
                rndm_indices = interleaved_indices.flatten()
        random_string = [ gateset.gates.keys()[i] for i in rndm_indices ] 
        random_string_group = [ gateset_to_group_labels[gateset.gates.keys()[i]] for i in rndm_indices ] 
        #print(random_string)
        inversion_group_element = generated_group.get_inv(generated_group.product(random_string_group))
        inversion_sequence = compilation[inversion_group_element]
        #print(inversion_sequence)
        random_string.extend(inversion_sequence)
        #print(random_string)
        
    if not inverse:
        if gateset:
            rndm_indices = rndm.randint(0,len(gateset.gates.keys()),m)
            gateLabels = list(gateset.gates.keys())
            if interleaved:
                interleaved_index = gateLabels.index(interleaved)
                interleaved_indices = interleaved_index*_np.ones((m,2),int)
                interleaved_indices[:,0] = rndm_indices
                rndm_indices = interleaved_indices.flatten()           
            random_string = [gateLabels[i] for i in rndm_indices ]
            
        else:
            rndm_indices = rndm.randint(0,len(group),m)
            if interleaved:
                interleaved_index = group.label_indices[interleaved]
                interleaved_indices = interleaved_index*_np.ones((m,2),int)
                interleaved_indices[:,0] = rndm_indices
                rndm_indices = interleaved_indices.flatten()
            random_string = [ group.labels[i] for i in rndm_indices ] 
            
    return _objs.GateString(random_string)

def create_random_gatestrings(m_list, K_m, group_or_gateset, inverse=True, 
                              interleaved = None, alias_maps=None, seed=None, 
                              randState=None):
    """
    Makes a list of random RB sequences.
    
    Parameters
    ----------
    m_list : list or array of ints
        The set of lengths for the random sequences (with the total
        number of Cliffords in each sequence given by m_list + 1). Minimal
        allowed length is therefore 1 (a random CLifford followed by its 
        inverse).

    clifford_group : MatrixGroup
        Which Clifford group to use.

    K_m : int or dict
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
        
    assert hasattr(group_or_gateset, 'gates') or hasattr(group_or_gateset, 
           'product'), 'group_or_gateset must be a MatrixGroup or Gateset'
    
    
    if inverse:
        if hasattr(group_or_gateset, 'gates'):
            group_or_gateset = _rbobjs.MatrixGroup(group_or_gateset.gates.values(),
                                  group_or_gateset.gates.keys())
    if isinstance(K_m,int):
        K_m_dict = {m : K_m for m in m_list }
    else: K_m_dict = K_m
    assert hasattr(K_m_dict, 'keys'),'K_m must be a dict or int!'

    string_lists = {'uncompiled': []} # GateStrings with uncompiled labels
    if alias_maps is not None:
        for gstyp in alias_maps.keys(): string_lists[gstyp] = []

    for m in m_list:
        K = K_m_dict[m]
        strs_for_this_m = [ create_random_gatestring(m, group_or_gateset,
            inverse=inverse,interleaved=interleaved,randState=rndm) for i in range(K) ]
        string_lists['uncompiled'].append(strs_for_this_m)
        if alias_maps is not None:
            for gstyp,alias_map in alias_maps.items(): 
                string_lists[gstyp].append(
                    _cnst.translate_gatestring_list(strs_for_this_m,alias_map))

    if alias_maps is None:
        return string_lists['uncompiled'] #only list of lists is uncompiled one
    else:
        return string_lists #note we also return this if alias_maps == {}

def create_random_interleaved_gatestrings(m_list, K_m, group_or_gateset, interleaved_list,
                                          inverse=True, alias_maps=None):
    
    # Currently no random number generator seed allowed, as needs to have different seed for each
    # call of create_random_gatestrings().
    all_random_string_lists = {}
    alias_maps_mod = {} if (alias_maps is None) else alias_maps      
    random_string_lists = create_random_gatestrings(m_list, K_m, 
                          group_or_gateset,inverse,interleaved = None, 
                          alias_maps = alias_maps_mod,)

    if alias_maps is None: 
        all_random_string_lists['baseline'] = random_string_lists['uncompiled']
    else:
        all_random_string_lists['baseline'] = random_string_lists
        
    for interleaved in interleaved_list:
        random_string_lists = \
                       create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = interleaved, alias_maps = alias_maps_mod)

        if alias_maps is None: 
            all_random_string_lists[interleaved] = random_string_lists['uncompiled']
        else:
            all_random_string_lists[interleaved] = random_string_lists
            
        return all_random_string_lists          

def write_empty_rb_files(filename, m_list, K_m, group_or_gateset, 
                         inverse=True, interleaved_list=None, alias_maps=None, 
                         seed=None, randState=None):
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
    base_filename = filename
    if interleaved_list is not None:
        base_filename = filename+'_baseline' 
        
    # line below ensures random_string_lists is *always* a dictionary
    alias_maps_mod = {} if (alias_maps is None) else alias_maps      
    random_string_lists = \
        create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = None, alias_maps = alias_maps_mod, 
                                  seed=seed, randState=randState)
    #always write uncompiled gates to empty dataset (in future have this be an arg?)
    _io.write_empty_dataset(base_filename+'.txt', list(
            _itertools.chain(*random_string_lists['uncompiled'])))
    for gstyp,strLists in random_string_lists.items():
        _io.write_gatestring_list(base_filename +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
        
    if interleaved_list is None:
        if alias_maps is None: 
            return random_string_lists['uncompiled'] 
            #mimic list_random_rb_clifford_strings return value
        else: return random_string_lists
        
    else:
        all_random_string_lists = {}
        if alias_maps is None: 
            all_random_string_lists['baseline'] = random_string_lists['uncompiled']
        else:
            all_random_string_lists['baseline'] = random_string_lists
        
        for interleaved in interleaved_list:
            # No seed allowed here currently, as currently no way to make it different to
            # the seed for the baseline decay
            filename_interleaved = filename+'_interleaved_'+interleaved
            random_string_lists = \
                       create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = interleaved, alias_maps = alias_maps_mod)
            _io.write_empty_dataset(filename_interleaved+'.txt', list(
                _itertools.chain(*random_string_lists['uncompiled'])))
            for gstyp,strLists in random_string_lists.items():
                _io.write_gatestring_list(filename_interleaved +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
                
            if alias_maps is None: 
                all_random_string_lists[interleaved] = random_string_lists['uncompiled']
            else:
                all_random_string_lists[interleaved] = random_string_lists
            
        return all_random_string_lists   

def do_randomized_benchmarking(dataset, gatestrings,
                               fit = 'standard',
                               success_outcomelabel = 'plus',
                               fit_parameters_dict = None,
                               dim = 2, 
                               weight_data = False,
                               pre_avg = True,
                               infinite_data = False,
                               one_freq_adjust=False):
    
    """
    Computes randomized benchmarking (RB) parameters (but not error bars).

    Parameters
    ----------
    dataset : DataSet
        The data to extract counts from.

    gatestrings : list of GateStrings
        The complete list of RB sequences in terms of Clifford operations,
        i.e., the labels in each `GateString` denote a Clifford operation.

    weight_data : bool, optional
        Whether or not to compute and use weights for each data point for the fit 
        procedures.  Default is False; only works when pre_avg = True.

    infinite_data : bool, optional
        Whether or not the dataset is generated using no sampling error.  Default is
        False; only works when weight_data = True.
        
    one_freq_adjust : bool, optional
        TODO: argument description

    success_outcomelabel : str, optional
        The outcome label which denotes the *expected* outcome of preparing,
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
#   alias_maps = {}
#    if clifford_to_canonical is not None:
#        alias_maps['canonical'] = clifford_to_canonical
#        if canonical_to_primitive is not None:
#            alias_maps['primitive'] = _cnst.compose_alias_dicts(
#                clifford_to_canonical, canonical_to_primitive)
    
#    if clifford_to_primitive is not None:
#        assert (canonical_to_primitive is None), \
#            "primitive gates specified via clifford_to_primitive AND " + \
#            "canonical_to_primitive!"
#        alias_maps['primitive'] = clifford_to_primitive

    return do_rb_base(dataset, gatestrings, fit, fit_parameters_dict,
                      success_outcomelabel, dim, weight_data, 
                      pre_avg, infinite_data, one_freq_adjust)

def do_rb_base(dataset, gatestrings, fit = 'standard',fit_parameters_dict = None, 
               success_outcomelabel = 'plus', dim = 2, weight_data=False,pre_avg=True,
               infinite_data=False, one_freq_adjust=False):
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

    weight_data : bool, optional
        Whether or not to compute and use weights for each data point for the fit 
        procedures.  Default is False; only works when pre_avg = True.

    infinite_data : bool, optional
        Whether or not the dataset is generated using no sampling error.  Default is
        False; only works when weight_data = True.

    one_freq_adjust : bool, optional
        TODO: argument description

    success_outcomelabel : str, optional
        The outcome label which denotes the *expected* outcome of preparing,
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

    def do_fit(xdata,ydata,fit='standard',fit_parameters_dict=None,one_freq_dict=None,weights=None):
        xdata = _np.asarray(xdata) - 1 #discount Clifford-inverse
        ydata = _np.asarray(ydata)
        if weights is None:
            weights = _np.array([1.] * len(xdata))
        if one_freq_dict is None:
            def obj_func_full(params):
                A,Bs,f = params
                return _np.sum((A+(Bs-A)*f**xdata-ydata)**2 / weights)
        
            def obj_func_1st_order_full(params):
                A1, B1s, C1, f1 = params
                return _np.sum((A1+(B1s-A1+C1*xdata)*f1**xdata-ydata)**2 / weights)
        else:
            xdata_correction = _np.array(one_freq_dict['m_list']) - 1
            n_0_list = one_freq_dict['n_0_list']
            N_list = one_freq_dict['N_list']
            K_list = one_freq_dict['K_list']
            ydata_correction = []
            weights_correction = []
            indices_to_delete = []
            for m in xdata_correction:
                for i,j in enumerate(xdata):
                    if j==m:
                        ydata_correction.append(ydata[i])
                        weights_correction.append(weights[i])
#                        indices_to_delete.append([i])
                        indices_to_delete.append(i)
            ydata_correction = _np.array(ydata_correction)
            xdata = _np.delete(xdata,indices_to_delete)
            ydata = _np.delete(ydata,indices_to_delete)
            weights = _np.delete(weights,indices_to_delete)
#            print(one_freq_dict)
#            print('xdata:')
#            print(xdata)
#            print('ydata:')
#            print(ydata)
#            print('weights:')
#            print(weights)
            case_1_dict = {}
            case_3_dict = {}
            for m, n_0, N, K in zip(xdata_correction,n_0_list,N_list,K_list):            
                case_1_dict[m,n_0,N,K] = (n_0-1)**(n_0-1)*(N-n_0)**(N-n_0) / ((N-1)**(N-1))
                case_3_dict[m,n_0,N,K] = (n_0**n_0 * (N-n_0-1)**(N-n_0-1))/((N-1)**(N-1))
            def obj_func_full(params):
                A,Bs,f = params
###                print("A="+str(A))
###                print("Bs="+str(Bs))
###                print("f="+str(f))
                if (((A < 0 or A > 1) or (Bs < 0 or Bs > 1)) or (f <= 0 or f >= 1)):
                    return 1e6
                total_0 = _np.sum((A+(Bs-A)*f**xdata-ydata)**2 / weights)
#                correction_1 = _np.sum((A+(Bs-A)*f**xdata_correction - ydata_correction)**2  / weights_correction)
                correction_2 = 0
                for m, n_0, N, K in zip(xdata_correction,n_0_list,N_list,K_list):
#                    print("type(A)="+str(type(A)))
#                    print("type(Bs)="+str(type(Bs)))
#                    print("type(f)="+str(type(f)))
#                    print("type(m)="+str(type(m)))
#                    print("type(n_0)="+str(type(n_0)))
#                    print("type(N)="+str(type(N)))
#                    print("type(K)="+str(type(K)))
                    F = A+(Bs-A)*f**m
#                    print("A="+str(A))
#                    print("Bs="+str(Bs))
#                    print("f="+str(f))
#                    print("m="+str(m))
                    try:
                        assert len(F)==1
                        F=F[0]
                    except:
                        pass
#                    print("F="+str(F))
#                    print("K="+str(K))
#                    print("n_0="+str(n_0))
#                    print("N="+str(N))
                    if F < (n_0 - 1) / (N - 1):
###                        print('Case 1 m='+str(m))
###                        print('F='+str(F))
###                        print(case_1_dict[m,n_0,N,K])
#                        print("type(F)="+str(type(F)))
#                        print("F="+str(F))
#                        print("K="+str(K))
#                        print("n_0="+str(n_0))
#                        print("N="+str(N))
#                        correction_2 += K * _np.log(F * (n_0-1)**(n_0-1)*(N-n_0)**(N-n_0) / ((N-1)**(N-1)))#+1e-10)
                        correction_2 += K * _np.log(F * case_1_dict[m,n_0,N,K])#+1e-10
##                        correction_2 += (F * (n_0-1)**(n_0-1)*(N-n_0)**(N-n_0) / ((N-1)**(N-1)))**K
                    elif F >= (n_0 - 1) / (N - 1) and F < n_0 / (N-1):
###                        print('Case 2 m='+str(m))
###                        print('F='+str(F))
###                        print('1-F='+str(1-F))
###                        print(F**n_0 * (1-F)**(N-n_0))
#                        print(case_1_dict[m,n_0,N,K])
#                        print("F="+str(F))
#                        print("K="+str(K))
#                        print("n_0="+str(n_0))
#                        print("N="+str(N))
                        correction_2 += K * _np.log(F**n_0 * (1-F)**(N-n_0))
##                        correction_2 += (F**n_0 * (1-F)**(N-n_0))**K
                    elif F>= n_0 / (N-1):
###                        print('Case 3 m='+str(m))
###                        print('1-F='+str(1-F))
###                        print(case_3_dict[m,n_0,N,K])
                        correction_2 += K * _np.log((1-F) * case_3_dict[m,n_0,N,K])#+1e-10)
#                        correction_2 += K * _np.log((1-F) * (n_0**n_0 * (N-n_0-1)**(N-n_0-1))/((N-1)**(N-1)))#+1e-10)
##                        correction_2 += ((1-F) * (n_0**n_0 * (N-n_0-1)**(N-n_0-1))/((N-1)**(N-1)))**K
                    else:
                        #print((n_0 - 1) / (N - 1))
                        #print(n_0 / (N-1))
                        #print("F="+str(F))
                        #print("A="+str(A))
                        #print("Bs="+str(Bs))
                        #print("f="+str(f))
                        #print("m="+str(m))
                        raise ValueError("F does not fall within any physical bounds for m="+str(m)+"!")
#                return total_0 - correction_1 + correction_2
                return total_0 - correction_2

            def obj_func_1st_order_full(params):
#                print("1st order call")
                A1, B1s, C1, f1 = params
                total_0 = _np.sum((A1+(B1s-A1+C1*xdata)*f1**xdata-ydata)**2 / weights)
#                correction_1 = _np.sum((A1+(B1s-A1+C1*xdata_correction)*f1**xdata_correction-ydata_correction)**2 / weights)
                correction_2 = 0
                for m, n_0, N, K in zip(xdata_correction,n_0_list,N_list,K_list):
#                    print("type(A)="+str(type(A)))
#                    print("type(Bs)="+str(type(Bs)))
#                    print("type(f)="+str(type(f)))
#                    print("type(m)="+str(type(m)))
#                    print("type(n_0)="+str(type(n_0)))
#                    print("type(N)="+str(type(N)))
#                    print("type(K)="+str(type(K)))
                    F = A+(Bs-A)*f**m
#                    print("A="+str(A))
#                    print("Bs="+str(Bs))
#                    print("f="+str(f))
#                    print("m="+str(m))
                    try:
                        assert len(F)==1
                        F=F[0]
                    except:
                        pass
#                    print("F="+str(F))
#                    print("K="+str(K))
#                    print("n_0="+str(n_0))
#                    print("N="+str(N))
                    if F < (n_0 - 1) / (N - 1):
###                        print('Case 1 m='+str(m))
###                        print('F='+str(F))
###                        print(case_1_dict[m,n_0,N,K])
#                        print("type(F)="+str(type(F)))
#                        print("F="+str(F))
#                        print("K="+str(K))
#                        print("n_0="+str(n_0))
#                        print("N="+str(N))
#                        correction_2 += K * _np.log(F * (n_0-1)**(n_0-1)*(N-n_0)**(N-n_0) / ((N-1)**(N-1)))#+1e-10)
                        correction_2 += K * _np.log(F * case_1_dict[m,n_0,N,K])#+1e-10
##                        correction_2 += (F * (n_0-1)**(n_0-1)*(N-n_0)**(N-n_0) / ((N-1)**(N-1)))**K
                    elif F >= (n_0 - 1) / (N - 1) and F < n_0 / (N-1):
###                        print('Case 2 m='+str(m))
###                        print('F='+str(F))
###                        print('1-F='+str(1-F))
###                        print(F**n_0 * (1-F)**(N-n_0))
#                        print(case_1_dict[m,n_0,N,K])
#                        print("F="+str(F))
#                        print("K="+str(K))
#                        print("n_0="+str(n_0))
#                        print("N="+str(N))
                        correction_2 += K * _np.log(F**n_0 * (1-F)**(N-n_0))
##                        correction_2 += (F**n_0 * (1-F)**(N-n_0))**K
                    elif F>= n_0 / (N-1):
###                        print('Case 3 m='+str(m))
###                        print('1-F='+str(1-F))
###                        print(case_3_dict[m,n_0,N,K])
                        correction_2 += K * _np.log((1-F) * case_3_dict[m,n_0,N,K])#+1e-10)
#                        correction_2 += K * _np.log((1-F) * (n_0**n_0 * (N-n_0-1)**(N-n_0-1))/((N-1)**(N-1)))#+1e-10)
##                        correction_2 += ((1-F) * (n_0**n_0 * (N-n_0-1)**(N-n_0-1))/((N-1)**(N-1)))**K
                    else:
                        #print((n_0 - 1) / (N - 1))
                        #print(n_0 / (N-1))
                        #print("F="+str(F))
                        #print("A="+str(A))
                        #print("Bs="+str(Bs))
                        #print("f="+str(f))
                        #print("m="+str(m))
                        raise ValueError("F does not fall within any physical bounds for m="+str(m)+"!")
#                return total_0 - correction_1 + correction_2
                return total_0 - correction_2

        def obj_func_1d(f):
            A = 0.5
            Bs = 1.
            return obj_func_full([A,Bs,f])
        
        if fit_parameters_dict is None:
            fit_parameters_dict = {}        
        if  fit == 'standard' or fit =='first order':
            if 'f0' not in fit_parameters_dict.keys():
                fit_parameters_dict['f0'] = 0.99
            if 'f_bnd' not in fit_parameters_dict.keys():
                fit_parameters_dict['f_bnd'] = [0.,1.]              
            if 'A0' not in fit_parameters_dict.keys():
                fit_parameters_dict['A0'] = 0.5
            if 'A_bnd' not in fit_parameters_dict.keys():
                fit_parameters_dict['A_bnd'] = [None,None]            
            if 'ApB0' not in fit_parameters_dict.keys():
                fit_parameters_dict['ApB0'] = 1.0             
            if 'ApB_bnd' not in fit_parameters_dict.keys():
                fit_parameters_dict['ApB_bnd'] = [None,None]
        if  fit == 'first order': 
            if 'C0' not in fit_parameters_dict.keys():
                fit_parameters_dict['C0'] = 0.0
            if 'C_bnd' not in fit_parameters_dict.keys():
                fit_parameters_dict['C_bnd'] = [None,None]
        
        if fit == 'standard' or fit == 'first order':
            f0 = fit_parameters_dict['f0']
            initial_soln = _minimize(obj_func_1d,f0, method='L-BFGS-B',
                                     bounds=[(0.,1.)])
            f0b = initial_soln.x[0]
            A0 = fit_parameters_dict['A0']
            ApB0 = fit_parameters_dict['ApB0']
            f_bnd =  fit_parameters_dict['f_bnd']
            A_bnd = fit_parameters_dict['A_bnd']
            ApB_bnd = fit_parameters_dict['ApB_bnd']
            
            p0 = [A0,ApB0,f0b] 
            final_soln_standard = _minimize(obj_func_full,p0, method='L-BFGS-B',
                               bounds=[A_bnd,ApB_bnd,f_bnd])
            A,Bs,f = final_soln_standard.x
            results_dict = {'A': A,'B': Bs-A,'f': f,'r': _rbutils.p_to_r(f,dim)}
            if fit == 'first order':
                C0 = fit_parameters_dict['C0']
                C_bnd = fit_parameters_dict['C_bnd']
                p0 = [A,Bs,C0,f]        
                final_soln_1storder = _minimize(obj_func_1st_order_full, p0, 
                               method='L-BFGS-B',
                               bounds=[A_bnd,ApB_bnd,C_bnd,f_bnd])
        
                A,Bs,C,f = final_soln_1storder.x
                results_dict = {'A': A,'B': Bs-A, 'C':C, 
                                'f': f,'r': _rbutils.p_to_r(f,dim)}
                
        return results_dict

    #Note: assumes dataset contains gate strings which use *base* labels
    base_lengths = list(map(len,gatestrings))
    occ_indices = _tools.compute_occurrence_indices(gatestrings)
    Ns = [ dataset.get_row(seq,k).total
           for seq,k in zip(gatestrings,occ_indices) ]
    successes = [ dataset.get_row(seq,k).fraction(success_outcomelabel) 
                  for seq,k in zip(gatestrings,occ_indices) ] 

    if pre_avg:
        base_lengths,base_successes,base_Ns = \
            preavg_by_length(base_lengths,successes,Ns)
        if weight_data : 
#            mkn_dict = _rbutils.dataset_to_mkn_dict(dataset,gatestrings,success_spamlabel)
#            weight_dict = _rbutils.mkn_dict_to_weighted_delta_f1_hat_dict(mkn_dict)
            if infinite_data == True:
                use_frequencies = True
            else:
                use_frequencies = False
            summary_dict = _rbutils.dataset_to_summary_dict(dataset,gatestrings,success_outcomelabel,use_frequencies)
#           weight_dict = _rbutils.mkn_dict_to_delta_f1_squared_dict(mkn_dict)
            weight_dict = _rbutils.summary_dict_to_delta_f1_squared_dict(summary_dict, infinite_data)
            weights = _np.array(map(lambda x : weight_dict[x],base_lengths))
            if _np.count_nonzero(weights) != len(weights):
                print("Warning: Zero weight detected!")
            if one_freq_adjust:
                one_freq_dict = _rbutils.summary_dict_to_one_freq_dict(summary_dict)
                if len(one_freq_dict['m_list']) == 0:
                    one_freq_dict = None
            else:
                one_freq_dict = None
        else : 
            weights = None
            one_freq_dict = None
    else:
        base_lengths,base_successes,base_Ns = base_lengths,successes,Ns
        weights = None
        one_freq_dict = None
    results = do_fit(base_lengths, base_successes,fit,fit_parameters_dict,one_freq_dict, weights)
    results.update({'gatestrings': gatestrings,
                          'lengths': base_lengths,
                          'successes': base_successes,
                          'counts': base_Ns })
#    result_dicts[basename] = base_results
    
#    for gstyp,alias_map in alias_maps.items():
#        if alias_map is None: continue #skip when map is None
#
#        gstyp_gatestrings = _cnst.translate_gatestring_list(
#            gatestrings, alias_map)
#        gstyp_lengths = list(map(len,gstyp_gatestrings))
#
#        if pre_avg:
#            gstyp_lengths,gstyp_successes,gstyp_Ns = \
#                preavg_by_length(gstyp_lengths,successes,Ns)
#        else:
#            gstyp_lengths,gstyp_successes,gstyp_Ns = gstyp_lengths,successes,Ns
#        gstyp_results = fit(gstyp_lengths, gstyp_successes, one_freq_dict, weights)
#        gstyp_results.update({'gatestrings': gstyp_gatestrings,
#                              'lengths': gstyp_lengths,
#                              'successes': gstyp_successes,
#                              'counts': gstyp_Ns })
#        result_dicts[gstyp] = gstyp_results

    results = _rbobjs.RBResults(dataset, results=results, fit=fit, success_outcomelabel=success_outcomelabel,
                                fit_parameters_dict=fit_parameters_dict, dim=dim, 
                                weight_data=weight_data, pre_avg=pre_avg, 
                                infinite_data=infinite_data, one_freq_adjust=one_freq_adjust)

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
    gateStrings = list(expRBdataset.keys(stripOccurrenceTags=True))
    Ns = [ expRBdataset[s].total for s in gateStrings ]
    return _cnst.generate_fake_data(gateset,gateStrings,Ns,sampleError='multinomial',
                                    collisionAction=expRBdataset.collisionAction,
                                    seed=seed)


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
    gateStrings = list(expRBdataset.keys(stripOccurrenceTags=True))
    return _cnst.generate_fake_data(gateset,gateStrings,N,sampleError='none',
                                    collisionAction=expRBdataset.collisionAction)

