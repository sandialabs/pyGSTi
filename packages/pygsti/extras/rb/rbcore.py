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
    Generate a random RB sequence.
    
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
    
    typ : {"generic", "canonical", "primitive"}
        What kind of gate set should the selected gate sequences be expressed
        as:

        - "generic" : Clifford gates are used, with labels "Gc0" through
        "Gc23".
        - "canonical" : The "canonical" gate set is used (so called because of
        its abundance in the literature for describing Clifford operations).
        This gate set contains the gates {I, X(pi/2), X(-pi/2), X(pi), Y(pi/2),
        Y(-pi/2), Y(pi)}
        - "primitive" : A gate set is used which is neither "generic" nor 
        "canonical".  E.g., {I, X(pi/2), Y(pi/2)}.  In this case, primD must
        be specified.
    
    clifford_to_canonical : dict, optional
        Dictionary mapping clifford labels (as defined in clifford_group) to 
        tuples of "canonical-clifford" labels.  This dictionary is required
        when `typ` is "canonical" or "primitive".

    canonical_to_primitive : dict, optional
        Dictionary mapping "canonical-clifford" labels, defined by the keys
        of `clifford_to_canonical` (typically {I, X(pi/2),  X(-pi/2), X(pi),
        Y(pi/2), Y(-pi/2), Y(pi)}) to tuples of "primitive" labels.  This
        dictionary is required only when `typ` is "primitive".
    
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    
    Returns
    -----------
    clifford_string_list : list
        List of gate strings; each gate string is an RB experiment.
    
    clifford_len_list : list
        List of Clifford lengths for clifford_string_list.  clifford_len_list[i] is
        the number of Clifford operations selected for the creation of
        clifford_string_list[i].
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

#        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
#            gatestr = []
#            for cliff in cliff_tup:
#                gatestr += CliffD[cliff]
#            cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]

#        if typ == 'canonical':
#            return canonical_string_list
#        elif typ == 'primitive':
#            primitive_string_list = _cnst.translate_gatestring_list(
#                canonical_string_list, canonical_to_primitive)
#            return primitive_string_list
#        else:
#            raise ValueError('typ must be "generic" or "canonical"'
#                             + ' or "primitive"!')


def write_empty_rb_files(filename, m_min, m_max, Delta_m, clifford_group, K_m,
                         alias_maps=None, seed=None, randState=None):
    """
    Wrapper for make_random_rb_cliff_string_lists.  Functionality is same as 
    random_RB_cliff_string_lists, except that both an empty data template file is written
    to disk as is the list recording the Clifford length of each gate sequence.
    See docstring for make_random_rb_cliff_string_lists for more details.
    """
    if alias_maps is None: alias_maps = {} # so below always returns a dict
    random_string_lists = \
        list_random_rb_clifford_strings(m_min, m_max, Delta_m, clifford_group,
                                        K_m, alias_maps, seed, randState)
    #always write cliffords to empty dataset (in future have this be an arg?)
    _io.write_empty_dataset(
        filename+'.txt', list(
            _itertools.chain(*random_string_lists['clifford'])))
    for gstyp,strLists in random_string_lists.items():
        _io.write_gatestring_list(filename +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
    return random_string_lists

                
            #for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            #            gatestr = []
            #            for cliff in cliff_tup:
            #                subgatestr = []
            #                for gate in CliffD[cliff]:
            #                    subgatestr += primD[gate]
            #                gatestr += subgatestr
            #            cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]
#    cliff_string_list =  _cnst.gatestring_list(cliff_string_list)
#    return cliff_string_list, cliff_len_list


#def process_rb_data(dataset, prim_seq_list, cliff_len_list, prim_dict=None,
#                    pre_avg=True, process_prim=False, process_cliff=False,
#                    f0 = [0.98],AB0 = [0.5,0.5]):


#def process_rb_data(dataset, clifford_gatestrings, cliff_to_canonical = None, 
#                    canonical_to_primitive = None, success_spamlabel = 'plus',
#                    dim = 2, pre_avg=True, f0 = [0.98], AB0 = [0.5,0.5]):
#    """
#    Process RB data, yielding an RB results object containing desired RB
#    quantities.  See docstring for rb_results for more details.
#
#    TODO: copy rb_resutls docstring here?
#    """
#
#
#    def preavg_by_length(lengths, success_probs, Ns):
#        bins = {}
#        for L,p,N in zip(lengths, success_probs, Ns):
#            if L not in bins:
#                bins[L] = _np.array([L,p,N,1],'d')
#            else:
#                bins[L] += _np.array([L,p,N,1],'d')
#        avgs = { L: ar/ar[3] for L,ar in bins.items() }
#        Ls = sorted(avgs.keys())
#        preavg_lengths =  [ avgs[L][0] for L in Ls ] 
#        preavg_psuccess = [ avgs[L][1] for L in Ls ] 
#        preavg_Ns =       [ avgs[L][2] for L in Ls ] 
#        return preavg_lengths, preavg_psuccess, preavg_Ns
#
#
#    def fit(xdata,ydata):
#        def obj_func_full(params):
#            A,B,f = params
#            return _np.sum((A+B*f**xdata-ydata)**2)
#
#        def obj_func_1d(f):
#            A = B = 0.5
#            return obj_func_full([A,B,f])
#
#        initial_soln = _minimize(obj_func_1d,f0, method='L-BFGS-B',
#                                 bounds=[(0.,1.)])
#        f0 = cliff_initial_soln.x[0]
#        p0 = AB0 + [f0]
#        final_soln = _minimize(obj_func_full,p0, method='L-BFGS-B',
#                               bounds=[(0.,1.),(0.0,1.),(0.,1.)])
#        A,B,f = final_soln.x
#        return {'A': A,'B': B,'f': f, 'F_avg': f_to_F_avg(f,dim),
#                'r': f_to_r(f,dim)}
#
#
#    #Note: assumes dataset contains gate strings which use *clifford* labels
#    cliff_lengths = list(map(len,clifford_gatestrings))
#    Ns = [ dataset[seq].total() for seq in clifford_gatestrings ]
#    successes = [ dataset[seq].fraction(success_spamlabel) 
#                  for seq in clifford_gatestrings ] 
#
#    if pre_avg:
#        cliff_lengths,cliff_successes,cliff_Ns = \
#            preavg_by_length(cliff_lengths,successes,Ns)
#    cliff_results = fit(cliff_lengths, cliff_successes)
#    cliff_results.update({'gatestrings': clifford_gatestrings,
#                          'lengths': cliff_lengths,
#                          'successes': cliff_successes,
#                          'counts': cliff_Ns })
#
#    if cliff_to_canonical is not None:
#        canonical_gatestrings = [ _objs.GateString(_itertools.chain(
#                    *[cliff_to_canonical[cliffLbl] for cliffLbl in gs]))
#                    for gs in clifford_gatestrings ]
#        canonical_lengths = list(map(len,canonical_gatestrings))
#
#        if pre_avg:
#            canonical_lengths,canonical_successes,canonical_Ns = \
#                preavg_by_length(canonical_lengths,successes,Ns)
#        canonical_results = fit(canonical_lengths, canonical_successes)
#        canonical_results.update({'gatestrings': canonical_gatestrings,
#                                  'lengths': canonical_lengths,
#                                  'successes': canonical_successes,
#                                  'counts': canonical_Ns })
#
#        if canonical_to_primitive is not None:
#            primitive_gatestrings = [ _objs.GateString(_itertools.chain(
#                 *[canonical_to_primitive[canonLbl] for canonLbl in gs]))
#                 for gs in canonical_gatestrings ]
#            primitive_lengths = list(map(len,primitive_gatestrings))
#
#            if pre_avg:
#                primitive_lengths,primitive_successes,primitive_Ns = \
#                    preavg_by_length(primitive_lengths,successes,Ns)
#            prim_results = fit(primitive_lengths, primitive_successes)
#            prim_results.update({'gatestrings': primitive_gatestrings,
#                                 'lengths': primitive_lengths,
#                                 'successes': primitive_successes,
#                                 'counts': primitive_Ns })
#
#    results = _rbobjs.RBResults(dataset, cliff_results, canonical_results,
#                                prim_results, dim, pre_avg, cliff_to_canonical,
#                                canonical_to_primitive)
#    return results


def do_randomized_benchmarking(dataset, clifford_gatestrings,
                               success_spamlabel = 'plus',
                               dim = 2, pre_avg=True, 
                               clifford_to_primitive = None,
                               clifford_to_canonical = None, 
                               canonical_to_primitive = None,
                               f0 = [0.98], AB0 = [0.5,0.5]):
    """
    TODO: docstring
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

    return do_rb_base(dataset, clifford_gatestrings, "clifford", alias_maps,
                      success_spamlabel, dim, pre_avg, f0, AB0)

def do_rb_base(dataset, base_gatestrings, basename, alias_maps=None,
               success_spamlabel = 'plus', dim = 2, pre_avg=True,
               f0 = [0.98], AB0 = [0.5,0.5]):
    """
    Process RB data, yielding an RB results object containing desired RB
    quantities.  See docstring for rb_results for more details.

    TODO: copy rb_resutls docstring here?
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


    def fit(xdata,ydata):
        def obj_func_full(params):
            A,B,f = params
            return _np.sum((A+B*f**xdata-ydata)**2)

        def obj_func_1d(f):
            A = B = 0.5
            return obj_func_full([A,B,f])

        initial_soln = _minimize(obj_func_1d,f0, method='L-BFGS-B',
                                 bounds=[(0.,1.)])
        f0b = initial_soln.x[0]
        p0 = AB0 + [f0b]
        final_soln = _minimize(obj_func_full,p0, method='L-BFGS-B',
                               bounds=[(0.,1.),(0.0,1.),(0.,1.)])
        A,B,f = final_soln.x
        return {'A': A,'B': B,'f': f, 'F_avg': _rbutils.f_to_F_avg(f,dim),
                'r': _rbutils.f_to_r(f,dim)}

    result_dicts = {}

    #Note: assumes dataset contains gate strings which use *base* labels
    base_lengths = list(map(len,base_gatestrings))
    Ns = [ dataset[seq].total() for seq in base_gatestrings ]
    successes = [ dataset[seq].fraction(success_spamlabel) 
                  for seq in base_gatestrings ] 

    if pre_avg:
        base_lengths,base_successes,base_Ns = \
            preavg_by_length(base_lengths,successes,Ns)
    base_results = fit(base_lengths, base_successes)
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
        gstyp_results = fit(gstyp_lengths, gstyp_successes)
        gstyp_results.update({'gatestrings': gstyp_gatestrings,
                              'lengths': gstyp_lengths,
                              'successes': gstyp_successes,
                              'counts': gstyp_Ns })
        result_dicts[gstyp] = gstyp_results

    results = _rbobjs.RBResults(dataset, result_dicts, basename, alias_maps,
                                success_spamlabel, dim, pre_avg, f0, AB0)
    return results




    

#
#    def 
#        prim_len_list = []
#        successes = []
#        N_list = []
#        for seq_num, seq in enumerate(self.prim_seq_list):
#            data_line = self.dataset[seq]
#            plus = data_line['plus']
#            minus = data_line['minus']
#            N = plus + minus
#            prim_length = len(seq)
#            prim_len_list.append(prim_length)
#            seq_success_prob = 1 - plus/float(N)
#            successes.append(seq_success_prob)
#            N_list.append(N)
#            if seq_success_prob < 0:
#                raise ValueError('Survival probability less than 0!')
#
#        if self.pre_avg:
#            cliff_zip = list(zip(self.cliff_len_list,successes,N_list))
#            cliff_zip = sorted(cliff_zip,key=lambda x: x[0])
#            #cliff_zip = _np.array(cliff_zip,dtype=[('length',int),('F',float),('N',float)])
#            #cliff_zip = _np.sort(cliff_zip,order='length')
#            cliff_avg = []
#            cliff_avg_len_list = []
#            total_N_list = []
#            total_N = 0
#            current_len = 0
#            total = 0
#            total_seqs = 0
#            for i in range(len(cliff_zip)):
#                tup = cliff_zip[i]
#                if tup[0] != current_len:
#                    if current_len != 0:
#                        cliff_avg_len_list.append(current_len)
#                        cliff_avg.append(float(total) / total_seqs)
#                        total_N_list.append(total_N)
#                    current_len = tup[0]
#                    total = 0
#                    total_seqs = 0
#                    total_N = 0
#                total += tup[1]
#                total_N += tup[2]
#                total_seqs += 1
#
#            self.total_N_list = _np.array(total_N_list)
#
#            prim_avg = []
#            prim_avg_len_list = []
#            current_len = 0
#            total = 0
#            total_seqs = 0
#
#            prim_zip = list(zip(prim_len_list,successes))
#
#            prim_zip = list(zip(self.prim_len_list,successes,N_list))
#            prim_zip = sorted(prim_zip,key=lambda x: x[0])
##            prim_zip = _np.array(prim_zip,dtype=[('length',int),('F',float),('N',float)])
##            prim_zip = _np.sort(prim_zip,order='length')
#
#            for i in range(len(cliff_zip)):
#                tup = prim_zip[i]
#                if tup[0] != current_len:
#                    if current_len != 0:
#                        prim_avg_len_list.append(current_len)
#                        prim_avg.append(float(total) / total_seqs)
#                    current_len = tup[0]
#                    total = 0
#                    total_seqs = 0
#                total += tup[1]
#                total_seqs += 1
#
#            self.cliff_len_list = cliff_avg_len_list
#            self.cliff_successes = cliff_avg
#
#            self.prim_len_list = prim_avg_len_list
#            self.prim_successes = prim_avg            
#        else:
#            self.prim_successes = successes
#            self.cliff_successes = successes
#
##        self.successes = successes
##        self.prim_len_list = prim_len_list
##        self.data_parsed = True
##    def parse_data_preavg(self):
##        if not self.data_parsed:
##            self.parse_data()
#
#
#    def analyze_data(self,rb_decay_func = rb_decay_WF,process_prim = False,
#                     process_cliff = False, f0 = [0.98], AB0=[0.5,0.5]):
#        """
#        Analyze RB data to compute fit parameters and in turn the RB error
#        rate.
#
#        TODO: docstring describing parameters
#        """
#
#        if process_prim:
#            xdata = self.prim_len_list
#            ydata = self.prim_successes
#            def obj_func_full(params):
#                A,B,f = params
#                val = _np.sum((A+B*f**xdata-ydata)**2)
#                return val
#            def obj_func_1d(f):
#                A = 0.5
#                B = 0.5
#                val = obj_func_full([A,B,f])
#                return val
#            self.prim_initial_soln = _minimize(obj_func_1d,f0,
#                                               method='L-BFGS-B',
#                                               bounds=[(0.,1.)])
#            f1 = [self.prim_initial_soln.x[0]]
#            p0 = AB0 + f1
#            self.prim_end_soln = _minimize(obj_func_full,p0,
#                                           method='L-BFGS-B',
#                                           bounds=[(0.,1.),(0.,1.),(0.,1.)])
#            A,B,f = self.prim_end_soln.x
##            results = _curve_fit(rb_decay_func,self.prim_len_list,self.prim_successes,p0 = p0)
##            A,B,f = results[0]
##            cov = results[1]
#            self.prim_A = A
#            self.prim_B = B
#            self.prim_f = f
##            self.prim_cov = cov
#            self.prim_F_avg = f_to_F_avg(self.prim_f)
#            self.prim_r = f_to_r(self.prim_f)
#            self.prim_analyzed = True
#        if process_cliff:
#            xdata = self.cliff_len_list
#            ydata = self.cliff_successes
#            def obj_func_full(params):
#                A,B,f = params
#                val = _np.sum((A+B*f**xdata-ydata)**2)
#                return val
#            def obj_func_1d(f):
#                A = 0.5
#                B = 0.5
#                val = obj_func_full([A,B,f])
#                return val
#            self.cliff_initial_soln = _minimize(obj_func_1d,f0,
#                                                method='L-BFGS-B',
#                                                bounds=[(0.,1.)])
#            f0 = self.cliff_initial_soln.x[0]
#            p0 = AB0 + [f0]
#            self.cliff_end_soln = _minimize(obj_func_full,p0,
#                                            method='L-BFGS-B',
#                                            bounds=[(0.,1.),(0.0,1.),(0.,1.)])
#            A,B,f = self.cliff_end_soln.x
##            results = _curve_fit(rb_decay_func,self.cliff_len_list,self.cliff_successes,p0 = p0)
##            A,B,f = results[0]
##            cov = results[1]
#            self.cliff_A = A
#            self.cliff_B = B
#            self.cliff_f = f
##            self.cliff_cov = cov
#            self.cliff_F_avg = f_to_F_avg(self.cliff_f)
#            self.cliff_r = f_to_r(self.cliff_f)
#            self.cliff_analyzed = True
#
#
#    results_obj = rb_results(dataset, prim_seq_list, cliff_nlen_list,
#                             prim_dict=prim_dict, pre_avg=pre_avg)
#    results_obj.parse_data()
#    results_obj.analyze_data(process_prim = process_prim,
#                             process_cliff = process_cliff,
#                             f0 = f0, AB0 = AB0)
#    return results_obj
