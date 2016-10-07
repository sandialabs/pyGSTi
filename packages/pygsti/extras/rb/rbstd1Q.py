from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines canonical 1-qubit quantities used in Randomized Benchmarking"""

from ... import construction as _cnst
from ... import objects as _objs
from ... import io as _io

import numpy as _np
from functools import reduce as _reduce



gs_cliff_canonical = _cnst.build_gateset(
    [2],[('Q0',)], ['Gi','Gxp2','Gxp','Gxmp2','Gyp2','Gyp','Gymp2'], 
    [ "I(Q0)","X(pi/2,Q0)", "X(pi,Q0)", "X(-pi/2,Q0)",
      "Y(pi/2,Q0)", "Y(pi,Q0)", "Y(-pi/2,Q0)"],
    prepLabels=["rho0"], prepExpressions=["0"],
    effectLabels=["E0"], effectExpressions=["1"], 
    spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )


CliffD = {}
#Definitions taken from arXiv:1508.06676v1
#0-indexing used instead of 1-indexing

CliffD[0] = ['Gi',]
CliffD[1] = ['Gyp2','Gxp2']
CliffD[2] = ['Gxmp2','Gymp2']
CliffD[3] = ['Gxp',]
CliffD[4] = ['Gymp2','Gxmp2']
CliffD[5] = ['Gxp2','Gymp2']
CliffD[6] = ['Gyp',]
CliffD[7] = ['Gymp2','Gxp2']
CliffD[8] = ['Gxp2','Gyp2']
CliffD[9] = ['Gxp','Gyp']
CliffD[10] = ['Gyp2','Gxmp2']
CliffD[11] = ['Gxmp2','Gyp2']
CliffD[12] = ['Gyp2','Gxp']
CliffD[13] = ['Gxmp2']
CliffD[14] = ['Gxp2','Gymp2','Gxmp2']
CliffD[15] = ['Gymp2']
CliffD[16] = ['Gxp2']
CliffD[17] = ['Gxp2','Gyp2','Gxp2']
CliffD[18] = ['Gymp2','Gxp']
CliffD[19] = ['Gxp2','Gyp']
CliffD[20] = ['Gxp2','Gymp2','Gxp2']
CliffD[21] = ['Gyp2']
CliffD[22] = ['Gxmp2','Gyp']
CliffD[23] = ['Gxp2','Gyp2','Gxmp2']

CliffMatD = {}
CliffMatInvD = {}
for i in range(24):
    CliffMatD[i] = gs_cliff_canonical.product(CliffD[i])
    CliffMatInvD[i] = _np.linalg.matrix_power(CliffMatD[i],-1)

gs_cliff_generic_1q = _cnst.build_gateset(
    [2],[('Q0',)], [], [],
    prepLabels=["rho0"], prepExpressions=["0"],
    effectLabels=["E0"], effectExpressions=["1"],
    spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

for i in list(CliffMatD.keys()):
    gate_lbl = 'Gc'+str(i)
    gs_cliff_generic_1q.gates[gate_lbl] = \
        _objs.FullyParameterizedGate(CliffMatD[i])

def makeCliffGroupTable():
    CliffGroupTable = _np.zeros([24,24], dtype=int)
    counter = 0
    for i in range(24):
        for j in range(24):
            test = _np.dot(CliffMatD[j],CliffMatD[i])
              #Dot in reverse order here for multiplication here because
              # gates are applied left to right.
            for k in range(24):
                diff = _np.linalg.norm(test-CliffMatD[k])
                if diff < 10**-10:
                    CliffGroupTable[i,j]=k
                    counter += 1
                    break
    assert counter==24**2, 'Logic Error!'
    return CliffGroupTable

CliffGroupTable = makeCliffGroupTable()

CliffInvTable = {}
for i in range(24):
    for j in range(24):
        if CliffGroupTable[i,j] == 0:
            CliffInvTable[i] = j
            
######################################################
#Need to figure out what to do with these functions
######################################################
def lookup_cliff_prod(i,j):
    """
    Auxiliary function for looking up the product of two Cliffords.
    """
    return CliffGroupTable[i,j]

def make_random_RB_cliff_string(m,seed=None):
    """
    Generate a random RB sequence.
    
    Parameters
    ----------
    m : int
        Sequence length is m+1 (because m Cliffords are chosen at random, then one additional
        Clifford is selected to invert the sequence).
    
    seed : int, optional
        Seed for the random number generator.
    
    Returns
    ----------
    cliff_string : list
        Random Clifford sequence of length m+1.  For ideal Cliffords, the sequence
        implements the identity operation.
    """
    if seed:
        _np.random.seed()
    cliff_string = _np.random.randint(0,24,m)
    effective_cliff = _reduce(lookup_cliff_prod,cliff_string)
    cliff_inv = CliffInvTable[effective_cliff]
    cliff_string = _np.append(cliff_string,cliff_inv)
    return cliff_string

def make_random_RB_cliff_string_lists(m_min, m_max, Delta_m, K_m_sched,
                                      generic_or_canonical_or_primitive,
                                      primD=None,seed=None):
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

    K_m_sched : int or OrderedDict
        If an integer, the fixed number of Clifford sequences to be sampled at
        each length m.  If an OrderedDict, then a mapping from Clifford
        sequence length m to number of Cliffords to be sampled at that length.
    
    generic_or_canonical_or_primitive : string
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
    
    primD : dictionary, optional
        A primitives dictionary, mapping the "canonical gate set" {I, X(pi/2),
        X(-pi/2), X(pi), Y(pi/2), Y(-pi/2), Y(pi)} to the gate set of
        primitives whose gate labels are to be used in the generated RB
        sequences.
    
    seed : int, optional
        Seed for random number generator; optional.
    
    Returns
    -----------
    cliff_string_list : list
        List of gate strings; each gate string is an RB experiment.
    
    cliff_len_list : list
        List of Clifford lengths for cliff_string_list.  cliff_len_list[i] is
        the number of Clifford operations selected for the creation of
        cliff_string_list[i].
    """
    if seed is not None:
        _np.random.seed(seed)
    cliff_string_list = []
    if not isinstance(K_m_sched,_OrderedDict):
        print("K_m_sched is not an OrderedDict, so Wallman and Flammia" +
              " error bars are not valid.")
        if not isinstance(K_m_sched,int):
            raise ValueError('K_m_sched must be an OrderedDict or an int!')
        K_m_sched_dict = _OrderedDict()
        for m in range(m_min, m_max+1,Delta_m):
            K_m_sched_dict[m] = K_m_sched
    else:
        K_m_sched_dict = K_m_sched
    for m in range(m_min,m_max+1,Delta_m):
        temp_list = []
        K_m = K_m_sched_dict[m]
        for i in range(K_m):
            temp_list.append(tuple(make_random_RB_cliff_string(m).tolist()))
#        temp_list = remove_duplicates(temp_list)
#        print len(temp_list)
        cliff_string_list += temp_list
    cliff_len_list = list(map(len,cliff_string_list))
    if generic_or_canonical_or_primitive == 'generic':
        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            cliff_string_list[cliff_tup_num] = ['Gc'+str(i) for i in cliff_tup]
    elif generic_or_canonical_or_primitive == 'canonical':
        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            gatestr = []
            for cliff in cliff_tup:
                gatestr += CliffD[cliff]
            cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]
    elif generic_or_canonical_or_primitive == 'primitive':
        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            gatestr = []
            for cliff in cliff_tup:
                subgatestr = []
                for gate in CliffD[cliff]:
                    subgatestr += primD[gate]
                gatestr += subgatestr
            cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]
    else:
        raise ValueError('generic_or_canonical_or_primitive must be "generic"'+
                         ' or "canonical" or "primitive"!')
    cliff_string_list =  _cnst.gatestring_list(cliff_string_list)
    return cliff_string_list, cliff_len_list

def write_empty_rb_files(filename,m_min,m_max,Delta_m,K_m,generic_or_canonical_or_primitive,primD=None,seed=None):
    """
    Wrapper for make_random_RB_cliff_string_lists.  Functionality is same as 
    random_RB_cliff_string_lists, except that both an empty data template file is written
    to disk as is the list recording the Clifford length of each gate sequence.
    See docstring for make_random_RB_cliff_string_lists for more details.
    """
    random_RB_cliff_string_lists, cliff_lens = \
        make_random_RB_cliff_string_lists(
            m_min, m_max, Delta_m, K_m, generic_or_canonical_or_primitive,
            primD=primD,seed=seed)
    _io.write_empty_dataset(filename+'.txt',random_RB_cliff_string_lists)
#    seq_len_list = map(len,random_RB_cliff_string_lists)
    temp_file = open(filename+'_cliff_seq_lengths.pkl','w')
    _pickle.dump(cliff_lens,temp_file) 
#    for cliff_len in cliff_lens:
#        temp_file.write(str(cliff_len)+'\n')
    temp_file.close()        
    return random_RB_cliff_string_lists, cliff_lens
#Want to keep track of both Clifford sequences and primitive sequences.

def process_rb_data(dataset, prim_seq_list, cliff_len_list, prim_dict=None,
                    pre_avg=True, process_prim=False, process_cliff=False,
                    f0 = [0.98],AB0 = [0.5,0.5]):
    """
    Process RB data, yielding an RB results object containing desired RB
    quantities.  See docstring for rb_results for more details.

    TODO: copy rb_resutls docstring here?
    """
    results_obj = rb_results(dataset, prim_seq_list, cliff_len_list,
                             prim_dict=prim_dict, pre_avg=pre_avg)
    results_obj.parse_data()
    results_obj.analyze_data(process_prim = process_prim,
                             process_cliff = process_cliff,
                             f0 = f0, AB0 = AB0)
    return results_obj
    
# def rb_decay_WF_rate(dataset,avg_gates_per_cliff=None,showPlot=False,xlim=None,ylim=None,saveFigPath=None,printData=False,p0=[0.5,0.5,0.98]):
#     RBlengths = []
#     RBsuccesses = []
#     for key in list(dataset.keys()):
#         dataLine = dataset[key]
#         plus = dataLine['plus']
#         minus = dataLine['minus']
#         N = plus + minus
#         key_len = len(key)
#         RBlengths.append(key_len)
#         RBsuccessProb=1 - dataLine['plus']/float(N)
#         RBsuccesses.append(RBsuccessProb)
#         if dataLine['plus']/float(N) > 1:
#             print(key)
#         if printData:
#             print(key_len,RBsuccessProb)
#     results = _curve_fit(rb_decay_WF,RBlengths,RBsuccesses,p0=p0)
#     A,B,f = results[0]
#     cov = results[1]
#     if saveFigPath or showPlot:
#         newplot = _plt.figure()
#         newplotgca = newplot.gca()
#         newplotgca.plot(RBlengths,RBsuccesses,'.')
#         newplotgca.plot(range(max(RBlengths)),
#                         rb_decay_WF(_np.arange(max(RBlengths)),A,B,f),'+')
#         newplotgca.set_xlabel('RB sequence length (non-Clifford)')
#         newplotgca.set_ylabel('Success rate')
#         newplotgca.set_title('RB success')
#         if xlim:
#             _plt.xlim(xlim)
#         if ylim:
#             _plt.ylim(ylim)
#     if saveFigPath:
#         newplot.savefig(saveFigPath)
#     print "f (for gates) =",f
#     if avg_gates_per_cliff:
#         print "f (for Cliffords) = f^(avg. gates/Cliffords) =",f**avg_gates_per_cliff
#     return A,B,f,cov
