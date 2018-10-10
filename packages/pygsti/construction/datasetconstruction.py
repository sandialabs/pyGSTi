""" Functions for creating datasets """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import numpy.random as _rndm
import warnings as _warnings
import collections as _collections
import itertools as _itertools

from ..objects import gatestring as _gs
from ..objects import dataset as _ds
from ..baseobjs import label as _lbl
from ..tools import compattools as _compat
from . import gatestringconstruction as _gstrc

from pprint import pprint

def generate_fake_data(gatesetOrDataset, gatestring_list, nSamples,
                       sampleError="none", seed=None, randState=None,
                       aliasDict=None, collisionAction="aggregate",
                       comm=None, memLimit=None):
    """Creates a DataSet using the probabilities obtained from a gateset.

    Parameters
    ----------
    gatesetOrDataset : GateSet or DataSet object
        If a GateSet, the gate set whose probabilities generate the data.
        If a DataSet, the data set whose frequencies generate the data.

    gatestring_list : list of (tuples or GateStrings) or None
        Each tuple or GateString contains gate labels and
        specifies a gate sequence whose counts are included
        in the returned DataSet. e.g. ``[ (), ('Gx',), ('Gx','Gy') ]``

    nSamples : int or list of ints or None
        The simulated number of samples for each gate string.  This only has
        effect when  ``sampleError == "binomial"`` or ``"multinomial"``.  If an
        integer, all gate strings have this number of total samples. If a list,
        integer elements specify the number of samples for the corresponding
        gate string.  If ``None``, then `gatesetOrDataset` must be a
        :class:`~pygsti.objects.DataSet`, and total counts are taken from it
        (on a per-gatestring basis).

    sampleError : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error: counts are floating point numbers such
          that the exact probabilty can be found by the ratio of count / total.
        - "round" - same as "none", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution.  Distribution has parameters p = probability of the
          gate string and n = number of samples.  This can only be used when
          there are exactly two SPAM labels in gatesetOrDataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = probability of the gate string
          using the k-th SPAM label and n = number of samples.

    seed : int, optional
        If not ``None``, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    randState : numpy.random.RandomState
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.

    aliasDict : dict, optional
        A dictionary mapping single gate labels into tuples of one or more
        other gate labels which translate the given gate strings before values
        are computed using `gatesetOrDataset`.  The resulting Dataset, however,
        contains the *un-translated* gate strings as keys.

    collisionAction : {"aggregate", "keepseparate"}
        Determines how duplicate gate sequences are handled by the resulting
        `DataSet`.  Please see the constructor documentation for `DataSet`.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors and ensuring that the *same* dataset is
        generated on each processor.

    memLimit : int, optional
        A rough memory limit in bytes which is used to determine job allocation
        when there are multiple processors.

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.

    """
    NTOL = 10
    TOL = 1 / (10**-NTOL)

    if isinstance(gatesetOrDataset, _ds.DataSet):
        dsGen = gatesetOrDataset
        gsGen = None
        dataset = _ds.DataSet( collisionAction=collisionAction,
                               outcomeLabelIndices = dsGen.olIndex ) # keep same outcome labels
    else:
        gsGen = gatesetOrDataset
        dsGen = None
        dataset = _ds.DataSet( collisionAction=collisionAction )

                
    if aliasDict:
        aliasDict = { _lbl.Label(ky): tuple((_lbl.Label(el) for el in val))
                      for ky,val in aliasDict.items() } # convert to use Labels

    if gsGen:
        trans_gatestring_list = [ _gstrc.translate_gatestring(s, aliasDict)
                                  for s in gatestring_list ]
        all_probs = gsGen.bulk_probs(trans_gatestring_list, comm=comm, memLimit=memLimit)
        #all_dprobs = gsGen.bulk_dprobs(gatestring_list) #DEBUG - not needed here!!!

    if comm is None or comm.Get_rank() == 0: # only root rank computes
                
        if sampleError in ("binomial","multinomial"):
            if randState is None:
                rndm = _rndm.RandomState(seed) # ok if seed is None
            else:
                rndm = randState

        for k,s in enumerate(gatestring_list):

            #print("DB GEN %d of %d (len %d)" % (k,len(gatestring_list),len(s)))
            trans_s = _gstrc.translate_gatestring(s, aliasDict)
            if gsGen:
                ps = all_probs[trans_s]

                if sampleError in ("binomial","multinomial"):
                    #Adjust to probabilities if needed (and warn if not close to in-bounds)
                    for ol in ps:
                        if ps[ol] < 0:
                            if ps[ol] < -TOL: _warnings.warn("Clipping probs < 0 to 0")
                            ps[ol] = 0.0
                        elif ps[ol] > 1:
                            if ps[ol] > (1+TOL): _warnings.warn("Clipping probs > 1 to 1")
                            ps[ol] = 1.0
            else:
                ps = _collections.OrderedDict([ (ol,frac) for ol,frac
                                                in dsGen[trans_s].fractions.items()])

            if gsGen and sampleError in ("binomial","multinomial"):
                #Check that sum ~= 1 (and nudge if needed) since binomial and
                #  multinomial random calls assume this.
                psum = sum(ps.values())
                adjusted = False
                if psum > 1+TOL:
                    adjusted = True
                    _warnings.warn("Adjusting sum(probs) > 1 to 1")
                if psum < 1-TOL:
                    adjusted = True
                    _warnings.warn("Adjusting sum(probs) < 1 to 1")

                # A cleaner probability cleanup.. lol
                OVERTOL  = 1.0 + TOL
                UNDERTOL = 1.0 - TOL
                normalized = lambda : UNDERTOL <= sum(ps.values()) <= OVERTOL
                if not normalized():
                    m = max(ps.values())
                    ps = {lbl : round(p/m, NTOL) for lbl, p in ps.items()}
                    print(sum(ps.values()))

                assert normalized(), 'psum={}'.format(sum(ps.values()))
                if adjusted:
                    _warnings.warn('Adjustment finished')

            if nSamples is None and dsGen is not None:
                N = dsGen[trans_s].total #use the number of samples from the generating dataset
                 #Note: total() accounts for other intermediate-measurment branches automatically
            else:
                try:
                    N = nSamples[k] #try to treat nSamples as a list
                except:
                    N = nSamples #if not indexable, nSamples should be a single number

            #Weight the number of samples according to a WeightedGateString
            if isinstance(s, _gs.WeightedGateString):
                nWeightedSamples = int(round(s.weight * N))
            else:
                nWeightedSamples = N

            counts = {} #don't use an ordered dict here - add_count_dict will sort keys
            labels = [ol for ol, _ in sorted(list(ps.items()), key=lambda x: x[1]) ]
              # "outcome labels" - sort by prob for consistent generation
            if sampleError == "binomial":
                
                if len(labels) == 1: #Special case when labels[0] == 1.0 (100%)
                    counts[labels[0]] = nWeightedSamples
                else:
                    assert(len(labels) == 2)
                    ol0,ol1 = labels[0], labels[1]
                    counts[ol0] = rndm.binomial(nWeightedSamples, ps[ol0])
                    counts[ol1] = nWeightedSamples - counts[ol0]

            elif sampleError == "multinomial":
                countsArray = rndm.multinomial(nWeightedSamples,
                        [ps[ol] for ol in labels], size=1) # well-ordered list of probs
                for i,ol in enumerate(labels):
                    counts[ol] = countsArray[0,i]
            else:
                for outcomeLabel,p in ps.items():
                    pc = _np.clip(p,0,1)
                    if sampleError == "none":
                        counts[outcomeLabel] = float(nWeightedSamples * pc)
                    elif sampleError == "round":
                        counts[outcomeLabel] = int(round(nWeightedSamples*pc))
                    else: raise ValueError("Invalid sample error parameter: '%s'  Valid options are 'none', 'round', 'binomial', or 'multinomial'" % sampleError)

            dataset.add_count_dict(s, counts)
        dataset.done_adding_data()

    if comm is not None: # broadcast to non-root procs
        dataset = comm.bcast(dataset if (comm.Get_rank() == 0) else None,root=0)

    return dataset


def merge_outcomes(dataset,label_merge_dict):
    """
    Creates a DataSet which merges certain outcomes in input DataSet;
    used, for example, to aggregate a 2-qubit 4-outcome DataSet into a 1-qubit 2-outcome
    DataSet.

    Parameters
    ----------
    dataset : DataSet object
        The input DataSet whose results will be compiled according to the rules
        set forth in label_merge_dict

    label_merge_dict : dictionary
        The dictionary whose keys define the new DataSet outcomes, and whose items
        are lists of input DataSet outcomes that are to be summed together.  For example,
        if a two-qubit DataSet has outcome labels "00", "01", "10", and "11", and
        we want to ''aggregate out'' the second qubit, we could use label_merge_dict =
        {'0':['00','01'],'1':['10','11']}.  When doing this, however, it may be better
        to use :function:`filter_qubits` which also updates the gate sequences.

    Returns
    -------
    merged_dataset : DataSet object
        The DataSet with outcomes merged according to the rules given in label_merge_dict.
    """

    new_outcomes = label_merge_dict.keys()
    merged_dataset = _ds.DataSet(outcomeLabels=new_outcomes)
    merge_dict_old_outcomes = [outcome for sublist in label_merge_dict.values() for outcome in sublist]
    if not set(dataset.get_outcome_labels()).issubset( merge_dict_old_outcomes ):
        raise ValueError(("`label_merge_dict` must account for all the outcomes in original dataset."
                          " It's missing directives for:\n%s") % 
                         '\n'.join(set(dataset.get_outcome_labels()) - set(merge_dict_old_outcomes)))

    for key in dataset.keys():
        linecounts = dataset[key].counts
        count_dict = {}
        for new_outcome in new_outcomes:
            count_dict[new_outcome] = 0
            for old_outcome in label_merge_dict[new_outcome]:
                count_dict[new_outcome] += linecounts.get(old_outcome,0)
        merged_dataset.add_count_dict(key,count_dict,aux=dataset[key].aux)
    merged_dataset.done_adding_data()
    return merged_dataset

def create_qubit_merge_dict(nQubits, qubits_to_keep):
    """ 
    Creates a dictionary appropriate for use with :function:`merge_outcomes`,
    that aggregates all but the specified `qubits_to_keep` when the outcome
    labels are those of `nQubits` qubits (i.e. strings of 0's and 1's).

    Parameters
    ----------
    nQubits : int
        The total number of qubits
        
    qubits_to_keep : list
        A list of integers specifying which qubits should be kept, that is,
        *not* aggregated, when the returned dictionary is passed to 
        `merge_outcomes`.

    Returns
    -------
    dict
    """
    outcome_labels = [''.join(map(str,t)) for t in itertools.product([0,1], repeat=nQubits)]
    return create_merge_dict(qubits_to_keep, outcome_labels)


def create_merge_dict(indices_to_keep, outcome_labels):
    """ 
    Creates a dictionary appropriate for use with :function:`merge_outcomes`,
    that aggregates all but the specified `indices_to_keep`.

    In particular, each element of `outcome_labels` should be a n-character
    string (or a 1-tuple of such a string).  The returned dictionary's keys
    will be all the unique results of keeping only the characters indexed by
    `indices_to_keep` from each outcome label.   The dictionary's values will
    be a list of all the original outcome labels which reduce to the key value
    when the non-`indices_to_keep` characters are removed.

    For example, if `outcome_labels == ['00','01','10','11']` and 
    `indices_to_keep == [1]` then this function returns the dict
    `{'0': ['00','10'], '1': ['01','11'] }`.

    Note: if the elements of `outcome_labels` are 1-tuples then so are the
    elements of the returned dictionary's values.

    Parameters
    ----------
    indices_to_keep : list
        A list of integer indices specifying which character positions should be
        kept (i.e. *not* aggregated together by `merge_outcomes`).

    outcome_labels : list
        A list of the outcome labels to potentially merge.  This can be a list
        of strings or of 1-tuples containing strings.

    Returns
    -------
    dict
    """
    merge_dict = _collections.defaultdict(list)
    for ol in outcome_labels:
        if _compat.isstr(ol):
            reduced = ''.join([ol[i] for i in indices_to_keep])
        else: 
            assert(len(ol) == 1) #should be a 1-tuple
            reduced = (''.join([ol[0][i] for i in indices_to_keep]),) # a tuple
        merge_dict[reduced].append(ol)
    return dict(merge_dict) # return a *normal* dict


def filter_dataset(dataset,sectors_to_keep,sindices_to_keep=None,new_sectors=None):
    """
    Creates a DataSet that restricts is the restriction of `dataset`
    to the sectors identified by `sectors_to_keep`.

    More specifically, this function aggregates (sums) outcomes in `dataset`
    which differ only in sectors (usually qubits - see below)  *not* in 
    `sectors_to_keep`, and removes any gate labels which act specifically on
    sectors not in `sectors_to_keep` (e.g. an idle gate acting on *all* 
    sectors because it's `.sslbls` is None will *not* be removed).

    Here "sectors" are state-space labels, present in the gate strings of 
    `dataset`.  Each sector also corresponds to a particular character position
    within the outcomes labels of `dataset`.  Thus, for this function to work,
    the outcome labels of `dataset` must all be 1-tuples whose sole element is
    an n-character string such that each character represents the outcome of
    a single sector.  If the state-space labels are integers, then they can
    serve as both a label and an outcome-string position.  The argument
    `new_sectors` may be given to rename the kept state-space labels in the
    returned `DataSet`'s gate strings.

    A typical case is when the state-space is that of *n* qubits, and the
    state space labels the intergers 0 to *n-1*.  As stated above, in this
    case there is no need to specify `sindices_to_keep`.  One may want to
    "rebase" the indices to 0 in the returned data set using `new_sectors`
    (E.g. `sectors_to_keep == [4,5,6]` and `new_sectors == [0,1,2]`).
    
    Parameters
    ----------
    dataset : DataSet object
        The input DataSet whose data will be processed.

    sectors_to_keep : list or tuple
        The state-space labels (strings or integers) of the "sectors" to keep in
        the returned DataSet.

    sindices_to_keep : list or tuple, optional
        The 0-based indices of the labels in `sectors_to_keep` which give the
        postiions of the corresponding letters in each outcome string (see above).
        If the state space labels are integers (labeling *qubits*) thath are also
        letter-positions, then this may be left as `None`.  For example, if the 
        outcome strings of `dataset` are '00','01','10',and '11' and the first
        position refers to qubit "Q1" and the second to qubit "Q2" (present in
        gate labels), then to extract just "Q2" data `sectors_to_keep` should be
        `["Q2"]` and `sindices_to_keep` should be `[1]`.

    new_sectors : list or tuple, optional
        New sectors names to map the elements of `sectors_to_keep` onto in the 
        output DataSet's gate strings.  None means the labels are not renamed.
        This can be useful if, for instance, you want to run a 2-qubit protocol
        that expects the qubits to be labeled "0" and "1" on qubits "4" and "5" 
        of a larger set.  Simply set `sectors_to_keep == [4,5]` and 
        `new_sectors == [0,1]`.

    Returns
    -------
    filtered_dataset : DataSet object
        The DataSet with outcomes and gate strings filtered as described above.
    """
    if sindices_to_keep is None:
        sindices_to_keep = sectors_to_keep

    ds_merged = merge_outcomes(dataset, create_merge_dict(sindices_to_keep,
                                             dataset.get_outcome_labels()))
    ds_merged = ds_merged.copy_nonstatic()
    ds_merged.process_gate_strings(lambda s: _gstrc.filter_gatestring(
                                       s, sectors_to_keep, new_sectors))
    ds_merged.done_adding_data()
    return ds_merged
