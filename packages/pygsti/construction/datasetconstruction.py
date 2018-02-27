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
from . import gatestringconstruction as _gstrc

def generate_fake_data(gatesetOrDataset, gatestring_list, nSamples,
                       sampleError="none", seed=None, randState=None,
                       aliasDict=None, collisionAction="aggregate", comm=None):
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


    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.

    """
    TOL = 1e-10
    
    if isinstance(gatesetOrDataset, _ds.DataSet):
        dsGen = gatesetOrDataset
        gsGen = None
        dataset = _ds.DataSet( collisionAction=collisionAction )
    else:
        gsGen = gatesetOrDataset
        dsGen = None
        dataset = _ds.DataSet( collisionAction=collisionAction )

    if comm is None or comm.Get_rank() == 0: # only root rank computes
        if sampleError in ("binomial","multinomial"):
            if randState is None:
                rndm = _rndm.RandomState(seed) # ok if seed is None
            else:
                rndm = randState

        if gsGen:
            trans_gatestring_list = [ _gstrc.translate_gatestring(s, aliasDict)
                                      for s in gatestring_list ]
            all_probs = gsGen.bulk_probs(trans_gatestring_list)
            #all_dprobs = gsGen.bulk_dprobs(gatestring_list) #DEBUG - not needed here!!!
                
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
                if psum > 1:
                    if psum > 1+TOL: _warnings.warn("Adjusting sum(probs) > 1 to 1")
                    extra_p = (psum-1.0) * (1.000000001) # to sum < 1+eps (numerical prec insurance)
                    for lbl in ps:
                        if extra_p > 0:
                            x = min(ps[lbl],extra_p)
                            ps[lbl] -= x; extra_p -= x
                        else: break
                #TODO: add adjustment if psum < 1?
                assert(1.-TOL <= sum(ps.values()) <= 1.+TOL)
                    
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
            labels = sorted(list(ps.keys())) # "outcome labels" - sort for consistent generation
            if sampleError == "binomial":
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
        we want to ''trace out'' the second qubit, we could use label_merge_dict =
        {'0':['00','01'],'1':['10','11']}.
    
    Returns
    -------
    merged_dataset : DataSet object
        The DataSet with outcomes merged according to the rules given in label_merge_dict.
    """

    new_outcomes = label_merge_dict.keys()
    merged_dataset = _ds.DataSet(outcomeLabels=new_outcomes)
    if sorted([outcome for sublist in label_merge_dict.values() for outcome in sublist]) != sorted(dataset.get_outcome_labels()):
        print('Warning: There is a mismatch between original outcomes in label_merge_dict and outcomes in original dataset.')
    for key in dataset.keys():
        dataline = dataset[key]
        count_dict = {}
        for new_outcome in new_outcomes:
            count_dict[new_outcome] = 0
            for old_outcome in label_merge_dict[new_outcome]:
                count_dict[new_outcome] += dataline[old_outcome]
        merged_dataset.add_count_dict(key,count_dict)
    merged_dataset.done_adding_data()
    return merged_dataset

