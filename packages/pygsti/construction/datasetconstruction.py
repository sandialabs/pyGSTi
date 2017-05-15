from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for creating datasets """

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
                       aliasDict=None, collisionAction="aggregate",
                       measurementGates=None):
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

    measurementGates : dict, optional
        If not None, a dictrionary whose keys are user-defined "measurement
        labels" and whose values are lists of gate labels.  The gate labels 
        in each list define the set of gates which describe the the operation
        that is performed contingent on a *specific outcome* of the measurement
        labelled by the key.  For example, `{ 'Zmeasure': ['Gmz_plus','Gmz_minus'] }`.


    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.

    """
    TOL = 1e-10
    
    if isinstance(gatesetOrDataset, _ds.DataSet):
        dsGen = gatesetOrDataset #dataset
        gsGen = None
        dataset = _ds.DataSet( spamLabels=dsGen.get_spam_labels(),
                               collisionAction=collisionAction,
                               measurementGates=measurementGates)
    else:
        gsGen = gatesetOrDataset #dataset
        dsGen = None
        dataset = _ds.DataSet( spamLabels=gsGen.get_spam_labels(),
                               collisionAction=collisionAction,
                               measurementGates=measurementGates)

    if sampleError in ("binomial","multinomial"):
        if randState is None:
            rndm = _rndm.RandomState(seed) # ok if seed is None
        else:
            rndm = randState


    for k,s0 in enumerate(gatestring_list):

        all_ps = _collections.OrderedDict() #holds probs for *all* intermediate measuement branches
        for s in expand_intermediate_measurements(s0, measurementGates):
            trans_s = _gstrc.translate_gatestring(s, aliasDict)
            if gsGen:
                ps = gsGen.probs(trans_s) 
                  # a dictionary of probabilities; keys = spam labels

                if sampleError in ("binomial","multinomial"):
                    #Adjust to probabilities if needed (and warn if not close to in-bounds)
                    for sl in ps: 
                        if ps[sl] < 0:
                            if ps[sl] < -TOL: _warnings.warn("Clipping probs < 0 to 0")
                            ps[sl] = 0.0
                        elif ps[sl] > 1: 
                            if ps[sl] > (1+TOL): _warnings.warn("Clipping probs > 1 to 1")
                            ps[sl] = 1.0
            else:
                ps = { sl: dsGen[trans_s].fraction(sl) 
                       for sl in dsGen.get_spam_labels() }
                
            for sl in sorted(list(ps.keys())):
                all_ps[(s,sl)] = ps[sl] #add to all_ps

        if gsGen and sampleError in ("binomial","multinomial"):
            #Check that sum ~= 1 (and nudge if needed) since binomial and
            #  multinomial random calls assume this.
            psum = sum(all_ps.values())
            if psum > 1:
                if psum > 1+TOL: _warnings.warn("Adjusting sum(probs) > 1 to 1")
                extra_p = (psum-1.0) * (1.000000001) # to sum < 1+eps (numerical prec insurance)
                for lbl in all_ps:
                    if extra_p > 0:
                        x = min(all_ps[lbl],extra_p)
                        all_ps[lbl] -= x; extra_p -= x
                    else: break
            #TODO: add adjustment if psum < 1?
            assert(1.-TOL <= sum(all_ps.values()) <= 1.+TOL)
                
        if nSamples is None and dsGen is not None:
            trans_s0 = _gstrc.translate_gatestring(s0, aliasDict)
            N = dsGen[trans_s0].total() #use the number of samples from the generating dataset
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

        counts = _collections.defaultdict(dict)
        labels = list(all_ps.keys()) # "label" == (gatestring, spamLabel)
        if sampleError == "binomial":
            assert(len(labels) == 2)
            gstr0,sl0 = labels[0]
            gstr1,sl1 = labels[1]
            counts[gstr0][sl0] = rndm.binomial(nWeightedSamples, all_ps[labels[0]])
            counts[gstr1][sl1] = nWeightedSamples - counts[gstr0][sl0]
            
        elif sampleError == "multinomial":
            countsArray = rndm.multinomial(nWeightedSamples,
                    list(all_ps.values()), size=1)
            for i,(gstr,sl) in enumerate(labels):
                counts[gstr][sl] = countsArray[0,i]
                
        else:
            for (gstr,spamLabel),p in all_ps.items():
                pc = _np.clip(p,0,1)
                if sampleError == "none":
                    counts[gstr][spamLabel] = float(nWeightedSamples * pc)
                elif sampleError == "round":
                    counts[gstr][spamLabel] = int(round(nWeightedSamples*pc))
                else: raise ValueError("Invalid sample error parameter: '%s'  Valid options are 'none', 'round', 'binomial', or 'multinomial'" % sampleError)

        for s in expand_intermediate_measurements(s0, measurementGates):
            dataset.add_count_dict(s, counts[s])
    dataset.done_adding_data()
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
        if a two-qubit DataSet has outcome labels "upup", "updn", "dnup", and "dndn", and
        we want to ''trace out'' the second qubit, we could use label_merge_dict =
        {'plus':['upup','updn'],'minus':['dnup','dndn']}.
    
    Returns
    -------
    merged_dataset : DataSet object
        The DataSet with outcomes merged according to the rules given in label_merge_dict.
    """

    new_effects = label_merge_dict.keys()
    merged_dataset = _ds.DataSet(spamLabels=new_effects)
    if sorted([effect for sublist in label_merge_dict.values() for effect in sublist]) != sorted(dataset.get_spam_labels()):
        print('Warning: There is a mismatch between original effects in label_merge_dict and original effects in original dataset.')
    for key in dataset.keys():
        dataline = dataset[key]
        count_dict = {}
        for new_effect in new_effects:
            count_dict[new_effect] = 0
            for old_effect in label_merge_dict[new_effect]:
                count_dict[new_effect] += dataline[old_effect]
        merged_dataset.add_count_dict(key,count_dict)
    merged_dataset.done_adding_data()
    return merged_dataset




def expand_intermediate_measurements(gatestring, measurementGates):
    """
    Gets an iterator over the intermediate-measurement branches of `gatestring`.

    When `gatestring` contains "measurement labels" (given by the keys of 
    `measurementGates`) there exist multiple "intermediate-measurement
    branches", each defined/constructed by replacing every measurement
    label by one of its "measurement gates" (given by one of the elements
    of the list-valued values of `measurementGates`).  The returned iterator
    loops over all such branches.  

    If `gatestring` does *not* contain any measurement labels, or if
    `measurementGates` is None, then the iterator just loops over the single
    string, `gatestring`.

    Parameters
    ----------
    gatestring : GateString
        The "base" gate sequence which may or may not contain measurement
        labels.

    measurementGates : dict
        If not None, a dictrionary whose keys are user-defined "measurement
        labels" and whose values are lists of "measurement gate" labels.  The
        labels in each list define the set of gates which describe the the
        operation that is performed contingent on a *specific outcome* of the
        measurement labelled by the key.  
        For example, `{ 'Zmeasure': ['Gmz_plus','Gmz_minus'] }`.

    Returns
    -------
    iterator
    """
    if measurementGates is None or len(gatestring) == 0:
        yield gatestring
        return
    
    gateLabelsByPos = [ measurementGates.get(lbl,[lbl]) for lbl in gatestring ]
    for gatelabels in _itertools.product(*gateLabelsByPos):
        yield _gs.GateString(tuple(gatelabels))

    
