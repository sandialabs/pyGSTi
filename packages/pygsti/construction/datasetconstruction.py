from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for creating datasets """

import numpy as _np
import numpy.random as _rndm

from ..objects import gatestring as _gs
from ..objects import dataset as _ds

def generate_fake_data(gatesetOrDataset, gatestring_list, nSamples,
                       sampleError="none", seed=None, randState=None):
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

    Returns
    -------
    DataSet
       A static data set filled with counts for the specified gate strings.

    """
    if isinstance(gatesetOrDataset, _ds.DataSet):
        dsGen = gatesetOrDataset #dataset
        gsGen = None
        dataset = _ds.DataSet( spamLabels=dsGen.get_spam_labels() )
    else:
        gsGen = gatesetOrDataset #dataset
        dsGen = None
        dataset = _ds.DataSet( spamLabels=gsGen.get_spam_labels() )

    if sampleError in ("binomial","multinomial"):
        if randState is None:
            rndm = _rndm.RandomState(seed) # ok if seed is None
        else:
            rndm = randState

    for k,s in enumerate(gatestring_list):
        if gsGen:
            ps = gsGen.probs(s) # a dictionary of probabilities; keys = spam labels
        else:
            ps = { sl: dsGen[s].fraction(sl) for sl in dsGen.get_spam_labels() }

        if nSamples is None and dsGen is not None:
            N = dsGen[s].total() #use the number of samples from the generating dataset
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

        counts = { }
        if sampleError == "binomial":
            assert(len(list(ps.keys())) == 2)
            spamLabel1, spamLabel2 = sorted(list(ps.keys())); p1 = ps[spamLabel1]
            if p1 < 0 and abs(p1) < 1e-6: p1 = 0
            if p1 > 1 and abs(p1-1.0) < 1e-6: p1 = 1
            if p1 < 0 or p1 > 1: print("Warning: probability == %g clipped to generate fake data" % p1)
            p1 = _np.clip(p1,0,1)
            counts[spamLabel1] = rndm.binomial(nWeightedSamples, p1) #numpy.clip(p1,0,1) )
            counts[spamLabel2] = nWeightedSamples - counts[spamLabel1]
        elif sampleError == "multinomial":
            #nOutcomes = len(list(ps.keys()))
            spamLabels = list(ps.keys())
            countsArray = rndm.multinomial(nWeightedSamples,
                    [ps[sl] for sl in spamLabels], size=1)
            for i,spamLabel in enumerate(spamLabels):
                counts[spamLabel] = countsArray[0,i]
        else:
            for (spamLabel,p) in ps.items():
                pc = _np.clip(p,0,1)
                if sampleError == "none":
                    counts[spamLabel] = float(nWeightedSamples * pc)
                elif sampleError == "round":
                    counts[spamLabel] = int(round(nWeightedSamples*pc))
                else: raise ValueError("Invalid sample error parameter: '%s'  Valid options are 'none', 'round', 'binomial', or 'multinomial'" % sampleError)

        dataset.add_count_dict(s, counts)
    dataset.done_adding_data()
    return dataset


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
       Seed for numpy's random number gernerator.

    Returns
    -------
    DataSet
    """

    rndm = _np.random.RandomState(seed)
    ds = _ds.DataSet(spamLabels=['plus','minus'])
    gateStrings = list(expRBdataset.keys())
    spamLabels = expRBdataset.get_spam_labels()

    possibleSpamLabels = gateset.get_spam_labels()
    assert( all([sl in possibleSpamLabels for sl in spamLabels]) )

    for s in gateStrings:
        N = expRBdataset[s].total()
        ps = gateset.probs(s)
        pList = [ ps[sl] for sl in spamLabels ]
        countsArray = rndm.multinomial(N, pList, 1)
        counts = { sl: countsArray[0,i] for i,sl in enumerate(spamLabels) }
        ds.add_count_dict(s, counts)
    ds.done_adding_data()
    return ds

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
    gateStrings = list(expRBdataset.keys())
    return generate_fake_data(gateset,gateStrings,N,sampleError='none')
