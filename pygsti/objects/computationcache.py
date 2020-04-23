""" Cache for distributed computation """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import numpy as _np


class ComputationCache(object):
    """
    A cache of information and computated quantities for calculations involving
    the same model, dataset, and set of circuits.
    """
    def __init__(self,
                 eval_tree=None, lookup=None, outcomes_lookup=None,
                 wrt_block_size=None, wrt_block_size2=None,
                 counts=None, total_counts=None):
        """
        Create a CompuationCache for storing computated quantities pertaining to a
        fixed model, dataset, and set of circuits.

        Note that the model, dataset, and circuits are *not* stored in this cache,
        and it is up to the user to ensure that a ComputationCache is properly
        associated with a *fixed* model, dataset, and ciruit list.

        Parameters
        ----------
        eval_tree : EvalTree
            An evaluation tree, storing the set of circuits in a simplified
            and structured way in order to improve the efficiency of later computations.

        lookup : collections.OrderedDict
            A dictionary whose keys are integer indices into the master circuit list and
            whose values are slices and/or integer-arrays into the space/axis of
            final elements returned by the 'bulk fill' forward simulator routines.  Thus,
            to get the  final elements corresponding to the ith circuits, use
            `filledArray[ lookup[i] ]`.

        outcomes : collections.OrderedDict
            A dictionary whose keys are integer indices into the master circuit list
            and whose values are lists of outcome labels (an outcome label is a tuple
            of POVM-effect and/or instrument-element labels).  Thus, to obtain
            what outcomes the i-th circuits's final elements (`filledArray[ lookup[i] ]`)
            correspond to, use `outcomes[i]`.
        
        wrt_block_size : int or float, optional
          The maximum average number of derivative columns to compute quantities for
          simultaneously.  None typically means to compute all the columns at once.

        wrt_block_size2 : int or float, optional
          The maximum average number of 2nd derivative columns to compute quantities for
          simultaneously (for those which use second derivatives, e.g. Hessians).

        counts : numpy.ndarray
            A 1D vector of outcome counts, one per computed element.

        total_counts : numpy.ndarray
            A 1D vector of total (per-circuit) outcome counts, one per computed element.
        """
        self.eval_tree = eval_tree
        self.lookup = lookup
        self.outcomes_lookup = outcomes_lookup
        self.wrt_block_size = wrt_block_size
        self.wrt_block_size2 = wrt_block_size2
        self.counts = counts
        self.total_counts = total_counts

    def has_evaltree(self):
        """
        Whether this cache constains an evaluation tree.

        Returns
        -------
        bool
        """
        return (self.eval_tree is not None)

    def add_evaltree(self, model, dataset=None, circuits_to_use=None, resource_alloc=None, subcalls=(), verbosity=0):
        """
        Add an evalution tree to this cache, based on the model, dataset,
        and circuits it is associated with (but which aren't stored within it).

        Parameters
        ----------
        model : Model
            The model to use.
        
        dataset : DataSet
            The dataset to use (for filtering cicuits, so that only circuits
            present in the data set are used)

        circuits_to_use : list of Circuits, optional
            The circuits to use.  If None, then all the circuits in `dataset` are used.

        resource_alloc : ResourceAllocation, optional
            The resources allocated to the computations that will use this
            cache.

        subcalls : list or tuple, optional
            A list of the (string-valued) sub-call names that will be called
            in the future.  This information allows the evaluation tree creation
            to estimate the amount of memory that will be required and better decide
            how to split up computations involving multiple derivative columns.

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        None
        """
        comm = resource_alloc.comm if resource_alloc else None
        mlim = resource_alloc.mem_limit if resource_alloc else None
        distribute_method = resource_alloc.distribute_method
        self.eval_tree, self.wrt_block_size, self.wrt_block_size2, self.lookup, self.outcomes_lookup = \
            model.bulk_evaltree_from_resources(circuits_to_use, comm, mlim, distribute_method,
                                               subcalls, dataset, verbosity)

    def has_count_vectors(self):
        """
        Whether this cache contains .counts and .total_counts vectors of
        per-element outcome counts and total circuit counts.

        Returns
        -------
        bool
        """
        return (self.counts is not None) and (self.total_counts is not None)

    def add_count_vectors(self, dataset, ds_circuits_to_use, circuit_weights=None):
        """
        Compute and add to this cache vectors of per-element outcome counts, taken from `dataset`.

        This cache must already have an evaluation tree (see :method:`has_evaltree`), as
        a tree defines the elements (one value per outcome) that are computed.

        This method adds the .counts and .total_counts members of the cache.

        Parameters
        ----------
        dataset : DataSet
            The dataset to take the counts from.

        ds_circuits_to_use : list of Circuits
            The circuits corresponding to those of the evaluation tree that should
            be used to query the `dataset`.  These are different from the "master"
            list of circuits used to construct the evaluation tree when, e.g. aliases
            are used.

        circuit_weights : numpy.ndarray, optional
            If not None, an array of per-circuit weights which multiply the
            counts extracted for each circuit.

        Returns
        -------
        None
        """
        assert(self.has_evaltree()), "Must `add_evaltree` before calling `add_count_vectors`!"
        nelements = self.eval_tree.num_final_elements()
        counts = _np.empty(nelements, 'd')
        totals = _np.empty(nelements, 'd')

        for (i, circuit) in enumerate(ds_circuits_to_use):
            cnts = dataset[circuit].counts
            totals[self.lookup[i]] = sum(cnts.values())  # dataset[opStr].total
            counts[self.lookup[i]] = [cnts.get(x, 0) for x in self.outcomes_lookup[i]]

        if circuit_weights is not None:
            for i in range(len(ds_circuits_to_use)):
                counts[self.lookup[i]] *= circuit_weights[i]  # dim nelements (K = nSpamLabels, M = nCircuits )
                totals[self.lookup[i]] *= circuit_weights[i]  # multiply N's by weights

        self.counts = counts
        self.total_counts = totals
