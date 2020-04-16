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
    def __init__(self,
                 eval_tree=None, lookup=None, outcomes_lookup=None, wrt_blk_size=None, wrt_blk_size2=None,
                 counts=None, total_counts=None):
        self.eval_tree = eval_tree
        self.lookup = lookup
        self.outcomes_lookup = outcomes_lookup
        self.wrt_blk_size = wrt_blk_size
        self.wrt_blk_size2 = wrt_blk_size2
        self.counts = counts
        self.total_counts = total_counts

    def has_evaltree(self):
        return (self.eval_tree is not None)

    def add_evaltree(self, model, dataset=None, circuits_to_use=None, resource_alloc=None, subcalls=(), verbosity=0):
        """TODO: docstring """
        comm = resource_alloc.comm if resource_alloc else None
        mlim = resource_alloc.mem_limit if resource_alloc else None
        distribute_method = resource_alloc.distribute_method
        self.eval_tree, self.wrt_blk_size, self.wrt_blk_size2, self.lookup, self.outcomes_lookup = \
            model.bulk_evaltree_from_resources(circuits_to_use, comm, mlim, distribute_method,
                                               subcalls, dataset, verbosity)

    def has_count_vectors(self):
        return (self.counts is not None) and (self.total_counts is not None)

    def add_count_vectors(self, dataset, circuits_to_use, ds_circuits_to_use, circuit_weights=None):
        assert(self.has_evaltree()), "Must `add_evaltree` before calling `add_count_vectors`!"
        nelements = self.eval_tree.num_final_elements()
        counts = _np.empty(nelements, 'd')
        totals = _np.empty(nelements, 'd')

        for (i, circuit) in enumerate(ds_circuits_to_use):
            cnts = dataset[circuit].counts
            totals[self.lookup[i]] = sum(cnts.values())  # dataset[opStr].total
            counts[self.lookup[i]] = [cnts.get(x, 0) for x in self.outcomes_lookup[i]]

        if circuit_weights is not None:
            for i in range(len(circuits_to_use)):
                counts[self.lookup[i]] *= circuit_weights[i]  # dim nelements (K = nSpamLabels, M = nCircuits )
                totals[self.lookup[i]] *= circuit_weights[i]  # multiply N's by weights

        self.counts = counts
        self.total_counts = totals
