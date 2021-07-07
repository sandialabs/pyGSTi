"""
Stencil label classes and supporting functions.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import pygsti.baseobjs.statespace as _statespace


class StencilLabel(object):

    @classmethod
    def cast(cls, obj):
        if obj is None: return StencilLabelTuple(None)
        if isinstance(obj, StencilLabel): return obj
        if isinstance(obj, tuple): return StencilLabelTuple(obj)
        if isinstance(obj, list): return StencilLabelSet(obj)
        raise ValueError("Cannot cast %s to a StencilLabel" % str(type(obj)))

    def __init__(self, local_state_space=None):
        self.local_state_space = local_state_space

    def _resolve_single_sslbls_tuple(self, sslbls, qubit_graph, state_space, target_lbls):
        if qubit_graph is None:  # without a graph, we need to ensure all the stencil_sslbls are valid
            assert (state_space.contains_labels(sslbls))
            return sslbls
        else:
            ret = [qubit_graph.resolve_relative_nodelabel(s, target_lbls) for s in sslbls]
            if any([x is None for x in ret]): return None  # signals there is a non-present dirs, e.g. end of chain
            return tuple(ret)

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        raise NotImplementedError("Derived classes should implement this!")

    def create_local_state_space(self, entire_state_space):
        """ Creates a local state space for an operator indexed using this stencil label """
        if self.local_state_space is not None:
            return self.local_state_space  # so the user can always override this space if needed
        else:
            return self._create_local_state_space(entire_state_space)

    def _create_local_state_space(self, entire_state_space):
        raise NotImplementedError("Derived classes should implement this!")

    def _create_local_state_space_for_sslbls(self, sslbls, entire_state_space):
        """ A helper function for derived class implementations of _create_local_state_space(...) """
        if entire_state_space.contains_labels(sslbls):  # absolute sslbls - get space directly
            return entire_state_space.create_subspace(sslbls)
        else:
            return entire_state_space.create_stencil_subspace(sslbls)
            # only works when state space has a common label dimension


class StencilLabelTuple(StencilLabel):
    def __init__(self, stencil_sslbls):
        self.sslbls = stencil_sslbls
        super(StencilLabelTuple, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        # Return a *list* of sslbls, since some stencil labels may resolve into multiple absolute sslbls
        if self.sslbls is None:
            return [None]  # sslbls=None is resolved to `None`
        return [self._resolve_single_sslbls_tuple(self.sslbls, qubit_graph, state_space, target_lbls)]

    def _create_local_state_space(self, entire_state_space):
        return self._create_local_state_space_for_sslbls(self.sslbls, entire_state_space)

    def __str__(self):
        return "StencilLabel(" + str(self.sslbls) + ")"


class StencilLabelSet(StencilLabel):
    def __init__(self, stencil_sslbls_set):
        self.sslbls_set = stencil_sslbls_set
        super(StencilLabelSet, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        # return a *list* of sslbls, since some stencil labels may resolve into multiple absolute sslbls
        return [self._resolve_single_sslbls_tuple(sslbls, qubit_graph, state_space, target_lbls)
                for sslbls in self.sslbls_set]

    def _create_local_state_space(self, entire_state_space):
        if len(self.sslbls_set) == 0: return None  # or an empty space?
        return self._create_local_state_space_for_sslbls(self.sslbls_set[0], entire_state_space)

    def __str__(self):
        return "StencilLabel{" + str(self.sslbls_set) + "}"


class StencilLabelAllCombos(StencilLabel):
    def __init__(self, possible_sslbls, num_to_choose, connected=False):
        self.possible_sslbls = possible_sslbls
        self.num_to_choose = num_to_choose
        self.connected = connected
        super(StencilLabelAllCombos, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        ret = []
        for chosen_sslbls in _itertools.combinations(self.possible_sslbls, self.num_to_choose):
            if self.connected and len(chosen_sslbls) == 2 \
                    and not qubit_graph.is_directly_connected(chosen_sslbls[0], chosen_sslbls[1]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph
            ret.append(self._resolve_single_sslbls_tuple(chosen_sslbls, qubit_graph, state_space, target_lbls))
        return ret  # return a *list* of

    def _create_local_state_space(self, entire_state_space):
        common_udim = entire_state_space.common_udimension
        if common_udim is None:
            raise ValueError(("All-combos stencil labels can only be used with state spaces that"
                              " have a common label dimension"))
        lbls = tuple(range(self.num_to_choose)); udims = (common_udim,) * self.num_to_choose
        return _statespace.ExplicitStateSpace(lbls, udims)

    def __str__(self):
        return ("StencilCombos(" + str(self.possible_sslbls) + (" connected-" if self.connected else " ")
                + "choose %d" % self.num_to_choose + ")")


class StencilLabelRadiusCombos(StencilLabel):
    def __init__(self, base_sslbls, radius, num_to_choose, connected=False):
        self.base_sslbls = base_sslbls
        self.radius = radius  # in "hops" along graph
        self.num_to_choose = num_to_choose
        self.connected = connected
        super(StencilLabelRadiusCombos, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        ret = []
        assert (qubit_graph is not None), "A qubit graph is required by StencilLabelRadiusCombos!"
        abs_base_sslbls = self._resolve_single_sslbls_tuple(self.base_sslbls, qubit_graph, state_space, target_lbls)
        radius_nodes = qubit_graph.radius(abs_base_sslbls, self.radius)

        for chosen_sslbls in _itertools.combinations(radius_nodes, self.num_to_choose):
            if self.connected and len(chosen_sslbls) == 2 \
                    and not qubit_graph.is_directly_connected(chosen_sslbls[0], chosen_sslbls[1]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph
            ret.append(self._resolve_single_sslbls_tuple(chosen_sslbls, qubit_graph, state_space, target_lbls))
        return ret  # return a *list* of sslbls

    def _create_local_state_space(self, entire_state_space):
        # A duplicate of StencilLabelAllCombos._create_local_state_space
        common_udim = entire_state_space.common_udimension
        if common_udim is None:
            raise ValueError(("All-combos stencil labels can only be used with state spaces that"
                              " have a common label dimension"))
        lbls = tuple(range(self.num_to_choose)); udims = (common_udim,) * self.num_to_choose
        return _statespace.ExplicitStateSpace(lbls, udims)

    def __str__(self):
        return ("StencilRadius(%schoose %d within %d hops from %s)" % (("connected-" if self.connected else ""),
                                                                       self.num_to_choose, self.radius,
                                                                       str(self.base_sslbls)))
