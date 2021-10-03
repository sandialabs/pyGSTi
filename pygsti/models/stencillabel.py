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
    """
    A generalization of a simple tuple of state space labels.

    A :class:`StencilLabel` can be a simple tuple of absolute state space labels, but
    it can also contain special labels identifying the target labels of a gate (`@<int>`)
    and qubit-graph directions relative to target labels, e.g. `@0+left`.  Furthermore,
    a stencil label can expand into multiple state-space label tuples, e.g. the 2-tuples
    of all the qubit-graph edges.

    Parameters
    ----------
    local_state_space : StateSpace
        A manually supplied local state space for this label, which is returned by
        :method:`create_local_state_space` instead of generating a local state space.
    """

    @classmethod
    def cast(cls, obj):
        """
        Convert an object into a stencil label if it isn't already.

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        StencilLabel
        """
        if obj is None: return StencilLabelTuple(None)
        if isinstance(obj, StencilLabel): return obj
        if isinstance(obj, tuple): return StencilLabelTuple(obj)
        if isinstance(obj, list): return StencilLabelSet(obj)
        raise ValueError("Cannot cast %s to a StencilLabel" % str(type(obj)))

    def __init__(self, local_state_space=None):
        self.local_state_space = local_state_space

    def _resolve_single_sslbls_tuple(self, sslbls, qubit_graph, state_space, target_lbls):
        if qubit_graph is None:  # without a graph, we need to ensure all the stencil_sslbls are valid
            # We still can resolve @<num> references to target labels, even without a graph
            try:
                resolved_sslbls = tuple([(target_lbls[int(s[1:])] if (isinstance(s, str) and s.startswith("@")) else s)
                                         for s in sslbls])
            except Exception:
                raise ValueError(("Could not resolve the relative ('@'-prefixed) labels in %s!"
                                  " Maybe needs a qubit graph?") % str(sslbls))

            assert (state_space.contains_labels(resolved_sslbls))
            return resolved_sslbls
        else:
            ret = [qubit_graph.resolve_relative_nodelabel(s, target_lbls) for s in sslbls]
            if any([x is None for x in ret]): return None  # signals there is a non-present dirs, e.g. end of chain
            return tuple(ret)

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        """
        Creates a list of all the state space label tuples this stencil label expands into.

        To perform the expansion, the qubit graph of the relevant processor is required, along
        with the target state space labels, which determine where the stencil is placed on the
        qubit graph.  The returned list of tuples contains *absolute* state space labels, meaning
        that they are labels in `state_space` and do not use any special directives.

        Parameters
        ----------
        qubit_graph : QubitGraph
            The qubit graph of the relevant processor, used to resolve any special stencil labels.

        state_space : StateSpace
            The state space for the entire processor.  This specifies what the state space labels are.

        target_lbls : tuple
            The target state space labels, specifying where the stencil is placed on the qubit graph
            before being expanded into absolute state space labels.

        Returns
        -------
        list
            The state space label tuples this stencil label expands into, e.g. `[('Q0','Q1'), ('Q1','Q2')]`.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def create_local_state_space(self, entire_state_space):
        """
        Creates a "local" state space for an operator indexed using this stencil label.

        When creating operator objects, a stencil label specifies where to place (embed)
        operators on some subset of the entire space.  When these to-be-embedded operations
        are constructed, they need to be supplied with a state space that just corresponds to
        the sub-space where they act -- a "local" state space.  A stencil label expands into
        one or more state space label tuples, and this function constructs a *single* local
        state space that is appropriate for any and all of these tuples (i.e. what is returned
        by :method:`compute_absolute_sslbls`), and that is therefore appropriate for constructing
        the to-be-embedded operation.  Importantly, this function can be called without knowing
        where this stencil label will be placed, that is, it doesn't require a "target labels"
        argument.

        Parameters
        ----------
        entire_state_space : StateSpace
            The entire state space of the relevant processor, specifying the state space labels
            and information about them.

        Returns
        -------
        StateSpace
        """
        if self.local_state_space is not None:
            return self.local_state_space  # so the user can always override this space if needed
        else:
            return self._create_local_state_space(entire_state_space)

    def _create_local_state_space(self, entire_state_space):
        """ Stub that derived classes implement - same function as :method:`create_local_state_space` """
        raise NotImplementedError("Derived classes should implement this!")

    def _create_local_state_space_for_sslbls(self, sslbls, entire_state_space):
        """
        A helper function for derived class implementations of _create_local_state_space(...)

        Constructs a sub-space of a large state space corresponding to a subset (`sslbls`) of the
        large state space's labels.  We call this the "local" state space for these labels.
        It's possible that `sslbls` contains special stencil labels, e.g. "@0", and in this case
        the entire state space must have labels with similar dimensions so it only matters how
        many labels (not which ones specifically) a local space is needed for.
        """
        if entire_state_space.contains_labels(sslbls):  # absolute sslbls - get space directly
            return entire_state_space.create_subspace(sslbls)
        else:
            return entire_state_space.create_stencil_subspace(sslbls)
            # only works when state space has a common label dimension


class StencilLabelTuple(StencilLabel):
    """
    A stencil label that is a single state space label tuple.

    This is the simplest type of stencil labels, and almost the same as just a tuple of
    state space labels.  It may contain, however, special stencil directives like `'@<num>'`
    and direction labels.

    Parameters
    ----------
    stencil_sslbls : tuple
        A tuple of state space labels.  May contain special stencil directives.
    """
    def __init__(self, stencil_sslbls):
        self.sslbls = stencil_sslbls
        super(StencilLabelTuple, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        """
        Creates a list of all the state space label tuples this stencil label expands into.

        See :method:`StencilLabel.compute_absolute_sslbls`
        """
        # Return a *list* of sslbls, since some stencil labels may resolve into multiple absolute sslbls
        if self.sslbls is None:
            return [None]  # sslbls=None is resolved to `None`
        return [self._resolve_single_sslbls_tuple(self.sslbls, qubit_graph, state_space, target_lbls)]

    def _create_local_state_space(self, entire_state_space):
        return self._create_local_state_space_for_sslbls(self.sslbls, entire_state_space)

    def __str__(self):
        return "StencilLabel(" + str(self.sslbls) + ")"


class StencilLabelSet(StencilLabel):
    """
    A stencil label that is explicitly a set of multiple state space label tuples.

    A :class:`StencilLabelSet` stencil label simply expands into the list/set of
    state space label tuples used to construct it.  It may contain special stencil
    directives, and is essentially just a list or tuple of :class:`StencilLabelTuple`
    objects.

    Parameters
    ----------
    stencil_sslbls_set : list or tuple or set
        A collection of the individual state space label tuples that this stencil
        label expands into.  May contain special stencil directives.
    """
    def __init__(self, stencil_sslbls_set):
        self.sslbls_set = stencil_sslbls_set
        super(StencilLabelSet, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        """
        Creates a list of all the state space label tuples this stencil label expands into.

        See :method:`StencilLabel.compute_absolute_sslbls`
        """
        # return a *list* of sslbls, since some stencil labels may resolve into multiple absolute sslbls
        return [self._resolve_single_sslbls_tuple(sslbls, qubit_graph, state_space, target_lbls)
                for sslbls in self.sslbls_set]

    def _create_local_state_space(self, entire_state_space):
        if len(self.sslbls_set) == 0: return None  # or an empty space?
        return self._create_local_state_space_for_sslbls(self.sslbls_set[0], entire_state_space)

    def __str__(self):
        return "StencilLabel{" + str(self.sslbls_set) + "}"


class StencilLabelAllCombos(StencilLabel):
    """
    A stencil label that expands into all the length-k combinations of a larger set of state space labels.

    For example, if `num_to_choose = 2` and `possible_sslbls = [1,2,3]` then this
    stencil label would expand into the pairs: `[(1,2), (2,3), (1,3)]`.  Optionally,
    the tuples can be restricted to only those that form a connected sub-graph of the
    qubit graph (eliminating `(1,3)` in our example if the qubits were in a 1-2-3 chain).

    Parameters
    ----------
    possible_sslbls : list or tuple
        The set of possible state space labels to take combinations of.

    num_to_choose : int
        The number of possible state space labels to choose when forming each
        state space label tuple.

    connected : bool, optional
        If `True`, restrict combinations to those that form a connected subgraph
        of the qubit graph.
    """
    def __init__(self, possible_sslbls, num_to_choose, connected=False):
        self.possible_sslbls = possible_sslbls
        self.num_to_choose = num_to_choose
        self.connected = connected
        super(StencilLabelAllCombos, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        """
        Creates a list of all the state space label tuples this stencil label expands into.

        See :method:`StencilLabel.compute_absolute_sslbls`
        """
        ret = []
        for chosen_sslbls in _itertools.combinations(self.possible_sslbls, self.num_to_choose):
            resolved_chosen_sslbls = self._resolve_single_sslbls_tuple(chosen_sslbls, qubit_graph,
                                                                       state_space, target_lbls)
            if self.connected and len(chosen_sslbls) == 2 \
               and qubit_graph.is_directly_connected(resolved_chosen_sslbls[0],
                                                     resolved_chosen_sslbls[1]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph
            ret.append(resolved_chosen_sslbls)
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
    """
    A stencil label specifying all length-k combinations of the labels within a radius of some "base" label(s).

    This stencil label depends heavily on the qubit graph.  It expands to all length-`num_to_choose`
    combinations of the qubit graphy nodes (state space labels) that lie within `radius` edge traversals
    of *any* of the node labels in `base_sslbls`.  If `connected=True` then the combinations are further
    filtered so that they must form connected subgraphs of the qubit graph.

    Parameters
    ----------
     base_sslbls : tuple
         The state space labels that form the "center" of the possible labels

    radius : int
        The maximum number of edge traversals (along the qubit graph) used to define the "radius" of
        labels about the base labels.

    num_to_choose : int
        The number of possible state space labels in each combination of the labels within the radius
        of possible labels.  This stencil label expands into potentially many state space label tuples
        all of this length.

    connected : bool, optional
        If `True`, restrict combinations to those that form a connected subgraph
        of the qubit graph.
    """
    def __init__(self, base_sslbls, radius, num_to_choose, connected=False):
        self.base_sslbls = base_sslbls
        self.radius = radius  # in "hops" along graph
        self.num_to_choose = num_to_choose
        self.connected = connected
        super(StencilLabelRadiusCombos, self).__init__()

    def compute_absolute_sslbls(self, qubit_graph, state_space, target_lbls):
        """
        Creates a list of all the state space label tuples this stencil label expands into.

        See :method:`StencilLabel.compute_absolute_sslbls`
        """
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
