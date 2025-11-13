"""
Base of the object-oriented model for modelpacks
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import gzip as _gzip
import pickle as _pickle
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from pathlib import Path as _Path

import numpy as _np
import collections as _collections

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.circuits.circuitconstruction import to_circuits as _circuit_list
from pygsti.models.modelconstruction import create_explicit_model_from_expressions as _build_explicit_model
from pygsti.circuits.gstcircuits import create_lsgst_circuit_lists as _make_lsgst_lists
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.polynomial import bulk_load_compact_polynomials as _bulk_load_compact_polys
from pygsti.protocols import gst as _gst
from pygsti.tools import optools as _ot
from pygsti.tools import basistools as _bt
from pygsti.tools.legacytools import deprecate as _deprecated_fn
from pygsti.processors import QubitProcessorSpec as _QubitProcessorSpec
from pygsti.models import Model as _Model
import os as _os
class ModelPack(_ABC):
    """
    ABC of all derived modelpack types

    Attributes
    ----------
    description : str
        a description of the model pack.

    gates : list
        a list of the gate labels of this model pack.

    _sslbls : tuple
        a tuple of the state space labels (usually *qubit* labels) of this model pack.
    """
    description = None
    gates = None
    _sslbls = None

    def __init__(self):
        self._gscache = {}

    @_abstractmethod
    def _target_model(self, sslbls, **kwargs):
        pass

    def _build_explicit_target_model(self, sslbls, gate_names, gate_expressions, **kwargs):
        """
        A helper function for derived classes which create explicit models.
        Updates gate names and expressions with a given set of state-space labels.
        """
        full_sslbls = [sslbls]  # put all sslbls in single tensor product block
        sslbl_map = {i: sslbl for i, sslbl in enumerate(sslbls)}
        updated_gatenames = [_Label(gn).map_state_space_labels(sslbl_map) for gn in gate_names]
        updated_gateexps = [gexp.format(*sslbls) for gexp in gate_expressions]
        return _build_explicit_model(full_sslbls, updated_gatenames, updated_gateexps, **kwargs)

    def target_model(self, gate_type="full", prep_type="auto", povm_type="auto", instrument_type="auto",
                     simulator="auto", evotype='default', qubit_labels=None):
        """
        Returns a copy of the target model in the given parameterization.

        Parameters
        ----------
        parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
            The gate and SPAM vector parameterization type. See
            :func:`Model.set_all_parameterizations` for all allowed values.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The simulator (or type) to be used for model calculations (leave as
            "auto" if you're not sure what this is).

        qubit_labels : tuple, optional
            A tuple of qubit labels, e.g. ('Q0', 'Q1') or (0, 1).  The default
            are the integers starting at 0.

        evotype : Evotype or str, optional
            The evolution type of this model, describing how states are
            represented.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        Returns
        -------
        Model
        """
        qubit_labels = self._sslbls if (qubit_labels is None) else tuple(qubit_labels)
        assert(len(qubit_labels) == len(self._sslbls)), \
            "Expected %d qubit labels and got: %s!" % (len(self._sslbls), str(qubit_labels))
        if gate_type == 'FOGI-GLND':
            assert hasattr(self, 'serialized_fogi_path'), 'This modelpack does not have a FOGI version yet. Please create it manually'
            assert povm_type == 'FOGI-GLND' or povm_type == 'auto', 'modelpack FOGI models have only been implemented for full FOGI-GLND.'
            assert prep_type == 'FOGI-GLND' or prep_type == 'auto', 'modelpack FOGI models have only been implemented for full FOGI-GLND.'
            path = _os.path.dirname(_os.path.abspath(__file__)) + '/serialized_fogi_models/' + self.serialized_fogi_path
            return _Model.read(path)

        cache_key = (gate_type, prep_type, povm_type, instrument_type, simulator, evotype, qubit_labels)
        if cache_key not in self._gscache:
            # cache miss
            mdl = self._target_model(qubit_labels, gate_type=gate_type, prep_type=prep_type, povm_type=povm_type,
                                     instrument_type=instrument_type, evotype=evotype)

            # Set the simulator (if auto, setter initializes to matrix or map)
            mdl.sim = simulator

            # finally cache result
            self._gscache[cache_key] = mdl

        return self._gscache[cache_key].copy()

    def processor_spec(self, qubit_labels=None):
        """
        Create a processor specification for this model pack with the given qubit labels.

        Parameters
        ----------
        qubit_labels : tuple, optional
            A tuple of qubit labels, e.g. ('Q0', 'Q1') or (0, 1).  The default
            are the integers starting at 0.

        Returns
        -------
        QubitProcessorSpec
        """
        static_target_model = self.target_model('static', qubit_labels=qubit_labels)  # assumed to be an ExplicitOpModel
        return static_target_model.create_processor_spec(qubit_labels if qubit_labels is not None else self._sslbls)

    def _get_cachefile_names(self, param_type, simulator):
        """ Get the standard cache file names for a modelpack """

        if param_type == "H+S terms":
            cachePath = _Path(__file__).absolute().parent / "caches"

            from pygsti.forwardsims.termforwardsim import TermForwardSimulator as _TermFSim
            assert(simulator == "auto" or isinstance(simulator, _TermFSim)), "Invalid `simulator` argument!"
            termOrder = 1 if simulator == "auto" else simulator.max_order
            fn = ("cacheHS%d." % termOrder) + self.__module__
            return cachePath / (fn + "_keys.pkz"), cachePath / (fn + "_vals.npz")
        else:
            raise ValueError("No cache files used for param-type=%s" % param_type)


class GSTModelPack(ModelPack):
    """
    ABC for modelpacks with GST information

    Attributes
    ----------
    _germs : list
        a list of "full" germ circuits, found by randomizing around the target model.

    _germs_lite : list
        a list of "lite" germ circuits, found without randomizing around the target model.

    _fiducials : list
        a list of the fiducial circuits in cases when the preparation and measurement
        fiducials are the same.

    _prepfiducials : list
        the preparation fiducials.

    _measfiducials : list
        the measurement fiducials.

    global_fidpairs : list
        a list of 2-tuples of integers indexing `_prepfiducials` and `_measfiducials` respectively,
        giving a list of global fiducial-pair-reduction results for `_germs`.

    global_fidpairs_lite : list
        a list of 2-tuples of integers indexing `_prepfiducials` and `_measfiducials` respectively,
        giving a list of global fiducial-pair-reduction results for `_germs_lite`.

    _pergerm_fidpairsdict : dict
        a dictionary with germ circuits (as tuples of labels) as keys and lists of 2-tuples as
        values.  The 2-tuples contain integers indexing `_prepfiducials` and `_measfiducials` respectively,
        and together this dictionary gives per-germ FPR results for `_germs`.

    _pergerm_fidpairsdict_lite : dict
        a dictionary with germ circuits (as tuples of labels) as keys and lists of 2-tuples as
        values.  The 2-tuples contain integers indexing `_prepfiducials` and `_measfiducials` respectively,
        and together this dictionary gives per-germ FPR results for `_germs_lite`.
    """
    _germs = None
    _germs_lite = None
    _fiducials = None
    _prepfiducials = None
    _measfiducials = None

    global_fidpairs = None
    global_fidpairs_lite = None
    _pergerm_fidpairsdict = None
    _pergerm_fidpairsdict_lite = None

    def __init__(self):
        self._gscache = {}

    def _indexed_circuits(self, prototype, index):
        if index is None: index = self._sslbls
        assert(len(index) == len(self._sslbls)), "Wrong number of labels in: %s" % str(index)
        if prototype is not None:
            return _circuit_list(_transform_circuittup_list(prototype, index), index)

    def _indexed_circuitdict(self, prototype, index):
        if index is None: index = self._sslbls
        assert(len(index) == len(self._sslbls)), "Wrong number of labels in: %s" % str(index)
        if prototype is not None:
            return {_Circuit(_transform_circuit_tup(k, index), line_labels=index): val for k, val in prototype.items()}

    def germs(self, qubit_labels=None, lite=True):
        """
        Returns the list of germ circuits for this model pack.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        lite : bool, optional
            Whether to return the "lite" set of germs, which amplifies all the errors of
            the target model to first order.  Setting `lite=False` will result in more
            (significantly more in 2+ qubit cases) germs which are selected to amplify
            all the errors of even small deviations from the target model.  Usually this
            added sensitivity is not worth the additional effort required to obtain data
            for the increased number of circuits, so the default is `lite=True`.

        Returns
        -------
        list of Circuits
        """
        if lite and self._germs_lite is not None:
            return self._indexed_circuits(self._germs_lite, qubit_labels)
        else:
            return self._indexed_circuits(self._germs, qubit_labels)

    def fiducials(self, qubit_labels=None):
        """
        Returns the list of fiducial circuits for this model pack.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        Returns
        -------
        list of Circuits
        """
        return self._indexed_circuits(self._fiducials, qubit_labels)

    def prep_fiducials(self, qubit_labels=None):
        """
        Returns the list of preparation fiducials for this model pack.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        Returns
        -------
        list of Circuits
        """
        return self._indexed_circuits(self._prepfiducials, qubit_labels)

    def meas_fiducials(self, qubit_labels=None):
        """
        Returns the list of measurement fiducials for this model pack.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        Returns
        -------
        list of Circuits
        """
        return self._indexed_circuits(self._measfiducials, qubit_labels)

    def pergerm_fidpair_dict(self, qubit_labels=None):
        """
        Returns the per-germ fiducial pair reduction (FPR) dictionary for this model pack.

        Note that these fiducial pairs correspond to the full (`lite=False`) set of germs.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        Returns
        -------
        dict
        """
        return self._indexed_circuitdict(self._pergerm_fidpairsdict, qubit_labels)

    def pergerm_fidpair_dict_lite(self, qubit_labels=None):
        """
        Returns the per-germ fiducial pair reduction (FPR) dictionary for this model pack.

        Note that these fiducial pairs correspond to the lite set of germs.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        Returns
        -------
        dict
        """
        return self._indexed_circuitdict(self._pergerm_fidpairsdict_lite, qubit_labels)

    @_deprecated_fn("create_gst_experiment_design")
    def get_gst_experiment_design(self, max_max_length, qubit_labels=None, fpr=False, lite=True,
                                  evotype='default', **kwargs):
        return self.create_gst_experiment_design(max_max_length, qubit_labels, fpr, lite, **kwargs)

    def create_gst_experiment_design(self, max_max_length, qubit_labels=None, fpr=False, lite=True, **kwargs):
        """
        Construct a :class:`protocols.gst.StandardGSTDesign` from this modelpack

        Parameters
        ----------
        max_max_length : number or list
            The greatest maximum-length to use. Equivalent to
            constructing a :class:`StandardGSTDesign` with a
            `max_lengths` list of powers of two less than or equal to
            the given value.  If a list is given, that this is treated
            as the raw list of maximum lengths, rather than just the maximum.

        qubit_labels : tuple, optional
            A tuple of qubit labels.  None means the integers starting at 0.

        fpr : bool, optional
            Whether to reduce the number of sequences using fiducial
            pair reduction (FPR).

        lite : bool, optional
            Whether to use a smaller "lite" list of germs. Unless you know
            you have a need to use the more pessimistic "full" set of germs,
            leave this set to True.


        Returns
        -------
        StandardGSTDesign
        """
        for k in kwargs.keys():
            if k not in ('germ_length_limits', 'keep_fraction', 'keep_seed', 'include_lgst', 'nest', 'circuit_rules',
                         'op_label_aliases', 'dscheck', 'action_if_missing', 'verbosity', 'add_default_protocol'):
                raise ValueError("Invalid argument '%s' to StandardGSTDesign constructor" % k)

        if qubit_labels is None: qubit_labels = self._sslbls
        assert(len(qubit_labels) == len(self._sslbls)), \
            "Expected %d qubit labels and got: %s!" % (len(self._sslbls), str(qubit_labels))

        if fpr:
            fidpairs = self.pergerm_fidpair_dict_lite(qubit_labels) if lite else \
                self.pergerm_fidpair_dict(qubit_labels)
            if fidpairs is None:
                raise ValueError("No FPR information for lite=%s" % lite)
        else:
            fidpairs = None

        if isinstance(max_max_length, (list, tuple)):
            max_lengths_list = max_max_length
        else:
            max_lengths_list = list(_gen_max_length(max_max_length))

        return _gst.StandardGSTDesign(
            self.processor_spec(qubit_labels),
            self.prep_fiducials(qubit_labels),
            self.meas_fiducials(qubit_labels),
            self.germs(qubit_labels, lite),
            max_lengths_list,
            kwargs.get('germ_length_limits', None),
            fidpairs,
            kwargs.get('keep_fraction', 1),
            kwargs.get('keep_seed', None),
            kwargs.get('include_lgst', True),
            kwargs.get('nest', True),
            kwargs.get('circuit_rules', None),
            kwargs.get('op_label_aliases', None),
            kwargs.get('dscheck', None),
            kwargs.get('action_if_missing', None),
            qubit_labels,
            kwargs.get('verbosity', 0),
            kwargs.get('add_default_protocol', False),
        )

    def create_gst_circuits(self, max_max_length, qubit_labels=None, fpr=False, lite=True, **kwargs):
        """
        Construct a :class:`pygsti.objects.CircuitList` from this modelpack.

        Parameters
        ----------
        max_max_length : number
            The greatest maximum-length to use. Equivalent to
            constructing a cicuit struct with a `max_lengths`
            list of powers of two less than or equal to
            the given value.

        qubit_labels : tuple, optional
            A tuple of qubit labels.  None means the integers starting at 0.

        fpr : bool, optional
            Whether to reduce the number of sequences using fiducial
            pair reduction (FPR).

        lite : bool, optional
            Whether to use a smaller "lite" list of germs. Unless you know
            you have a need to use the more pessimistic "full" set of germs,
            leave this set to True.

        Returns
        -------
         : class:`pygsti.objects.CircuitList`
        """
        if fpr:
            fidpairs = self.pergerm_fidpair_dict_lite(qubit_labels) if lite else \
                self.pergerm_fidpair_dict(qubit_labels)
            if fidpairs is None:
                raise ValueError("No FPR information for lite=%s" % lite)
        else:
            fidpairs = None

        qubit_labels = self._sslbls if (qubit_labels is None) else tuple(qubit_labels)
        assert(len(qubit_labels) == len(self._sslbls)), \
            "Expected %d qubit labels and got: %s!" % (len(self._sslbls), str(qubit_labels))

        lists = _make_lsgst_lists(self._target_model(qubit_labels, evotype='default'),  # Note: only need gate names
                                  self.prep_fiducials(qubit_labels),
                                  self.meas_fiducials(qubit_labels),
                                  self.germs(qubit_labels, lite),
                                  list(_gen_max_length(max_max_length)),
                                  fidpairs,
                                  **kwargs)
        return lists[-1]  # just return final list (for longest sequences)


class RBModelPack(ModelPack):
    """
    Quantities related to performing Randomized Benchmarking (RB) on a given gate-set or model.

    Attributes
    ----------
    _clifford_compilation : OrderedDict
        A dictionary whose keys are all the n-qubit Clifford gates, `"GcX"`, where
        `X` is an integer, and whose values are circuits (given as tuples of labels)
        specifying how to compile that Clifford out of the native gates.
    """
    _clifford_compilation = None

    def clifford_compilation(self, qubit_labels=None):
        """
        Return the Clifford-compilation dictionary for this model pack.

        This is a dictionary whose keys are all the n-qubit Clifford gates, `"GcX"`, where
        `X` is an integer, and whose values are circuits (given as tuples of labels)
        specifying how to compile that Clifford out of the native gates.

        Parameters
        ----------
        qubit_labels : tuple, optional
            If not None, a tuple of the qubit labels to use in the returned circuits. If None,
            then the default labels are used, which are often the integers beginning with 0.

        Returns
        -------
        dict
        """
        if qubit_labels is None: qubit_labels = self._sslbls
        assert(len(qubit_labels) == len(self._sslbls)), "Wrong number of labels in: %s" % str(qubit_labels)
        return {clifford_name: _Circuit(_transform_circuit_tup(circuittup_of_native_gates,
                                                               qubit_labels), line_labels=qubit_labels)
                for clifford_name, circuittup_of_native_gates in self._clifford_compilation.items()}


# Helper functions
def _load_calccache(key_path, val_path):
    """
    The complement to _write_calccache, this function loads a cache
    dictionary from key and value filenames.

    Parameters
    ----------
    key_path, val_path : str
        key and value filenames.

    Returns
    -------
    dict
        The cache of calculated (compact) polynomials.
    """
    #print("Loading cache..."); t0 = _time.time()
    with _gzip.open(key_path, "rb") as f:
        keys = _pickle.load(f)
    npfile = _np.load(val_path)
    vals = _bulk_load_compact_polys(npfile['vtape'], npfile['ctape'], keep_compact=True)
    calc_cache = {k: v for k, v in zip(keys, vals)}
    #print("Done in %.1fs" % (_time.time()-t0))
    return calc_cache


def _transform_layer_tup(layer_tup, index_tup):
    if len(layer_tup) > 1:
        if isinstance(layer_tup[0], str):
            # layer_tup  == a gate label
            # Note: perhaps use the Label object here in future to support more complex labels
            lbl, *idx = layer_tup
            return (lbl, *[index_tup[i] for i in idx])
        else:
            # layer_tup = tuple of gate labels
            return tuple([_transform_layer_tup(comp, index_tup) for comp in layer_tup])
    else:
        return layer_tup  # e.g. ()


def _transform_circuit_tup(circuit_tup, index_tup):
    return tuple(_transform_layer_tup(layer_lbl, index_tup) for layer_lbl in circuit_tup)


def _transform_circuittup_list(circuittup_list, index_tup):
    """ Transform the indices of the tuples in a circuit list with the given index factory """
    for circuit_tup in circuittup_list:
        yield _transform_circuit_tup(circuit_tup, index_tup)


def _gen_max_length(max_max_length):
    i = 1
    while i <= max_max_length:
        yield i
        i *= 2
