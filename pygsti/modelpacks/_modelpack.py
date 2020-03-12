""" Base of the object-oriented model for modelpacks """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from abc import ABC as _ABC, abstractmethod as _abstractmethod
from pathlib import Path as _Path
import numpy as _np
import gzip as _gzip
import pickle as _pickle

from ..objects.polynomial import bulk_load_compact_polys as _bulk_load_compact_polys
from ..objects.circuit import Circuit as _Circuit
from ..construction.circuitconstruction import circuit_list as _circuit_list
from ..construction.modelconstruction import build_explicit_model as _build_explicit_model
from ..construction.stdlists import make_lsgst_structs as _make_lsgst_structs
from ..protocols import gst as _gst


class ModelPack(_ABC):
    """ ABC of all derived modelpack types"""
    description = None
    gates = None
    _sslbls = None

    def __init__(self):
        self._gscache = {}

    @_abstractmethod
    def _target_model(self, sslbls):
        pass

    def _build_explicit_target_model(self, sslbls, gate_names, gate_expressions, **kwargs):
        """
        A helper function for derived classes which create explicit models.
        Updates gate names and expressions with a given set of state-space labels.
        """
        def update_gatename(gn):
            return gn[0:1] + tuple([sslbls[i] for i in gn[1:]])
        full_sslbls = [sslbls]  # put all sslbls in single tensor product block
        updated_gatenames = [update_gatename(gn) for gn in gate_names]
        updated_gateexps = [gexp.format(*sslbls) for gexp in gate_expressions]
        return _build_explicit_model(full_sslbls, updated_gatenames, updated_gateexps, **kwargs)

    def target_model(self, parameterization_type="full", sim_type="auto", qubit_labels=None):
        """
        Returns a copy of the target model in the given parameterization.

        Parameters
        ----------
        parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
            The gate and SPAM vector parameterization type. See
            :function:`Model.set_all_parameterizations` for all allowed values.

        sim_type : {"auto", "matrix", "map", "termorder" }
            The simulator type to be used for model calculations (leave as
            "auto" if you're not sure what this is).

        qubit_labels : tuple, optional
            A tuple of qubit labels, e.g. ('Q0', 'Q1') or (0, 1).  The default
            are the integers starting at 0.

        Returns
        -------
        Model
        """
        qubit_labels = self._sslbls if (qubit_labels is None) else tuple(qubit_labels)
        assert(len(qubit_labels) == len(self._sslbls)), \
            "Expected %d qubit labels and got: %s!" % (len(self._sslbls), str(qubit_labels))

        if (parameterization_type, sim_type, qubit_labels) not in self._gscache:
            # cache miss
            mdl = self._target_model(qubit_labels)
            mdl.set_all_parameterizations(parameterization_type)  # automatically sets sim_type
            if parameterization_type == "H+S terms":
                assert (sim_type == "auto" or sim_type in ("termorder", "termgap", "termdirect")), \
                    "Invalid `sim_type` argument for H+S terms: %s!" % sim_type
                if sim_type == "auto":
                    simt = "termorder"
                    simt_kwargs = {'max_order': 1}
                else:
                    simt = sim_type  # don't update sim_type b/c gscache
                    simt_kwargs = {'max_order': 1}  # TODO: update so user can specify other args?

                key_path, val_path = self._get_cachefile_names(parameterization_type, simt)
                if key_path.exists() and val_path.exists():
                    simt_kwargs['cache'] = _load_calccache(key_path, val_path)

                mdl.set_simtype(simt, **simt_kwargs)
            else:
                if sim_type != "auto":
                    mdl.set_simtype(sim_type)

            # finally cache result
            self._gscache[(parameterization_type, sim_type, qubit_labels)] = mdl

        return self._gscache[(parameterization_type, sim_type, qubit_labels)].copy()

    def _get_cachefile_names(self, param_type, sim_type):
        """ Get the standard cache file names for a modelpack """

        if param_type == "H+S terms":
            cachePath = _Path(__file__).absolute().parent / "caches"

            assert (sim_type == "auto" or sim_type.startswith("termorder:")), "Invalid `sim_type` argument!"
            termOrder = 1 if sim_type == "auto" else int(sim_type.split(":")[1])
            fn = ("cacheHS%d." % termOrder) + self.__module__
            return cachePath / (fn + "_keys.pkz"), cachePath / (fn + "_vals.npz")
        else:
            raise ValueError("No cache files used for param-type=%s" % param_type)


class GSTModelPack(ModelPack):
    """ ABC for modelpacks with GST information"""
    _germs = None
    _germs_lite = None
    _fiducials = None
    _prepfiducials = None
    _measfiducials = None

    global_fidPairs = None
    global_fidPairs_lite = None
    _pergerm_fidPairsDict = None
    _pergerm_fidPairsDict_lite = None

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
        if lite and self._germs_lite is not None:
            return self._indexed_circuits(self._germs_lite, qubit_labels)
        else:
            return self._indexed_circuits(self._germs, qubit_labels)

    def fiducials(self, qubit_labels=None):
        return self._indexed_circuits(self._fiducials, qubit_labels)

    def prep_fiducials(self, qubit_labels=None):
        return self._indexed_circuits(self._prepfiducials, qubit_labels)

    def meas_fiducials(self, qubit_labels=None):
        return self._indexed_circuits(self._measfiducials, qubit_labels)

    def pergerm_fidpair_dict(self, qubit_labels=None):
        return self._indexed_circuitdict(self._pergerm_fidPairsDict, qubit_labels)

    def pergerm_fidpair_dict_lite(self, qubit_labels=None):
        return self._indexed_circuitdict(self._pergerm_fidPairsDict_lite, qubit_labels)

    def get_gst_experiment_design(self, max_max_length, qubit_labels=None, fpr=False, lite=True, **kwargs):
        """ Construct a :class:`protocols.gst.StandardGSTDesign` from this modelpack

        Parameters
        ----------
        max_max_length : number
            The greatest maximum-length to use. Equivalent to
            constructing a :class:`StandardGSTDesign` with a
            `max_lengths` list of powers of two less than or equal to
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

        **kwargs :
            Additional arguments to pass to :class:`StandardGSTDesign`

        Returns
        -------
        :class:`StandardGSTDesign`
        """
        for k in kwargs.keys():
            if k not in ('germ_length_limits', 'keep_fraction', 'keep_seed', 'include_lgst', 'nest', 'sequence_rules',
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

        return _gst.StandardGSTDesign(
            self._target_model(qubit_labels),
            self.prep_fiducials(qubit_labels),
            self.meas_fiducials(qubit_labels),
            self.germs(qubit_labels, lite),
            list(_gen_max_length(max_max_length)),
            kwargs.get('germ_length_limits', None),
            fidpairs,
            kwargs.get('keep_fraction', 1),
            kwargs.get('keep_seed', None),
            kwargs.get('include_lgst', True),
            kwargs.get('nest', True),
            kwargs.get('sequence_rules', None),
            kwargs.get('op_label_aliases', None),
            kwargs.get('dscheck', None),
            kwargs.get('action_if_missing', None),
            qubit_labels,
            kwargs.get('verbosity', 0),
            kwargs.get('add_default_protocol', False),
        )

    def get_gst_circuits_struct(self, max_max_length, qubit_labels=None, fpr=False, lite=True, **kwargs):
        """ Construct a :class:`pygsti.objects.LsGermsStructure` from this modelpack.

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

        **kwargs :
            Additional arguments to pass to :function:`make_lsgst_structs`

        Returns
        -------
        :class:`pygsti.objects.LsGermsStructure`
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

        structs = _make_lsgst_structs(self._target_model(qubit_labels),  # Note: only need gate names here
                                      self.prep_fiducials(qubit_labels),
                                      self.meas_fiducials(qubit_labels),
                                      self.germs(qubit_labels, lite),
                                      list(_gen_max_length(max_max_length)),
                                      fidpairs,
                                      **kwargs)
        return structs[-1]  # just return final struct (for longest sequences)


class RBModelPack(ModelPack):
    _clifford_compilation = None

    def clifford_compilation(self, qubit_labels=None):
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
        # This assumes simple layer labels that == a gate label
        # We could use the Label object here in future to support more complex labels
        lbl, *idx = layer_tup
        return (lbl, *[index_tup[i] for i in idx])
    else:
        return layer_tup


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
