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


class ModelPack(_ABC):
    """ ABC of all derived modelpack types"""
    description = None

    def __init__(self):
        self._gscache = {("full", "auto"): self._target_model}

    @property
    @_abstractmethod
    def _target_model(self):
        pass

    def target_model(self, parameterization_type="full", sim_type="auto"):
        """
        Returns a copy of the target model in the given parameterization.

        Parameters
        ----------
        parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
            The gate and SPAM vector parameterization type. See
            :function:`Model.set_all_parameterizations` for all allowed values.

        sim_type : {"auto", "matrix", "map", "termorder:X" }
            The simulator type to be used for model calculations (leave as
            "auto" if you're not sure what this is).

        Returns
        -------
        Model
        """

        if (parameterization_type, sim_type) not in self._gscache:
            # cache miss
            mdl = self._target_model.copy()
            mdl.set_all_parameterizations(parameterization_type)  # automatically sets sim_type
            if parameterization_type == "H+S terms":
                assert(sim_type == "auto" or sim_type.startswith("termorder:")), "Invalid `sim_type` argument!"
                simt = "termorder:1" if sim_type == "auto" else sim_type  # don't update sim_type b/c gscache
                calc_cache = {}  # the default

                key_path, val_path = self._get_cachefile_names(parameterization_type, simt)
                if key_path.exists() and val_path.exists():
                    calc_cache = _load_calccache(key_path, val_path)

                mdl.set_simtype(simt, calc_cache)
            else:
                if sim_type != "auto":
                    mdl.set_simtype(sim_type)

            # finally cache result
            self._gscache[(parameterization_type, sim_type)] = mdl

        return self._gscache[(parameterization_type, sim_type)].copy()

    def _get_cachefile_names(self, param_type, sim_type):
        """ Get the standard cache file names for a modelpack """

        if param_type == "H+S terms":
            cachePath = _Path(__file__).absolute().parent / "caches"
            # cachePath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
            #                           "caches")

            assert(sim_type == "auto" or sim_type.startswith("termorder:")), "Invalid `sim_type` argument!"
            termOrder = 1 if sim_type == "auto" else int(sim_type.split(":")[1])
            fn = ("cacheHS%d." % termOrder) + self.__module__
            return cachePath / (fn + "_keys.pkz"), cachePath / (fn + "_vals.npz")
            # fn = _os.path.join(cachePath, fn)
            # return fn + "_keys.pkz", fn + "_vals.npz"
        else:
            raise ValueError("No cache files used for param-type=%s" % param_type)


# XXX should this be GSTModelPack?
class SMQModelPack(ModelPack):
    """ ABC for standard multi-qubit modelpacks """
    gates = None
    fiducials = None
    effectStrs = None
    prepStrs = None
    germs = None

    def get_gst_inputs(max_max_length, drift_analysis=False):
        """ TODO """
        pass  # TODO

    def get_gst_circuits_struct(max_max_length):
        """ TODO """
        pass  # TODO


class RPEModelPack(ModelPack):
    """ ABC for robust phase estimation modelpacks """
    def get_rpe_inputs(max_max_length):
        pass # TODO


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
