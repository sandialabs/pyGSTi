"""
Defines the CHPForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla
import time as _time
import itertools as _itertools
import collections as _collections

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools.matrixtools import _fas
from .profiler import DummyProfiler as _DummyProfiler
from .label import Label as _Label
from .weakforwardsim import WeakForwardSimulator as _WeakForwardSimulator
from .forwardsim import _bytes_for_array_types
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .objectivefns import RawChi2Function as _RawChi2Function
from .objectivefns import RawPoissonPicDeltaLogLFunction as _RawPoissonPicDeltaLogLFunction
_dummy_profiler = _DummyProfiler()


class SimpleCHPWeakForwardSimulator(_WeakForwardSimulator):

    def _compute_circuit_outcome_frequencies(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        """
        Compute probabilities of a multiple "outcomes" for a single circuit.

        The outcomes correspond to `circuit` sandwiched between `rholabel` (a state preparation)
        and the multiple effect labels in `elabels`.

        Parameters
        ----------

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        use_scaling : bool, optional
            Whether to use a post-scaled product internally.  If False, this
            routine will run slightly faster, but with a chance that the
            product will overflow and the subsequent trace operation will
            yield nan as the returned probability.

        time : float, optional
            The *start* time at which `circuit` is evaluated.
        """
        assert(time is None), "CHPForwardSimulator cannot be used to simulate time-dependent circuits"

        expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        for spc, spc_outcomes in expanded_circuit_outcomes.items():  # spc is a SeparatePOVMCircuit
            indices = [outcome_to_index[o] for o in spc_outcomes]
            rholabel = spc.circuit_without_povm[0]
            circuit_ops = spc.circuit_without_povm[1:]
            rho, Es = self._rho_es_from_spam_tuples(rholabel, spc.full_effect_labels)
            #shapes: rho = (N,1), Es = (len(elabels),N)

            G = self.product(circuit_ops, False)
            if self.model.evotype == "statevec":
                ps = _np.real(_np.abs(_np.dot(Es, _np.dot(G, rho)))**2)
            else:  # evotype == "densitymx"
                ps = _np.real(_np.dot(Es, _np.dot(G, rho)))
            array_to_fill[indices] = ps.flat

