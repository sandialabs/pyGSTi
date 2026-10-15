"""
DFE Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import defaultdict
import numpy as _np

from pygsti.protocols import protocol as _proto
from pygsti.protocols import vb as _vb
from pygsti import tools as _tools
from pygsti.algorithms import dfe as _dfe
from pygsti.protocols import rb as _rb


class DFEDesign(_vb.BenchmarkingDesign):
    """
    An experiment design for direct fidelity estimation of a set of Clifford circuits.

    For each circuit, `num_samples` DFE circuits are sampled with
    :func:`~pygsti.algorithms.dfe.sample_dfe_circuit`.

    This currently only works if `circuits` are all on the same qubits
    b/c it is built on BenchmarkingDesign. Code will likely silently
    break if this isn't true
    """
    def __init__(self, pspec, circuits, clifford_compilations, num_samples,
                 descriptor='A DFE experiment', add_default_protocol=False, seed=None):

        circuit_lists = []
        measurements = []
        signs = []

        rand_state = _np.random.RandomState(seed)  # Ok if seed is None

        for c in circuits:
            dfe_circuits_for_c = []
            measurements_for_c = []
            signs_for_c = []
            for sample_num in range(num_samples):
                dfe_circ, meas, sign = _dfe.sample_dfe_circuit(pspec, c, clifford_compilations,
                                                               seed=rand_state.randint(0, 2**31))
                dfe_circuits_for_c.append(dfe_circ)
                measurements_for_c.append(meas)
                signs_for_c.append(int(sign))

            circuit_lists.append(dfe_circuits_for_c)
            measurements.append(measurements_for_c)
            signs.append(signs_for_c)

        self._init_foundation(circuit_lists, measurements, signs, num_samples, descriptor,
                              add_default_protocol)

    def _init_foundation(self, circuit_lists, measurements, signs, num_samples, descriptor,
                         add_default_protocol):
        # Pair these attributes with circuit data so that we serialize/truncate properly
        self.paired_with_circuit_attrs = ["measurements", "signs"]

        dummy_depths = list(range(len(circuit_lists)))
        qubit_labels = circuit_lists[0][0].line_labels
        super().__init__(dummy_depths, circuit_lists, signs, qubit_labels, remove_duplicates=False)
        self.measurements = measurements
        self.signs = signs
        self.circuits_per_depth = num_samples
        self.descriptor = descriptor

        # This is a hack to do the data processing automatically
        defaultfit = 'A-fixed'
        self.add_default_protocol(_rb.RB(name='RB', defaultfit=defaultfit))
