"""
Crosstalk-free GST on many qubits, by factorizing the likelihood over circuit lanes.

For a crosstalk-free model -- gates act only on their target qubits, state preparation and
measurement are products of single-qubit errors -- the outcome distribution of a circuit is
the product of the distributions of its *lanes* (groups of qubits linked by gates).  So the
log-likelihood of tensor-product data, such as that of a
:class:`~pygsti.protocols.SimultaneousGSTDesign`, is exactly a sum of lane-sized terms, and
the model can be fit, gauge-optimized and reported on without ever simulating more than one
lane (typically one or two qubits) at a time.

Main entry points:

- :func:`run_crosstalk_free_gst` -- iterative (chi^2, then log-likelihood) fit.
- :class:`LaneModels` -- lane models that share one global parameter vector, built either by
  restricting a many-qubit model (:meth:`LaneModels.from_model`) or straight from
  sub-processor-specs by a factory (needed beyond ~12 qubits).
- :func:`create_general_crosstalk_free_model` -- the general crosstalk-free model.
- :func:`gauge_optimize` -- gauge optimization over the per-qubit product of TP gauge groups.
- :func:`error_summary` -- per-gate and per-qubit error metrics.
- :class:`LaneDecomposition`, :func:`simulate_lanewise_marginals` -- lane-marginal data,
  which avoids holding joint many-qubit datasets.
- :func:`add_single_qubit_instruments` -- mid-circuit measurements, as single-qubit instruments.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .lanes import LaneDecomposition, split_circuit_into_lanes
from .lanemodels import LaneModels, check_factorizable, member_objects, member_parameter_indices, restrict_model
from .parallel import LanePool
from .fit import CrosstalkFreeGSTResults, FactorizedObjective, levenberg_marquardt, run_crosstalk_free_gst
from .simulate import lane_outcome_distributions, simulate_lanewise_data, simulate_lanewise_marginals
from .gaugeopt import GaugeOptResult, apply_product_gauge, gauge_optimize
from .models import add_single_qubit_instruments, create_general_crosstalk_free_model, ideal_z_measurement
from .report import error_summary, format_error_summary
