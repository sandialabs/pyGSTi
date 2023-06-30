"""
RPE Protocol objects
"""
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************

import argparse as _argparse
import collections as _collections

import numpy as _np

from pygsti.protocols import protocol as _proto
from pygsti.algorithms.robust_phase_estimation import RobustPhaseEstimation as _RobustPhaseEstimation


class RobustPhaseEstimationDesign(_proto.CircuitListsDesign):
    """
    Experimental design for robust phase estimation (RPE).

    Produces an Experiment Design to test the phase that develops on a target
    gate, by applying it req_lengths times to a states prepared by sin_prep
    and cos_prep circuits, and then measured in the computational basis after
    (respective) action by sin_meas and cos_meas circuits.  outcomes_pos and
    outcomes_neg determine which of those computational basis states count
    towards each of the probabilities

    `P^{γ'γ}_{Ns} = |<γ' y| U^N |γ x>|² = |<γ' x| U^N |-γ y>|² = (1 ± sin(θ))/2`
    `P^{γ'γ}_{Nc} = |<γ' x| U^N |γ x>|² = |<γ' y| U^N | γ y>|² = (1 ± cos(θ))/2`

    (Computational basis state measurements in neither of these sets are
    silently dropped.)

    In the above, the +x refers to the `|E_0> + |E_1>` combination of eigenstates
    of U, *not* of computational basis states.  For instance, if U is rotation
    in the X basis, then cos_prep and cos_meas could be simply the identity:

    `|± U> = |0> ± |1>`

    where `|±U>` are the eigenstates of U, so that, in the notation of the above,

    `|+x> = |+U> + |-U> = |0>`

    The circuit would then calculate

    `P^+_{Nc} = |<+x| U^N | +x>|²`

    provided that cos_outcomes_pos = [0] and cos_outcomes_neg = [1].

    Parameters
    ----------
    gate : <TODO typ>
        <TODO description>

    req_lengths : <TODO typ>
        <TODO description>

    sin_prep : <TODO typ>
        <TODO description>

    sin_meas : <TODO typ>
        <TODO description>

    sin_outcomes_pos : <TODO typ>
        <TODO description>

    sin_outcomes_neg : <TODO typ>
        <TODO description>

    cos_prep : <TODO typ>
        <TODO description>

    cos_meas : <TODO typ>
        <TODO description>

    cos_outcomes_pos : <TODO typ>
        <TODO description>

    cos_outcomes_neg : <TODO typ>
        <TODO description>
        
    """

    def __init__(
        self,
        gate,
        req_lengths,
        sin_prep,
        sin_meas,
        sin_outcomes_pos,
        sin_outcomes_neg,
        cos_prep,
        cos_meas,
        cos_outcomes_pos,
        cos_outcomes_neg,
        *,
        qubit_labels=None,
        req_counts=None
    ):
        """
        Produces an Experiment Design to test the phase that develops on a target
        gate, by applying it req_lengths times to a states prepared by sin_prep
        and cos_prep circuits, and then measured in the computational basis after
        (respective) action by sin_meas and cos_meas circuits.  outcomes_pos and
        outcomes_neg determine which of those computational basis states count
        towards each of the probabilities

        `P^{γ'γ}_{Ns} = |<γ' y| U^N |γ x>|² = |<γ' x| U^N |-γ y>|² = (1 ± sin(θ))/2`
        `P^{γ'γ}_{Nc} = |<γ' x| U^N |γ x>|² = |<γ' y| U^N | γ y>|² = (1 ± cos(θ))/2`

        (Computational basis state measurements in neither of these sets are
        silently dropped.)

        In the above, the +x refers to the `|E_0> + |E_1>` combination of eigenstates
        of U, *not* of computational basis states.  For instance, if U is rotation
        in the X basis, then cos_prep and cos_meas could be simply the identity:

        `|± U> = |0> ± |1>`

        where `|±U>` are the eigenstates of U, so that, in the notation of the above,

        `|+x> = |+U> + |-U> = |0>`

        The circuit would then calculate

        `P^+_{Nc} = |<+x| U^N | +x>|²`

        provided that cos_outcomes_pos = [0] and cos_outcomes_neg = [1].
        """

        # TODO: Serialize these more naturally
        self.sin_prep = (sin_prep,)
        self.sin_meas = (sin_meas,)
        self.sin_outcomes_pos = sin_outcomes_pos
        self.sin_outcomes_neg = sin_outcomes_neg
        self.cos_prep = (cos_prep,)
        self.cos_meas = (cos_meas,)
        self.cos_outcomes_pos = cos_outcomes_pos
        self.cos_outcomes_neg = cos_outcomes_neg
        self.gate = (gate,)

        # What length circuits do we want to run?
        self.req_counts = req_counts
        self.req_lengths = req_lengths

        # Actually build the circuits.
        sin_circs = []
        cos_circs = []
        for n in req_lengths:
            sin_circs.append(sin_prep + gate * n + sin_meas)
            cos_circs.append(cos_prep + gate * n + cos_meas)

        super().__init__([sin_circs, cos_circs], qubit_labels=qubit_labels)
        self.auxfile_types["sin_prep"] = "text-circuit-list"
        self.auxfile_types["sin_meas"] = "text-circuit-list"
        self.auxfile_types["cos_prep"] = "text-circuit-list"
        self.auxfile_types["cos_meas"] = "text-circuit-list"
        self.auxfile_types["gate"] = "text-circuit-list"


class RobustPhaseEstimation(_proto.Protocol):
    """
    Robust phase estimation (RPE) protocol
    """

    def _parse_row(self, row, outcomes_pos, outcomes_neg):
        pos = 0
        neg = 0
        for i in outcomes_pos:
            pos += row[i]
        for i in outcomes_neg:
            neg += row[i]

        return pos, neg

    def parse_dataset(self, design, dataset):
        """
        <TODO summary>

        Parameters
        ----------
        design : <TODO typ>
            <TODO description>

        dataset : <TODO typ>
            <TODO description>
            
        """
        measured = _collections.OrderedDict()
        for n, sin_circ, cos_circ in zip(design.req_lengths, *design.circuit_lists):
            m = measured[n] = _np.zeros(4, dtype=int)
            m[:2] = self._parse_row(
                dataset[sin_circ], design.sin_outcomes_pos, design.sin_outcomes_neg
            )
            m[2:] = self._parse_row(
                dataset[cos_circ], design.cos_outcomes_pos, design.cos_outcomes_neg
            )
        return measured

    def compute_raw_angles(self, measured):
        """
        Determine the raw angles from the count data.

        This corresponds to the angle of `U^N`, i.e., it is N times the phase of U.

        Parameters
        ----------
        measured : <TODO typ>
            <TODO description>

        Returns
        -------
        <TODO typ>
        
        """

        angles = _collections.OrderedDict()

        # The ordering here is chosen to maintain compatibility.
        for n, (Cp_Ns, Cm_Ns, Cp_Nc, Cm_Nc) in measured.items():                                                        # noqa
            # See the description of RobustPhaseEstimationDesign.
            # We estimate P^+_{Ns} and P^-_{Nc} from the similarly named counts.
            # The MLE for these probabilities is:
            Pp_Ns = Cp_Ns / (Cp_Ns + Cm_Ns)                                                                             # noqa
            Pp_Nc = Cp_Nc / (Cp_Nc + Cm_Nc)                                                                             # noqa

            angles[n] = _np.arctan2(2 * Pp_Ns - 1, 2 * Pp_Nc - 1) % (2 * _np.pi)

        return angles

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        RobustPhaseEstimationResults
        """
        meas = self.parse_dataset(data.edesign, data.dataset)
        angles = self.compute_raw_angles(meas)

        _res = _RobustPhaseEstimation(_argparse.Namespace(raw_angles=angles, _measured=meas))

        ret = RobustPhaseEstimationResults(data, self, _res.angle_estimates)
        return ret


class RobustPhaseEstimationResults(_proto.ProtocolResults):
    """
    Results from the RPE protocol

    Parameters
    ----------
    data : <TODO typ>
        <TODO description>

    protocol_instance : <TODO typ>
        <TODO description>

    angle_estimates : <TODO typ>
        <TODO description>

    Attributes
    ----------
    angle_estimate : <TODO typ>
        <TODO description>

    measured_counts : <TODO typ>
        <TODO description>

    raw_angles : <TODO typ>
        <TODO description>
    """

    def __init__(self, data, protocol_instance, angle_estimates):
        """
        Produce an RPE results object, providing access to
          - angle_estimates for each generation, and
          - the RPE-estimated angle, angle_estimate, from the last generation
        """
        super().__init__(data, protocol_instance)

        self.angle_estimates = angle_estimates

    @property
    def angle_estimate(self):
        """
        <TODO summary>
        """
        return self.angle_estimates[-1]

    @property
    def measured_counts(self):
        """
        <TODO summary>
        """
        return self.protocol_instance.parse_dataset(
            self.data.edesign, self.data.dataset
        )

    @property
    def raw_angles(self):
        """
        <TODO summary>
        """
        return self.protocol_instance.rawn_angles(self.measured_counts)


# shorthands
RPEDesign = RobustPhaseEstimationDesign
RPE = RobustPhaseEstimation
RPEResults = RobustPhaseEstimationResults
