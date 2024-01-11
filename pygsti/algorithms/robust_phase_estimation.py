## user-exposure: low (??, EGN Tentative, probably only used in RPE internals, but check with Kenny)
"""
Robust Phase Estimation platform agnostic portion
"""
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************

import numpy


class RobustPhaseEstimation(object):
    """
    Runs the non-adaptive RPE algorithm.

    Runs the non-adaptive RPE algorithm using a dictionary of measurement results,
    `Q.raw_angles`, containing the angles calculated from the probabilities:
    
    `P^{γ'γ}_{Nₖs} = |<γ' y| U^Nₖ |γ x>|² = |<γ' x| U^Nₖ |-γ y>|² = (1 ± sin(θ))/2`
    `P^{γ'γ}_{Nₖc} = |<γ' x| U^Nₖ |γ x>|² = |<γ' y| U^Nₖ | γ y>|² = (1 ± cos(θ))/2`

    Expect `measured[Nₖ] = θ`.

    Overview:

    At each generation, use the previous estimated angle to select the `2π/L` window
    (of which the measurements cannot distinguish).

    Returns an result object. theta is the estimated angle, angle_estimates are
    the estimates from each generation.

    Parameters
    ----------
    q : <TODO typ>
        <TODO description>
    """

    def __init__(self, q):
        self.Q = q
        meas = self.raw_angles = q.raw_angles
        angle_estimates = self.angle_estimates = numpy.zeros(len(meas))

        # The 2π/Nₖ window that is selected is centered around previousAngle,
        # if it is in the middle of [0,2π], angles are naturally forced into
        # the principle range [0,2π].
        theta = numpy.pi

        # iterate over each `generation`
        for k, N in enumerate(meas):
            previousAngle = theta

            frac = 2 * numpy.pi / N

            theta = self.theta_n(N)

            # -> (previousAngle - theta ) // frac
            #       would push into the frac-sized bin that is closest, but the
            #       left side of the `0` bin is directly on previousAngle.
            #       Just being slightly smaller pushes you into a totally
            #       different bin, which is obviously wrong.
            # -> (previousAngle - theta + frac / 2 ) // frac
            #       centers the bins on previousAngle .

            theta += frac * (
                (previousAngle - theta + frac / 2) % (2 * numpy.pi) // frac
            )
            # accounts for wrap-around due to 2 pi periodicity.

            angle_estimates[k] = theta

    def theta_n(self, n):
        """
        Returns the equivalence class of the measurement Θ.

        By definition, Θ is equivalent when any integer multiple of 2π/N is added to it.

        Parameters
        ----------
        n : int
            The RPE 'N' parameter, used to determine the equivalence class.

        Returns
        -------
        float
        """

        # The measurement outcomes have probability:
        # P^{γ'γ}_{Ns} = |<γ' y| U^N |γ x>|² = |<γ' x| U^N |-γ y>|² = (1 ± sin(θ))/2
        # P^{γ'γ}_{Nc} = |<γ' x| U^N |γ x>|² = |<γ' y| U^N | γ y>|² = (1 ± cos(θ))/2

        return (self.raw_angles[n] % (2 * numpy.pi)) / n
