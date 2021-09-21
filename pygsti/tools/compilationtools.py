"""
Functions for manupilating gates in circuits
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


# There is probably an better way to do this.
def mod_2pi(theta):
    while (theta > _np.pi or theta <= -1 * _np.pi):
        if theta > _np.pi:
            theta = theta - 2 * _np.pi
        elif theta <= -1 * _np.pi:
            theta = theta + 2 * _np.pi
    return theta


def pauli_frame_randomize_unitary(theta1, theta2, theta3, net_pauli, recomp_pauli):
    #takes the z rotation angles for the compiled version of a random unitary and finds the angles for the compiled
    # version of the pauli frame randomized unitary
    #redefine the values so that when the net pauli commutes through, we get the original parameters
    if net_pauli == 1 or net_pauli == 3:
        theta2 *= -1
    if net_pauli == 1 or net_pauli == 2:
        theta3 *= -1
        theta1 *= -1

    #change angles to recompile the new pauli into the gate
    if recomp_pauli == 1 or recomp_pauli == 2:  # if x or y
        theta1 = -theta1 + _np.pi
        theta2 = theta2 + _np.pi
    if recomp_pauli == 2 or recomp_pauli == 3:  # if y or z
        theta1 = theta1 + _np.pi

    #make everything between -pi and pi.
    theta1 = mod_2pi(theta1)
    theta2 = mod_2pi(theta2)
    theta3 = mod_2pi(theta3)

    return (theta1, theta2, theta3)


def inv_recompile_unitary(theta1, theta2, theta3):
    """
    TODO
    """
    #makes a compiled version of the inverse of a compiled general unitary
    #negate angles for inverse based on central pauli, account for recompiling the X(-pi/2) into X(pi/2)
    theta1 = mod_2pi(_np.pi - theta1)
    theta2 = mod_2pi(-theta2)
    theta3 = mod_2pi(-theta3 + _np.pi)

    return (theta1, theta2, theta3)
