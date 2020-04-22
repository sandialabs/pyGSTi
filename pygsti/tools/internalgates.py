"""The standard unitaries and gate names, used internal compilers and short-hand model init"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.linalg as _spl
from . import optools as _gts
from . import symplectic as _symp


def get_internal_gate_unitaries():
    """
    The unitaries for the *internally* defined gates. These are gates that are used in
    some circuit-compilation methods internally (e.g., compiling multi-qubit Clifford
    gates), and under normal usage of functions/methods that internally make use of these
    labels, circuits containing these gate names will not be returned to the user -- they
    are first converted into gates with user-defined names and actions (with names starting
    with 'G').

    Note that some unitaries in this dict do not have unique names. E.g., the key 'I' is the
    1-qubit identity unitary, but so is 'C0' (which refers to the zeroth element of the 1-qubit
    Clifford group).

    Returns
    -------
    dict of numpy.ndarray objects that are complex, unitary matrices.
    """
    std_unitaries = {}
    # The 1-qubit Paulis
    std_unitaries['I'] = _np.array([[1, 0], [0, 1]], complex)
    std_unitaries['X'] = _np.array([[0, 1], [1, 0]], complex)
    std_unitaries['Y'] = _np.array([[0, -1.0j], [1.0j, 0]], complex)
    std_unitaries['Z'] = _np.array([[1, 0], [0, -1]], complex)
    # 5 gates constructed from Hadamard and Phase which each represent 1 of the 5 1-qubit Clifford gate classes
    # that cannot be converted to each other or the identity via Pauli operators.
    std_unitaries['H'] = (1 / _np.sqrt(2)) * _np.array([[1., 1.], [1., -1.]], complex)
    std_unitaries['P'] = _np.array([[1., 0.], [0., 1j]], complex)
    std_unitaries['HP'] = _np.dot(std_unitaries['H'], std_unitaries['P'])
    std_unitaries['PH'] = _np.dot(std_unitaries['P'], std_unitaries['H'])
    std_unitaries['HPH'] = _np.dot(std_unitaries['H'], _np.dot(std_unitaries['P'], std_unitaries['H']))
    # The 1-qubit Clifford group. The labelling is the same as in the the 1-qubit Clifford group generated
    # in pygsti.extras.rb.group, with the mapping 'Ci' - > 'Gci'. (we keep with the convention here of not have
    # hard-coded unitaries starting with a 'G'.)
    std_unitaries['C0'] = _np.array([[1, 0], [0, 1]], complex)
    std_unitaries['C1'] = _np.array([[1, -1j], [1, 1j]], complex) / _np.sqrt(2)
    std_unitaries['C2'] = _np.array([[1, 1], [1j, -1j]], complex) / _np.sqrt(2)
    std_unitaries['C3'] = _np.array([[0, 1], [1, 0]], complex)
    std_unitaries['C4'] = _np.array([[-1, -1j], [1, -1j]], complex) / _np.sqrt(2)
    std_unitaries['C5'] = _np.array([[1, 1], [-1j, 1j]], complex) / _np.sqrt(2)
    std_unitaries['C6'] = _np.array([[0, -1j], [1j, 0]], complex)
    std_unitaries['C7'] = _np.array([[1j, 1], [-1j, 1]], complex) / _np.sqrt(2)
    std_unitaries['C8'] = _np.array([[1j, -1j], [1, 1]], complex) / _np.sqrt(2)
    std_unitaries['C9'] = _np.array([[1, 0], [0, -1]], complex)
    std_unitaries['C10'] = _np.array([[1, 1j], [1, -1j]], complex) / _np.sqrt(2)
    std_unitaries['C11'] = _np.array([[1, -1], [1j, 1j]], complex) / _np.sqrt(2)
    std_unitaries['C12'] = _np.array([[1, 1], [1, -1]], complex) / _np.sqrt(2)
    std_unitaries['C13'] = _np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]], complex)
    std_unitaries['C14'] = _np.array([[1, 0], [0, 1j]], complex)
    std_unitaries['C15'] = _np.array([[1, 1], [-1, 1]], complex) / _np.sqrt(2)
    std_unitaries['C16'] = _np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], complex)
    std_unitaries['C17'] = _np.array([[0, 1], [1j, 0]], complex)
    std_unitaries['C18'] = _np.array([[1j, -1j], [-1j, -1j]], complex) / _np.sqrt(2)
    std_unitaries['C19'] = _np.array([[0.5 + 0.5j, -0.5 + 0.5j], [0.5 - 0.5j, -0.5 - 0.5j]], complex)
    std_unitaries['C20'] = _np.array([[0, -1j], [-1, 0]], complex)
    std_unitaries['C21'] = _np.array([[1, -1], [1, 1]], complex) / _np.sqrt(2)
    std_unitaries['C22'] = _np.array([[0.5 + 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]], complex)
    std_unitaries['C23'] = _np.array([[1, 0], [0, -1j]], complex)
    # Standard 2-qubit gates.
    std_unitaries['CPHASE'] = _np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [
                                        0., 0., 1., 0.], [0., 0., 0., -1.]], complex)
    std_unitaries['CNOT'] = _np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]], complex)
    std_unitaries['SWAP'] = _np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]], complex)

    return std_unitaries


def is_gate_this_standard_unitary(gate_unitary, standard_gate_name):
    """
    Returns True if the unitary `gate_unitary` is, up to phase, the standard gate specified
    by the name `standard_gate_name`. The correspondence between the standard names and
    unitaries is w.r.t the internally-used gatenames (see get_internal_gate_unitaries()).
    For example, one use of this function is to check whether some gate specifed by a user
    with the name 'Ghadamard' is the Hadamard gate, denoted internally by 'H'.

    Parameters
    ----------
    gate_unitary : complex np.array
        The unitary to test.

    standard_gate_name : str
        The standard gatename to check whether the unitary `gate_unitary` is (e.g., 'CNOT').

    Returns
    -------
    bool
        True if the `gate_unitary` is, up to phase, the unitary specified `standard_gate_name`.
        False otherwise.
    """
    std_unitaries = get_internal_gate_unitaries()
    if _np.shape(gate_unitary) != _np.shape(std_unitaries[standard_gate_name]):
        return False
    else:
        pm_input = _gts.unitary_to_pauligate(gate_unitary)
        pm_std = _gts.unitary_to_pauligate(std_unitaries[standard_gate_name])
        equal = _np.allclose(pm_input, pm_std)
        return equal

# Currently not needed, but might be added in.
#


def is_gate_pauli_equivalent_to_this_standard_unitary(gate_unitary, standard_gate_name):
    """
    Returns True if the unitary `gate_unitary` is, when pre- and post-multiplied by some
    Pauli and up to phase, the standard gate specified by the name `standard_gate_name`.
    The correspondence between the standard names and unitaries is w.r.t the internally-used
    gatenames (see get_internal_gate_unitaries()).

    Currently only supported for Clifford gates.

    Parameters
    ----------
    gate_unitary : complex np.array
        The unitary to test.

    standard_gate_name : str
        The standard gatename to check whether the unitary `gate_unitary` is (e.g., 'CNOT').

    Returns
    -------
    bool
        True if the `gate_unitary` is, up to phase and Pauli-multiplication, the unitary
        specified `standard_gate_name`. False otherwise.
    """
    std_symplectic_reps = _symp.get_internal_gate_symplectic_representations()
    gate_symplectic_rep = _symp.unitary_to_symplectic(gate_unitary)

    if _np.shape(gate_symplectic_rep[0]) != _np.shape(std_symplectic_reps[standard_gate_name][0]):
        return False
    else:
        return _np.allclose(gate_symplectic_rep[0], std_symplectic_reps[standard_gate_name][0])


def get_standard_gatename_unitaries():
    """
    Constructs and returns a dictionary of unitary matrices describing the
    action of "standard" gates. These gates (the keys of the returned
    dictionary) are:

    - Clifford Gates:

      - 'Gi' : the 1Q idle operation.
      - 'Gxpi2','Gypi2','Gzpi2' : 1Q pi/2 rotations around X, Y and Z.
      - 'Gxpi','Gypi','Gzpi' : 1Q pi rotations around X, Y and Z.
      - 'Gxpi2','Gypi2','Gzpi2' : 1Q pi/2 rotations around X, Y and Z.
      - 'Gxpi2','Gypi2','Gzpi2' : 1Q pi/2 rotations around X, Y and Z.
      - 'Gh' : Hadamard.
      - 'Gp', 'Gpdag' : phase and inverse phase (an alternative notation/name for Gzpi and Gzmpi2).
      - 'Gci' where i = 0, 1, ..., 23 : the 24 1-qubit Cliffor gates (all the gates above are included as one of these).
      - 'Gcphase','Gcnot','Gswap' : standard 2Q gates.

    - Non-Clifford gates:

      - 'Gt', 'Gtdag' : the T and inverse T gates (T is a Z rotation by pi/4).

    Mostly, pyGSTi does not assume that a gate with one of these names is indeed
    the unitary specified here. Instead, these names are intended as short-hand
    for defining ProcessorSpecs and n-qubit models. Moreover, when these names
    are used then conversion of circuits to QUIL or QISKIT is particular convenient,
    and does not require the user to specify the syntax conversion.

    Returns
    -------
    dict of numpy.ndarray objects.
    """
    std_unitaries = {}

    sigmax = _np.array([[0, 1], [1, 0]])
    sigmay = _np.array([[0, -1.0j], [1.0j, 0]])
    sigmaz = _np.array([[1, 0], [0, -1]])

    def u_op(exp):
        return _np.array(_spl.expm(-1j * exp / 2), complex)

    std_unitaries['Gi'] = _np.array([[1., 0.], [0., 1.]], complex)

    std_unitaries['Gx'] = std_unitaries['Gxpi2'] = u_op(_np.pi / 2 * sigmax)
    std_unitaries['Gy'] = std_unitaries['Gypi2'] = u_op(_np.pi / 2 * sigmay)
    std_unitaries['Gz'] = std_unitaries['Gzpi2'] = u_op(_np.pi / 2 * sigmaz)

    std_unitaries['Gxpi'] = _np.array([[0., 1.], [1., 0.]], complex)
    std_unitaries['Gypi'] = _np.array([[0., -1j], [1j, 0.]], complex)
    std_unitaries['Gzpi'] = _np.array([[1., 0.], [0., -1.]], complex)

    std_unitaries['Gxmpi2'] = u_op(-1 * _np.pi / 2 * sigmax)
    std_unitaries['Gympi2'] = u_op(-1 * _np.pi / 2 * sigmay)
    std_unitaries['Gzmpi2'] = u_op(-1 * _np.pi / 2 * sigmaz)

    H = (1 / _np.sqrt(2)) * _np.array([[1., 1.], [1., -1.]], complex)
    P = _np.array([[1., 0.], [0., 1j]], complex)
    Pdag = _np.array([[1., 0.], [0., -1j]], complex)

    std_unitaries['Gh'] = H
    std_unitaries['Gp'] = P
    std_unitaries['Gpdag'] = Pdag
    #std_unitaries['Ghp'] = _np.dot(H,P)
    #std_unitaries['Gph'] = _np.dot(P,H)
    #std_unitaries['Ghph'] = _np.dot(H,_np.dot(P,H))
    std_unitaries['Gt'] = _np.array([[1., 0.], [0., _np.exp(1j * _np.pi / 4)]], complex)
    std_unitaries['Gtdag'] = _np.array([[1., 0.], [0., _np.exp(-1j * _np.pi / 4)]], complex)
    # The 1-qubit Clifford group. The labelling is the same as in the the 1-qubit Clifford group generated
    # in pygsti.extras.rb.group, and also in the internal standard unitary (but with 'Gci' -> 'Ci')
    std_unitaries['Gc0'] = _np.array([[1, 0], [0, 1]], complex)  # This is Gi
    std_unitaries['Gc1'] = _np.array([[1, -1j], [1, 1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc2'] = _np.array([[1, 1], [1j, -1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc3'] = _np.array([[0, 1], [1, 0]], complex)  # This is Gxpi (up to phase)
    std_unitaries['Gc4'] = _np.array([[-1, -1j], [1, -1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc5'] = _np.array([[1, 1], [-1j, 1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc6'] = _np.array([[0, -1j], [1j, 0]], complex)  # This is Gypi (up to phase)
    std_unitaries['Gc7'] = _np.array([[1j, 1], [-1j, 1]], complex) / _np.sqrt(2)
    std_unitaries['Gc8'] = _np.array([[1j, -1j], [1, 1]], complex) / _np.sqrt(2)
    std_unitaries['Gc9'] = _np.array([[1, 0], [0, -1]], complex)  # This is Gzpi
    std_unitaries['Gc10'] = _np.array([[1, 1j], [1, -1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc11'] = _np.array([[1, -1], [1j, 1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc12'] = _np.array([[1, 1], [1, -1]], complex) / _np.sqrt(2)  # This is Gh
    std_unitaries['Gc13'] = _np.array([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]],
                                      complex)  # This is Gxmpi2 (up to phase)
    std_unitaries['Gc14'] = _np.array([[1, 0], [0, 1j]], complex)  # THis is Gzpi2 / Gp (up to phase)
    std_unitaries['Gc15'] = _np.array([[1, 1], [-1, 1]], complex) / _np.sqrt(2)  # This is Gympi2 (up to phase)
    std_unitaries['Gc16'] = _np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
                                      complex)  # This is Gxpi2 (up to phase)
    std_unitaries['Gc17'] = _np.array([[0, 1], [1j, 0]], complex)
    std_unitaries['Gc18'] = _np.array([[1j, -1j], [-1j, -1j]], complex) / _np.sqrt(2)
    std_unitaries['Gc19'] = _np.array([[0.5 + 0.5j, -0.5 + 0.5j], [0.5 - 0.5j, -0.5 - 0.5j]], complex)
    std_unitaries['Gc20'] = _np.array([[0, -1j], [-1, 0]], complex)
    std_unitaries['Gc21'] = _np.array([[1, -1], [1, 1]], complex) / _np.sqrt(2)  # This is Gypi2 (up to phase)
    std_unitaries['Gc22'] = _np.array([[0.5 + 0.5j, 0.5 - 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]], complex)
    std_unitaries['Gc23'] = _np.array([[1, 0], [0, -1j]], complex)  # This is Gzmpi2 / Gpdag (up to phase)
    # Two-qubit gates
    std_unitaries['Gcphase'] = _np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [
                                         0., 0., 1., 0.], [0., 0., 0., -1.]], complex)
    std_unitaries['Gcnot'] = _np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [
                                       0., 0., 0., 1.], [0., 0., 1., 0.]], complex)
    std_unitaries['Gswap'] = _np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [
                                       0., 1., 0., 0.], [0., 0., 0., 1.]], complex)

    return std_unitaries


def get_standard_gatenames_cirq_conversions():
    """
    A dictionary converting the gates with standard names
    (see get_standard_gatename_unitaries()) to the cirq
    names for these gates.

    By default, an idle operation will not be converted to a gate.
    If you want an idle to be converted to a `cirq.WaitGate`, you will have
    to modify this dictionary.

    Note that throughout pyGSTi the standard gatenames (e.g., 'Gh' for Hadamard)
    are not enforced to correspond to the expected unitaries. So, if the user
    as, say, defined 'Gh' to be something other than the Hadamard gate this
    conversion dictionary will be incorrect.

    Currently there are some standard gate names with no conversion to cirq.

    TODO: add Clifford gates with
    https://cirq.readthedocs.io/en/latest/generated/cirq.SingleQubitCliffordGate.html

    Returns
    -------
    dict mapping strings to string
    """
    try:
        import cirq
    except ImportError:
        raise ImportError("Cirq is required for this operation, and it does not appear to be installed.")

    std_gatenames_to_cirq = {}
    std_gatenames_to_cirq['Gi'] = None
    std_gatenames_to_cirq['Gxpi2'] = cirq.XPowGate(exponent=1 / 2)
    std_gatenames_to_cirq['Gxmpi2'] = cirq.XPowGate(exponent=-1 / 2)
    std_gatenames_to_cirq['Gxpi'] = cirq.X
    std_gatenames_to_cirq['Gzpi2'] = cirq.ZPowGate(exponent=1 / 2)
    std_gatenames_to_cirq['Gzmpi2'] = cirq.ZPowGate(exponent=-1 / 2)
    std_gatenames_to_cirq['Gzpi'] = cirq.Z
    std_gatenames_to_cirq['Gypi2'] = cirq.YPowGate(exponent=1 / 2)
    std_gatenames_to_cirq['Gympi2'] = cirq.YPowGate(exponent=-1 / 2)
    std_gatenames_to_cirq['Gypi'] = cirq.Y
    std_gatenames_to_cirq['Gp'] = std_gatenames_to_cirq['Gzpi2']
    std_gatenames_to_cirq['Gpdag'] = std_gatenames_to_cirq['Gzmpi2']
    std_gatenames_to_cirq['Gh'] = cirq.H
    std_gatenames_to_cirq['Gt'] = cirq.T
    std_gatenames_to_cirq['Gtdag'] = cirq.T**-1
    std_gatenames_to_cirq['Gcphase'] = cirq.CZ
    std_gatenames_to_cirq['Gcnot'] = cirq.CNOT
    std_gatenames_to_cirq['Gswap'] = cirq.SWAP

    return std_gatenames_to_cirq


def get_standard_gatenames_quil_conversions():
    """
    A dictionary converting the gates with standard names
    (see get_standard_gatename_unitaries()) to the QUIL
    names for these gates.

    Note that throughout pyGSTi the standard gatenames (e.g., 'Gh' for Hadamard)
    are not enforced to correspond to the expected unitaries. So, if the user
    as, say, defined 'Gh' to be something other than the Hadamard gate this
    conversion dictionary will be incorrect.

    Currently there are some standard gate names with no conversion to quil.

    Returns
    -------
    dict mapping strings to strings.
    """
    std_gatenames_to_quil = {}
    std_gatenames_to_quil['Gi'] = 'I'
    std_gatenames_to_quil['Gxpi2'] = 'RX(pi/2)'
    std_gatenames_to_quil['Gxmpi2'] = 'RX(-pi/2)'
    std_gatenames_to_quil['Gxpi'] = 'X'
    std_gatenames_to_quil['Gzpi2'] = 'RZ(pi/2)'
    std_gatenames_to_quil['Gzmpi2'] = 'RZ(-pi/2)'
    std_gatenames_to_quil['Gzpi'] = 'Z'
    std_gatenames_to_quil['Gypi2'] = 'RY(pi/2)'
    std_gatenames_to_quil['Gympi2'] = 'RY(-pi/2)'
    std_gatenames_to_quil['Gypi'] = 'Y'
    std_gatenames_to_quil['Gp'] = 'RZ(pi/2)'  # todo : check that this is correct, and shouldn't instead be -pi/2
    std_gatenames_to_quil['Gpdag'] = 'RZ(-pi/2)'  # todo : check that this is correct, and shouldn't instead be +pi/2
    std_gatenames_to_quil['Gh'] = 'H'
    std_gatenames_to_quil['Gt'] = 'RZ(pi/4)'  # todo : check that this is correct, and shouldn't instead be -pi/4
    std_gatenames_to_quil['Gtdag'] = 'RZ(-pi/4)'  # todo : check that this is correct, and shouldn't instead be +pi/4
    std_gatenames_to_quil['Gcphase'] = 'CZ'
    std_gatenames_to_quil['Gcnot'] = 'CNOT'

    std_gatenames_to_quil['Gc0'] = 'Gc0'
    std_gatenames_to_quil['Gc1'] = 'Gc1'
    std_gatenames_to_quil['Gc2'] = 'Gc2'
    std_gatenames_to_quil['Gc3'] = 'Gc3'
    std_gatenames_to_quil['Gc4'] = 'Gc4'
    std_gatenames_to_quil['Gc5'] = 'Gc5'
    std_gatenames_to_quil['Gc6'] = 'Gc6'
    std_gatenames_to_quil['Gc7'] = 'Gc7'
    std_gatenames_to_quil['Gc8'] = 'Gc8'
    std_gatenames_to_quil['Gc9'] = 'Gc9'
    std_gatenames_to_quil['Gc10'] = 'Gc10'
    std_gatenames_to_quil['Gc11'] = 'Gc11'
    std_gatenames_to_quil['Gc12'] = 'Gc12'
    std_gatenames_to_quil['Gc13'] = 'Gc13'
    std_gatenames_to_quil['Gc14'] = 'Gc14'
    std_gatenames_to_quil['Gc15'] = 'Gc15'
    std_gatenames_to_quil['Gc16'] = 'Gc16'
    std_gatenames_to_quil['Gc17'] = 'Gc17'
    std_gatenames_to_quil['Gc18'] = 'Gc18'
    std_gatenames_to_quil['Gc19'] = 'Gc19'
    std_gatenames_to_quil['Gc20'] = 'Gc20'
    std_gatenames_to_quil['Gc21'] = 'Gc21'
    std_gatenames_to_quil['Gc22'] = 'Gc22'
    std_gatenames_to_quil['Gc23'] = 'Gc23'

    return std_gatenames_to_quil


def get_standard_gatenames_openqasm_conversions():
    """
    A dictionary converting the gates with standard names
    (see get_standard_gatename_unitaries()) to the QASM
    names for these gates.

    Note that throughout pyGSTi the standard gatenames (e.g., 'Gh' for Hadamard)
    are not enforced to correspond to the expected unitaries. So, if the user
    has, say, defined 'Gh' to be something other than the Hadamard gate this
    conversion dictionary will be incorrect.

    Returns
    -------
    dict mapping strings to strings.
    """
    std_gatenames_to_qasm = {}
    std_gatenames_to_qasm['Gi'] = 'id'
    std_gatenames_to_qasm['Gxpi2'] = 'u3(1.570796326794897, 4.71238898038469, 1.570796326794897)'  # [1, 3, 1] * pi/2
    std_gatenames_to_qasm['Gxmpi2'] = 'u3(1.570796326794897, 1.570796326794897, 4.71238898038469)'  # [1, 1, 3] * pi/2
    std_gatenames_to_qasm['Gxpi'] = 'x'
    std_gatenames_to_qasm['Gzpi2'] = 'u3(0, 0, 1.570796326794897)'  # [0, 0, 1] * pi/2
    std_gatenames_to_qasm['Gzmpi2'] = 'u3(0, 0, 4.71238898038469)'  # [0, 0, 3] * pi/2
    std_gatenames_to_qasm['Gzpi'] = 'z'
    std_gatenames_to_qasm['Gypi2'] = 'u3(1.570796326794897, 0, 0)'  # [1, 0, 0] * pi/2
    std_gatenames_to_qasm['Gympi2'] = 'u3(1.570796326794897, 3.141592653589793, 3.141592653589793)'  # [1, 2, 2] * pi/2
    std_gatenames_to_qasm['Gypi'] = 'y'
    std_gatenames_to_qasm['Gp'] = 's'
    std_gatenames_to_qasm['Gpdag'] = 'sdg'
    std_gatenames_to_qasm['Gh'] = 'h'
    std_gatenames_to_qasm['Gt'] = 't'
    std_gatenames_to_qasm['Gtdag'] = 'tdg'
    std_gatenames_to_qasm['Gcphase'] = 'cz'
    std_gatenames_to_qasm['Gcnot'] = 'cx'
    std_gatenames_to_qasm['Gswap'] = 'swap'

    std_gatenames_to_qasm['Gc0'] = 'u3(0, 0, 0)'  # [0, 0, 0] * pi/2 (thi is Gi)
    std_gatenames_to_qasm['Gc1'] = 'u3(1.570796326794897, 0, 1.570796326794897)'  # [1, 0, 1] * pi/2
    std_gatenames_to_qasm['Gc2'] = 'u3(1.570796326794897, 1.570796326794897, 3.141592653589793)'  # [1, 1, 2] * pi/2
    std_gatenames_to_qasm['Gc3'] = 'u3(3.141592653589793, 0, 3.141592653589793)'  # [2, 0, 2] * pi/2 (this is Gxpi)
    std_gatenames_to_qasm['Gc4'] = 'u3(1.570796326794897, 3.141592653589793, 4.71238898038469)'  # [1, 2, 3] * pi/2
    std_gatenames_to_qasm['Gc5'] = 'u3(1.570796326794897, 4.71238898038469, 3.141592653589793)'  # [1, 3, 2] * pi/2
    std_gatenames_to_qasm['Gc6'] = 'u3(3.141592653589793, 0, 0)'  # [2, 0, 0] * pi/2 (this is Gypi)
    std_gatenames_to_qasm['Gc7'] = 'u3(1.570796326794897, 3.141592653589793, 1.570796326794897)'  # [1, 2, 1] * pi/2
    std_gatenames_to_qasm['Gc8'] = 'u3(1.570796326794897, 4.71238898038469, 0.)'  # [1, 3, 0] * pi/2
    std_gatenames_to_qasm['Gc9'] = 'u3(0, 0, 3.141592653589793)'  # [0, 0, 2] * pi/2 (this is Gzpi)
    std_gatenames_to_qasm['Gc10'] = 'u3(1.570796326794897, 0, 4.71238898038469)'  # [1, 0, 3] * pi/2
    std_gatenames_to_qasm['Gc11'] = 'u3(1.570796326794897, 1.570796326794897, 0.)'  # [1, 1, 0] * pi/2
    std_gatenames_to_qasm['Gc12'] = 'u3(1.570796326794897, 0., 3.141592653589793)'  # [1, 0, 2] * pi/2 (this is Gh)
    # [1, 1, 3] * pi/2 (this is Gxmpi2 )
    std_gatenames_to_qasm['Gc13'] = 'u3(1.570796326794897, 1.570796326794897, 4.71238898038469)'
    std_gatenames_to_qasm['Gc14'] = 'u3(0, 0, 1.570796326794897)'  # [0, 0, 1] * pi/2 (this is Gzpi2 / Gp)
    # [1, 2, 2] * pi/2 (the is Gympi2)
    std_gatenames_to_qasm['Gc15'] = 'u3(1.570796326794897, 3.141592653589793, 3.141592653589793)'
    # [1, 3, 1] * pi/2 (this is Gxpi2 )
    std_gatenames_to_qasm['Gc16'] = 'u3(1.570796326794897, 4.71238898038469, 1.570796326794897)'
    std_gatenames_to_qasm['Gc17'] = 'u3(3.141592653589793, 0, 1.570796326794897)'  # [2, 0, 1] * pi/2
    std_gatenames_to_qasm['Gc18'] = 'u3(1.570796326794897, 3.141592653589793, 0.)'  # [1, 2, 0] * pi/2
    std_gatenames_to_qasm['Gc19'] = 'u3(1.570796326794897, 4.71238898038469, 4.71238898038469)'  # [1, 3, 3] * pi/2
    std_gatenames_to_qasm['Gc20'] = 'u3(3.141592653589793, 0, 4.71238898038469)'  # [2, 0, 3] * pi/2
    std_gatenames_to_qasm['Gc21'] = 'u3(1.570796326794897, 0, 0)'  # [1, 0, 0] * pi/2 (this is Gypi2)
    std_gatenames_to_qasm['Gc22'] = 'u3(1.570796326794897, 1.570796326794897, 1.570796326794897)'  # [1, 1, 1] * pi/2
    std_gatenames_to_qasm['Gc23'] = 'u3(0, 0, 4.71238898038469)'  # [0, 0, 3] * pi/2 (this is Gzmpi2 / Gpdag)

    return std_gatenames_to_qasm


def qasm_u3(theta, phi, lamb, output='unitary'):
    """
    The u3 1-qubit gate of QASM, returned as a unitary
    if output = 'unitary' and as a processmatrix in the
    Pauli basis if out = 'superoperater.'
    """
    u3_unitary = _np.array([[_np.cos(theta / 2), -1 * _np.exp(1j * lamb) * _np.sin(theta / 2)],
                            [_np.exp(1j * phi) * _np.sin(theta / 2), _np.exp(1j * (lamb + phi)) * _np.cos(theta / 2)]])

    if output == 'unitary':
        return u3_unitary

    elif output == 'superoperator':
        u3_superoperator = _gts.unitary_to_pauligate(u3_unitary)
        return u3_superoperator

    else: raise ValueError("The `output` string is invalid!")
