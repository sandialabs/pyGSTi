"""Functions which compute named quantities for Models and Datasets."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Named quantities as well as their confidence-region error bars are
 computed by the functions in this module. These quantities are
 used primarily in reports, so we refer to these quantities as
 "reportables".
"""
import numpy as _np
import scipy.linalg as _spl
import warnings as _warnings

from .. import tools as _tools
from .. import algorithms as _alg
from ..objects.basis import Basis as _Basis, DirectSumBasis as _DirectSumBasis
from ..objects.label import Label as _Lbl
from ..objects.reportableqty import ReportableQty as _ReportableQty
from ..objects import modelfunction as _modf

import pkgutil
_CVXPY_AVAILABLE = pkgutil.find_loader('cvxpy') is not None

FINITE_DIFF_EPS = 1e-7


def _nullFn(*arg):
    return None


def _projectToValidProb(p, tol=1e-9):
    if p < tol: return tol
    if p > 1 - tol: return 1 - tol
    return p


def _make_reportable_qty_or_dict(f0, df=None, nonMarkovianEBs=False):
    """ Just adds special processing with f0 is a dict, where we
        return a dict or ReportableQtys rather than a single
        ReportableQty of the dict.
    """
    if isinstance(f0, dict):
        #special processing for dict -> df is dict of error bars
        # and we return a dict of ReportableQtys
        if df:
            return {ky: _ReportableQty(f0[ky], df[ky], nonMarkovianEBs) for ky in f0}
        else:
            return {ky: _ReportableQty(f0[ky], None, False) for ky in f0}
    else:
        return _ReportableQty(f0, df, nonMarkovianEBs)


def evaluate(modelFn, cri=None, verbosity=0):
    """
    Evaluate a ModelFunction object using confidence region information

    Parameters
    ----------
    modelFn : ModelFunction
        The function to evaluate

    cri : ConfidenceRegionFactoryView, optional
        View for computing confidence intervals.

    verbosity : int, optional
        Amount of detail to print to stdout.

    Returns
    -------
    ReportableQty or dict
        If `modelFn` does returns a dict of ReportableQty objects, otherwise
        a single ReportableQty.
    """
    if modelFn is None:  # so you can set fn to None when they're missing (e.g. diamond norm)
        return _ReportableQty(_np.nan)

    if cri:
        nmEBs = bool(cri.get_errobar_type() == "non-markovian")
        df, f0 = cri.get_fn_confidence_interval(
            modelFn, returnFnVal=True,
            verbosity=verbosity)
        return _make_reportable_qty_or_dict(f0, df, nmEBs)
    else:
        return _make_reportable_qty_or_dict(modelFn.evaluate(modelFn.base_model))


def spam_dotprods(rhoVecs, povms):
    """SPAM dot products (concatenates POVMS)"""
    nEVecs = sum(len(povm) for povm in povms)
    ret = _np.empty((len(rhoVecs), nEVecs), 'd')
    for i, rhoVec in enumerate(rhoVecs):
        j = 0
        for povm in povms:
            for EVec in povm.values():
                ret[i, j] = _np.vdot(EVec.todense(), rhoVec.todense()); j += 1
                # todense() gives a 1D array, so no need to transpose EVec
    return ret


Spam_dotprods = _modf.spamfn_factory(spam_dotprods)  # init args == (model)


def choi_matrix(gate, mxBasis):
    """Choi matrix"""
    return _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)


Choi_matrix = _modf.opfn_factory(choi_matrix)  # init args == (model, opLabel)


def choi_evals(gate, mxBasis):
    """Choi matrix eigenvalues"""
    choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
    choi_eigvals = _np.linalg.eigvals(choi)
    return _np.array(sorted(choi_eigvals))


Choi_evals = _modf.opfn_factory(choi_evals)  # init args == (model, opLabel)


def choi_trace(gate, mxBasis):
    """Trace of the Choi matrix"""
    choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
    return _np.trace(choi)


Choi_trace = _modf.opfn_factory(choi_trace)  # init args == (model, opLabel)


class Gate_eigenvalues(_modf.ModelFunction):
    """Gate eigenvalues"""

    def __init__(self, model, oplabel):
        self.oplabel = oplabel
        _modf.ModelFunction.__init__(self, model, [("gate", oplabel)])

    def evaluate(self, model):
        """Evaluate at `model`"""
        evals, evecs = _np.linalg.eig(model.operations[self.oplabel].todense())

        ev_list = list(enumerate(evals))
        ev_list.sort(key=lambda tup: abs(tup[1]), reverse=True)
        indx, evals = zip(*ev_list)
        evecs = evecs[:, indx]  # sort evecs according to evals

        self.G0 = model.operations[self.oplabel]
        self.evals = _np.array(evals)
        self.evecs = evecs
        self.inv_evecs = _np.linalg.inv(evecs)

        return self.evals

    def evaluate_nearby(self, nearby_model):
        """Evaluate at a nearby model"""
        #avoid calling minweight_match again
        dMx = nearby_model.operations[self.oplabel] - self.G0
        #evalsM = evals0 + Uinv * (M-M0) * U
        return _np.array([self.evals[k] + _np.dot(self.inv_evecs[k, :], _np.dot(dMx, self.evecs[:, k]))
                          for k in range(dMx.shape[0])])
    # ref for eigenvalue derivatives: https://www.win.tue.nl/casa/meetings/seminar/previous/_abstract051019_files/Presentation.pdf                              # noqa


class Circuit_eigenvalues(_modf.ModelFunction):
    """Circuit eigenvalues"""

    def __init__(self, model, circuit):
        self.circuit = circuit
        _modf.ModelFunction.__init__(self, model, ["all"])

    def evaluate(self, model):
        """Evaluate at `model`"""
        Mx = model.product(self.circuit)
        evals, evecs = _np.linalg.eig(Mx)

        ev_list = list(enumerate(evals))
        ev_list.sort(key=lambda tup: abs(tup[1]), reverse=True)
        indx, evals = zip(*ev_list)
        evecs = evecs[:, indx]  # sort evecs according to evals

        self.Mx = Mx
        self.evals = _np.array(evals)
        self.evecs = evecs
        self.inv_evecs = _np.linalg.inv(evecs)

        return self.evals

    def evaluate_nearby(self, nearby_model):
        """Evaluate at nearby model"""
        #avoid calling minweight_match again
        Mx = nearby_model.product(self.circuit)
        dMx = Mx - self.Mx
        #evalsM = evals0 + Uinv * (M-M0) * U
        return _np.array([self.evals[k] + _np.dot(self.inv_evecs[k, :], _np.dot(dMx, self.evecs[:, k]))
                          for k in range(dMx.shape[0])])
    # ref for eigenvalue derivatives: https://www.win.tue.nl/casa/meetings/seminar/previous/_abstract051019_files/Presentation.pdf                              # noqa


#def circuit_eigenvalues(model, circuit):
#    return _np.array(sorted(_np.linalg.eigvals(model.product(circuit)),
#                            key=lambda ev: abs(ev), reverse=True))
#Circuit_eigenvalues = _modf.modelfn_factory(circuit_eigenvalues)
## init args == (model, circuit)


def rel_circuit_eigenvalues(modelA, modelB, circuit):
    """Eigenvalues of dot(productB(circuit)^-1, productA(circuit))"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    rel_op = _np.dot(_np.linalg.inv(B), A)  # "relative gate" == target^{-1} * gate
    return _np.linalg.eigvals(rel_op)


Rel_circuit_eigenvalues = _modf.modelfn_factory(rel_circuit_eigenvalues)
# init args == (modelA, modelB, circuit)


def circuit_fro_diff(modelA, modelB, circuit):
    """ Frobenius distance btwn productA(circuit) and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return fro_diff(A, B, modelB.basis)


Circuit_fro_diff = _modf.modelfn_factory(circuit_fro_diff)
# init args == (modelA, modelB, circuit)


def circuit_entanglement_infidelity(modelA, modelB, circuit):
    """ Entanglement infidelity btwn productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return entanglement_infidelity(A, B, modelB.basis)


Circuit_entanglement_infidelity = _modf.modelfn_factory(circuit_entanglement_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_avg_gate_infidelity(modelA, modelB, circuit):
    """ Average gate infidelity between productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return avg_gate_infidelity(A, B, modelB.basis)


Circuit_avg_gate_infidelity = _modf.modelfn_factory(circuit_avg_gate_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_jt_diff(modelA, modelB, circuit):
    """ Jamiolkowski trace distance between productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return jt_diff(A, B, modelB.basis)


Circuit_jt_diff = _modf.modelfn_factory(circuit_jt_diff)
# init args == (modelA, modelB, circuit)

if _CVXPY_AVAILABLE:

    class Circuit_half_diamond_norm(_modf.ModelFunction):
        """ 1/2 diamond norm of difference between productA(circuit)
            and productB(circuit)"""

        def __init__(self, modelA, modelB, circuit):
            self.circuit = circuit
            self.B = modelB.product(circuit)
            self.d = int(round(_np.sqrt(modelA.dim)))
            _modf.ModelFunction.__init__(self, modelA, ["all"])

        def evaluate(self, model):
            """Evaluate this function at `model`"""
            A = model.product(self.circuit)
            dm, W = _tools.diamonddist(A, self.B, model.basis,
                                       return_x=True)
            self.W = W
            return 0.5 * dm

        def evaluate_nearby(self, nearby_model):
            """Evaluate at a nearby model"""
            mxBasis = nearby_model.basis
            JAstd = self.d * _tools.fast_jamiolkowski_iso_std(
                nearby_model.product(self.circuit), mxBasis)
            JBstd = self.d * _tools.fast_jamiolkowski_iso_std(self.B, mxBasis)
            Jt = (JBstd - JAstd).T
            return 0.5 * _np.trace(_np.dot(Jt.real, self.W.real) + _np.dot(Jt.imag, self.W.imag))

    #def circuit_half_diamond_norm(modelA, modelB, circuit):
    #    A = modelA.product(circuit) # "gate"
    #    B = modelB.product(circuit) # "target gate"
    #    return half_diamond_norm(A, B, modelB.basis)
    #Circuit_half_diamond_norm = _modf.modelfn_factory(circuit_half_diamond_norm)
    #  # init args == (modelA, modelB, circuit)

else:
    circuit_half_diamond_norm = None
    Circuit_half_diamond_norm = _nullFn


def circuit_nonunitary_entanglement_infidelity(modelA, modelB, circuit):
    """ Nonunitary entanglement infidelity between productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return nonunitary_entanglement_infidelity(A, B, modelB.basis)


Circuit_nonunitary_entanglement_infidelity = _modf.modelfn_factory(circuit_nonunitary_entanglement_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_nonunitary_avg_gate_infidelity(modelA, modelB, circuit):
    """ Nonunitary average gate infidelity between productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return nonunitary_avg_gate_infidelity(A, B, modelB.basis)


Circuit_nonunitary_avg_gate_infidelity = _modf.modelfn_factory(circuit_nonunitary_avg_gate_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_eigenvalue_entanglement_infidelity(modelA, modelB, circuit):
    """ Eigenvalue entanglement infidelity between productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return eigenvalue_entanglement_infidelity(A, B, modelB.basis)


Circuit_eigenvalue_entanglement_infidelity = _modf.modelfn_factory(circuit_eigenvalue_entanglement_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_eigenvalue_avg_gate_infidelity(modelA, modelB, circuit):
    """ Eigenvalue average gate infidelity between productA(circuit)
        and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return eigenvalue_avg_gate_infidelity(A, B, modelB.basis)


Circuit_eigenvalue_avg_gate_infidelity = _modf.modelfn_factory(circuit_eigenvalue_avg_gate_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_eigenvalue_nonunitary_entanglement_infidelity(modelA, modelB, circuit):
    """ Eigenvalue nonunitary entanglement infidelity between
        productA(circuit) and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return eigenvalue_nonunitary_entanglement_infidelity(A, B, modelB.basis)


Circuit_eigenvalue_nonunitary_entanglement_infidelity = _modf.modelfn_factory(
    circuit_eigenvalue_nonunitary_entanglement_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_eigenvalue_nonunitary_avg_gate_infidelity(modelA, modelB, circuit):
    """ Eigenvalue nonunitary average gate infidelity between
        productA(circuit) and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return eigenvalue_nonunitary_avg_gate_infidelity(A, B, modelB.basis)


Circuit_eigenvalue_nonunitary_avg_gate_infidelity = _modf.modelfn_factory(
    circuit_eigenvalue_nonunitary_avg_gate_infidelity)
# init args == (modelA, modelB, circuit)


def circuit_eigenvalue_diamondnorm(modelA, modelB, circuit):
    """ Eigenvalue diamond distance between
        productA(circuit) and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return eigenvalue_diamondnorm(A, B, modelB.basis)


Circuit_eigenvalue_diamondnorm = _modf.modelfn_factory(circuit_eigenvalue_diamondnorm)
# init args == (modelA, modelB, circuit)


def circuit_eigenvalue_nonunitary_diamondnorm(modelA, modelB, circuit):
    """ Eigenvalue nonunitary diamond distance between
        productA(circuit) and productB(circuit)"""
    A = modelA.product(circuit)  # "gate"
    B = modelB.product(circuit)  # "target gate"
    return eigenvalue_nonunitary_diamondnorm(A, B, modelB.basis)


Circuit_eigenvalue_nonunitary_diamondnorm = _modf.modelfn_factory(circuit_eigenvalue_nonunitary_diamondnorm)
# init args == (modelA, modelB, circuit)


def povm_entanglement_infidelity(modelA, modelB, povmlbl):
    """
    POVM entanglement infidelity between `modelA` and `modelB`, equal to
    `1 - entanglement_fidelity(POVM_MAP)` where `POVM_MAP` is the extension
    of the POVM from the classical space of k-outcomes to the space of
    (diagonal) k by k density matrices.
    """
    return 1.0 - _tools.povm_fidelity(modelA, modelB, povmlbl)


POVM_entanglement_infidelity = _modf.povmfn_factory(povm_entanglement_infidelity)
# init args == (model1, modelB, povmlbl)


def povm_jt_diff(modelA, modelB, povmlbl):
    """
    POVM Jamiolkowski trace distance between `modelA` and `modelB`, equal to
    `Jamiolkowski_trace_distance(POVM_MAP)` where `POVM_MAP` is the extension
    of the POVM from the classical space of k-outcomes to the space of
    (diagonal) k by k density matrices.
    """
    return _tools.povm_jtracedist(modelA, modelB, povmlbl)


POVM_jt_diff = _modf.povmfn_factory(povm_jt_diff)
# init args == (model1, modelB, povmlbl)

if _CVXPY_AVAILABLE:

    def povm_half_diamond_norm(modelA, modelB, povmlbl):
        """
        Half the POVM diamond distance between `modelA` and `modelB`, equal
        to `half_diamond_dist(POVM_MAP)` where `POVM_MAP` is the extension
        of the POVM from the classical space of k-outcomes to the space of
        (diagonal) k by k density matrices.
        """
        return 0.5 * _tools.povm_diamonddist(modelA, modelB, povmlbl)
    POVM_half_diamond_norm = _modf.povmfn_factory(povm_half_diamond_norm)
else:
    povm_half_diamond_norm = None
    POVM_half_diamond_norm = _nullFn


def decomposition(gate):
    """
    DEPRECATED: Decompose a 1Q `gate` into rotations about axes.
    """
    decompDict = _tools.decompose_gate_matrix(gate)
    if decompDict['isValid']:
        #angleQty   = decompDict.get('pi rotations',0)
        #diagQty    = decompDict.get('decay of diagonal rotation terms',0)
        #offdiagQty = decompDict.get('decay of off diagonal rotation terms',0)
        errBarDict = {'pi rotations': None,
                      'decay of diagonal rotation terms': None,
                      'decay of off diagonal rotation terms': None}
        return _ReportableQty(decompDict, errBarDict)
    else:
        return _ReportableQty({})


def upper_bound_fidelity(gate, mxBasis):
    """ Upper bound on entanglement fidelity """
    return _tools.fidelity_upper_bound(gate)[0]


Upper_bound_fidelity = _modf.opfn_factory(upper_bound_fidelity)
# init args == (model, opLabel)


def closest_ujmx(gate, mxBasis):
    """ Jamiolkowski state of closest unitary to `gate` """
    closestUOpMx = _alg.find_closest_unitary_opmx(gate)
    return _tools.jamiolkowski_iso(closestUOpMx, mxBasis, mxBasis)


Closest_ujmx = _modf.opfn_factory(closest_ujmx)
# init args == (model, opLabel)


def maximum_fidelity(gate, mxBasis):
    """ Fidelity between `gate` and its closest unitary"""
    closestUOpMx = _alg.find_closest_unitary_opmx(gate)
    closestUJMx = _tools.jamiolkowski_iso(closestUOpMx, mxBasis, mxBasis)
    choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
    return _tools.fidelity(closestUJMx, choi)


Maximum_fidelity = _modf.opfn_factory(maximum_fidelity)
# init args == (model, opLabel)


def maximum_trace_dist(gate, mxBasis):
    """ Jamiolkowski trace distance between `gate` and its closest unitary"""
    closestUOpMx = _alg.find_closest_unitary_opmx(gate)
    #closestUJMx = _tools.jamiolkowski_iso(closestUOpMx, mxBasis, mxBasis)
    _tools.jamiolkowski_iso(closestUOpMx, mxBasis, mxBasis)
    return _tools.jtracedist(gate, closestUOpMx)


Maximum_trace_dist = _modf.opfn_factory(maximum_trace_dist)
# init args == (model, opLabel)


def angles_btwn_rotn_axes(model):
    """
    Array of angles between the rotation axes of the gates of `model`.

    Returns
    -------
    numpy.ndarray
        Of size `(nOperations,nGate)` where `nOperations=len(model.operations)`
    """
    opLabels = list(model.operations.keys())
    angles_btwn_rotn_axes = _np.zeros((len(opLabels), len(opLabels)), 'd')

    for i, gl in enumerate(opLabels):
        decomp = _tools.decompose_gate_matrix(model.operations[gl].todense())
        rotnAngle = decomp.get('pi rotations', 'X')
        axisOfRotn = decomp.get('axis of rotation', None)

        for j, gl_other in enumerate(opLabels[i + 1:], start=i + 1):
            decomp_other = _tools.decompose_gate_matrix(model.operations[gl_other])
            rotnAngle_other = decomp_other.get('pi rotations', 'X')

            if str(rotnAngle) == 'X' or abs(rotnAngle) < 1e-4 or \
               str(rotnAngle_other) == 'X' or abs(rotnAngle_other) < 1e-4:
                angles_btwn_rotn_axes[i, j] = _np.nan
            else:
                axisOfRotn_other = decomp_other.get('axis of rotation', None)
                if axisOfRotn is not None and axisOfRotn_other is not None:
                    real_dot = _np.clip(_np.real(_np.dot(axisOfRotn, axisOfRotn_other)), -1.0, 1.0)
                    angles_btwn_rotn_axes[i, j] = _np.arccos(real_dot) / _np.pi
                else:
                    angles_btwn_rotn_axes[i, j] = _np.nan

            angles_btwn_rotn_axes[j, i] = angles_btwn_rotn_axes[i, j]
    return angles_btwn_rotn_axes


Angles_btwn_rotn_axes = _modf.modelfn_factory(angles_btwn_rotn_axes)
# init args == (model)


def entanglement_fidelity(A, B, mxBasis):
    """Entanglement fidelity between A and B"""
    return _tools.entanglement_fidelity(A, B, mxBasis)


Entanglement_fidelity = _modf.opsfn_factory(entanglement_fidelity)
# init args == (model1, model2, opLabel)


def entanglement_infidelity(A, B, mxBasis):
    """Entanglement infidelity between A and B"""
    return 1 - _tools.entanglement_fidelity(A, B, mxBasis)


Entanglement_infidelity = _modf.opsfn_factory(entanglement_infidelity)
# init args == (model1, model2, opLabel)


def closest_unitary_fidelity(A, B, mxBasis):  # assume vary model1, model2 fixed
    """Entanglement infidelity between closest unitaries to A and B"""
    decomp1 = _tools.decompose_gate_matrix(A)
    decomp2 = _tools.decompose_gate_matrix(B)

    if decomp1['isUnitary']:
        closestUGateMx1 = A
    else: closestUGateMx1 = _alg.find_closest_unitary_opmx(A)

    if decomp2['isUnitary']:
        closestUGateMx2 = B
    else: closestUGateMx2 = _alg.find_closest_unitary_opmx(A)

    closeChoi1 = _tools.jamiolkowski_iso(closestUGateMx1)
    closeChoi2 = _tools.jamiolkowski_iso(closestUGateMx2)
    return _tools.fidelity(closeChoi1, closeChoi2)


Closest_unitary_fidelity = _modf.opsfn_factory(closest_unitary_fidelity)
# init args == (model1, model2, opLabel)


def fro_diff(A, B, mxBasis):  # assume vary model1, model2 fixed
    """ Frobenius distance between A and B """
    return _tools.frobeniusdist(A, B)


Fro_diff = _modf.opsfn_factory(fro_diff)
# init args == (model1, model2, opLabel)


def jt_diff(A, B, mxBasis):  # assume vary model1, model2 fixed
    """ Jamiolkowski trace distance between A and B"""
    return _tools.jtracedist(A, B, mxBasis)


Jt_diff = _modf.opsfn_factory(jt_diff)
# init args == (model1, model2, opLabel)


if _CVXPY_AVAILABLE:

    class Half_diamond_norm(_modf.ModelFunction):
        """Half the diamond distance bewteen `modelA.operations[opLabel]` and
           `modelB.operations[opLabel]` """

        def __init__(self, modelA, modelB, oplabel):
            self.oplabel = oplabel
            self.B = modelB.operations[oplabel].todense()
            self.d = int(round(_np.sqrt(modelA.dim)))
            _modf.ModelFunction.__init__(self, modelA, [("gate", oplabel)])

        def evaluate(self, model):
            """Evaluate at `modelA = model` """
            gl = self.oplabel
            dm, W = _tools.diamonddist(model.operations[gl].todense(),
                                       self.B, model.basis, return_x=True)
            self.W = W
            return 0.5 * dm

        def evaluate_nearby(self, nearby_model):
            """Evaluates at a nearby model"""
            gl = self.oplabel; mxBasis = nearby_model.basis
            JAstd = self.d * _tools.fast_jamiolkowski_iso_std(
                nearby_model.operations[gl].todense(), mxBasis)
            JBstd = self.d * _tools.fast_jamiolkowski_iso_std(self.B, mxBasis)
            Jt = (JBstd - JAstd).T
            return 0.5 * _np.trace(_np.dot(Jt.real, self.W.real) + _np.dot(Jt.imag, self.W.imag))

    def half_diamond_norm(A, B, mxBasis):
        return 0.5 * _tools.diamonddist(A, B, mxBasis)
    #Half_diamond_norm = _modf.opsfn_factory(half_diamond_norm)
    ## init args == (model1, model2, opLabel)

else:
    half_diamond_norm = None
    Half_diamond_norm = _nullFn


def std_unitarity(A, B, mxBasis):
    """ A gauge-invariant quantity that behaves like the unitarity """
    Lambda = _np.dot(A, _np.linalg.inv(B))
    return _tools.unitarity(Lambda, mxBasis)


def eigenvalue_unitarity(A, B):
    """ A gauge-invariant quantity that behaves like the unitarity """
    Lambda = _np.dot(A, _np.linalg.inv(B))
    d2 = Lambda.shape[0]
    lmb = _np.linalg.eigvals(Lambda)
    return float(_np.real(_np.vdot(lmb, lmb)) - 1.0) / (d2 - 1.0)


def nonunitary_entanglement_infidelity(A, B, mxBasis):
    """ Returns (d^2 - 1)/d^2 * (1 - sqrt(U)), where U is the unitarity of A*B^{-1} """
    if isinstance(mxBasis, _DirectSumBasis): return -1  # deal w/block-dims later
    d2 = A.shape[0]; U = std_unitarity(A, B, mxBasis)
    return (d2 - 1.0) / d2 * (1.0 - _np.sqrt(U))


Nonunitary_entanglement_infidelity = _modf.opsfn_factory(nonunitary_entanglement_infidelity)
# init args == (model1, model2, opLabel)


def nonunitary_avg_gate_infidelity(A, B, mxBasis):
    """ Returns (d - 1)/d * (1 - sqrt(U)), where U is the unitarity of A*B^{-1} """
    if isinstance(mxBasis, _DirectSumBasis): return -1  # deal w/block-dims later
    d2 = A.shape[0]; d = int(round(_np.sqrt(d2)))
    U = std_unitarity(A, B, mxBasis)
    return (d - 1.0) / d * (1.0 - _np.sqrt(U))


Nonunitary_avg_gate_infidelity = _modf.opsfn_factory(nonunitary_avg_gate_infidelity)
# init args == (model1, model2, opLabel)


def eigenvalue_nonunitary_entanglement_infidelity(A, B, mxBasis):
    """ Returns (d^2 - 1)/d^2 * (1 - sqrt(U)), where U is the eigenvalue-unitarity of A*B^{-1} """
    d2 = A.shape[0]; U = eigenvalue_unitarity(A, B)
    return (d2 - 1.0) / d2 * (1.0 - _np.sqrt(U))


Eigenvalue_nonunitary_entanglement_infidelity = _modf.opsfn_factory(eigenvalue_nonunitary_entanglement_infidelity)
# init args == (model1, model2, opLabel)


def eigenvalue_nonunitary_avg_gate_infidelity(A, B, mxBasis):
    """ Returns (d - 1)/d * (1 - sqrt(U)), where U is the eigenvalue-unitarity of A*B^{-1} """
    d2 = A.shape[0]; d = int(round(_np.sqrt(d2)))
    U = eigenvalue_unitarity(A, B)
    return (d - 1.0) / d * (1.0 - _np.sqrt(U))


Eigenvalue_nonunitary_avg_gate_infidelity = _modf.opsfn_factory(eigenvalue_nonunitary_avg_gate_infidelity)
# init args == (model1, model2, opLabel)


def eigenvalue_entanglement_infidelity(A, B, mxBasis):
    """ Eigenvalue entanglement infidelity between A and B """
    d2 = A.shape[0]
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    _, pairs = _tools.minweight_match(evA, evB, lambda x, y: abs(x - y),
                                      return_pairs=True)  # just to get pairing
    mlPl = abs(_np.sum([_np.conjugate(evB[j]) * evA[i] for i, j in pairs]))
    return 1.0 - mlPl / float(d2)


Eigenvalue_entanglement_infidelity = _modf.opsfn_factory(eigenvalue_entanglement_infidelity)
# init args == (model1, model2, opLabel)


def eigenvalue_avg_gate_infidelity(A, B, mxBasis):
    """ Eigenvalue average gate infidelity between A and B """
    d2 = A.shape[0]; d = int(round(_np.sqrt(d2)))
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    _, pairs = _tools.minweight_match(evA, evB, lambda x, y: abs(x - y),
                                      return_pairs=True)  # just to get pairing
    mlPl = abs(_np.sum([_np.conjugate(evB[j]) * evA[i] for i, j in pairs]))
    return (d2 - mlPl) / float(d * (d + 1))


Eigenvalue_avg_gate_infidelity = _modf.opsfn_factory(eigenvalue_avg_gate_infidelity)
# init args == (model1, model2, opLabel)


def eigenvalue_diamondnorm(A, B, mxBasis):
    """ Eigenvalue diamond distance between A and B """
    d2 = A.shape[0]
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    return (d2 - 1.0) / d2 * _np.max(_tools.minweight_match(evA, evB, lambda x, y: abs(x - y),
                                                            return_pairs=False))


Eigenvalue_diamondnorm = _modf.opsfn_factory(eigenvalue_diamondnorm)
# init args == (model1, model2, opLabel)


def eigenvalue_nonunitary_diamondnorm(A, B, mxBasis):
    """ Eigenvalue nonunitary diamond distance between A and B """
    d2 = A.shape[0]
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    return (d2 - 1.0) / d2 * _np.max(_tools.minweight_match(evA, evB, lambda x, y: abs(abs(x) - abs(y)),
                                                            return_pairs=False))


Eigenvalue_nonunitary_diamondnorm = _modf.opsfn_factory(eigenvalue_nonunitary_diamondnorm)
# init args == (model1, model2, opLabel)


def avg_gate_infidelity(A, B, mxBasis):
    """ Returns the average gate infidelity between A and B, where B is the "target" operation."""
    return _tools.average_gate_infidelity(A, B, mxBasis)


Avg_gate_infidelity = _modf.opsfn_factory(avg_gate_infidelity)
# init args == (model1, model2, opLabel)


def model_model_angles_btwn_axes(A, B, mxBasis):  # Note: default 'gm' basis
    """ Angle between the rotation axes of A and B (1-qubit gates)"""
    decomp = _tools.decompose_gate_matrix(A)
    decomp2 = _tools.decompose_gate_matrix(B)
    axisOfRotn = decomp.get('axis of rotation', None)
    rotnAngle = decomp.get('pi rotations', 'X')
    axisOfRotn2 = decomp2.get('axis of rotation', None)
    rotnAngle2 = decomp2.get('pi rotations', 'X')

    if rotnAngle == 'X' or abs(rotnAngle) < 1e-4 or \
       rotnAngle2 == 'X' or abs(rotnAngle2) < 1e-4:
        return _np.nan

    if axisOfRotn is None or axisOfRotn2 is None:
        return _np.nan

    real_dot = _np.clip(_np.real(_np.dot(axisOfRotn, axisOfRotn2)), -1.0, 1.0)
    return _np.arccos(abs(real_dot)) / _np.pi
    #Note: abs() allows axis to be off by 180 degrees -- if showing *angle* as
    #      well, must flip sign of angle of rotation if you allow axis to
    #      "reverse" by 180 degrees.


Model_model_angles_btwn_axes = _modf.opsfn_factory(model_model_angles_btwn_axes)
# init args == (model1, model2, opLabel)


def rel_eigvals(A, B, mxBasis):
    """ Eigenvalues of B^{-1} * A"""
    target_op_inv = _np.linalg.inv(B)
    rel_op = _np.dot(target_op_inv, A)
    return _np.linalg.eigvals(rel_op).astype("complex")  # since they generally *can* be complex


Rel_eigvals = _modf.opsfn_factory(rel_eigvals)
# init args == (model1, model2, opLabel)


def rel_logTiG_eigvals(A, B, mxBasis):
    """ Eigenvalues of log(B^{-1} * A)"""
    rel_op = _tools.error_generator(A, B, mxBasis, "logTiG")
    return _np.linalg.eigvals(rel_op).astype("complex")  # since they generally *can* be complex


Rel_logTiG_eigvals = _modf.opsfn_factory(rel_logTiG_eigvals)
# init args == (model1, model2, opLabel)


def rel_logGTi_eigvals(A, B, mxBasis):
    """ Eigenvalues of log(A * B^{-1})"""
    rel_op = _tools.error_generator(A, B, mxBasis, "logGTi")
    return _np.linalg.eigvals(rel_op).astype("complex")  # since they generally *can* be complex


Rel_logGTi_eigvals = _modf.opsfn_factory(rel_logGTi_eigvals)
# init args == (model1, model2, opLabel)


def rel_logGmlogT_eigvals(A, B, mxBasis):
    """ Eigenvalues of log(A) - log(B)"""
    rel_op = _tools.error_generator(A, B, mxBasis, "logG-logT")
    return _np.linalg.eigvals(rel_op).astype("complex")  # since they generally *can* be complex


Rel_logGmlogT_eigvals = _modf.opsfn_factory(rel_logGmlogT_eigvals)
# init args == (model1, model2, opLabel)


def rel_gate_eigenvalues(A, B, mxBasis):  # DUPLICATE of rel_eigvals TODO
    """ Eigenvalues of B^{-1} * A """
    rel_op = _np.dot(_np.linalg.inv(B), A)  # "relative gate" == target^{-1} * gate
    return _np.linalg.eigvals(rel_op).astype("complex")  # since they generally *can* be complex


Rel_gate_eigenvalues = _modf.opsfn_factory(rel_gate_eigenvalues)
# init args == (model1, model2, opLabel)


def errgen_and_projections(errgen, mxBasis):
    """
    Project `errgen` on all of the standard sets of error generators.

    Returns
    -------
    dict
        Dictionary of 'error generator', '*X* projections', and
        '*X* projection power' keys, where *X* is 'hamiltonian',
        'stochastic', and 'affine'.
    """
    ret = {}
    egnorm = _np.linalg.norm(errgen.flatten())
    ret['error generator'] = errgen
    proj, scale = \
        _tools.std_errgen_projections(
            errgen, "hamiltonian", mxBasis, mxBasis, return_scale_fctr=True)
    ret['hamiltonian projections'] = proj
    ret['hamiltonian projection power'] = float(_np.sum(proj**2) / scale**2) / egnorm**2 \
        if (abs(scale) > 1e-8 and abs(egnorm) > 1e-8) else 0
    #sum of squared projections of normalized error generator onto normalized projectors

    proj, scale = \
        _tools.std_errgen_projections(
            errgen, "stochastic", mxBasis, mxBasis, return_scale_fctr=True)
    ret['stochastic projections'] = proj
    ret['stochastic projection power'] = float(_np.sum(proj**2) / scale**2) / egnorm**2 \
        if (abs(scale) > 1e-8 and abs(egnorm) > 1e-8) else 0
    #sum of squared projections of normalized error generator onto normalized projectors

    proj, scale = \
        _tools.std_errgen_projections(
            errgen, "affine", mxBasis, mxBasis, return_scale_fctr=True)
    ret['affine projections'] = proj
    ret['affine projection power'] = float(_np.sum(proj**2) / scale**2) / egnorm**2 \
        if (abs(scale) > 1e-8 and abs(egnorm) > 1e-8) else 0
    #sum of squared projections of normalized error generator onto normalized projectors
    return ret


def logTiG_and_projections(A, B, mxBasis):
    """
    Projections of `log(B^{-1}*A)`.  Returns a dictionary of quantities with
    keys 'error generator', '*X* projections', and '*X* projection power',
    where *X* is 'hamiltonian', 'stochastic', and 'affine'.
    """
    errgen = _tools.error_generator(A, B, mxBasis, "logTiG")
    return errgen_and_projections(errgen, mxBasis)


LogTiG_and_projections = _modf.opsfn_factory(logTiG_and_projections)
# init args == (model1, model2, opLabel)


def logGTi_and_projections(A, B, mxBasis):
    """
    Projections of `log(A*B^{-1})`.  Returns a dictionary of quantities with
    keys 'error generator', '*X* projections', and '*X* projection power',
    where *X* is 'hamiltonian', 'stochastic', and 'affine'.
    """
    errgen = _tools.error_generator(A, B, mxBasis, "logGTi")
    return errgen_and_projections(errgen, mxBasis)


LogGTi_and_projections = _modf.opsfn_factory(logGTi_and_projections)
# init args == (model1, model2, opLabel)


def logGmlogT_and_projections(A, B, mxBasis):
    """
    Projections of `log(A)-log(B)`.  Returns a dictionary of quantities with
    keys 'error generator', '*X* projections', and '*X* projection power',
    where *X* is 'hamiltonian', 'stochastic', and 'affine'.
    """
    errgen = _tools.error_generator(A, B, mxBasis, "logG-logT")
    return errgen_and_projections(errgen, mxBasis)


LogGmlogT_and_projections = _modf.opsfn_factory(logGmlogT_and_projections)
# init args == (model1, model2, opLabel)


def robust_logGTi_and_projections(modelA, modelB, syntheticIdleStrs):
    """
    Projections of `log(A*B^{-1})` using a gauge-robust technique.
    Returns a dictionary of quantities with keys '*G* error generator',
    '*G* *X* projections', and '*G* *X* projection power',
    where *G* is a operation label and *X* is 'hamiltonian', 'stochastic', and
    'affine'.
    """
    ret = {}
    mxBasis = modelB.basis  # target model is more likely to have a valid basis
    Id = _np.identity(modelA.dim, 'd')
    opLabels = [gl for gl, gate in modelB.operations.items() if not _np.allclose(gate, Id)]
    nOperations = len(opLabels)

    error_superops = []; ptype_counts = {}; ptype_scaleFctrs = {}
    error_labels = []
    for ptype in ("hamiltonian", "stochastic", "affine"):
        lindbladMxs = _tools.std_error_generators(modelA.dim, ptype,
                                                  mxBasis)
        lindbladMxBasis = _Basis.cast(mxBasis, modelA.dim)

        lindbladMxs = lindbladMxs[1:]  # skip [0] == Identity
        lbls = lindbladMxBasis.labels[1:]

        scaleFctr = _tools.std_scale_factor(modelA.dim, ptype)
        #if ptype == "hamiltonian": scaleFctr *= 2.0 #HACK (DEAL LATER)
        #if ptype == "affine": scaleFctr *= 0.5 #HACK
        ptype_counts[ptype] = len(lindbladMxs)
        ptype_scaleFctrs[ptype] = scaleFctr  # UNUSED?
        error_superops.extend([_tools.change_basis(eg, "std", mxBasis) for eg in lindbladMxs])
        error_labels.extend(["%s(%s)" % (ptype[0], lbl) for lbl in lbls])
    nSuperOps = len(error_superops)
    assert(len(error_labels) == nSuperOps)

    #DEBUG
    #print("DB: %d gates (%s)" % (nOperations, str(opLabels)))
    #print("DB: %d superops; counts = " % nSuperOps, ptype_counts)
    #print("DB: factors = ",ptype_scaleFctrs)
    #for i,x in enumerate(error_superops):
    #    print("DB: Superop vec[%d] norm = %g" % (i,_np.linalg.norm(x)))
    #    print("DB: Choi Superop[%d] = " % i)
    #    _tools.print_mx(_tools.jamiolkowski_iso(x, mxBasis, mxBasis), width=4, prec=1)
    #    print("")

    def get_projection_vec(errgen):
        proj = []
        for ptype in ("hamiltonian", "stochastic", "affine"):
            proj.append(_tools.std_errgen_projections(
                errgen, ptype, mxBasis, mxBasis)[1:])  # skip [0] == Identity
        return _np.concatenate(proj)

    def firstOrderNoise(opstr, errSupOp, glWithErr):
        noise = _np.zeros((modelB.dim, modelB.dim), 'd')
        for n, gl in enumerate(opstr):
            if gl == glWithErr:
                noise += _np.dot(modelB.product(opstr[n + 1:]),
                                 _np.dot(errSupOp, modelB.product(opstr[:n + 1])))
        #DEBUG
        #print("first order noise (%s,%s) Choi superop : " % (str(opstr),glWithErr))
        #_tools.print_mx( _tools.jamiolkowski_iso(noise, mxBasis, mxBasis) ,width=4,prec=1)

        return noise  # _tools.jamiolkowski_iso(noise, mxBasis, mxBasis)

    def errorGeneratorJacobian(opstr):
        jac = _np.empty((nSuperOps, nSuperOps * nOperations), 'complex')  # should be real, but we'll check

        for i, gl in enumerate(opLabels):
            for k, errOnGate in enumerate(error_superops):
                noise = firstOrderNoise(opstr, errOnGate, gl)
                jac[:, i * nSuperOps + k] = [_np.vdot(errOut.flatten(), noise.flatten()) for errOut in error_superops]

                #DEBUG CHECK
                check = [_np.trace(_np.dot(
                    _tools.jamiolkowski_iso(errOut, mxBasis, mxBasis).conj().T,
                    _tools.jamiolkowski_iso(noise, mxBasis, mxBasis))) * 4  # for 1-qubit...
                    for errOut in error_superops]
                assert(_np.allclose(jac[:, i * nSuperOps + k], check))

        assert(_np.linalg.norm(jac.imag) < 1e-6), "error generator jacobian should be real!"
        return jac.real

    runningJac = None; runningY = None
    for s in syntheticIdleStrs:
        Sa = modelA.product(s)
        Sb = modelB.product(s)
        assert(_np.linalg.norm(Sb - _np.identity(modelB.dim, 'd')) < 1e-6), \
            "Synthetic idle %s is not an idle!!" % str(s)
        SIerrgen = _tools.error_generator(Sa, Sb, mxBasis, "logGTi")
        SIproj = get_projection_vec(SIerrgen)
        jacSI = errorGeneratorJacobian(s)
        #print("DB jacobian for %s = \n" % str(s)); _tools.print_mx(jacSI, width=4, prec=1) #DEBUG
        if runningJac is None:
            runningJac = jacSI
            runningY = SIproj
        else:
            runningJac = _np.concatenate((runningJac, jacSI), axis=0)
            runningY = _np.concatenate((runningY, SIproj), axis=0)

        rank = _np.linalg.matrix_rank(runningJac)

        print("DB: Added synthetic idle %s => rank=%d <?> %d (shape=%s; %s)" %
              (str(s), rank, nSuperOps * nOperations, str(runningJac.shape), str(runningY.shape)))

        #if rank >= nSuperOps*nOperations: #then we can extract error terms for the gates
        #    # J*vec_opErrs = Y => vec_opErrs = (J^T*J)^-1 J^T*Y (lin least squares)
        #    J,JT = runningJac, runningJac.T
        #    vec_opErrs = _np.dot( _np.linalg.inv(_np.dot(JT,J)), _np.dot(JT,runningY))
        #    return vec_to_projdict(vec_opErrs)
    #raise ValueError("Not enough synthetic idle sequences to extract gauge-robust error rates.")

    # J*vec_opErrs = Y => U*s*Vt * vecErrRates = Y  => Vt*vecErrRates = s^{-1}*U^-1*Y
    # where shapes are: U = (M,K), s = (K,K), Vt = (K,N),
    #   so Uinv*Y = (K,) and s^{-1}*Uinv*Y = (K,), and rows of Vt specify the linear combos
    #   corresponding to values in s^{-1}*Uinv*Y that are != 0
    ret = {}
    RANK_TOL = 1e-8; COEFF_TOL = 1e-1
    U, s, Vt = _np.linalg.svd(runningJac)
    rank = _np.count_nonzero(s > RANK_TOL)
    vals = _np.dot(_np.diag(1.0 / s[0:rank]), _np.dot(U[:, 0:rank].conj().T, runningY))
    op_error_labels = ["%s.%s" % (gl, errLbl) for gl in opLabels for errLbl in error_labels]
    assert(len(op_error_labels) == runningJac.shape[1])
    for combo, val in zip(Vt[0:rank, :], vals):
        combo_str = " + ".join(["%.1f*%s" % (c, errLbl)
                                for c, errLbl in zip(combo, op_error_labels) if abs(c) > COEFF_TOL])
        ret[combo_str] = val
    return ret


Robust_LogGTi_and_projections = _modf.modelfn_factory(robust_logGTi_and_projections)
# init args == (modelA, modelB, syntheticIdleStrs)


def general_decomposition(modelA, modelB):
    """
    Decomposition of gates in `modelA` using those in `modelB` as their
    targets.  This function uses a generalized decomposition algorithm that
    can gates acting on a Hilbert space of any dimension.

    Returns
    -------
    dict
    """
    # B is target model usually but must be "gatsetB" b/c of decorator coding...
    decomp = {}
    opLabels = list(modelA.operations.keys())  # operation labels
    mxBasis = modelB.basis  # B is usually the target which has a well-defined basis

    for gl in opLabels:
        gate = modelA.operations[gl].todense()
        targetOp = modelB.operations[gl].todense()
        gl = str(gl)  # Label -> str for decomp-dict keys

        target_evals = _np.linalg.eigvals(targetOp)
        if _np.any(_np.isclose(target_evals, -1.0)):
            target_logG = _tools.unitary_superoperator_matrix_log(targetOp, mxBasis)
            logG = _tools.approximate_matrix_log(gate, target_logG)
        else:
            logG = _tools.real_matrix_log(gate, "warn")
            if _np.linalg.norm(logG.imag) > 1e-6:
                _warnings.warn("Truncating imaginary logarithm!")
                logG = _np.real(logG)

        decomp[gl + ' log inexactness'] = _np.linalg.norm(_spl.expm(logG) - gate)

        hamProjs, hamGens = _tools.std_errgen_projections(
            logG, "hamiltonian", mxBasis, mxBasis, return_generators=True)
        norm = _np.linalg.norm(hamProjs)
        decomp[gl + ' axis'] = hamProjs / norm if (norm > 1e-15) else hamProjs

        decomp[gl + ' angle'] = norm * 2.0 / _np.pi
        # Units: hamProjs (and norm) are already in "Hamiltonian-coefficient" units,
        # (see 'std_scale_factor' fn), but because of convention the "angle" is equal
        # to *twice* this coefficient (e.g. a X(pi/2) rotn is exp( i pi/4 X ) ),
        # thus the factor of 2.0 above.

        basis_mxs = mxBasis.elements
        scalings = [(_np.linalg.norm(hamGens[i]) / _np.linalg.norm(_tools.hamiltonian_to_lindbladian(mx))
                     if _np.linalg.norm(hamGens[i]) > 1e-10 else 0.0)
                    for i, mx in enumerate(basis_mxs)]
        #really want hamProjs[i] * lindbladian_to_hamiltonian(hamGens[i]) but fn doesn't exists (yet)
        hamMx = sum([s * c * bmx for s, c, bmx in zip(scalings, hamProjs, basis_mxs)])
        decomp[gl + ' hamiltonian eigenvalues'] = _np.array(_np.linalg.eigvals(hamMx))

    for gl in opLabels:
        for gl_other in opLabels:
            rotnAngle = decomp[str(gl) + ' angle']
            rotnAngle_other = decomp[str(gl_other) + ' angle']

            if gl == gl_other or abs(rotnAngle) < 1e-4 or abs(rotnAngle_other) < 1e-4:
                decomp[str(gl) + "," + str(gl_other) + " axis angle"] = 10000.0  # sentinel for irrelevant angle

            real_dot = _np.clip(
                _np.real(_np.dot(decomp[str(gl) + ' axis'].flatten(),
                                 decomp[str(gl_other) + ' axis'].flatten())),
                -1.0, 1.0)
            angle = _np.arccos(real_dot) / _np.pi
            decomp[str(gl) + "," + str(gl_other) + " axis angle"] = angle

    return decomp


General_decomposition = _modf.modelfn_factory(general_decomposition)
# init args == (modelA, modelB)


def average_gateset_infidelity(modelA, modelB):
    """ Average model infidelity """
    # B is target model usually but must be "modelB" b/c of decorator coding...
    #TEMPORARILY disabled b/c RB analysis is broken
    #from ..extras.rb import theory as _rbtheory
    return -1.0  # _rbtheory.gateset_infidelity(modelA, modelB)


Average_gateset_infidelity = _modf.modelfn_factory(average_gateset_infidelity)
# init args == (modelA, modelB)


def predicted_rb_number(modelA, modelB):
    """
    Prediction of RB number based on estimated (A) and target (B) models
    """
    #TEMPORARILY disabled b/c RB analysis is broken
    #from ..extras.rb import theory as _rbtheory
    return -1.0  # _rbtheory.predicted_RB_number(modelA, modelB)


Predicted_rb_number = _modf.modelfn_factory(predicted_rb_number)
# init args == (modelA, modelB)


def vec_fidelity(A, B, mxBasis):
    """ State fidelity between SPAM vectors A and B """
    rhoMx1 = _tools.vec_to_stdmx(A, mxBasis)
    rhoMx2 = _tools.vec_to_stdmx(B, mxBasis)
    return _tools.fidelity(rhoMx1, rhoMx2)


Vec_fidelity = _modf.vecsfn_factory(vec_fidelity)
# init args == (model1, model2, label, typ)


def vec_infidelity(A, B, mxBasis):
    """ State infidelity fidelity between SPAM vectors A and B """
    rhoMx1 = _tools.vec_to_stdmx(A, mxBasis)
    rhoMx2 = _tools.vec_to_stdmx(B, mxBasis)
    return 1 - _tools.fidelity(rhoMx1, rhoMx2)


Vec_infidelity = _modf.vecsfn_factory(vec_infidelity)
# init args == (model1, model2, label, typ)


def vec_tr_diff(A, B, mxBasis):  # assume vary model1, model2 fixed
    """ Trace distance between SPAM vectors A and B """
    rhoMx1 = _tools.vec_to_stdmx(A, mxBasis)
    rhoMx2 = _tools.vec_to_stdmx(B, mxBasis)
    return _tools.tracedist(rhoMx1, rhoMx2)


Vec_tr_diff = _modf.vecsfn_factory(vec_tr_diff)
# init args == (model1, model2, label, typ)


def vec_as_stdmx(vec, mxBasis):
    """ SPAM vectors as a standard density matrix """
    return _tools.vec_to_stdmx(vec, mxBasis)


Vec_as_stdmx = _modf.vecfn_factory(vec_as_stdmx)
# init args == (model, label, typ)


def vec_as_stdmx_eigenvalues(vec, mxBasis):
    """ Eigenvalues of the density matrix corresponding to a SPAM vector """
    mx = _tools.vec_to_stdmx(vec, mxBasis)
    return _np.linalg.eigvals(mx)


Vec_as_stdmx_eigenvalues = _modf.vecfn_factory(vec_as_stdmx_eigenvalues)
# init args == (model, label, typ)


def info_of_opfn_by_name(name):
    """
    Returns a nice human-readable name and tooltip for a given gate-function
    abbreviation.

    Parameters
    ----------
    name : str
        An appreviation for a gate-function name.  Allowed values are:

        - "inf" :     entanglement infidelity
        - "agi" :     average gate infidelity
        - "trace" :   1/2 trace distance
        - "diamond" : 1/2 diamond norm distance
        - "nuinf" :   non-unitary entanglement infidelity
        - "nuagi" :   non-unitary entanglement infidelity
        - "evinf" :     eigenvalue entanglement infidelity
        - "evagi" :     eigenvalue average gate infidelity
        - "evnuinf" :   eigenvalue non-unitary entanglement infidelity
        - "evnuagi" :   eigenvalue non-unitary entanglement infidelity
        - "evdiamond" : eigenvalue 1/2 diamond norm distance
        - "evnudiamond" : eigenvalue non-unitary 1/2 diamond norm distance
        - "frob" :    frobenius distance
        - "unmodeled" : unmodeled "wildcard" budget

    Returns
    -------
    nicename : str
    tooltip : str
    """
    info = {
        "inf": ("Entanglement|Infidelity",
                "1.0 - <psi| 1 x Lambda(psi) |psi>"),
        "agi": ("Avg. Gate|Infidelity",
                "d/(d+1) (entanglement infidelity)"),
        "trace": ("1/2 Trace|Distance",
                  "0.5 | Chi(A) - Chi(B) |_tr"),
        "diamond": ("1/2 Diamond-Dist",
                    "0.5 sup | (1 x (A-B))(rho) |_tr"),
        "nuinf": ("Non-unitary|Ent. Infidelity",
                  "(d^2-1)/d^2 [1 - sqrt( unitarity(A B^-1) )]"),
        "nuagi": ("Non-unitary|Avg. Gate Infidelity",
                  "(d-1)/d [1 - sqrt( unitarity(A B^-1) )]"),
        "evinf": ("Eigenvalue|Ent. Infidelity",
                  "min_P 1 - |lambda_a P lambda_b^dag|/d^2  "
                  "[P = permutation, (lambda_a,lambda_b) = eigenvalues of A and B]"),
        "evagi": ("Eigenvalue|Avg. Gate Infidelity",
                  "min_P (d^2 - |lambda_a P lambda_b^dag|)/d(d+1)  "
                  "[P = permutation, (lambda_a,lambda_b) = eigenvalues of A and B]"),
        "evnuinf": ("Eigenvalue Non-U.|Ent. Infidelity",
                    "(d^2-1)/d^2 [1 - sqrt( eigenvalue_unitarity(A B^-1) )]"),
        "evnuagi": ("Eigenvalue Non-U.|Avg. Gate Infidelity",
                    "(d-1)/d [1 - sqrt( eigenvalue_unitarity(A B^-1) )]"),
        "evdiamond": ("Eigenvalue|1/2 Diamond-Dist",
                      "(d^2-1)/d^2 max_i { |a_i - b_i| } "
                      "where (a_i,b_i) are corresponding eigenvalues of A and B."),
        "evnudiamond": ("Eigenvalue Non-U.|1/2 Diamond-Dist",
                        "(d^2-1)/d^2 max_i { | |a_i| - |b_i| | } "
                        "where (a_i,b_i) are corresponding eigenvalues of A and B."),
        "frob": ("Frobenius|Distance",
                 "sqrt( sum( (A_ij - B_ij)^2 ) )"),
        "unmodeled": ("Un-modeled|Error",
                      "The per-operation budget used to account for un-modeled errors (model violation)")
    }
    if name in info:
        return info[name]
    else:
        raise ValueError("Invalid name: %s" % name)


def evaluate_opfn_by_name(name, model, targetModel, opLabelOrString,
                          confidenceRegionInfo):
    """
    Evaluates that gate-function named by the abbreviation `name`.

    Parameters
    ----------
    name : str
        An appreviation for a gate-function name.  Allowed values are the
        same as those of :func:`info_of_opfn_by_name`.

    model, targetModel : Model
        The models to compare.  Only the element or product given by
        `opLabelOrString` is compared using the named gate-function.

    opLabelOrString : str or Circuit or tuple
        The operation label or sequence of labels to compare.  If a sequence
        of labels is given, then the "virtual gate" computed by taking the
        product of the specified gate matrices is compared.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region  used to compute error
        intervals.

    Returns
    -------
    ReportableQty
    """
    gl = opLabelOrString
    b = bool(isinstance(gl, _Lbl) or isinstance(gl, str))  # whether this is a operation label or a string

    if name == "inf":
        fn = Entanglement_infidelity if b else \
            Circuit_entanglement_infidelity
    elif name == "agi":
        fn = Avg_gate_infidelity if b else \
            Circuit_avg_gate_infidelity
    elif name == "trace":
        fn = Jt_diff if b else \
            Circuit_jt_diff
    elif name == "diamond":
        fn = Half_diamond_norm if b else \
            Circuit_half_diamond_norm
    elif name == "nuinf":
        fn = Nonunitary_entanglement_infidelity if b else \
            Circuit_nonunitary_entanglement_infidelity
    elif name == "nuagi":
        fn = Nonunitary_avg_gate_infidelity if b else \
            Circuit_nonunitary_avg_gate_infidelity
    elif name == "evinf":
        fn = Eigenvalue_entanglement_infidelity if b else \
            Circuit_eigenvalue_entanglement_infidelity
    elif name == "evagi":
        fn = Eigenvalue_avg_gate_infidelity if b else \
            Circuit_eigenvalue_avg_gate_infidelity
    elif name == "evnuinf":
        fn = Eigenvalue_nonunitary_entanglement_infidelity if b else \
            Circuit_eigenvalue_nonunitary_entanglement_infidelity
    elif name == "evnuagi":
        fn = Eigenvalue_nonunitary_avg_gate_infidelity if b else \
            Circuit_eigenvalue_nonunitary_avg_gate_infidelity
    elif name == "evdiamond":
        fn = Eigenvalue_diamondnorm if b else \
            Circuit_eigenvalue_diamondnorm
    elif name == "evnudiamond":
        fn = Eigenvalue_nonunitary_diamondnorm if b else \
            Circuit_eigenvalue_nonunitary_diamondnorm
    elif name == "frob":
        fn = Fro_diff if b else \
            Circuit_fro_diff

    return evaluate(fn(model, targetModel, gl), confidenceRegionInfo)
