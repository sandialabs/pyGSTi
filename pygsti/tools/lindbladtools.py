"""
Utility functions relevant to Lindblad forms and projections
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
from collections.abc import Iterable
import numpy as _np
import scipy.sparse as _sps
from typing import Dict, Tuple, Union

from pygsti.tools import matrixtools as _mt
from pygsti.tools.basistools import basis_matrices
from pygsti.baseobjs.errorgenlabel import (
    ElementaryErrorgenLabel as _EEGLabel,
    LocalElementaryErrorgenLabel as _LocalEEGLabel,
    GlobalElementaryErrorgenLabel as _GlobalEEGLabel
)


def create_elementary_errorgen_dual(typ, p, q=None, sparse=False, normalization_factor='auto'):
    """
    Construct a "dual" elementary error generator matrix in the "standard" (matrix-unit) basis.

    The elementary error generator that is dual to the one computed by calling
    :func:`create_elementary_errorgen` with the same argument.  This dual element
    can be used to find the coefficient of the original, or "primal" elementary generator.
    For example, if `A = sum(c_i * E_i)`, where `E_i` are the elementary error generators given
    by :func:`create_elementary_errorgen`), then `c_i = dot(D_i.conj(), A)` where `D_i`
    is the dual to `E_i`.

    There are four different types of dual elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j/(2d^2) * [ p, rho ]`
    Stochastic:   `L(rho) = 1/(d^2) p * rho * p`
    Correlation:  `L(rho) = 1/(2d^2) ( p * rho * q + q * rho * p)`
    Active:       `L(rho) = 1j/(2d^2) ( p * rho * q - q * rho * p)`

    where `d` is the dimension of the Hilbert space, e.g. 2 for a single qubit.  Square
    brackets denotes the commutator and curly brackets the anticommutator.
    `L` is returned as a superoperator matrix that acts on vectorized density matrices.

    Parameters
    ----------
    typ : {'H','S','C','A'}
        The type of dual error generator to construct.

    p : numpy.ndarray
        d-dimensional basis matrix.

    q : numpy.ndarray, optional
        d-dimensional basis matrix; must be non-None if and only if `typ` is `'C'` or `'A'`.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = p.shape[0]; d2 = d**2
    pdag = p.T.conjugate()
    qdag = q.T.conjugate() if (q is not None) else None

    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    # Loop through the standard basis as all possible input density matrices
    for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
        # Only difference between H/S/C/A is how they transform input density matrices
        if typ == 'H':
            rho1 = -1j * (p @ rho0 - rho0 @ p)  # -1j / (2 * d2) *
        elif typ == 'S':
            rho1 = (p @ rho0 @ pdag)  # 1 / d2 *
        elif typ == 'C':
            rho1 = (p @ rho0 @ qdag + q @ rho0 @ pdag)  # 1 / (2 * d2) *
        elif typ == 'A':
            rho1 = 1j * (p @ rho0 @ qdag - q @ rho0 @ pdag)  # 1j / (2 * d2)
        elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    return_normalization = bool(normalization_factor == 'auto_return')
    if normalization_factor in ('auto', 'auto_return'):
        primal = create_elementary_errorgen(typ, p, q, sparse)
        if sparse:
            normalization_factor = _np.vdot(elem_errgen.toarray().flatten(), primal.toarray().flatten())
        else:
            normalization_factor = _np.vdot(elem_errgen.flatten(), primal.flatten())
    elem_errgen *= _np.real_if_close(1 / normalization_factor).item()  # item() -> scalar

    if sparse: elem_errgen = elem_errgen.tocsr()
    return (elem_errgen, normalization_factor) if return_normalization else elem_errgen


def create_elementary_errorgen(typ, p, q=None, sparse=False):
    """
    Construct an elementary error generator as a matrix in the "standard" (matrix-unit) basis.

    There are four different types of elementary error generators: 'H' (Hamiltonian),
    'S' (stochastic), 'C' (correlation), and 'A' (active).  See arxiv:2103.01928.
    Each type transforms an input density matrix differently.  The action of an elementary
    error generator `L` on an input density matrix `rho` is given by:

    Hamiltonian:  `L(rho) = -1j * [ p, rho ]`
    Stochastic:   `L(rho) = p * rho * p - rho`
    Correlation:  `L(rho) = p * rho * q + q * rho * p - 0.5 {{p,q}, rho}`
    Active:       `L(rho) = 1j( p * rho * q - q * rho * p + 0.5 {[p,q], rho} )`

    Square brackets denotes the commutator and curly brackets the anticommutator.
    `L` is returned as a superoperator matrix that acts on vectorized density matrices.

    Parameters
    ----------
    typ : {'H','S','C','A'}
        The type of error generator to construct.

    p : numpy.ndarray
        d-dimensional basis matrix.

    q : numpy.ndarray, optional
        d-dimensional basis matrix; must be non-None if and only if `typ` is `'C'` or `'A'`.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = p.shape[0]; d2 = d**2
    if sparse:
        elem_errgen = _sps.lil_matrix((d2, d2), dtype=p.dtype)
    else:
        elem_errgen = _np.empty((d2, d2), dtype=p.dtype)

    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"
    assert((typ in 'HS' and q is None) or (typ in 'CA' and q is not None)), \
        "Wrong number of basis elements provided for %s-type elementary errorgen!" % typ

    pdag = p.T.conjugate()
    qdag = q.T.conjugate() if (q is not None) else None

    if typ in 'CA':
        pq_plus_qp = pdag @ q + qdag @ p
        pq_minus_qp = pdag @ q - qdag @ p

    # Loop through the standard basis as all possible input density matrices
    for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
        # Only difference between H/S/C/A is how they transform input density matrices
        if typ == 'H':
            rho1 = -1j * (p @ rho0 - rho0 @ p)  # Add "/2" to have PP ham gens match previous versions of pyGSTi
        elif typ == 'S':
            pdag_p = (pdag @ p)
            rho1 = p @ rho0 @ pdag - 0.5 * (pdag_p @ rho0 + rho0 @ pdag_p)
        elif typ == 'C':
            rho1 = p @ rho0 @ qdag + q @ rho0 @ pdag - 0.5 * (pq_plus_qp @ rho0 + rho0 @ pq_plus_qp)
        elif typ == 'A':
            rho1 = 1j * (p @ rho0 @ qdag - q @ rho0 @ pdag + 0.5 * (pq_minus_qp @ rho0 + rho0 @ pq_minus_qp))

        elem_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse: elem_errgen = elem_errgen.tocsr()
    return elem_errgen

def create_lindbladian_term_errorgen(typ, Lm, Ln=None, sparse=False):  # noqa N803
    """
    Construct the superoperator for a term in the common Lindbladian expansion of an error generator.

    Mathematically, for d-dimensional matrices Lm and Ln, this routine
    constructs the d^2-dimension Lindbladian matrix L whose action is
    given by:

    `L(rho) = -i [Lm, rho] `    (when `typ == 'H'`)

    or

    `L(rho) = Ln*rho*Lm^dag - 1/2(rho*Lm^dag*Ln + Lm^dag*Ln*rho)`    (`typ == 'O'`)

    where rho is a density matrix.  L is returned as a superoperator
    matrix that acts on a vectorized density matrices.

    Parameters
    ----------
    typ : {'H', 'O'}
        The type of error generator to construct.

    Lm : numpy.ndarray
        d-dimensional basis matrix.

    Ln : numpy.ndarray, optional
        d-dimensional basis matrix.

    sparse : bool, optional
        Whether to construct a sparse or dense (the default) matrix.

    Returns
    -------
    ndarray or Scipy CSR matrix
    """
    d = Lm.shape[0]; d2 = d**2
    if sparse:
        lind_errgen = _sps.lil_matrix((d2, d2), dtype=Lm.dtype)
    else:
        lind_errgen = _np.empty((d2, d2), dtype=Lm.dtype)

    assert(typ in ('H', 'O')), "`typ` must be one of 'H' or 'O'"
    assert((typ in 'H' and Ln is None) or (typ in 'O' and Ln is not None)), \
        "Wrong number of basis elements provided for %s-type lindblad term errorgen!" % typ

    if typ in 'O':
        Lm_dag = _np.conjugate(_np.transpose(Lm))
        Lmdag_Ln = Lm_dag @ Ln

    # Loop through the standard basis as all possible input density matrices
    for i, rho0 in enumerate(basis_matrices('std', d2)):  # rho0 == input density mx
        # Only difference between H/S/C/A is how they transform input density matrices
        if typ == 'H':
            rho1 = -1j * (Lm @ rho0 - rho0 @ Lm)
        elif typ == 'O':
            rho1 = Ln @ rho0 @ Lm_dag - 0.5 * (Lmdag_Ln @ rho0 + rho0 @ Lmdag_Ln)
        else: raise ValueError("Invalid lindblad term errogen type!")
        lind_errgen[:, i] = rho1.flatten()[:, None] if sparse else rho1.flatten()

    if sparse: lind_errgen = lind_errgen.tocsr()
    return lind_errgen

#############################################
### Error generator commutation/BCH tools ###
#############################################

## Some Pauli string tools first (multiplication, commutation, anticommutation)
def paulistr_multiply(pstr1: Iterable[str], pstr2: Iterable[str]) -> Tuple[complex, str]:
    '''
    Compute "symbolically" the product sigma_pstr1 * sigma_pstr2
    
    Parameters
    ----------
    pstr1: 
        Pauli string consisting of "IXYZ"
    
    pstr2: 
        Pauli string consisting of "IXYZ"
    
    Returns
    ---------
    output : tuple
        Tuple corresponding to the product of sigma_pstr1 * sigma_pstr2
        output[0] : complex
            Coefficient of the product. Possible values are 1, -1, 1j, -1j.
        output[1] : list
            Pauli string of the product.
    '''
    assert len(pstr1) == len(pstr2), "Pauli strings must have same length"
    
    idx_str_map = {k:i for i,k in enumerate("IXYZ")}
    try:
        ilist = [idx_str_map[p] for p in pstr1]
        jlist = [idx_str_map[p] for p in pstr2]
    except KeyError as e:
        raise ValueError("Pauli strings must have IXYZ labels") from e
    
    output_idxs = []
    sgn_ctr = 0
    for k in range(len(ilist)):
        if ilist[k]==jlist[k]:#If local Paulis match, get identity
            output_idxs.append(0)
        elif ilist[k] == 0:#If one local Pauli is identity, get the other local Pauli
            output_idxs.append(jlist[k])
        elif jlist[k] == 0:
            output_idxs.append(ilist[k])
        else:#If neither local Pauli is identity nor do they match, get the third non-trivial Pauli.
            output_idxs.append({1,2,3}.difference({ilist[k],jlist[k]}).pop())
            if (jlist[k] - ilist[k]) % 3 == 1:#Are the local Paulis cyclic ((1,2), (2,3), or (3,1))?
                sgn_ctr += 1
            else:
                #They must be anti-cyclic ((2,1), (3,2), or (1,3))
                sgn_ctr -= 1
            
    str_idx_map = {v:k for k,v in idx_str_map.items()}
    output = (1j**sgn_ctr,''.join([str_idx_map[idx] for idx in output_idxs]))
    
    return output


def paulistr_commutator(pstr1: Iterable[str], pstr2: Iterable[str]) -> Tuple[Union[int, complex], str]:
    '''
    Compute "symbolically" the commutator [sigma_pstr1, sigma_pstr2]
    
    Parameters
    ----------
    pstr1:
        Pauli string consisting of "IXYZ"
    
    pstr2: 
        Pauli string consisting of "IXYZ"
    
    Returns
    ---------
    output : tuple
        Tuple corresponding to the commutator of [sigma_pstr1, sigma_pstr2]
        output[0] : int/complex
            Coefficient of the commutator.  
            If commutator is 0, then output[0]==0.
            Possible values are 0, 2j, -2j.  
            (Can't have output[0] = 2 or -2 because an even number of differences means inputs commute.)
        output[1] : list
            Pauli string of the commutator.
    '''
    output = paulistr_multiply(pstr1, pstr2)
    if output[0].imag == 0:# Even number of differences, commutator is 0
        return (0,output[1])
    
    # Double weight from product for the two terms in the commutator
    return (2*output[0], output[1])

def paulistr_anticommutator(pstr1: Iterable[str], pstr2: Iterable[str]) -> Tuple[int, str]:
    '''
    Compute "symbolically" the anticommutator {sigma_pstr1, sigma_pstr2}
    
    Parameters
    ----------
    pstr1: 
        Pauli string consisting of "IXYZ"
    
    pstr2: 
        Pauli string consisting of "IXYZ"
    
    Returns
    ---------
    output : tuple
        Tuple corresponding to the anticommutator of {sigma_pstr1, sigma_pstr2}
        output[0] : int
            Coefficient of the anticommutator.  
            If anticommutator is 0, then output[0]==0.
            Possible values are 0 and 2.
        output[1] : list
            Pauli string of the anticommutator.
    '''
    output = paulistr_multiply(pstr1, pstr2)
    if output[0].real == 0:# Odd number of differences, anticommutator is 0
        return (0,output[1])
    
    # Double weight from product for the two terms in the anticommutator
    return (2, output[1])

def _errorgen_commutator_symbolic_term(
    lbl1: _LocalEEGLabel, lbl2: _LocalEEGLabel
) -> Dict[_LocalEEGLabel, complex]:
    """Compute the symbolic commutator of two elementary error generators.
    
    **Note:** This is using the code currently in feature-Propagatable-ErrorGens
    rather than the original notes. I'm assuming that the code is more debugged
    than the notes, but I'm marking differences in red.
    
    $$\large
    \begin{align*}
    [H_P, H_Q] &= -iH_{[P,Q]}\\
    [H_P, S_Q] &= iC_{Q,[Q,P]}\\
    [H_P, C_{A,B}] &= iC_{[A,P],B} + iC_{\textcolor{red}{[B,P]},A}\\
    [H_P, A_{A,B}] &= -iA_{\textcolor{red}{[P,A]},B} - iA_{A,[P,B]}\\
    [S_P, S_Q] &= 0\\
    [S_P, C_{A,B}] &= -iA_{\textcolor{red}{PA},BP} -iA_{PB,\textcolor{red}{AP}} -\frac{i}{2}\big(A_{\{A,B\}P,P} + A_{P,P\{A,B\}}\big)\\
    [S_P, A_{A,B}] &= iC_{PA,BP} -iC_{PB,AP} -\frac{textcolor{red}{1}}{2}A_{P,[P,[A,B]]}\\
    [C_{A,B}, C_{P,Q}] &= -i\big(A_{AP,QB} + A_{AQ,\textcolor{red}{PB}} \textcolor{red}{+} A_{BP,QA} \textcolor{red}{+} A_{BQ,PA}\big)\\
    &-\frac{i}{2}\big(A_{[P,\{A,B\}],Q} + A_{[Q,\{A,B\}],P} + A_{[\{P,Q\},A],B} + A_{[\{P,Q\},B],A}\big)\\
    &+\frac{i}{4}H_{[\{A,B\},\{P,Q\}]}\\
    [C_{A,B}, A_{P,Q}] &= i\big(C_{AP,QB} - C_{AQ,PB} + C_{BP,QA} - C_{PA,BQ}\big)\\
    &+\frac{1}{2}\big(A_{[A,[P,Q]],B} + A_{[B,[P,Q]],A} + iC_{[P,\{A,B\}],Q} - \textcolor{red}{i}C_{[Q,\{A,B\}],P}\big)\\
    &-\frac{1}{4}H_{[[P,Q],\{A,B\}]}\\
    [A_{A,B}, A_{P,Q}] &= -i\big(A_{QB,AP} + A_{PA,BQ} + A_{BP,QA} + A_{AQ,PB}\big)\\
    &+\frac{1}{2}\big(C_{[B,[P,Q]],A} - C_{[A,[P,Q]],B} + C_{[P,[A,B]],Q} - C_{[Q,[A,B]],P}\big)\\
    &+\frac{\textcolor{red}{1}}{4}H_{[[P,Q],[A,B]]}
    \end{align*}
    $$

    Parameters
    ----------
    lbl1:
        The first error generator label
    
    lbl2:
        The second error generator label
    
    Returns
    -------
    terms
        Dicts with the resulting error generator labels as keys
        and corresponding weights as values
    """
    lbl1 = _LocalEEGLabel.cast(lbl1)
    lbl2 = _LocalEEGLabel.cast(lbl2)
    
    # Sanity checks. If this is not true, the else below will be an infinite recursion
    assert lbl1.errorgen_type in "HSCA"
    assert lbl2.errorgen_type in "HSCA"
    
    # So that we can += and have a default value of 0
    # Should take care of most of our collision logic
    terms = defaultdict(complex)
    
    if lbl1.errorgen_type == "H" and lbl2.errorgen_type == "H":
        # [H_P, H_Q] = -i H_[P,Q]
        P = lbl1.basis_element_labels[0]
        Q = lbl2.basis_element_labels[0]
        
        wt_cPQ, cPQ = paulistr_commutator(P, Q)
        if wt_cPQ:
            terms[_LocalEEGLabel("H", [cPQ])] += -1j*wt_cPQ
    elif lbl1.errorgen_type == "H" and lbl2.errorgen_type == "S":
        # [H_P, S_Q] = iC_{Q,[Q,P]}
        P = lbl1.basis_element_labels[0]
        Q = lbl2.basis_element_labels[0]
        
        wt_cQP, cQP = paulistr_commutator(Q, P)
        if wt_cQP:
            terms[_LocalEEGLabel("C", [Q, cQP])] += 1j*wt_cQP
    elif lbl1.errorgen_type == "H" and lbl2.errorgen_type == "C":
        # [H_P, C_{A,B}] = iC_{[A,P],B} + iC_{\textcolor{red}{[B,P]},A}
        P = lbl1.basis_element_labels[0]
        A,B = lbl2.basis_element_labels
        
        wt_cAP, cAP = paulistr_commutator(A, P)
        if wt_cAP:
            terms[_LocalEEGLabel("C", [cAP, B])] += 1j*wt_cAP
        
        wt_cBP, cBP = paulistr_commutator(B, P)
        if wt_cBP:
            terms[_LocalEEGLabel("C", [cBP, A])] += 1j*wt_cBP
    elif lbl1.errorgen_type == "H" and lbl2.errorgen_type == "A":
        # [H_P, A_{A,B}] = -iA_{\textcolor{red}{[P,A]},B} - iA_{A,[P,B]}
        P = lbl1.basis_element_labels[0]
        A,B = lbl2.basis_element_labels
        
        wt_cPA, cPA = paulistr_commutator(P, A)
        if wt_cPA:
            terms[_LocalEEGLabel("A", [cPA, B])] += -1j*wt_cPA
        
        wt_cPB, cPB = paulistr_commutator(P, B)
        if wt_cPB:
            terms[_LocalEEGLabel("C", [A, cPB])] += -1j*wt_cPB
    elif lbl1.errorgen_type == "S" and lbl2.errorgen_type == "S":
        pass
    elif lbl1.errorgen_type == "S" and lbl2.errorgen_type == "C":
        # [S_P, C_{A,B}] = -iA_{\textcolor{red}{PA},BP} -iA_{PB,\textcolor{red}{AP}}
        # -\frac{i}{2}\big(A_{\{A,B\}P,P} + A_{P,P\{A,B\}}\big)
        P = lbl1.basis_element_labels[0]
        A,B = lbl2.basis_element_labels
        
        wt_PA, PA = paulistr_multiply(P, A)
        wt_PB, PB = paulistr_multiply(P, B)
        wt_AP, AP = paulistr_multiply(A, P)
        wt_BP, BP = paulistr_multiply(B, P)
        
        # We have A_{PA,BP} + A_{PB,AP} = A_{PA,BP} - A_{AP,PB}
        # These cancel if PA == AP and PB == BP
        if PA != AP or PB != BP:
            terms[_LocalEEGLabel("A", [PA, BP])] += -1j*wt_PA*wt_BP
            terms[_LocalEEGLabel("A", [PB, AP])] += -1j*wt_PA*wt_BP
        
        wt_aAB, aAB = paulistr_anticommutator(A, B)
        if wt_aAB:
            wt_aABP, aABP = paulistr_multiply(aAB, P)
            if wt_aABP:
                terms[_LocalEEGLabel("A", [aABP, P])] += -0.5*1j*wt_aAB*wt_aABP
            
            wt_PaAB, PaAB = paulistr_multiply(P, aAB)
            if wt_PaAB:
                terms[_LocalEEGLabel("A", [P, PaAB])] += -0.5*1j*wt_aAB*wt_PaAB
    elif lbl1.errorgen_type == "S" and lbl2.errorgen_type == "A":
        # [S_P, A_{A,B}] = iC_{PA,BP} -iC_{PB,AP} -\frac{textcolor{red}{1}}{2}A_{P,[P,[A,B]]}
        P = lbl1.basis_element_labels[0]
        A,B = lbl2.basis_element_labels
            
        wt_PA, PA = paulistr_multiply(P, A)
        wt_PB, PB = paulistr_multiply(P, B)
        wt_AP, AP = paulistr_multiply(A, P)
        wt_BP, BP = paulistr_multiply(B, P)
        
        # We have C_{PA,BP} - C_{PB,AP} = C_{PA,BP} - C_{AP,PB}
        # These cancel if PA == AP and PB == BP
        if PA != AP or PB != BP:
            terms[_LocalEEGLabel("C", [PA, BP])] += -1j*wt_PA*wt_BP
            terms[_LocalEEGLabel("C", [PB, AP])] += -1j*wt_PA*wt_BP
        
        wt_cAB, cAB = paulistr_commutator(A, B)
        wt_cPcAB, cPcAB = paulistr_commutator(P, cAB)
        if wt_cAB and wt_cPcAB:
            terms[_LocalEEGLabel("A", [P, cPcAB])] += -0.5*wt_cAB*wt_cPcAB
    elif lbl1.errorgen_type == "C" and lbl2.errorgen_type == "C":
        # [C_{A,B}, C_{P,Q}] = -i\big(A_{AP,QB} + A_{AQ,\textcolor{red}{PB}}
        # \textcolor{red}{+} A_{BP,QA} \textcolor{red}{+} A_{BQ,PA}\big)\\
        # -\frac{i}{2}\big(A_{[P,\{A,B\}],Q} + A_{[Q,\{A,B\}],P} + A_{[\{P,Q\},A],B} + A_{[\{P,Q\},B],A}\big)\\
        # +\frac{i}{4}H_{[\{A,B\},\{P,Q\}]}\\
        A,B = lbl1.basis_element_labels
        P,Q = lbl2.basis_element_labels

        wt_PA, PA = paulistr_multiply(P, A)
        wt_PB, PB = paulistr_multiply(P, B)
        wt_AP, AP = paulistr_multiply(A, P)
        wt_BP, BP = paulistr_multiply(B, P)
        
        wt_QA, QA = paulistr_multiply(Q, A)
        wt_QB, QB = paulistr_multiply(Q, B)
        wt_AQ, AQ = paulistr_multiply(A, Q)
        wt_BQ, BQ = paulistr_multiply(B, Q)

        # We have A_{AP,QB} + A_{BQ,PA} = A_{AP,QB} - A_{PA,BQ}
        # These cancel if PA == AP and QB == BQ
        if PA != AP or QB != BQ:
            terms[_LocalEEGLabel("A", [AP, QB])] += -1j*wt_AP*wt_QB
            terms[_LocalEEGLabel("A", [BQ, PA])] += -1j*wt_BQ*wt_PA
        
        # We also have A_{AQ,PB} + A_{BP,QA} = A_{AQ,PB} - A_{QA,BP}
        # These cancel if PB == BP and QA == AQ
        if PB != BP or QA != AQ:
            terms[_LocalEEGLabel("A", [AQ, PB])] += -1j*wt_AQ*wt_PB
            terms[_LocalEEGLabel("A", [BP, QA])] += -1j*wt_BP*wt_QA
        
        wt_aAB, aAB = paulistr_anticommutator(A, B)
        if wt_aAB:
            wt_cPaAB, cPaAB = paulistr_commutator(P, aAB)
            if wt_cPaAB:
                terms[_LocalEEGLabel("A", [cPaAB, Q])] += -0.5*1j*wt_aAB*wt_cPaAB

            wt_cQaAB, cQaAB = paulistr_commutator(Q, aAB)
            if wt_cQaAB:
                terms[_LocalEEGLabel("A", [cQaAB, P])] += -0.5*1j*wt_aAB*wt_cQaAB
        
        wt_aPQ, aPQ = paulistr_anticommutator(P, Q)
        if wt_aPQ:
            wt_caPQA, caPQA = paulistr_commutator(aPQ, A)
            if wt_caPQA:
                terms[_LocalEEGLabel("A", [caPQA, B])] += -0.5*1j*wt_aPQ*wt_caPQA
            
            wt_caPQB, caPQB = paulistr_commutator(aPQ, B)
            if wt_caPQB:
                terms[_LocalEEGLabel("A", [caPQB, A])] += -0.5*1j*wt_aPQ*wt_caPQB
        
        if wt_aAB and wt_aPQ:
            wt_caABaPQ, caABaPQ = paulistr_commutator(aAB, aPQ)
            if wt_caABaPQ:
                terms[_LocalEEGLabel("H", [caABaPQ])] += 0.25*1j*wt_aAB*wt_aPQ*wt_caABaPQ
    elif lbl1.errorgen_type == "C" and lbl2.errorgen_type == "A":
        # [C_{A,B}, A_{P,Q}] = i\big(C_{AP,QB} - C_{AQ,PB} + C_{BP,QA} - C_{PA,BQ}\big)\\
        # +\frac{1}{2}\big(A_{[A,[P,Q]],B} + A_{[B,[P,Q]],A} + iC_{[P,\{A,B\}],Q} - \textcolor{red}{i}C_{[Q,\{A,B\}],P}\big)\\
        # -\frac{1}{4}H_{[[P,Q],\{A,B\}]}\\
        A,B = lbl1.basis_element_labels
        P,Q = lbl2.basis_element_labels

        wt_PA, PA = paulistr_multiply(P, A)
        wt_PB, PB = paulistr_multiply(P, B)
        wt_AP, AP = paulistr_multiply(A, P)
        wt_BP, BP = paulistr_multiply(B, P)
        
        wt_QA, QA = paulistr_multiply(Q, A)
        wt_QB, QB = paulistr_multiply(Q, B)
        wt_AQ, AQ = paulistr_multiply(A, Q)
        wt_BQ, BQ = paulistr_multiply(B, Q)

        # We have C_{AP,QB} - C_{PA,BQ}
        # These cancel if PA == AP and QB == BQ
        if PA != AP or QB != BQ:
            terms[_LocalEEGLabel("C", [AP, QB])] += 1j*wt_AP*wt_QB
            terms[_LocalEEGLabel("C", [BQ, PA])] += -1j*wt_BQ*wt_PA
        
        # We also have -C_{AQ,PB} + C_{BP,QA}
        # These cancel if PB == BP and QA == AQ
        if PB != BP or QA != AQ:
            terms[_LocalEEGLabel("C", [AQ, PB])] += -1j*wt_AQ*wt_PB
            terms[_LocalEEGLabel("C", [BP, QA])] += 1j*wt_BP*wt_QA
        
        wt_cPQ, cPQ = paulistr_commutator(P, Q)
        if wt_cPQ:
            wt_cAcPQ, cAcPQ = paulistr_commutator(A, cPQ)
            if wt_cAcPQ:
                terms[_LocalEEGLabel("A", [cAcPQ, B])] += 0.5*wt_cPQ*wt_cAcPQ
            
            wt_cBcPQ, cBcPQ = paulistr_commutator(B, cPQ)
            if wt_cBcPQ:
                terms[_LocalEEGLabel("A", [cBcPQ, A])] += 0.5*wt_cPQ*wt_cBcPQ
        
        wt_aAB, aAB = paulistr_anticommutator(A, B)
        if wt_aAB:
            wt_cPaAB, cPaAB = paulistr_commutator(P, aAB)
            if wt_cPaAB:
                terms[_LocalEEGLabel("C", [cPaAB, Q])] += 0.5*1j*wt_aAB*wt_cPaAB

            wt_cQaAB, cQaAB = paulistr_commutator(Q, aAB)
            if wt_cQaAB:
                terms[_LocalEEGLabel("C", [cQaAB, P])] += -0.5*1j*wt_aAB*wt_cQaAB
        
        if wt_aAB and wt_cPQ:
            wt_ccPQaAB, ccPQaAB = paulistr_commutator(cPQ, aAB)
            if wt_ccPQaAB:
                terms[_LocalEEGLabel("H", [ccPQaAB])] += 0.25*wt_aAB*wt_cPQ*wt_ccPQaAB
    elif lbl1.errorgen_type == "A" and lbl2.errorgen_type == "A":
        # [A_{A,B}, A_{P,Q}] = -i\big(A_{QB,AP} + A_{PA,BQ} + A_{BP,QA} + A_{AQ,PB}\big)\\
        # +\frac{1}{2}\big(C_{[B,[P,Q]],A} - C_{[A,[P,Q]],B} + C_{[P,[A,B]],Q} - C_{[Q,[A,B]],P}\big)\\
        # +\frac{\textcolor{red}{1}}{4}H_{[[P,Q],[A,B]]}
        A,B = lbl1.basis_element_labels
        P,Q = lbl2.basis_element_labels

        wt_PA, PA = paulistr_multiply(P, A)
        wt_PB, PB = paulistr_multiply(P, B)
        wt_AP, AP = paulistr_multiply(A, P)
        wt_BP, BP = paulistr_multiply(B, P)
        
        wt_QA, QA = paulistr_multiply(Q, A)
        wt_QB, QB = paulistr_multiply(Q, B)
        wt_AQ, AQ = paulistr_multiply(A, Q)
        wt_BQ, BQ = paulistr_multiply(B, Q)

        # We have A_{QB,AP} + A_{PA,BQ} = A_{QB,AP} - A_{BQ,PA}
        # These cancel if PA == AP and QB == BQ
        if PA != AP or QB != BQ:
            terms[_LocalEEGLabel("A", [QB, AP])] += -1j*wt_QB*wt_AP
            terms[_LocalEEGLabel("A", [BQ, PA])] += -1j*wt_BQ*wt_PA
        
        # We also have A_{BP,QA} + A_{AQ,PB} = A_{BP,QA} - A_{PB,AQ}
        # These cancel if PB == BP and QA == AQ
        if PB != BP or QA != AQ:
            terms[_LocalEEGLabel("A", [BP, QA])] += -1j*wt_BP*wt_QA
            terms[_LocalEEGLabel("A", [AQ, PB])] += -1j*wt_AQ*wt_PB
        
        wt_cPQ, cPQ = paulistr_commutator(P, Q)
        if wt_cPQ:
            wt_cBcPQ, cBcPQ = paulistr_commutator(B, cPQ)
            if wt_cBcPQ:
                terms[_LocalEEGLabel("C", [cBcPQ, A])] += 0.5*wt_cPQ*wt_cBcPQ
            
            wt_cAcPQ, cAcPQ = paulistr_commutator(A, cPQ)
            if wt_cAcPQ:
                terms[_LocalEEGLabel("C", [cAcPQ, B])] += -0.5*wt_cPQ*wt_cAcPQ
        
        wt_cAB, cAB = paulistr_commutator(A, B)
        if wt_cAB:
            wt_cPcAB, cPcAB = paulistr_commutator(P, cAB)
            if wt_cPaAB:
                terms[_LocalEEGLabel("C", [cPcAB, Q])] += 0.5*wt_cAB*wt_cPcAB

            wt_cQcAB, cQcAB = paulistr_commutator(Q, cAB)
            if wt_cQcAB:
                terms[_LocalEEGLabel("C", [cQcAB, P])] += -0.5*wt_cAB*wt_cQcAB
        
        if wt_cAB and wt_cPQ:
            wt_ccPQcAB, ccPQcAB = paulistr_commutator(cPQ, cAB)
            if wt_ccPQcAB:
                terms[_LocalEEGLabel("H", [ccPQcAB])] += 0.25*wt_cAB*wt_cPQ*wt_ccPQcAB
    else:
        # This should be a negative of one of the cases above, i.e. [B,A] = -[A,B]
        neg_terms = _errorgen_commutator_symbolic_term(lbl2, lbl1)
        for k,v in neg_terms.items():
            terms[k] += -1*v
    
    return terms