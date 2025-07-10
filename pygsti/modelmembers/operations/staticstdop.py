"""
The StaticStandardOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import numpy as _np

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers.errorgencontainer import NoErrorGeneratorInterface as _NoErrorGeneratorInterface
from pygsti.modelmembers import term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.tools import internalgates as _itgs


class StaticStandardOp(_LinearOperator, _NoErrorGeneratorInterface):
    """
    An operation that is completely fixed, or "static" (i.e. that posesses no parameters)
    that can be constructed from "standard" gate names (as defined in pygsti.tools.internalgates).

    Parameters
    ----------
    name : str
        Standard gate name

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """
    def __init__(self, name, basis='pp', evotype="default", state_space=None):
        self.name = name

        #Create default state space if needed
        std_unitaries = _itgs.standard_gatename_unitaries()
        if name not in std_unitaries:
            raise ValueError("'%s' does not name a standard operation" % self.name)
        state_space = _statespace.default_space_for_udim(std_unitaries[name].shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)
        basis = _Basis.cast(basis, state_space.dim)  # basis for Hilbert-Schmidt (superop) space

        evotype = _Evotype.cast(evotype, state_space=state_space)
        rep = evotype.create_standard_rep(name, basis, state_space)
        _LinearOperator.__init__(self, rep, evotype)

    def to_dense(self, on_space: SpaceT='minimal'):
        """
        Return the dense array used to represent this operation within its evolution type.

        Note: for efficiency, this doesn't copy the underlying data, so
        the caller should copy this data before modifying it.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        numpy.ndarray
        """
        return self._rep.to_dense(on_space)  # standard rep needs to implement this

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the operation's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the operation's parent (usually a :class:`Model`), not the
        operation's local parameter array (i.e. that returned from `to_vector`).

        Parameters
        ----------
        order : int
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        return_coeff_polys : bool
            Whether a parallel list of locally-indexed (using variable indices
            corresponding to *this* object's parameters rather than its parent's)
            polynomial coefficients should be returned as well.

        Returns
        -------
        terms : list
            A list of :class:`RankOneTerm` objects.
        coefficients : list
            Only present when `return_coeff_polys == True`.
            A list of *compact* polynomial objects, meaning that each element
            is a `(vtape,ctape)` 2-tuple formed by concatenating together the
            output of :meth:`Polynomial.compact`.
        """
        #Same as unitary op -- assume this op acts as a single unitary term -- consolidate in FUTURE?
        if order == 0:  # only 0-th order term exists
            coeff = _Polynomial({(): 1.0}, max_polynomial_vars)
            terms = [_term.RankOnePolynomialOpTerm.create_from(coeff, self, self,
                                                               self._evotype, self.state_space)]
            if return_coeff_polys:
                coeffs_as_compact_polys = coeff.compact(complex_coeff_tape=True)
                return terms, coeffs_as_compact_polys
            else:
                return terms
        else:
            if return_coeff_polys:
                vtape = _np.empty(0, _np.int64)
                ctape = _np.empty(0, complex)
                return [], (vtape, ctape)
            else:
                return []

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this operator's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this operator in a Taylor series.

        Returns
        -------
        float
        """
        return 1.0

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this operator's terms.

        Computes the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        return _np.empty((0,), 'd')

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
            module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['name'] = self.name
        mm_dict['basis'] = self._rep.basis.to_nice_serialization()

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        basis = _Basis.from_nice_serialization(mm_dict['basis'])
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        return cls(mm_dict['name'], basis, mm_dict['evotype'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return self.name == other.name  # also compare self._rep.basis (?)

    def __str__(self):
        s = "%s with name %s and evotype %s\n" % (self.__class__.__name__, self.name, self._evotype)
        #TODO: move this to __str__ methods of reps??
        #if self._evotype in ['statevec', 'densitymx', 'svterm', 'cterm']:
        #    s += _mt.mx_to_string(self.base, width=4, prec=2)
        #elif self._evotype == 'chp':
        #    s += 'CHP operations: ' + ','.join(self._rep.chp_ops) + '\n'
        return s

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        return "%s gate" % self.name
