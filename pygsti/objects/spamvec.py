"""
Defines classes with represent SPAM operations, along with supporting
functionality.
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
import scipy.sparse as _sps
import collections as _collections
import numbers as _numbers
import itertools as _itertools
import functools as _functools
import copy as _copy

from .. import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools import symplectic as _symp
from .basis import Basis as _Basis
from .protectedarray import ProtectedArray as _ProtectedArray
from . import modelmember as _modelmember

from . import term as _term
from . import stabilizer as _stabilizer
from .polynomial import Polynomial as _Polynomial
from . import replib
from .opcalc import bulk_eval_compact_polys_complex as _bulk_eval_compact_polys_complex

IMAG_TOL = 1e-8  # tolerance for imaginary part being considered zero


def optimize_spamvec(vecToOptimize, targetVec):
    """
    Optimize the parameters of vecToOptimize so that the
      the resulting SPAM vector is as close as possible to
      targetVec.

    This is trivial for the case of FullSPAMVec
      instances, but for other types of parameterization
      this involves an iterative optimization over all the
      parameters of vecToOptimize.

    Parameters
    ----------
    vecToOptimize : SPAMVec
      The vector to optimize. This object gets altered.

    targetVec : SPAMVec
      The SPAM vector used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(vecToOptimize, StaticSPAMVec):
        return  # nothing to optimize

    if isinstance(vecToOptimize, FullSPAMVec):
        if(targetVec.dim != vecToOptimize.dim):  # special case: gates can have different overall dimension
            vecToOptimize.dim = targetVec.dim  # this is a HACK to allow model selection code to work correctly
        vecToOptimize.set_value(targetVec)  # just copy entire overall matrix since fully parameterized
        return

    assert(targetVec.dim == vecToOptimize.dim)  # vectors must have the same overall dimension
    targetVector = _np.asarray(targetVec)

    def _objective_func(param_vec):
        vecToOptimize.from_vector(param_vec)
        return _mt.frobeniusnorm(vecToOptimize - targetVector)

    x0 = vecToOptimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    vecToOptimize.from_vector(minSol.x)
    #print("DEBUG: optimized vector to min frobenius distance %g" % _mt.frobeniusnorm(vecToOptimize-targetVector))


def convert(spamvec, toType, basis, extra=None):
    """
    Convert SPAM vector to a new type of parameterization, potentially
    creating a new SPAMVec object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    spamvec : SPAMVec
        SPAM vector to convert

    toType : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `spamvec`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.


    Returns
    -------
    SPAMVec
       The converted SPAM vector, usually a distinct
       object from the object passed as input.
    """
    if toType == "full":
        if isinstance(spamvec, FullSPAMVec):
            return spamvec  # no conversion necessary
        else:
            typ = spamvec._prep_or_effect if isinstance(spamvec, SPAMVec) else "prep"
            return FullSPAMVec(spamvec.todense(), typ=typ)

    elif toType == "TP":
        if isinstance(spamvec, TPSPAMVec):
            return spamvec  # no conversion necessary
        else:
            return TPSPAMVec(spamvec.todense())
            # above will raise ValueError if conversion cannot be done

    elif toType == "TrueCPTP":  # a non-lindbladian CPTP spamvec that hasn't worked well...
        if isinstance(spamvec, CPTPSPAMVec):
            return spamvec  # no conversion necessary
        else:
            return CPTPSPAMVec(spamvec, basis)
            # above will raise ValueError if conversion cannot be done

    elif toType == "static":
        if isinstance(spamvec, StaticSPAMVec):
            return spamvec  # no conversion necessary
        else:
            typ = spamvec._prep_or_effect if isinstance(spamvec, SPAMVec) else "prep"
            return StaticSPAMVec(spamvec, typ=typ)

    elif toType == "static unitary":
        dmvec = _bt.change_basis(spamvec.todense(), basis, 'std')
        purevec = _gt.dmvec_to_state(dmvec)
        return StaticSPAMVec(purevec, "statevec", spamvec._prep_or_effect)

    elif _gt.is_valid_lindblad_paramtype(toType):

        if extra is None:
            purevec = spamvec  # right now, we don't try to extract a "closest pure vec"
            # to spamvec - below will fail if spamvec isn't pure.
        else:
            purevec = extra  # assume extra info is a pure vector

        nQubits = _np.log2(spamvec.dim) / 2.0
        bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis
        typ = spamvec._prep_or_effect if isinstance(spamvec, SPAMVec) else "prep"

        return LindbladSPAMVec.from_spamvec_obj(
            spamvec, typ, toType, None, proj_basis, basis,
            truncate=True, lazy=True)

    elif toType == "clifford":
        if isinstance(spamvec, StabilizerSPAMVec):
            return spamvec  # no conversion necessary

        purevec = spamvec.flatten()  # assume a pure state (otherwise would
        # need to change Model dim)
        return StabilizerSPAMVec.from_dense_purevec(purevec)

    else:
        raise ValueError("Invalid toType argument: %s" % toType)


def _convert_to_lindblad_base(vec, typ, new_evotype, mxBasis="pp"):
    """
    Attempts to convert `vec` to a static (0 params) SPAMVec with
    evoution type `new_evotype`.  Used to convert spam vecs to
    being LindbladSPAMVec objects.
    """
    if vec._evotype == new_evotype and vec.num_params() == 0:
        return vec  # no conversion necessary
    if new_evotype == "densitymx":
        return StaticSPAMVec(vec.todense(), "densitymx", typ)
    if new_evotype in ("svterm", "cterm"):
        if isinstance(vec, ComputationalSPAMVec):  # special case when conversion is easy
            return ComputationalSPAMVec(vec._zvals, new_evotype, typ)
        elif vec._evotype == "densitymx":
            # then try to extract a (static) pure state from vec wth
            # evotype 'statevec' or 'stabilizer' <=> 'svterm', 'cterm'
            if isinstance(vec, DenseSPAMVec):
                dmvec = _bt.change_basis(vec, mxBasis, 'std')
                purestate = StaticSPAMVec(_gt.dmvec_to_state(dmvec), 'statevec', typ)
            elif isinstance(vec, PureStateSPAMVec):
                purestate = vec.pure_state_vec  # evotype 'statevec'
            else:
                raise ValueError("Unable to obtain pure state from density matrix type %s!" % type(vec))

            if new_evotype == "cterm":  # then purestate 'statevec' => 'stabilizer' (if possible)
                if typ == "prep":
                    purestate = StabilizerSPAMVec.from_dense_purevec(purestate.todense())
                else:  # type == "effect"
                    purestate = StabilizerEffectVec.from_dense_purevec(purestate.todense())

            return PureStateSPAMVec(purestate, new_evotype, mxBasis, typ)

    raise ValueError("Could not convert %s (evotype %s) to %s w/0 params!" %
                     (str(type(vec)), vec._evotype, new_evotype))


def finite_difference_deriv_wrt_params(spamvec, wrtFilter=None, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a SPAMVec object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the spam vector with respect to a single
    parameter, matching the format expected from the spam vectors's
    `deriv_wrt_params` method.

    Parameters
    ----------
    spamvec : SPAMVec
        The spam vector object to compute a Jacobian for.

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    numpy.ndarray
        An M by N matrix where M is the number of gate elements and
        N is the number of gate parameters.
    """
    dim = spamvec.get_dimension()
    spamvec2 = spamvec.copy()
    p = spamvec.to_vector()
    fd_deriv = _np.empty((dim, spamvec.num_params()), 'd')  # assume real (?)

    for i in range(spamvec.num_params()):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        spamvec2.from_vector(p_plus_dp, close=True)
        fd_deriv[:, i:i + 1] = (spamvec2 - spamvec) / eps

    fd_deriv.shape = [dim, spamvec.num_params()]
    if wrtFilter is None:
        return fd_deriv
    else:
        return _np.take(fd_deriv, wrtFilter, axis=1)


def check_deriv_wrt_params(spamvec, deriv_to_check=None, wrtFilter=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a SPAMVec object.

    This routine is meant to be used as an aid in testing and debugging
    SPAMVec classes by comparing the finite-difference Jacobian that
    *should* be returned by `spamvec.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    spamvec : SPAMVec
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `spamvec.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a LinearOperator
        class's `deriv_wrt_params()` method itself as a part of testing.

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    None
    """
    fd_deriv = finite_difference_deriv_wrt_params(spamvec, wrtFilter, eps)
    if deriv_to_check is None:
        deriv_to_check = spamvec.deriv_wrt_params()

    #print("Deriv shapes = %s and %s" % (str(fd_deriv.shape),
    #                                    str(deriv_to_check.shape)))
    #print("finite difference deriv = \n",fd_deriv)
    #print("deriv_wrt_params deriv = \n",deriv_to_check)
    #print("deriv_wrt_params - finite diff deriv = \n",
    #      deriv_to_check - fd_deriv)

    for i in range(deriv_to_check.shape[0]):
        for j in range(deriv_to_check.shape[1]):
            diff = abs(deriv_to_check[i, j] - fd_deriv[i, j])
            if diff > 5 * eps:
                print("deriv_chk_mismatch: (%d,%d): %g (comp) - %g (fd) = %g" %
                      (i, j, deriv_to_check[i, j], fd_deriv[i, j], diff))

    if _np.linalg.norm(fd_deriv - deriv_to_check) > 100 * eps:
        raise ValueError("Failed check of deriv_wrt_params:\n"
                         " norm diff = %g" %
                         _np.linalg.norm(fd_deriv - deriv_to_check))


class SPAMVec(_modelmember.ModelMember):
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    def __init__(self, rep, evotype, typ):
        """ Initialize a new SPAM Vector """
        if isinstance(rep, int):  # For operators that have no representation themselves (term ops)
            dim = rep             # allow passing an integer as `rep`.
            rep = None
        else:
            dim = rep.dim
        super(SPAMVec, self).__init__(dim, evotype)
        self._rep = rep
        self._prep_or_effect = typ

    @property
    def size(self):
        """
        Return the number of independent elements in this gate (when viewed as a dense array)
        """
        return self.dim

    @property
    def outcomes(self):
        """
        Return the z-value outcomes corresponding to this effect SPAM vector
        in the context of a stabilizer-state simulation.
        """
        raise NotImplementedError("'outcomes' property is not implemented for %s objects" % self.__class__.__name__)

    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        raise ValueError("Cannot set the value of a %s directly!" % self.__class__.__name__)

    def set_time(self, t):
        """
        Sets the current time for a time-dependent operator.  For time-independent
        operators (the default), this function does absolutely nothing.

        Parameters
        ----------
        t : float
            The current time.

        Returns
        -------
        None
        """
        pass

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        raise NotImplementedError("todense(...) not implemented for %s objects!" % self.__class__.__name__)

#    def torep(self, typ, outrep=None):
#        """
#        Return a "representation" object for this SPAM vector.
#        Such objects are primarily used internally by pyGSTi to compute
#        things like probabilities more efficiently.
#
#        Parameters
#        ----------
#        typ : {'prep','effect'}
#            The type of representation (for cases when the vector type is
#            not already defined).
#
#        outrep : StateRep
#            If not None, an existing state representation appropriate to this
#            SPAM vector that may be used instead of allocating a new one.
#
#        Returns
#        -------
#        StateRep
#        """
#        if typ == "prep":
#            if self._evotype == "statevec":
#                return replib.SVStateRep(self.todense())
#            elif self._evotype == "densitymx":
#                return replib.DMStateRep(self.todense())
#            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
#                                      (self._evotype, self.__class__.__name__))
#        elif typ == "effect":
#            if self._evotype == "statevec":
#                return replib.SVEffectRep_Dense(self.todense())
#            elif self._evotype == "densitymx":
#                return replib.DMEffectRep_Dense(self.todense())
#            raise NotImplementedError("torep(%s) not implemented for %s objects!" %
#                                      (self._evotype, self.__class__.__name__))
#        else:
#            raise ValueError("Invalid `typ` argument for torep(): %s" % typ)

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_poly_coeffs=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

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
            output of :method:`Polynomial.compact`.
        """
        raise NotImplementedError("get_taylor_order_terms(...) not implemented for %s objects!" %
                                  self.__class__.__name__)

    def get_highmagnitude_terms(self, min_term_mag, force_firstorder=True, max_taylor_order=3, max_poly_vars=100):
        """
        Get the terms (from a Taylor expansion of this SPAM vector) that have
        magnitude above `min_term_mag` (the magnitude of a term is taken to
        be the absolute value of its coefficient), considering only those
        terms up to some maximum Taylor expansion order, `max_taylor_order`.

        Note that this function also *sets* the magnitudes of the returned
        terms (by calling `term.set_magnitude(...)`) based on the current
        values of this SPAM vector's parameters.  This is an essential step
        to using these terms in pruned-path-integral calculations later on.

        Parameters
        ----------
        min_term_mag : float
            the threshold for term magnitudes: only terms with magnitudes above
            this value are returned.

        force_firstorder : bool, optional
            if True, then always return all the first-order Taylor-series terms,
            even if they have magnitudes smaller than `min_term_mag`.  This
            behavior is needed for using GST with pruned-term calculations, as
            we may begin with a guess model that has no error (all terms have
            zero magnitude!) and still need to compute a meaningful jacobian at
            this point.

        max_taylor_order : int, optional
            the maximum Taylor-order to consider when checking whether term-
            magnitudes exceed `min_term_mag`.

        Returns
        -------
        highmag_terms : list
            A list of the high-magnitude terms that were found.  These
            terms are *sorted* in descending order by term-magnitude.
        first_order_indices : list
            A list of the indices into `highmag_terms` that mark which
            of these terms are first-order Taylor terms (useful when
            we're forcing these terms to always be present).
        """
        #NOTE: SAME as for LinearOperator class -- TODO consolidate in FUTURE
        #print("DB: SPAM get_high_magnitude_terms")
        v = self.to_vector()
        taylor_order = 0
        terms = []; last_len = -1; first_order_magmax = 1.0
        while len(terms) > last_len:  # while we keep adding something
            if taylor_order > 1 and first_order_magmax**taylor_order < min_term_mag:
                break  # there's no way any terms at this order reach min_term_mag - exit now!

            MAX_CACHED_TERM_ORDER = 1
            if taylor_order <= MAX_CACHED_TERM_ORDER:
                #print("order ",taylor_order," : ",len(terms), "terms")
                terms_at_order, cpolys = self.get_taylor_order_terms(taylor_order, max_poly_vars, True)
                coeffs = _bulk_eval_compact_polys_complex(
                    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
                mags = _np.abs(coeffs)
                last_len = len(terms)
                #OLD: terms_at_order = [ t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order) ]

                if taylor_order == 1:
                    #OLD: first_order_magmax = max([t.magnitude for t in terms_at_order])
                    first_order_magmax = max(mags)

                    if force_firstorder:
                        terms.extend([(taylor_order, t.copy_with_magnitude(mag))
                                      for coeff, mag, t in zip(coeffs, mags, terms_at_order)])
                    else:
                        for mag, t in zip(mags, terms_at_order):
                            if mag >= min_term_mag:
                                terms.append((taylor_order, t.copy_with_magnitude(mag)))
                else:
                    for mag, t in zip(mags, terms_at_order):
                        if mag >= min_term_mag:
                            terms.append((taylor_order, t.copy_with_magnitude(mag)))

            else:
                terms.extend([(taylor_order, t) for t in
                              self.get_taylor_order_terms_above_mag(taylor_order,
                                                                    max_poly_vars, min_term_mag)])

            taylor_order += 1
            if taylor_order > max_taylor_order: break

        #Sort terms based on magnitude
        sorted_terms = sorted(terms, key=lambda t: t[1].magnitude, reverse=True)
        first_order_indices = [i for i, t in enumerate(sorted_terms) if t[0] == 1]
        return [t[1] for t in sorted_terms], first_order_indices

    def get_taylor_order_terms_above_mag(self, order, max_poly_vars, min_term_mag):
        """ TODO: docstring """
        v = self.to_vector()
        terms_at_order, cpolys = self.get_taylor_order_terms(order, max_poly_vars, True)
        coeffs = _bulk_eval_compact_polys_complex(
            cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        terms_at_order = [t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order)]
        return [t for t in terms_at_order if t.magnitude >= min_term_mag]

    def frobeniusdist2(self, otherSpamVec, typ, transform=None,
                       inv_transform=None):
        """
        Return the squared frobenius difference between this spam vector and
        `otherSpamVec`, optionally transforming this vector first using
        `transform` and `inv_transform` (depending on the value of `typ`).

        Parameters
        ----------
        otherSpamVec : SPAMVec
            The other spam vector

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed.

        transform, inv_transform : numpy.ndarray
            The transformation (if not None) to be performed.

        Returns
        -------
        float
        """
        vec = self.todense()
        if typ == 'prep':
            if inv_transform is None:
                return _gt.frobeniusdist2(vec, otherSpamVec.todense())
            else:
                return _gt.frobeniusdist2(_np.dot(inv_transform, vec),
                                          otherSpamVec.todense())
        elif typ == "effect":
            if transform is None:
                return _gt.frobeniusdist2(vec, otherSpamVec.todense())
            else:
                return _gt.frobeniusdist2(_np.dot(_np.transpose(transform),
                                                  vec), otherSpamVec.todense())
        else: raise ValueError("Invalid 'typ' argument: %s" % typ)

    def residuals(self, otherSpamVec, typ, transform=None, inv_transform=None):
        """
        Return a vector of residuals between this spam vector and
        `otherSpamVec`, optionally transforming this vector first using
        `transform` and `inv_transform` (depending on the value of `typ`).

        Parameters
        ----------
        otherSpamVec : SPAMVec
            The other spam vector

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed.

        transform, inv_transform : numpy.ndarray
            The transformation (if not None) to be performed.

        Returns
        -------
        float
        """
        vec = self.todense()
        if typ == 'prep':
            if inv_transform is None:
                return _gt.residuals(vec, otherSpamVec.todense())
            else:
                return _gt.residuals(_np.dot(inv_transform, vec),
                                     otherSpamVec.todense())
        elif typ == "effect":
            if transform is None:
                return _gt.residuals(vec, otherSpamVec.todense())
            else:
                return _gt.residuals(_np.dot(_np.transpose(transform),
                                             vec), otherSpamVec.todense())

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for preparation
        or  effect SPAM vectors, respectively.

        Note that this is equivalent to state preparation vectors getting
        mapped: `rho -> inv(S) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * S`.

        Generally, the transform function updates the *parameters* of
        the SPAM vector such that the resulting vector is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        if typ == 'prep':
            Si = S.get_transform_matrix_inverse()
            self.set_value(_np.dot(Si, self.todense()))
        elif typ == 'effect':
            Smx = S.get_transform_matrix()
            self.set_value(_np.dot(_np.transpose(Smx), self.todense()))
            #Evec^T --> ( Evec^T * S )^T
        else:
            raise ValueError("Invalid typ argument: %s" % typ)

    def depolarize(self, amount):
        """
        Depolarize this SPAM vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the SPAMVec such that the resulting vector is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. All but the
            first element of the spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        if isinstance(amount, float) or _compat.isint(amount):
            D = _np.diag([1] + [1 - amount] * (self.dim - 1))
        else:
            assert(len(amount) == self.dim - 1)
            D = _np.diag([1] + list(1.0 - _np.array(amount, 'd')))
        self.set_value(_np.dot(D, self.todense()))

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0  # no parameters

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd')  # no parameters

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(len(v) == 0)  # should be no parameters, and nothing to do

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dtype = complex if self._evotype == 'statevec' else 'd'
        derivMx = _np.zeros((self.dim, 0), dtype)
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrtFilter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.num_params() > 0

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the vectors's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        if not self.has_nonzero_hessian():
            return _np.zeros(self.size, self.num_params(), self.num_params())

        # FUTURE: create a finite differencing hessian method?
        raise NotImplementedError("hessian_wrt_params(...) is not implemented for %s objects" % self.__class__.__name__)

    #Note: no __str__ fn

    @staticmethod
    def convert_to_vector(V):
        """
        Static method that converts a vector-like object to a 2D numpy
        dim x 1 column array.

        Parameters
        ----------
        V : array_like

        Returns
        -------
        numpy array
        """
        if isinstance(V, SPAMVec):
            vector = V.todense().copy()
            vector.shape = (vector.size, 1)
        elif isinstance(V, _np.ndarray):
            vector = V.copy()
            if len(vector.shape) == 1:  # convert (N,) shape vecs to (N,1)
                vector.shape = (vector.size, 1)
        else:
            try:
                len(V)
                # XXX this is an abuse of exception handling
            except:
                raise ValueError("%s doesn't look like an array/list" % V)
            try:
                d2s = [len(row) for row in V]
            except TypeError:  # thrown if len(row) fails because no 2nd dim
                d2s = None

            if d2s is not None:
                if any([len(row) != 1 for row in V]):
                    raise ValueError("%s is 2-dimensional but 2nd dim != 1" % V)

                typ = 'd' if _np.all(_np.isreal(V)) else 'complex'
                try:
                    vector = _np.array(V, typ)  # vec is already a 2-D column vector
                except TypeError:
                    raise ValueError("%s doesn't look like an array/list" % V)
            else:
                typ = 'd' if _np.all(_np.isreal(V)) else 'complex'
                vector = _np.array(V, typ)[:, None]  # make into a 2-D column vec

        assert(len(vector.shape) == 2 and vector.shape[1] == 1)
        return vector.flatten()  # HACK for convention change -> (N,) instead of (N,1)


class DenseSPAMVec(SPAMVec):
    """
    Excapulates a parameterization of a state preparation OR POVM effect
    vector. This class is the  common base class for all specific
    parameterizations of a SPAM vector.
    """

    def __init__(self, vec, evotype, prep_or_effect):
        """ Initialize a new SPAM Vector """
        dtype = complex if evotype == "statevec" else 'd'
        vec = _np.asarray(vec, dtype=dtype)
        vec.shape = (vec.size,)  # just store 1D array flatten
        vec = _np.require(vec, requirements=['OWNDATA', 'C_CONTIGUOUS'])

        if prep_or_effect == "prep":
            if evotype == "statevec":
                rep = replib.SVStateRep(vec)
            elif evotype == "densitymx":
                rep = replib.DMStateRep(vec)
            else:
                raise ValueError("Invalid evotype for DenseSPAMVec: %s" % evotype)
        elif prep_or_effect == "effect":
            if evotype == "statevec":
                rep = replib.SVEffectRep_Dense(vec)
            elif evotype == "densitymx":
                rep = replib.DMEffectRep_Dense(vec)
            else:
                raise ValueError("Invalid evotype for DenseSPAMVec: %s" % evotype)
        else:
            raise ValueError("Invalid `prep_or_effect` argument: %s" % prep_or_effect)

        super(DenseSPAMVec, self).__init__(rep, evotype, prep_or_effect)
        assert(self.base1D.flags['C_CONTIGUOUS'] and self.base1D.flags['OWNDATA'])

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        #don't use scratch since we already have memory allocated
        return self.base1D  # *must* be a numpy array for Cython arg conversion

    @property
    def base1D(self):
        return self._rep.base

    @property
    def base(self):
        bv = self.base1D.view()
        bv.shape = (bv.size, 1)  # 'base' is by convention a (N,1)-shaped array
        return bv

    def __copy__(self):
        # We need to implement __copy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __copy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def __deepcopy__(self, memo):
        # We need to implement __deepcopy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __deepcopy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, _copy.deepcopy(v, memo))
        return cpy

    #Access to underlying array
    def __getitem__(self, key):
        self.dirty = True
        return self.base.__getitem__(key)

    def __getslice__(self, i, j):
        self.dirty = True
        return self.__getitem__(slice(i, j))  # Called for A[:]

    def __setitem__(self, key, val):
        self.dirty = True
        return self.base.__setitem__(key, val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        if '_rep' in self.__dict__:  # sometimes in loading __getattr__ gets called before the instance is loaded
            ret = getattr(self.base, attr)
        else:
            raise AttributeError("No attribute:", attr)
        self.dirty = True
        return ret

    #Mimic array
    def __pos__(self): return self.base
    def __neg__(self): return -self.base
    def __abs__(self): return abs(self.base)
    def __add__(self, x): return self.base + x
    def __radd__(self, x): return x + self.base
    def __sub__(self, x): return self.base - x
    def __rsub__(self, x): return x - self.base
    def __mul__(self, x): return self.base * x
    def __rmul__(self, x): return x * self.base
    def __truediv__(self, x): return self.base / x
    def __rtruediv__(self, x): return x / self.base
    def __floordiv__(self, x): return self.base // x
    def __rfloordiv__(self, x): return x // self.base
    def __pow__(self, x): return self.base ** x
    def __eq__(self, x): return self.base == x
    def __len__(self): return len(self.base)
    def __int__(self): return int(self.base)
    def __long__(self): return int(self.base)
    def __float__(self): return float(self.base)
    def __complex__(self): return complex(self.base)

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__, self.dim)
        s += _mt.mx_to_string(self.todense(), width=4, prec=2)
        return s


class StaticSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, vec, evotype="auto", typ="prep"):
        """
        Initialize a StaticSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.

        evotype : {"densitymx", "statevec"}
            the evolution type being used.

        typ : {"prep", "effect"}
            whether this is a state preparation or an effect (measurement)
            SPAM vector.
        """
        vec = SPAMVec.convert_to_vector(vec)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(vec) else "densitymx"
        elif evotype == "statevec":
            vec = _np.asarray(vec, complex)  # ensure all statevec vecs are complex (densitymx could be either?)

        assert(evotype in ("statevec", "densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)
        DenseSPAMVec.__init__(self, vec, evotype, typ)


class FullSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized, that is,
      each element of the SPAM vector is an independent parameter.
    """

    def __init__(self, vec, evotype="auto", typ="prep"):
        """
        Initialize a FullSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.

        evotype : {"densitymx", "statevec"}
            the evolution type being used.

        typ : {"prep", "effect"}
            whether this is a state preparation or an effect (measurement)
            SPAM vector.
        """
        vec = SPAMVec.convert_to_vector(vec)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(vec) else "densitymx"
        assert(evotype in ("statevec", "densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)
        DenseSPAMVec.__init__(self, vec, evotype, typ)

    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        vec = SPAMVec.convert_to_vector(vec)
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        self.base1D[:] = vec
        self.dirty = True

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 2 * self.size if self._evotype == "statevec" else self.size

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        if self._evotype == "statevec":
            return _np.concatenate((self.base1D.real, self.base1D.imag), axis=0)
        else:
            return self.base1D

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        if self._evotype == "statevec":
            self.base1D[:] = v[0:self.dim] + 1j * v[self.dim:]
        else:
            self.base1D[:] = v
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        if self._evotype == "statevec":
            derivMx = _np.concatenate((_np.identity(self.dim, complex),
                                       1j * _np.identity(self.dim, complex)), axis=1)
        else:
            derivMx = _np.identity(self.dim, 'd')

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrtFilter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class TPSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is fully parameterized except for the first
    element, which is frozen to be 1/(d**0.25).  This is so that, when the SPAM
    vector is interpreted in the Pauli or Gell-Mann basis, the represented
    density matrix has trace == 1.  This restriction is frequently used in
    conjuction with trace-preserving (TP) gates.
    """

    #Note: here we assume that the first basis element is (1/sqrt(x) * I),
    # where I the d-dimensional identity (where len(vector) == d**2). So
    # if Tr(basisEl*basisEl) == Tr(1/x*I) == d/x must == 1, then we must
    # have x == d.  Thus, we multiply this first basis element by
    # alpha = 1/sqrt(d) to obtain a trace-1 matrix, i.e., finding alpha
    # s.t. Tr(alpha*[1/sqrt(d)*I]) == 1 => alpha*d/sqrt(d) == 1 =>
    # alpha = 1/sqrt(d) = 1/(len(vec)**0.25).
    def __init__(self, vec):
        """
        Initialize a TPSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.
        """
        vector = SPAMVec.convert_to_vector(vec)
        firstEl = len(vector)**-0.25
        if not _np.isclose(vector[0], firstEl):
            raise ValueError("Cannot create TPSPAMVec: "
                             "first element must equal %g!" % firstEl)
        DenseSPAMVec.__init__(self, vec, "densitymx", "prep")
        assert(isinstance(self.base, _ProtectedArray))

    @property
    def base(self):
        bv = self.base1D.view()
        bv.shape = (bv.size, 1)
        return _ProtectedArray(bv, indicesToProtect=(0, 0))

    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        vec = SPAMVec.convert_to_vector(vec)
        firstEl = (self.dim)**-0.25
        if(vec.size != self.dim):
            raise ValueError("Argument must be length %d" % self.dim)
        if not _np.isclose(vec[0], firstEl):
            raise ValueError("Cannot create TPSPAMVec: "
                             "first element must equal %g!" % firstEl)
        self.base1D[1:] = vec[1:]
        self.dirty = True

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.dim - 1

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.base1D[1:]  # .real in case of complex matrices?

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(_np.isclose(self.base1D[0], (self.dim)**-0.25))
        self.base1D[1:] = v
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        derivMx = _np.identity(self.dim, 'd')  # TP vecs assumed real
        derivMx = derivMx[:, 1:]  # remove first col ( <=> first-el parameters )
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrtFilter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class ComplementSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is parameterized by
    `I - sum(other_spam_vecs)` where `I` is a (static) identity element
    and `other_param_vecs` is a list of other spam vectors in the same parent
    :class:`POVM`.  This only *partially* implements the SPAMVec interface
    (some methods such as `to_vector` and `from_vector` will thunk down to base
    class versions which raise `NotImplementedError`), as instances are meant to
    be contained within a :class:`POVM` which takes care of vectorization.
    """

    def __init__(self, identity, other_spamvecs):
        """
        Initialize a ComplementSPAMVec object.

        Parameters
        ----------
        identity : array_like or SPAMVec
            a 1D numpy array representing the static identity operation from
            which the sum of the other vectors is subtracted.

        other_spamvecs : list of SPAMVecs
            A list of the "other" parameterized SPAM vectors which are
            subtracted from `identity` to compute the final value of this
            "complement" SPAM vector.
        """
        self.identity = FullSPAMVec(
            SPAMVec.convert_to_vector(identity))  # so easy to transform
        # or depolarize by parent POVM

        self.other_vecs = other_spamvecs
        #Note: we assume that our parent will do the following:
        # 1) set our gpindices to indicate how many parameters we have
        # 2) set the gpindices of the elements of other_spamvecs so
        #    that they index into our local parameter vector.

        DenseSPAMVec.__init__(self, self.identity, "densitymx", "effect")  # dummy
        self._construct_vector()  # reset's self.base

    def _construct_vector(self):
        self.base1D.flags.writeable = True
        self.base1D[:] = self.identity.base1D - sum([vec.base1D for vec in self.other_vecs])
        self.base1D.flags.writeable = False

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        raise ValueError(("ComplementSPAMVec.to_vector() should never be called"
                          " - use TPPOVM.to_vector() instead"))

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this SPAM vector using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of parameters.

        Returns
        -------
        None
        """
        #Rely on prior .from_vector initialization of self.other_vecs, so
        # we just construct our vector based on them.
        #Note: this is needed for finite-differencing in map-based calculator
        self._construct_vector()
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        if len(self.other_vecs) == 0: return _np.zeros((self.dim, 0), 'd')  # Complement vecs assumed real
        Np = len(self.gpindices_as_array())
        neg_deriv = _np.zeros((self.dim, Np), 'd')
        for ovec in self.other_vecs:
            local_inds = _modelmember._decompose_gpindices(
                self.gpindices, ovec.gpindices)
            #Note: other_vecs are not copies but other *sibling* effect vecs
            # so their gpindices index the same space as this complement vec's
            # does - so we need to "_decompose_gpindices"
            neg_deriv[:, local_inds] += ovec.deriv_wrt_params()
        derivMx = -neg_deriv

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrtFilter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class CPTPSPAMVec(DenseSPAMVec):
    """
    Encapsulates a SPAM vector that is parameterized through the Cholesky
    decomposition of it's standard-basis representation as a density matrix
    (not a Liouville vector).  The resulting SPAM vector thus represents a
    positive density matrix, and additional constraints on the parameters
    also guarantee that the trace == 1.  This SPAM vector is meant for
    use with CPTP processes, hence the name.
    """

    def __init__(self, vec, basis, truncate=False):
        """
        Initialize a CPTPSPAMVec object.

        Parameters
        ----------
        vec : array_like or SPAMVec
            a 1D numpy array representing the SPAM operation.  The
            shape of this array sets the dimension of the SPAM op.

        basis : {"std", "gm", "pp", "qt"} or Basis
            The basis `vec` is in.  Needed because this parameterization
            requires we construct the density matrix corresponding to
            the Lioville vector `vec`.

        trunctate : bool, optional
            Whether or not a non-positive, trace=1 `vec` should
            be truncated to force a successful construction.
        """
        vector = SPAMVec.convert_to_vector(vec)
        basis = _Basis.cast(basis, len(vector))

        self.basis = basis
        self.basis_mxs = basis.elements  # shape (len(vec), dmDim, dmDim)
        self.basis_mxs = _np.rollaxis(self.basis_mxs, 0, 3)  # shape (dmDim, dmDim, len(vec))
        assert(self.basis_mxs.shape[-1] == len(vector))

        # set self.params and self.dmDim
        self._set_params_from_vector(vector, truncate)

        #scratch space
        self.Lmx = _np.zeros((self.dmDim, self.dmDim), 'complex')

        DenseSPAMVec.__init__(self, vector, "densitymx", "prep")

    def _set_params_from_vector(self, vector, truncate):
        density_mx = _np.dot(self.basis_mxs, vector)
        density_mx = density_mx.squeeze()
        dmDim = density_mx.shape[0]
        assert(dmDim == density_mx.shape[1]), "Density matrix must be square!"

        trc = _np.trace(density_mx)
        assert(truncate or _np.isclose(trc, 1.0)), \
            "`vec` must correspond to a trace-1 density matrix (truncate == False)!"

        if not _np.isclose(trc, 1.0):  # truncate to trace == 1
            density_mx -= _np.identity(dmDim, 'd') / dmDim * (trc - 1.0)

        #push any slightly negative evals of density_mx positive
        # so that the Cholesky decomp will work.
        evals, U = _np.linalg.eig(density_mx)
        Ui = _np.linalg.inv(U)

        assert(truncate or all([ev >= -1e-12 for ev in evals])), \
            "`vec` must correspond to a positive density matrix (truncate == False)!"

        pos_evals = evals.clip(1e-16, 1e100)
        density_mx = _np.dot(U, _np.dot(_np.diag(pos_evals), Ui))
        try:
            Lmx = _np.linalg.cholesky(density_mx)
        except _np.linalg.LinAlgError:  # Lmx not postitive definite?
            pos_evals = evals.clip(1e-12, 1e100)  # try again with 1e-12
            density_mx = _np.dot(U, _np.dot(_np.diag(pos_evals), Ui))
            Lmx = _np.linalg.cholesky(density_mx)

        #check TP condition: that diagonal els of Lmx squared add to 1.0
        Lmx_norm = _np.trace(_np.dot(Lmx.T.conjugate(), Lmx))  # sum of magnitude^2 of all els
        assert(_np.isclose(Lmx_norm, 1.0)), \
            "Cholesky decomp didn't preserve trace=1!"

        self.dmDim = dmDim
        self.params = _np.empty(dmDim**2, 'd')
        for i in range(dmDim):
            assert(_np.linalg.norm(_np.imag(Lmx[i, i])) < IMAG_TOL)
            self.params[i * dmDim + i] = Lmx[i, i].real  # / paramNorm == 1 as asserted above
            for j in range(i):
                self.params[i * dmDim + j] = Lmx[i, j].real
                self.params[j * dmDim + i] = Lmx[i, j].imag

    def _construct_vector(self):
        dmDim = self.dmDim

        #  params is an array of length dmDim^2-1 that
        #  encodes a lower-triangular matrix "Lmx" via:
        #  Lmx[i,i] = params[i*dmDim + i] / param-norm  # i = 0...dmDim-2
        #     *last diagonal el is given by sqrt(1.0 - sum(L[i,j]**2))
        #  Lmx[i,j] = params[i*dmDim + j] + 1j*params[j*dmDim+i] (i > j)

        param2Sum = _np.vdot(self.params, self.params)  # or "dot" would work, since params are real
        paramNorm = _np.sqrt(param2Sum)  # also the norm of *all* Lmx els

        for i in range(dmDim):
            self.Lmx[i, i] = self.params[i * dmDim + i] / paramNorm
            for j in range(i):
                self.Lmx[i, j] = (self.params[i * dmDim + j] + 1j * self.params[j * dmDim + i]) / paramNorm

        Lmx_norm = _np.trace(_np.dot(self.Lmx.T.conjugate(), self.Lmx))  # sum of magnitude^2 of all els
        assert(_np.isclose(Lmx_norm, 1.0)), "Violated trace=1 condition!"

        #The (complex, Hermitian) density matrix is build by
        # assuming Lmx is its Cholesky decomp, which makes
        # the density matrix is pos-def.
        density_mx = _np.dot(self.Lmx, self.Lmx.T.conjugate())
        assert(_np.isclose(_np.trace(density_mx), 1.0)), "density matrix must be trace == 1"

        # write density matrix in given basis: = sum_i alpha_i B_i
        # ASSUME that basis is orthogonal, i.e. Tr(Bi^dag*Bj) = delta_ij
        basis_mxs = _np.rollaxis(self.basis_mxs, 2)  # shape (dmDim, dmDim, len(vec))
        vec = _np.array([_np.trace(_np.dot(M.T.conjugate(), density_mx)) for M in basis_mxs])

        #for now, assume Liouville vector should always be real (TODO: add 'real' flag later?)
        assert(_np.linalg.norm(_np.imag(vec)) < IMAG_TOL)
        vec = _np.real(vec)

        self.base1D.flags.writeable = True
        self.base1D[:] = vec[:]  # so shape is (dim,1) - the convention for spam vectors
        self.base1D.flags.writeable = False

    def set_value(self, vec):
        """
        Attempts to modify SPAMVec parameters so that the specified raw
        SPAM vector becomes vec.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        try:
            self._set_params_from_vector(vec, truncate=False)
            self.dirty = True
        except AssertionError as e:
            raise ValueError("Error initializing the parameters of this "
                             "CPTPSPAMVec object: " + str(e))

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        assert(self.dmDim**2 == self.dim)  # should at least be true without composite bases...
        return self.dmDim**2

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.params

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(len(v) == self.num_params())
        self.params[:] = v[:]
        self._construct_vector()
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dmDim = self.dmDim
        nP = len(self.params)
        assert(nP == dmDim**2)  # number of parameters

        # v_i = trace( B_i^dag * Lmx * Lmx^dag )
        # d(v_i) = trace( B_i^dag * (dLmx * Lmx^dag + Lmx * (dLmx)^dag) )  #trace = linear so commutes w/deriv
        #               /
        # where dLmx/d[ab] = {
        #               \
        L, Lbar = self.Lmx, self.Lmx.conjugate()
        F1 = _np.tril(_np.ones((dmDim, dmDim), 'd'))
        F2 = _np.triu(_np.ones((dmDim, dmDim), 'd'), 1) * 1j
        conj_basis_mxs = self.basis_mxs.conjugate()

        # Derivative of vector wrt params; shape == [vecLen,dmDim,dmDim] *not dealing with TP condition yet*
        # (first get derivative assuming last diagonal el of Lmx *is* a parameter, then use chain rule)
        dVdp = _np.einsum('aml,mb,ab->lab', conj_basis_mxs, Lbar, F1)  # only a >= b nonzero (F1)
        dVdp += _np.einsum('mal,mb,ab->lab', conj_basis_mxs, L, F1)    # ditto
        dVdp += _np.einsum('bml,ma,ab->lab', conj_basis_mxs, Lbar, F2)  # only b > a nonzero (F2)
        dVdp += _np.einsum('mbl,ma,ab->lab', conj_basis_mxs, L, F2.conjugate())  # ditto

        dVdp.shape = [dVdp.shape[0], nP]  # jacobian with respect to "p" params,
        # which don't include normalization for TP-constraint

        #Now get jacobian of actual params wrt the params used above. Denote the actual
        # params "P" in variable names, so p_ij = P_ij / sqrt(sum(P_xy**2))
        param2Sum = _np.vdot(self.params, self.params)
        paramNorm = _np.sqrt(param2Sum)  # norm of *all* Lmx els (note lastDiagEl
        dpdP = _np.identity(nP, 'd')

        # all p_ij params ==  P_ij / paramNorm = P_ij / sqrt(sum(P_xy**2))
        # and so have derivs wrt *all* Pxy elements.
        for ij in range(nP):
            for kl in range(nP):
                if ij == kl:
                    # dp_ij / dP_ij = 1.0 / (sum(P_xy**2))^(1/2) - 0.5 * P_ij / (sum(P_xy**2))^(3/2) * 2*P_ij
                    #               = 1.0 / (sum(P_xy**2))^(1/2) - P_ij^2 / (sum(P_xy**2))^(3/2)
                    dpdP[ij, ij] = 1.0 / paramNorm - self.params[ij]**2 / paramNorm**3
                else:
                    # dp_ij / dP_kl = -0.5 * P_ij / (sum(P_xy**2))^(3/2) * 2*P_kl
                    #               = - P_ij * P_kl / (sum(P_xy**2))^(3/2)
                    dpdP[ij, kl] = - self.params[ij] * self.params[kl] / paramNorm**3

        #Apply the chain rule to get dVdP:
        dVdP = _np.dot(dVdp, dpdP)  # shape (vecLen, nP) - the jacobian!
        dVdp = dpdP = None  # free memory!

        assert(_np.linalg.norm(_np.imag(dVdP)) < IMAG_TOL)
        derivMx = _np.real(dVdP)

        if wrtFilter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrtFilter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return True

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the vectors's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        raise NotImplementedError("TODO: add hessian computation for CPTPSPAMVec")


class TensorProdSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is a tensor-product of other SPAM vectors.
    """

    def __init__(self, typ, factors, povmEffectLbls=None):
        """
        Initialize a TensorProdSPAMVec object.

        Parameters
        ----------
        typ : {"prep","effect"}
            The type of spam vector - either a product of preparation
            vectors ("prep") or of POVM effect vectors ("effect")

        factors : list of SPAMVecs or POVMs
            if `typ == "prep"`, a list of the component SPAMVecs; if
            `typ == "effect"`, a list of "reference" POVMs into which
            `povmEffectLbls` indexes.

        povmEffectLbls : array-like
            Only non-None when `typ == "effect"`.  The effect label of each
            factor POVM which is tensored together to form this SPAM vector.
        """
        assert(len(factors) > 0), "Must have at least one factor!"

        self.factors = factors  # do *not* copy - needs to reference common objects
        self.Np = sum([fct.num_params() for fct in factors])
        if typ == "effect":
            self.effectLbls = _np.array(povmEffectLbls)
        elif typ == "prep":
            assert(povmEffectLbls is None), '`povmEffectLbls` must be None when `typ != "effects"`'
            self.effectLbls = None
        else: raise ValueError("Invalid `typ` argument: %s" % typ)

        dim = _np.product([fct.dim for fct in factors])
        evotype = self.factors[0]._evotype

        #Arrays for speeding up kron product in effect reps
        if evotype in ("statevec", "densitymx") and typ == "effect":  # types that require fast kronecker prods
            max_factor_dim = max(fct.dim for fct in factors)
            self._fast_kron_array = _np.ascontiguousarray(
                _np.empty((len(factors), max_factor_dim), complex if evotype == "statevec" else 'd'))
            self._fast_kron_factordims = _np.ascontiguousarray(
                _np.array([fct.dim for fct in factors], _np.int64))
        else:  # "stabilizer", "svterm", "cterm"
            self._fast_kron_array = None
            self._fast_kron_factordims = None

        #Create representation
        if evotype == "statevec":
            if typ == "prep":  # prep-type vectors can be represented as dense effects too
                rep = replib.SVStateRep(_np.ascontiguousarray(_np.zeros(dim, complex)))
                #sometimes: return replib.SVEffectRep_Dense(self.todense()) ???
            else:  # "effect"
                rep = replib.SVEffectRep_TensorProd(self._fast_kron_array, self._fast_kron_factordims,
                                                    len(self.factors), self._fast_kron_array.shape[1], dim)
        elif evotype == "densitymx":
            if typ == "prep":
                vec = _np.require(_np.zeros(dim, 'd'), requirements=['OWNDATA', 'C_CONTIGUOUS'])
                rep = replib.DMStateRep(vec)
                #sometimes: return replib.DMEffectRep_Dense(self.todense()) ???
            else:  # "effect"
                rep = replib.DMEffectRep_TensorProd(self._fast_kron_array, self._fast_kron_factordims,
                                                    len(self.factors), self._fast_kron_array.shape[1], dim)
        elif evotype == "stabilizer":
            if typ == "prep":
                #Rep is stabilizer-rep tuple, just like StabilizerSPAMVec
                sframe_factors = [f.todense() for f in self.factors]  # StabilizerFrame for each factor
                rep = _stabilizer.sframe_kronecker(sframe_factors).torep()

                #Sometimes ???
                # prep-type vectors can be represented as dense effects too; this just means that self.factors
                # => self.factors should all be StabilizerSPAMVec objs
                #else:  # self._prep_or_effect == "effect", so each factor is a StabilizerEffectVec
                #  outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in self.factors])), _np.int64)
                #  return replib.SBEffectRep(outcomes)

            else:  # self._prep_or_effect == "effect", so each factor is a StabilizerZPOVM
                # like above, but get a StabilizerEffectVec from each StabilizerZPOVM factor
                factorPOVMs = self.factors
                factorVecs = [factorPOVMs[i][self.effectLbls[i]] for i in range(1, len(factorPOVMs))]
                outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in factorVecs])), _np.int64)
                rep = replib.SBEffectRep(outcomes)

                #OLD - now can remove outcomes prop?
                #raise ValueError("Cannot convert Stabilizer tensor product effect to a representation!")
                # should be using effect.outcomes property...

        else:  # self._evotype in ("svterm","cterm")
            rep = dim  # no reps for term-based evotypes

        SPAMVec.__init__(self, rep, evotype, typ)
        self._update_rep()  # initializes rep data
        #sets gpindices, so do before stuff below

        if typ == "effect":
            #Set our parent and gpindices based on those of factor-POVMs, which
            # should all be owned by a TensorProdPOVM object.
            # (for now say we depend on *all* the POVMs parameters (even though
            #  we really only depend on one element of each POVM, which may allow
            #  using just a subset of each factor POVMs indices - but this is tricky).
            self.set_gpindices(_slct.list_to_slice(
                _np.concatenate([fct.gpindices_as_array()
                                 for fct in factors], axis=0), True, False),
                               factors[0].parent)  # use parent of first factor
            # (they should all be the same)
        else:
            # don't init our own gpindices (prep case), since our parent
            # is likely to be a Model and it will init them correctly.
            #But do set the indices of self.factors, since they're now
            # considered "owned" by this product-prep-vec (different from
            # the "effect" case when the factors are shared).
            off = 0
            for fct in factors:
                assert(isinstance(fct, SPAMVec)), "Factors must be SPAMVec objects!"
                N = fct.num_params()
                fct.set_gpindices(slice(off, off + N), self); off += N
            assert(off == self.Np)

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        if self._prep_or_effect == "prep":
            for i, factor_dim in enumerate(self._fast_kron_factordims):
                self._fast_kron_array[i][0:factor_dim] = self.factors[i].todense()
        else:
            factorPOVMs = self.factors
            for i, (factor_dim, Elbl) in enumerate(zip(self._fast_kron_factordims, self.effectLbls)):
                self._fast_kron_array[i][0:factor_dim] = factorPOVMs[i][Elbl].todense()

    def _update_rep(self):
        if self._evotype in ("statevec", "densitymx"):
            if self._prep_or_effect == "prep":
                self._rep.base[:] = self.todense()
            else:
                self._fill_fast_kron()  # updates effect reps
        elif self._evotype == "stabilizer":
            if self._prep_or_effect == "prep":
                #we need to update self._rep, which is a SBStateRep object.  For now, we
                # kinda punt and just create a new rep and copy its data over to the existing
                # rep instead of having some kind of update method within SBStateRep...
                # (TODO FUTURE - at least a .copy_from method?)
                sframe_factors = [f.todense() for f in self.factors]  # StabilizerFrame for each factor
                new_rep = _stabilizer.sframe_kronecker(sframe_factors).torep()
                self._rep.smatrix[:, :] = new_rep.smatrix[:, :]
                self._rep.pvectors[:, :] = new_rep.pvectors[:, :]
                self._rep.amps[:, :] = new_rep.amps[:, :]
            else:
                pass  # I think the below (e.g. 'outcomes') is not altered by any parameters
                #factorPOVMs = self.factors
                #factorVecs = [factorPOVMs[i][self.effectLbls[i]] for i in range(1, len(factorPOVMs))]
                #outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in factorVecs])), _np.int64)
                #rep = replib.SBEffectRep(outcomes)

    def todense(self):
        """
        Return this SPAM vector as a (dense) numpy array.
        """
        if self._evotype in ("statevec", "densitymx"):
            if len(self.factors) == 0: return _np.empty(0, complex if self._evotype == "statevec" else 'd')
            #NOTE: moved a fast version of todense to replib - could use that if need a fast todense call...

            if self._prep_or_effect == "prep":
                ret = self.factors[0].todense()  # factors are just other SPAMVecs
                for i in range(1, len(self.factors)):
                    ret = _np.kron(ret, self.factors[i].todense())
            else:
                factorPOVMs = self.factors
                ret = factorPOVMs[0][self.effectLbls[0]].todense()
                for i in range(1, len(factorPOVMs)):
                    ret = _np.kron(ret, factorPOVMs[i][self.effectLbls[i]].todense())

            return ret
        elif self._evotype == "stabilizer":

            if self._prep_or_effect == "prep":
                # => self.factors should all be StabilizerSPAMVec objs
                #Return stabilizer-rep tuple, just like StabilizerSPAMVec
                sframe_factors = [f.todense() for f in self.factors]
                return _stabilizer.sframe_kronecker(sframe_factors)

            else:  # self._prep_or_effect == "effect", so each factor is a StabilizerEffectVec
                raise ValueError("Cannot convert Stabilizer tensor product effect to an array!")
                # should be using effect.outcomes property...
        else:  # self._evotype in ("svterm","cterm")
            raise NotImplementedError("todense() not implemented for %s evolution type" % self._evotype)

    #def torep(self, typ, outrep=None):
    #    """
    #    Return a "representation" object for this SPAM vector.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Parameters
    #    ----------
    #    typ : {'prep','effect'}
    #        The type of representation (for cases when the vector type is
    #        not already defined).
    #
    #    outrep : StateRep
    #        If not None, an existing state representation appropriate to this
    #        SPAM vector that may be used instead of allocating a new one.
    #
    #    Returns
    #    -------
    #    StateRep
    #    """
    #    assert(len(self.factors) > 0), "Cannot get representation of a TensorProdSPAMVec with no factors!"
    #    assert(self._prep_or_effect in ('prep', 'effect')), "Invalid internal type: %s!" % self._prep_or_effect
    #
    #    #FUTURE: use outrep as scratch for rep constructor?
    #    if self._evotype == "statevec":
    #        if self._prep_or_effect == "prep":  # prep-type vectors can be represented as dense effects too
    #            if typ == "prep":
    #                return replib.SVStateRep(self.todense())
    #            else:
    #                return replib.SVEffectRep_Dense(self.todense())
    #        else:
    #            return replib.SVEffectRep_TensorProd(self._fast_kron_array, self._fast_kron_factordims,
    #                                                 len(self.factors), self._fast_kron_array.shape[1], self.dim)
    #    elif self._evotype == "densitymx":
    #        if self._prep_or_effect == "prep":
    #            if typ == "prep":
    #                return replib.DMStateRep(self.todense())
    #            else:
    #                return replib.DMEffectRep_Dense(self.todense())
    #
    #        else:
    #            return replib.DMEffectRep_TensorProd(self._fast_kron_array, self._fast_kron_factordims,
    #                                                 len(self.factors), self._fast_kron_array.shape[1], self.dim)
    #
    #    elif self._evotype == "stabilizer":
    #        if self._prep_or_effect == "prep":
    #            # prep-type vectors can be represented as dense effects too; this just means that self.factors
    #            if typ == "prep":
    #                # => self.factors should all be StabilizerSPAMVec objs
    #                #Return stabilizer-rep tuple, just like StabilizerSPAMVec
    #                sframe_factors = [f.todense() for f in self.factors]  # StabilizerFrame for each factor
    #                return _stabilizer.sframe_kronecker(sframe_factors).torep()
    #            else:  # self._prep_or_effect == "effect", so each factor is a StabilizerEffectVec
    #                outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in self.factors])), _np.int64)
    #                return replib.SBEffectRep(outcomes)
    #
    #        else:  # self._prep_or_effect == "effect", so each factor is a StabilizerZPOVM
    #            # like above, but get a StabilizerEffectVec from each StabilizerZPOVM factor
    #            factorPOVMs = self.factors
    #            factorVecs = [factorPOVMs[i][self.effectLbls[i]] for i in range(1, len(factorPOVMs))]
    #            outcomes = _np.array(list(_itertools.chain(*[f.outcomes for f in factorVecs])), _np.int64)
    #            return replib.SBEffectRep(outcomes)
    #
    #            #OLD - now can remove outcomes prop?
    #            #raise ValueError("Cannot convert Stabilizer tensor product effect to a representation!")
    #            # should be using effect.outcomes property...
    #
    #    else:  # self._evotype in ("svterm","cterm")
    #        raise NotImplementedError("torep() not implemented for %s evolution type" % self._evotype)

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

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
            output of :method:`Polynomial.compact`.
        """
        from .operation import EmbeddedOp as _EmbeddedGateMap
        terms = []
        fnq = [int(round(_np.log2(f.dim))) // 2 for f in self.factors]  # num of qubits per factor
        # assumes density matrix evolution
        total_nQ = sum(fnq)  # total number of qubits

        for p in _lt.partition_into(order, len(self.factors)):
            if self._prep_or_effect == "prep":
                factor_lists = [self.factors[i].get_taylor_order_terms(pi, max_poly_vars) for i, pi in enumerate(p)]
            else:
                factorPOVMs = self.factors
                factor_lists = [factorPOVMs[i][Elbl].get_taylor_order_terms(pi, max_poly_vars)
                                for i, (pi, Elbl) in enumerate(zip(p, self.effectLbls))]

            # When possible, create COLLAPSED factor_lists so each factor has just a single
            # (SPAMVec) pre & post op, which can be formed into the new terms'
            # TensorProdSPAMVec ops.
            # - DON'T collapse stabilizer states & clifford ops - can't for POVMs
            collapsible = False  # bool(self._evotype =="svterm") # need to use reps for collapsing now... TODO?

            if collapsible:
                factor_lists = [[t.collapse_vec() for t in fterms] for fterms in factor_lists]

            for factors in _itertools.product(*factor_lists):
                # create a term with a TensorProdSPAMVec - Note we always create
                # "prep"-mode vectors, since even when self._prep_or_effect == "effect" these
                # vectors are created with factor (prep- or effect-type) SPAMVecs not factor POVMs
                # we workaround this by still allowing such "prep"-mode
                # TensorProdSPAMVecs to be represented as effects (i.e. in torep('effect'...) works)
                coeff = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
                pre_op = TensorProdSPAMVec("prep", [f.pre_ops[0] for f in factors
                                                    if (f.pre_ops[0] is not None)])
                post_op = TensorProdSPAMVec("prep", [f.post_ops[0] for f in factors
                                                     if (f.post_ops[0] is not None)])
                term = _term.RankOnePolyPrepTerm.simple_init(coeff, pre_op, post_op, self._evotype)

                if not collapsible:  # then may need to add more ops.  Assume factor ops are clifford gates
                    # Embed each factors ops according to their target qubit(s) and just daisy chain them
                    stateSpaceLabels = tuple(range(total_nQ)); curQ = 0
                    for f, nq in zip(factors, fnq):
                        targetLabels = tuple(range(curQ, curQ + nq)); curQ += nq
                        term.pre_ops.extend([_EmbeddedGateMap(stateSpaceLabels, targetLabels, op)
                                             for op in f.pre_ops[1:]])  # embed and add ops
                        term.post_ops.extend([_EmbeddedGateMap(stateSpaceLabels, targetLabels, op)
                                              for op in f.post_ops[1:]])  # embed and add ops

                terms.append(term)

        if return_coeff_polys:
            def _decompose_indices(x):
                return tuple(_modelmember._decompose_gpindices(
                    self.gpindices, _np.array(x, _np.int64)))

            poly_coeffs = [t.coeff.map_indices(_decompose_indices) for t in terms]  # with *local* indices
            tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
            if len(tapes) > 0:
                vtape = _np.concatenate([t[0] for t in tapes])
                ctape = _np.concatenate([t[1] for t in tapes])
            else:
                vtape = _np.empty(0, _np.int64)
                ctape = _np.empty(0, complex)
            coeffs_as_compact_polys = (vtape, ctape)
            #self.local_term_poly_coeffs[order] = coeffs_as_compact_polys #FUTURE?
            return terms, coeffs_as_compact_polys
        else:
            return terms  # Cache terms in FUTURE?

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.Np

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        if self._prep_or_effect == "prep":
            return _np.concatenate([fct.to_vector() for fct in self.factors], axis=0)
        else:
            raise ValueError(("'`to_vector` should not be called on effect-like"
                              " TensorProdSPAMVecs (instead it should be called"
                              " on the POVM)"))

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        if self._prep_or_effect == "prep":
            for sv in self.factors:
                sv.from_vector(v[sv.gpindices], close, nodirty)  # factors hold local indices

        elif all([self.effectLbls[i] == list(povm.keys())[0]
                  for i, povm in enumerate(self.factors)]):
            #then this is the *first* vector in the larger TensorProdPOVM
            # and we should initialize all of the factorPOVMs
            for povm in self.factors:
                local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, povm.gpindices)
                povm.from_vector(v[local_inds], close, nodirty)

        #Update representation, which may be a dense matrix or
        # just fast-kron arrays or a stabilizer state.
        self._update_rep()

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        assert(self._evotype in ("statevec", "densitymx"))
        typ = complex if self._evotype == "statevec" else 'd'
        derivMx = _np.zeros((self.dim, self.num_params()), typ)

        #Product rule to compute jacobian
        for i, fct in enumerate(self.factors):  # loop over the spamvec/povm we differentiate wrt
            vec = fct if (self._prep_or_effect == "prep") else fct[self.effectLbls[i]]

            if vec.num_params() == 0: continue  # no contribution
            deriv = vec.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this gate...
            deriv.shape = (vec.dim, vec.num_params())

            if i > 0:  # factors before ith
                if self._prep_or_effect == "prep":
                    pre = self.factors[0].todense()
                    for vecA in self.factors[1:i]:
                        pre = _np.kron(pre, vecA.todense())
                else:
                    pre = self.factors[0][self.effectLbls[0]].todense()
                    for j, fctA in enumerate(self.factors[1:i], start=1):
                        pre = _np.kron(pre, fctA[self.effectLbls[j]].todense())
                deriv = _np.kron(pre[:, None], deriv)  # add a dummy 1-dim to 'pre' and do kron properly...

            if i + 1 < len(self.factors):  # factors after ith
                if self._prep_or_effect == "prep":
                    post = self.factors[i + 1].todense()
                    for vecA in self.factors[i + 2:]:
                        post = _np.kron(post, vecA.todense())
                else:
                    post = self.factors[i + 1][self.effectLbls[i + 1]].todense()
                    for j, fctA in enumerate(self.factors[i + 2:], start=i + 2):
                        post = _np.kron(post, fctA[self.effectLbls[j]].todense())
                deriv = _np.kron(deriv, post[:, None])  # add a dummy 1-dim to 'post' and do kron properly...

            if self._prep_or_effect == "prep":
                local_inds = fct.gpindices  # factor vectors hold local indices
            else:  # in effect case, POVM-factors hold global indices (b/c they're meant to be shareable)
                local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, fct.gpindices)

            assert(local_inds is not None), \
                "Error: gpindices has not been initialized for factor %d - cannot compute derivative!" % i
            derivMx[:, local_inds] += deriv

        derivMx.shape = (self.dim, self.num_params())  # necessary?
        if wrtFilter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrtFilter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False

    def __str__(self):
        s = "Tensor product %s vector with length %d\n" % (self._prep_or_effect, self.dim)
        #ar = self.todense()
        #s += _mt.mx_to_string(ar, width=4, prec=2)

        if self._prep_or_effect == "prep":
            # factors are just other SPAMVecs
            s += " x ".join([_mt.mx_to_string(fct.todense(), width=4, prec=2) for fct in self.factors])
        else:
            # factors are POVMs
            s += " x ".join([_mt.mx_to_string(fct[self.effectLbls[i]].todense(), width=4, prec=2)
                             for i, fct in enumerate(self.factors)])
        return s


class PureStateSPAMVec(SPAMVec):
    """
    Encapsulates a SPAM vector that is a pure state but evolves according to
    one of the density matrix evolution types ("denstiymx", "svterm", and
    "cterm").  It is parameterized by a contained pure-state SPAMVec which
    evolves according to a state vector evolution type ("statevec" or
    "stabilizer").
    """

    def __init__(self, pure_state_vec, evotype='densitymx', dm_basis='pp', typ="prep"):
        """
        Initialize a PureStateSPAMVec object.

        Parameters
        ----------
        pure_state_vec : array_like or SPAMVec
            a 1D numpy array or object representing the pure state.  This object
            sets the parameterization and dimension of this SPAM vector (if
            `pure_state_vec`'s dimension is `d`, then this SPAM vector's
            dimension is `d^2`).  Assumed to be a complex vector in the
            standard computational basis.

        evotype : {'densitymx', 'svterm', 'cterm'}
            The evolution type of this SPAMVec.  Note that the evotype of
            `pure_state_vec` must be compatible with this value.  In particular,
            `pure_state_vec` must have an evotype of `"statevec"` (then allowed
            values are `"densitymx"` and `"svterm"`) or `"stabilizer"` (then
            the only allowed value is `"cterm"`).

        dm_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this SPAM vector - that is, for the *density matrix*
            corresponding to `pure_state_vec`.  Allowed values are Matrix-unit
            (std),  Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
            (or a custom basis object).
        """
        if not isinstance(pure_state_vec, SPAMVec):
            pure_state_vec = StaticSPAMVec(SPAMVec.convert_to_vector(pure_state_vec), 'statevec')
        self.pure_state_vec = pure_state_vec
        self.basis = dm_basis  # only used for dense conversion

        pure_evo = pure_state_vec._evotype
        if pure_evo == "statevec":
            if evotype not in ("densitymx", "svterm"):
                raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
                                  " when `pure_state_vec` evotype is 'statevec'"))
        elif pure_evo == "stabilizer":
            if evotype not in ("cterm",):
                raise ValueError(("`evotype` arg must be 'densitymx' or 'svterm'"
                                  " when `pure_state_vec` evotype is 'statevec'"))
        else:
            raise ValueError("`pure_state_vec` evotype must be 'statevec' or 'stabilizer' (not '%s')" % pure_evo)

        dim = self.pure_state_vec.dim**2
        rep = dim  # no representation yet... maybe this should be a dense vector
        # (in "densitymx" evotype case -- use todense)? TODO

        #Create representation
        SPAMVec.__init__(self, rep, evotype, typ)

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        dmVec_std = _gt.state_to_dmvec(self.pure_state_vec.todense())
        return _bt.change_basis(dmVec_std, 'std', self.basis)

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

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
            output of :method:`Polynomial.compact`.
        """
        if self.num_params() > 0:
            raise ValueError(("PureStateSPAMVec.get_taylor_order_terms(...) is only "
                              "implemented for the case when its underlying "
                              "pure state vector has 0 parameters (is static)"))

        if order == 0:  # only 0-th order term exists (assumes static pure_state_vec)
            purevec = self.pure_state_vec
            coeff = _Polynomial({(): 1.0}, max_poly_vars)
            if self._prep_or_effect == "prep":
                terms = [_term.RankOnePolyPrepTerm.simple_init(coeff, purevec, purevec, self._evotype)]
            else:
                terms = [_term.RankOnePolyEffectTerm.simple_init(coeff, purevec, purevec, self._evotype)]

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

    #TODO REMOVE
    #def get_direct_order_terms(self, order, base_order):
    #    """
    #    Parameters
    #    ----------
    #    order : int
    #        The order of terms to get.
    #
    #    Returns
    #    -------
    #    list
    #        A list of :class:`RankOneTerm` objects.
    #    """
    #    if self.num_params() > 0:
    #        raise ValueError(("PureStateSPAMVec.get_taylor_order_terms(...) is only "
    #                          "implemented for the case when its underlying "
    #                          "pure state vector has 0 parameters (is static)"))
    #
    #    if order == 0: # only 0-th order term exists (assumes static pure_state_vec)
    #        if self._evotype == "svterm":  tt = "dense"
    #        elif self._evotype == "cterm": tt = "clifford"
    #        else: raise ValueError("Invalid evolution type %s for calling `get_taylor_order_terms`" % self._evotype)
    #
    #        purevec = self.pure_state_vec
    #        terms = [ _term.RankOneTerm(1.0, purevec, purevec, tt) ]
    #    else:
    #        terms = []
    #    return terms

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.pure_state_vec.num_params()

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.pure_state_vec.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        self.pure_state_vec.from_vector(v, close, nodirty)
        #Update dense rep if one is created (TODO)

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.
        An empty 2D array in the StaticSPAMVec case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        raise NotImplementedError("Still need to work out derivative calculation of PureStateSPAMVec")

    def has_nonzero_hessian(self):
        """
        Returns whether this SPAM vector has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return self.pure_state_vec.has_nonzero_hessian()

    def __str__(self):
        s = "Pure-state spam vector with length %d holding:\n" % self.dim
        s += "  " + str(self.pure_state_vec)
        return s


class LindbladSPAMVec(SPAMVec):
    """ A Lindblad-parameterized SPAMVec (that is also expandable into terms)"""

    @classmethod
    def from_spamvec_obj(cls, spamvec, typ, paramType="GLND", purevec=None,
                         proj_basis="pp", mxBasis="pp", truncate=True,
                         lazy=False):
        """
        Creates a LindbladSPAMVec from an existing SPAMVec object
        and some additional information.

        This function is different from `from_spam_vector` in that it assumes
        that `spamvec` is a :class:`SPAMVec`-derived object, and if `lazy=True`
        and if `spamvec` is already a matching LindbladSPAMVec, it
        is returned directly.  This routine is primarily used in spam vector
        conversion functions, where conversion is desired only when necessary.

        Parameters
        ----------
        spamvec : SPAMVec
            The spam vector object to "convert" to a
            `LindbladSPAMVec`.

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

        paramType : str, optional
            The high-level "parameter type" of the gate to create.  This
            specifies both which Lindblad parameters are included and what
            type of evolution is used.  Examples of valid values are
            `"CPTP"`, `"H+S"`, `"S terms"`, and `"GLND clifford terms"`.

        purevec : numpy array or SPAMVec object, optional
            A SPAM vector which represents a pure-state, taken as the "ideal"
            reference state when constructing the error generator of the
            returned `LindbladSPAMVec`.  Note that this vector
            still acts on density matrices (if it's a SPAMVec it should have
            a "densitymx", "svterm", or "cterm" evolution type, and if it's
            a numpy array it should have the same dimension as `spamvec`).
            If None, then it is taken to be `spamvec`, and so `spamvec` must
            represent a pure state in this case.

        proj_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct the Lindblad-term error generators onto
            which the SPAM vector's error generator is projected.  Allowed values
            are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `spamvec` cannot
            be realized by the specified set of Lindblad projections.

        lazy : bool, optional
            If True, then if `spamvec` is already a LindbladSPAMVec
            with the requested details (given by the other arguments), then
            `spamvec` is returned directly and no conversion/copying is
            performed. If False, then a new object is always returned.

        Returns
        -------
        LindbladSPAMVec
        """

        if not isinstance(spamvec, SPAMVec):
            spamvec = StaticSPAMVec(spamvec, typ=typ)  # assume spamvec is just a vector

        if purevec is None:
            purevec = spamvec  # right now, we don't try to extract a "closest pure vec"
            # to spamvec - below will fail if spamvec isn't pure.
        elif not isinstance(purevec, SPAMVec):
            purevec = StaticSPAMVec(purevec, typ=typ)  # assume spamvec is just a vector

        #Break paramType in to a "base" type and an evotype
        from .operation import LindbladOp as _LPGMap
        bTyp, evotype, nonham_mode, param_mode = _LPGMap.decomp_paramtype(paramType)

        ham_basis = proj_basis if (("H+" in bTyp) or bTyp in ("CPTP", "GLND")) else None
        nonham_basis = proj_basis

        def beq(b1, b2):
            """ Check if bases have equal names """
            b1 = b1.name if isinstance(b1, _Basis) else b1
            b2 = b2.name if isinstance(b2, _Basis) else b2
            return b1 == b2

        def normeq(a, b):
            if a is None and b is None: return True
            if a is None or b is None: return False
            return _mt.safenorm(a - b) < 1e-6  # what about possibility of Clifford gates?

        if isinstance(spamvec, LindbladSPAMVec) \
           and spamvec._evotype == evotype and spamvec.typ == typ \
           and beq(ham_basis, spamvec.error_map.ham_basis) and beq(nonham_basis, spamvec.error_map.other_basis) \
           and param_mode == spamvec.error_map.param_mode and nonham_mode == spamvec.error_map.nonham_mode \
           and beq(mxBasis, spamvec.error_map.matrix_basis) and lazy:
            #normeq(gate.pure_state_vec,purevec) \ # TODO: more checks for equality?!
            return spamvec  # no creation necessary!
        else:
            #Convert vectors (if possible) to SPAMVecs
            # of the appropriate evotype and 0 params.
            bDiff = spamvec is not purevec
            spamvec = _convert_to_lindblad_base(spamvec, typ, evotype, mxBasis)
            purevec = _convert_to_lindblad_base(purevec, typ, evotype, mxBasis) if bDiff else spamvec
            assert(spamvec._evotype == evotype)
            assert(purevec._evotype == evotype)

            return cls.from_spam_vector(
                spamvec, purevec, typ, ham_basis, nonham_basis,
                param_mode, nonham_mode, truncate, mxBasis, evotype)

    @classmethod
    def from_spam_vector(cls, spamVec, pureVec, typ,
                         ham_basis="pp", nonham_basis="pp", param_mode="cptp",
                         nonham_mode="all", truncate=True, mxBasis="pp",
                         evotype="densitymx"):
        """
        Creates a Lindblad-parameterized spamvec from a state vector and a basis
        which specifies how to decompose (project) the vector's error generator.

        spamVec : SPAMVec
            the SPAM vector to initialize from.  The error generator that
            tranforms `pureVec` into `spamVec` forms the parameterization
            of the returned LindbladSPAMVec.

        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have the same dimension as `spamVec` - 4 in the case
            of a single qubit) which represents a pure-state preparation or
            projection.  This is used as the "base" preparation/projection
            when computing the error generator that will be parameterized.
            Note that this argument must be specified, as there is no natural
            default value (like the identity in the case of gates).

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        nonham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            SPAM vector's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `gate` cannot
            be realized by the specified set of Lindblad projections.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the spamvec being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (spamvec is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but stabilizer states.

        Returns
        -------
        LindbladSPAMVec
        """
        #Compute a (errgen, pureVec) pair from the given
        # (spamVec, pureVec) pair.

        assert(pureVec is not None), "Must supply `pureVec`!"  # since there's no good default?

        if not isinstance(spamVec, SPAMVec):
            spamVec = StaticSPAMVec(spamVec, evotype, typ)  # assume spamvec is just a vector
        if not isinstance(pureVec, SPAMVec):
            pureVec = StaticSPAMVec(pureVec, evotype, typ)  # assume spamvec is just a vector
        d2 = pureVec.dim

        #Determine whether we're using sparse bases or not
        sparse = None
        if ham_basis is not None:
            if isinstance(ham_basis, _Basis): sparse = ham_basis.sparse
            elif not isinstance(ham_basis, str) and len(ham_basis) > 0:
                sparse = _sps.issparse(ham_basis[0])
        if sparse is None and nonham_basis is not None:
            if isinstance(nonham_basis, _Basis): sparse = nonham_basis.sparse
            elif not isinstance(nonham_basis, str) and len(nonham_basis) > 0:
                sparse = _sps.issparse(nonham_basis[0])
        if sparse is None: sparse = False  # the default

        if spamVec is None or spamVec is pureVec:
            if sparse: errgen = _sps.csr_matrix((d2, d2), dtype='d')
            else: errgen = _np.zeros((d2, d2), 'd')
        else:
            #Construct "spam error generator" by comparing *dense* vectors
            pvdense = pureVec.todense()
            svdense = spamVec.todense()
            errgen = _gt.spam_error_generator(svdense, pvdense, mxBasis)
            if sparse: errgen = _sps.csr_matrix(errgen)

        assert(pureVec._evotype == evotype), "`pureVec` must have evotype == '%s'" % evotype

        from .operation import LindbladErrorgen as _LErrorgen
        from .operation import LindbladOp as _LPGMap
        from .operation import LindbladDenseOp as _LPOp

        errgen = _LErrorgen.from_error_generator(errgen, ham_basis,
                                                 nonham_basis, param_mode, nonham_mode,
                                                 mxBasis, truncate, evotype)
        errcls = _LPOp if (pureVec.dim <= 64 and evotype == "densitymx") else _LPGMap
        errmap = errcls(None, errgen)

        return cls(pureVec, errmap, typ)

    @classmethod
    def from_lindblad_terms(cls, pureVec, Ltermdict, typ, basisdict=None,
                            param_mode="cptp", nonham_mode="all", truncate=True,
                            mxBasis="pp", evotype="densitymx"):
        """
        Create a Lindblad-parameterized spamvec with a given set of Lindblad terms.

        Parameters
        ----------
        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have dimension 4 in the case of a single qubit) which
            represents a pure-state preparation or projection.  This is used as
            the "base" preparation or projection that is followed or preceded
            by, respectively, the parameterized Lindblad-form error generator.

        Ltermdict : dict
            A dictionary specifying which Linblad terms are present in the gate
            parameteriztion.  Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
            (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
            have a single basis label (so key is a 2-tuple) whereas Stochastic
            tuples with 1 basis label indicate a *diagonal* term, and are the
            only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
            Stochastic term tuples can include 2 basis labels to specify
            "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
            strings or integers.  Values are complex coefficients (error rates).

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.

        basisdict : dict, optional
            A dictionary mapping the basis labels (strings or ints) used in the
            keys of `Ltermdict` to basis matrices (numpy arrays or Scipy sparse
            matrices).

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            SPAM vector's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given dictionary of
            Lindblad terms doesn't conform to the constrains.

        mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the spamvec being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (spamvec is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but stabilizer states.

        Returns
        -------
        LindbladOp
        """
        #Need a dimension for error map construction (basisdict could be completely empty)
        if not isinstance(pureVec, SPAMVec):
            pureVec = StaticSPAMVec(pureVec, evotype, typ)  # assume spamvec is just a vector
        d2 = pureVec.dim

        from .operation import LindbladOp as _LPGMap
        errmap = _LPGMap(d2, Ltermdict, basisdict, param_mode, nonham_mode,
                         truncate, mxBasis, evotype)
        return cls(pureVec, errmap, typ)

    def __init__(self, pureVec, errormap, typ):
        """
        Initialize a LindbladSPAMVec object.

        Essentially a pure state preparation or projection that is followed
        or preceded by, respectively, the action of LindbladDenseOp.

        Parameters
        ----------
        pureVec : numpy array or SPAMVec
            An array or SPAMVec in the *full* density-matrix space (this
            vector will have dimension 4 in the case of a single qubit) which
            represents a pure-state preparation or projection.  This is used as
            the "base" preparation or projection that is followed or preceded
            by, respectively, the parameterized Lindblad-form error generator.
            (This argument is *not* copied if it is a SPAMVec.  A numpy array
             is converted to a new StaticSPAMVec.)

        errormap : MapOperator
            The error generator action and parameterization, encapsulated in
            a gate object.  Usually a :class:`LindbladOp`
            or :class:`ComposedOp` object.  (This argument is *not* copied,
            to allow LindbladSPAMVecs to share error generator
            parameters with other gates and spam vectors.)

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.
        """
        from .operation import LindbladOp as _LPGMap
        evotype = errormap._evotype
        assert(evotype in ("densitymx", "svterm", "cterm")), \
            "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        if not isinstance(pureVec, SPAMVec):
            pureVec = StaticSPAMVec(pureVec, evotype, typ)  # assume spamvec is just a vector

        assert(pureVec._evotype == evotype), \
            "`pureVec` evotype must match `errormap` ('%s' != '%s')" % (pureVec._evotype, evotype)
        assert(pureVec.num_params() == 0), "`pureVec` 'reference' must have *zero* parameters!"

        d2 = pureVec.dim
        self.state_vec = pureVec
        self.error_map = errormap
        self.terms = {} if evotype in ("svterm", "cterm") else None
        self.local_term_poly_coeffs = {} if evotype in ("svterm", "cterm") else None
        # TODO REMOVE self.direct_terms = {} if evotype in ("svterm","cterm") else None
        # TODO REMOVE self.direct_term_poly_coeffs = {} if evotype in ("svterm","cterm") else None

        #Create representation
        if evotype == "densitymx":
            assert(self.state_vec._prep_or_effect == typ), "LindbladSPAMVec prep/effect mismatch with given statevec!"

            if typ == "prep":
                dmRep = self.state_vec._rep
                errmapRep = self.error_map._rep
                rep = errmapRep.acton(dmRep)  # FUTURE: do this acton in place somehow? (like C-reps do)
                #maybe make a special _Errgen *state* rep?

            else:  # effect
                dmEffectRep = self.state_vec._rep
                errmapRep = self.error_map._rep
                rep = replib.DMEffectRep_Errgen(errmapRep, dmEffectRep, id(self.error_map))
                # an effect that applies a *named* errormap before computing with dmEffectRep

        else:
            rep = d2  # no representations for term-based evotypes

        SPAMVec.__init__(self, rep, evotype, typ)  # sets self.dim

    def _update_rep(self):
        if self._evotype == "densitymx":
            if self._prep_or_effect == "prep":
                # _rep is a DMStateRep
                dmRep = self.state_vec._rep
                errmapRep = self.error_map._rep
                self._rep.base[:] = errmapRep.acton(dmRep).base[:]  # copy from "new_rep"

            else:  # effect
                # _rep is a DMEffectRep_Errgen, which just holds references to the
                # effect and error map's representations (which we assume have been updated)
                # so there's no need to update anything here
                pass

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.error_map]

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that embedded gate has its
        # parent reset correctly.
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.state_vec, self.error_map.copy(parent), self._prep_or_effect)
        return self._copy_gpindices(copyOfMe, parent)

    def set_gpindices(self, gpindices, parent, memo=None):
        """
        Set the parent and indices into the parent's parameter vector that
        are used by this ModelMember object.

        Parameters
        ----------
        gpindices : slice or integer ndarray
            The indices of this objects parameters in its parent's array.

        parent : Model or ModelMember
            The parent whose parameter array gpindices references.

        Returns
        -------
        None
        """
        self.terms = {}  # clear terms cache since param indices have changed now
        self.local_term_poly_coeffs = {}
        # TODO REMOVE self.direct_terms = {}
        # TODO REMOVE self.direct_term_poly_coeffs = {}
        _modelmember.ModelMember.set_gpindices(self, gpindices, parent, memo)

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if self._prep_or_effect == "prep":
            #error map acts on dmVec
            return _np.dot(self.error_map.todense(), self.state_vec.todense())
        else:
            #Note: if this is an effect vector, self.error_map is the
            # map that acts on the *state* vector before dmVec acts
            # as an effect:  E.T -> dot(E.T,errmap) ==> E -> dot(errmap.T,E)
            return _np.dot(self.error_map.todense().conjugate().T, self.state_vec.todense())

    #def torep(self, typ, outvec=None):
    #    """
    #    Return a "representation" object for this SPAMVec.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    StateRep
    #    """
    #    if self._evotype == "densitymx":
    #
    #        if typ == "prep":
    #            dmRep = self.state_vec.torep(typ)
    #            errmapRep = self.error_map.torep()
    #            return errmapRep.acton(dmRep)  # FUTURE: do this acton in place somehow? (like C-reps do)
    #            #maybe make a special _Errgen *state* rep?
    #
    #        else:  # effect
    #            dmEffectRep = self.state_vec.torep(typ)
    #            errmapRep = self.error_map.torep()
    #            return replib.DMEffectRep_Errgen(errmapRep, dmEffectRep, id(self.error_map))
    #            # an effect that applies a *named* errormap before computing with dmEffectRep
    #
    #    else:
    #        #framework should not be calling "torep" on states w/a term-based evotype...
    #        # they should call torep on the *terms* given by get_taylor_order_terms(...)
    #        raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
    #                         (self._evotype, self.__class__.__name__))

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

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
            output of :method:`Polynomial.compact`.
        """
        if order not in self.terms:
            if self._evotype not in ('svterm', 'cterm'):
                raise ValueError("Invalid evolution type %s for calling `get_taylor_order_terms`" % self._evotype)
            assert(self.gpindices is not None), "LindbladSPAMVec must be added to a Model before use!"

            state_terms = self.state_vec.get_taylor_order_terms(0, max_poly_vars); assert(len(state_terms) == 1)
            stateTerm = state_terms[0]
            err_terms, cpolys = self.error_map.get_taylor_order_terms(order, max_poly_vars, True)
            if self._prep_or_effect == "prep":
                terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
            else:  # "effect"
                # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
                # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
                # too - the difference comes when this SPAMVec is used as an effect rather than a prep.
                terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's

            #OLD: now this is done within calculator when possible b/c not all terms can be collapsed
            #terms = [ t.collapse() for t in terms ] # collapse terms for speed
            # - resulting in terms with just a single pre/post op, each == a pure state

            #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen
            self.local_term_poly_coeffs[order] = cpolys
            self.terms[order] = terms

        if return_coeff_polys:
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def get_taylor_order_terms_above_mag(self, order, max_poly_vars, min_term_mag):
        state_terms = self.state_vec.get_taylor_order_terms(0, max_poly_vars); assert(len(state_terms) == 1)
        stateTerm = state_terms[0]
        stateTerm = stateTerm.copy_with_magnitude(1.0)
        #assert(stateTerm.coeff == Polynomial_1.0) # TODO... so can assume local polys are same as for errorgen

        err_terms = self.error_map.get_taylor_order_terms_above_mag(
            order, max_poly_vars, min_term_mag / stateTerm.magnitude)

        #This gives the appropriate logic, but *both* prep or effect results in *same* expression, so just run it:
        #if self._prep_or_effect == "prep":
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        #else:  # "effect"
        #    # Effect terms are special in that all their pre/post ops act in order on the *state* before the final
        #    # effect is used to compute a probability.  Thus, constructing the same "terms" as above works here
        #    # too - the difference comes when this SPAMVec is used as an effect rather than a prep.
        #    terms = [_term.compose_terms((stateTerm, t)) for t in err_terms]  # t ops occur *after* stateTerm's
        terms = [_term.compose_terms_with_mag((stateTerm, t), stateTerm.magnitude * t.magnitude)
                 for t in err_terms]  # t ops occur *after* stateTerm's
        return terms

    def get_total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this SPAM vector's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this SPAM vector in a Taylor series.

        Returns
        -------
        float
        """
        # return (sum of absvals of *all* term coeffs)
        return self.error_map.get_total_term_magnitude()  # error map is only part with terms

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        return self.error_map.get_total_term_magnitude_deriv()

    def deriv_wrt_params(self, wrtFilter=None):
        """
        Construct a matrix whose columns are the derivatives of the SPAM vector
        with respect to a single param.  Thus, each column is of length
        get_dimension and there is one column per SPAM vector parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dmVec = self.state_vec.todense()

        derrgen = self.error_map.deriv_wrt_params(wrtFilter)  # shape (dim*dim, nParams)
        derrgen.shape = (self.dim, self.dim, derrgen.shape[1])  # => (dim,dim,nParams)

        if self._prep_or_effect == "prep":
            #derror map acts on dmVec
            #return _np.einsum("ijk,j->ik", derrgen, dmVec) # return shape = (dim,nParams)
            return _np.tensordot(derrgen, dmVec, (1, 0))  # return shape = (dim,nParams)
        else:
            # self.error_map acts on the *state* vector before dmVec acts
            # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
            #return _np.einsum("jik,j->ik", derrgen.conjugate(), dmVec) # return shape = (dim,nParams)
            return _np.tensordot(derrgen.conjugate(), dmVec, (0, 0))  # return shape = (dim,nParams)

    def hessian_wrt_params(self, wrtFilter1=None, wrtFilter2=None):
        """
        Construct the Hessian of this SPAM vector with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrtFilter1, wrtFilter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the vectors's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension, num_params1, num_params2)
        """
        dmVec = self.state_vec.todense()

        herrgen = self.error_map.hessian_wrt_params(wrtFilter1, wrtFilter2)  # shape (dim*dim, nParams1, nParams2)
        herrgen.shape = (self.dim, self.dim, herrgen.shape[1], herrgen.shape[2])  # => (dim,dim,nParams1, nParams2)

        if self._prep_or_effect == "prep":
            #derror map acts on dmVec
            #return _np.einsum("ijkl,j->ikl", herrgen, dmVec) # return shape = (dim,nParams)
            return _np.tensordot(herrgen, dmVec, (1, 0))  # return shape = (dim,nParams)
        else:
            # self.error_map acts on the *state* vector before dmVec acts
            # as an effect:  E.dag -> dot(E.dag,errmap) ==> E -> dot(errmap.dag,E)
            #return _np.einsum("jikl,j->ikl", herrgen.conjugate(), dmVec) # return shape = (dim,nParams)
            return _np.tensordot(herrgen.conjugate(), dmVec, (0, 0))  # return shape = (dim,nParams)

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.error_map.num_params()

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.error_map.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params().

        Returns
        -------
        None
        """
        self.error_map.from_vector(v, close, nodirty)
        self._update_rep()
        if not nodirty: self.dirty = True

    def transform(self, S, typ):
        """
        Update SPAM (column) vector V as inv(S) * V or S^T * V for preparation
        or  effect SPAM vectors, respectively.

        Note that this is equivalent to state preparation vectors getting
        mapped: `rho -> inv(S) * rho` and the *transpose* of effect vectors
        being mapped as `E^T -> E^T * S`.

        Generally, the transform function updates the *parameters* of
        the SPAM vector such that the resulting vector is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        S : GaugeGroupElement
            A gauge group element which specifies the "S" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        #Defer everything to LindbladOp's
        # `spam_tranform` function, which applies either
        # error_map -> inv(S) * error_map ("prep" case) OR
        # error_map -> error_map * S      ("effect" case)
        self.error_map.spam_transform(S, typ)
        self._update_rep()
        self.dirty = True

    def depolarize(self, amount):
        """
        Depolarize this SPAM vector by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the SPAMVec such that the resulting vector is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the spam vector. All but
            the first element of the spam vector (often corresponding to the
            identity element) are multiplied by `amount` (if a float) or
            the corresponding `amount[i]` (if a tuple).

        Returns
        -------
        None
        """
        self.error_map.depolarize(amount)
        self._update_rep()


class StabilizerSPAMVec(SPAMVec):
    """
    A stabilizer state preparation represented internally using a compact
    representation of its stabilizer group.
    """

    @classmethod
    def from_dense_purevec(cls, purevec):
        """
        Create a new StabilizerSPAMVec from a pure-state vector.

        Currently, purevec must be a single computational basis state (it
        cannot be a superpostion of multiple of them).

        Parameters
        ----------
        purevec : numpy.ndarray
            A complex-valued state vector specifying a pure state in the
            standard computational basis.  This vector has length 2^n for
            n qubits.

        Returns
        -------
        StabilizerSPAMVec
        """
        nqubits = int(round(_np.log2(len(purevec))))
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, purevec.flat):
                return cls(nqubits, zvals)
        raise ValueError(("Given `purevec` must be a z-basis product state - "
                          "cannot construct StabilizerSPAMVec"))

    def __init__(self, nqubits, zvals=None, sframe=None):
        """
        Initialize a StabilizerSPAMVec object.

        Parameters
        ----------
        nqubits : int
            Number of qubits

        zvals : iterable, optional
            An iterable over anything that can be cast as True/False
            to indicate the 0/1 value of each qubit in the Z basis.
            If None, the all-zeros state is created.

        sframe : StabilizerFrame, optional
            A complete stabilizer frame to initialize this state from.
            If this is not None, then `nqubits` and `zvals` must be None.
        """
        if sframe is not None:
            assert(nqubits is None and zvals is None), "`nqubits` and `zvals` must be None when `sframe` isn't!"
            self.sframe = sframe
        else:
            self.sframe = _stabilizer.StabilizerFrame.from_zvals(nqubits, zvals)
        rep = self.sframe.torep()  # dim == 2**nqubits
        SPAMVec.__init__(self, rep, "stabilizer", "prep")

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array of shape
        (2^(nqubits), 1).  The memory in `scratch` maybe used when
        it is not-None.
        """
        statevec = self.sframe.to_statevec()
        statevec.shape = (statevec.size, 1)
        return statevec

    #def torep(self, typ, outvec=None):
    #    """
    #    Return a "representation" object for this SPAMVec.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    SBStateRep
    #    """
    #    return self.sframe.torep()

    def __str__(self):
        s = "Stabilizer spam vector for %d qubits with rep:\n" % (self.sframe.nqubits)
        s += str(self.sframe)
        return s


class StabilizerEffectVec(SPAMVec):  # FUTURE: merge this with ComptationalSPAMVec (w/evotype == "stabilizer")?
    """
    A dummy SPAM vector that points to a set/product of 1-qubit POVM
    outcomes from stabilizer-state measurements.
    """

    @classmethod
    def from_dense_purevec(cls, purevec):
        """
        Create a new StabilizerEffectVec from a pure-state vector.

        Currently, purevec must be a single computational basis state (it
        cannot be a superpostion of multiple of them).

        Parameters
        ----------
        purevec : numpy.ndarray
            A complex-valued state vector specifying a pure state in the
            standard computational basis.  This vector has length 2^n for
            n qubits.

        Returns
        -------
        StabilizerSPAMVec
        """
        nqubits = int(round(_np.log2(len(purevec))))
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, purevec.flat):
                return cls(zvals)
        raise ValueError(("Given `purevec` must be a z-basis product state - "
                          "cannot construct StabilizerEffectVec"))

    def __init__(self, outcomes):
        """
        Initialize a StabilizerEffectVec object.

        Parameters
        ----------
        outcomes : iterable
            A list or other iterable of integer 0 or 1 outcomes specifying
            which POVM effect vector this object represents within the
            full `stabilizerPOVM`
        """
        self._outcomes = _np.ascontiguousarray(_np.array(outcomes, int), _np.int64)
        #Note: dtype='i' => int in Cython, whereas dtype=int/np.int64 => long in Cython
        rep = replib.SBEffectRep(self._outcomes)  # dim == 2**nqubits == 2**len(outcomes)
        SPAMVec.__init__(self, rep, "stabilizer", "effect")

    #def torep(self, typ, outvec=None):
    #    # changes to_statevec/to_dmvec -> todense, and have
    #    # torep create an effect rep object...
    #    return replib.SBEffectRep(_np.ascontiguousarray(self._outcomes, _np.int64))

    def todense(self):
        """
        Return this SPAM vector as a dense state vector of shape
        (2^(nqubits), 1)

        Returns
        -------
        numpy array
        """
        v = (_np.array([1, 0], 'd'), _np.array([0, 1], 'd'))  # (v0,v1) - eigenstates of sigma_z
        statevec = _functools.reduce(_np.kron, [v[i] for i in self.outcomes])
        statevec.shape = (statevec.size, 1)
        return statevec

    @property
    def outcomes(self):
        """ The 0/1 outcomes identifying this effect within its StabilizerZPOVM """
        return self._outcomes

    def __str__(self):
        nQubits = len(self.outcomes)
        s = "Stabilizer effect vector for %d qubits with outcome %s" % (nQubits, str(self.outcomes))
        return s


class ComputationalSPAMVec(SPAMVec):
    """
    A static SPAM vector that is tensor product of 1-qubit Z-eigenstates.

    This is called a "computational basis state" in many contexts.
    """

    @classmethod
    def from_dense_vec(cls, vec, evotype):
        """
        Create a new ComputationalSPAMVec from a dense vector.

        Parameters
        ----------
        vec : numpy.ndarray
            A state vector specifying a computational basis state in the
            standard basis.  This vector has length 2^n or 4^n for
            n qubits depending on `evotype`.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of the resulting SPAM vector.  This value
            must be consistent with `len(vec)`, in that `"statevec"` and
            `"stabilizer"` expect 2^n whereas the rest expect 4^n.

        Returns
        -------
        ComputationalSPAMVec
        """
        if evotype in ('stabilizer', 'statevec'):
            nqubits = int(round(_np.log2(len(vec))))
            v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
            v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        else:
            nqubits = int(round(_np.log2(len(vec)) / 2))
            v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
            v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec

        v = (v0, v1)
        for zvals in _itertools.product(*([(0, 1)] * nqubits)):
            testvec = _functools.reduce(_np.kron, [v[i] for i in zvals])
            if _np.allclose(testvec, vec.flat):
                return cls(zvals, evotype)
        raise ValueError(("Given `vec` is not a z-basis product state - "
                          "cannot construct ComputatinoalSPAMVec"))

    def __init__(self, zvals, evotype, typ="prep"):
        """
        Initialize a ComputationalSPAMVec object.

        Parameters
        ----------
        zvals : iterable
            A list or other iterable of integer 0 or 1 outcomes specifying
            which computational basis element this object represents.  The
            length of `zvals` gives the total number of qubits.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The type of evolution being performed.

        typ : {"prep","effect"}
            Whether this is a state preparation or POVM effect vector.
        """
        self._zvals = _np.ascontiguousarray(_np.array(zvals, _np.int64))

        nqubits = len(self._zvals)
        if evotype in ("densitymx", "svterm", "cterm"):
            dim = 4**nqubits
        elif evotype in ("statevec", "stabilizer"):
            dim = 2**nqubits
        else: raise ValueError("Invalid `evotype`: %s" % evotype)
        self._evotype = evotype  # set this before call to SPAMVec.__init__ so self.todense() can work...

        if typ == "prep":
            if evotype == "statevec":
                rep = replib.SVStateRep(self.todense())
            elif evotype == "densitymx":
                vec = _np.require(self.todense(), requirements=['OWNDATA', 'C_CONTIGUOUS'])
                rep = replib.DMStateRep(vec)
            elif evotype == "stabilizer":
                sframe = _stabilizer.StabilizerFrame.from_zvals(len(self._zvals), self._zvals)
                rep = sframe.torep()
            else:
                rep = dim  # no representations for term-based evotypes

        elif typ == "effect":
            if evotype == "statevec":
                rep = replib.SVEffectRep_Computational(self._zvals, dim)
            elif evotype == "densitymx":
                rep = replib.DMEffectRep_Computational(self._zvals, dim)
            elif evotype == "stabilizer":
                rep = replib.SBEffectRep(self._zvals)
            else:
                rep = dim  # no representations for term-based evotypes
        else:
            raise ValueError("Invalid `typ` argument for torep(): %s" % typ)

        SPAMVec.__init__(self, rep, evotype, typ)

    def todense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.  The memory
        in `scratch` maybe used when it is not-None.
        """
        if self._evotype == "densitymx":
            v0 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, 1), 'd')  # '0' qubit state as Pauli dmvec
            v1 = 1.0 / _np.sqrt(2) * _np.array((1, 0, 0, -1), 'd')  # '1' qubit state as Pauli dmvec
        elif self._evotype in ("statevec", "stabilizer"):
            v0 = _np.array((1, 0), complex)  # '0' qubit state as complex state vec
            v1 = _np.array((0, 1), complex)  # '1' qubit state as complex state vec
        elif self._evotype in ("svterm", "cterm"):
            raise NotImplementedError("todense() is not implemented for evotype %s!" %
                                      self._evotype)
        else: raise ValueError("Invalid `evotype`: %s" % self._evotype)

        v = (v0, v1)
        return _functools.reduce(_np.kron, [v[i] for i in self._zvals])

    #def torep(self, typ, outvec=None):
    #    if typ == "prep":
    #        if self._evotype == "statevec":
    #            return replib.SVStateRep(self.todense())
    #        elif self._evotype == "densitymx":
    #            return replib.DMStateRep(self.todense())
    #        elif self._evotype == "stabilizer":
    #            sframe = _stabilizer.StabilizerFrame.from_zvals(len(self._zvals), self._zvals)
    #            return sframe.torep()
    #        raise NotImplementedError("torep(%s) not implemented for %s objects!" %
    #                                  (self._evotype, self.__class__.__name__))
    #    elif typ == "effect":
    #        if self._evotype == "statevec":
    #            return replib.SVEffectRep_Computational(self._zvals, self.dim)
    #        elif self._evotype == "densitymx":
    #            return replib.DMEffectRep_Computational(self._zvals, self.dim)
    #        elif self._evotype == "stabilizer":
    #            return replib.SBEffectRep(_np.ascontiguousarray(self._zvals, _np.int64))
    #        raise NotImplementedError("torep(%s) not implemented for %s objects!" %
    #                                  (self._evotype, self.__class__.__name__))
    #    else:
    #        raise ValueError("Invalid `typ` argument for torep(): %s" % typ)

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this SPAM vector.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that it is a state
        preparation followed by or POVM effect preceded by actions on a
        density matrix `rho` of the form:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the
        SPAMVec's parameters, where the polynomial's variable indices index the
        *global* parameters of the SPAMVec's parent (usually a :class:`Model`)
        , not the SPAMVec's local parameter array (i.e. that returned from
        `to_vector`).


        Parameters
        ----------
        order : int
            The order of terms to get.

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
            output of :method:`Polynomial.compact`.
        """
        if order == 0:  # only 0-th order term exists
            if self._evotype == "svterm":
                purevec = ComputationalSPAMVec(self._zvals, "statevec", self._prep_or_effect)
            elif self._evotype == "cterm":
                purevec = ComputationalSPAMVec(self._zvals, "stabilizer", self._prep_or_effect)
            else: raise ValueError("Invalid evolution type %s for calling `get_taylor_order_terms`" % self._evotype)

            coeff = _Polynomial({(): 1.0}, max_poly_vars)
            if self._prep_or_effect == "prep":
                terms = [_term.RankOnePolyPrepTerm.simple_init(coeff, purevec, purevec, self._evotype)]
            else:
                terms = [_term.RankOnePolyEffectTerm.simple_init(coeff, purevec, purevec, self._evotype)]

            if return_coeff_polys:
                coeffs_as_compact_polys = coeff.compact(complex_coeff_tape=True)
                return terms, coeffs_as_compact_polys
            else:
                return terms  # Cache terms in FUTURE?
        else:
            if return_coeff_polys:
                vtape = _np.empty(0, _np.int64)
                ctape = _np.empty(0, complex)
                return [], (vtape, ctape)
            else:
                return []

    def num_params(self):
        """
        Get the number of independent parameters which specify this SPAM vector.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 0  # no parameters

    def to_vector(self):
        """
        Get the SPAM vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return _np.array([], 'd')  # no parameters

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the SPAM vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(len(v) == 0)  # should be no parameters, and nothing to do

    def __str__(self):
        nQubits = len(self._zvals)
        s = "Computational Z-basis SPAM vec for %d qubits w/z-values: %s" % (nQubits, str(self._zvals))
        return s
