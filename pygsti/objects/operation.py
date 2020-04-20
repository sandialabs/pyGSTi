""" Defines classes which represent gates, as well as supporting functions """
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
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import functools as _functools
import itertools as _itertools
import copy as _copy
import warnings as _warnings
import collections as _collections
import numbers as _numbers

from .. import optimize as _opt
from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import jamiolkowski as _jt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import slicetools as _slct
from ..tools import symplectic as _symp
from ..tools import lindbladtools as _lbt
from . import gaugegroup as _gaugegroup
from . import modelmember as _modelmember
from . import stabilizer as _stabilizer
from .protectedarray import ProtectedArray as _ProtectedArray
from .basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis, EmbeddedBasis as _EmbeddedBasis, \
    ExplicitBasis as _ExplicitBasis

from . import term as _term
from .polynomial import Polynomial as _Polynomial
from . import replib
from . import opcalc
from .opcalc import compact_deriv as _compact_deriv, \
    bulk_eval_compact_polys_complex as _bulk_eval_compact_polys_complex, \
    abs_sum_bulk_eval_compact_polys_complex as _abs_sum_bulk_eval_compact_polys_complex

TOL = 1e-12
IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


def optimize_operation(op_to_optimize, target_op):
    """
    Optimize the parameters of op_to_optimize so that the
      the resulting operation matrix is as close as possible to
      target_op's matrix.

    This is trivial for the case of FullDenseOp
      instances, but for other types of parameterization
      this involves an iterative optimization over all the
      parameters of op_to_optimize.

    Parameters
    ----------
    op_to_optimize : LinearOperator
      The gate to optimize.  This object gets altered.

    target_op : LinearOperator
      The gate whose matrix is used as the target.

    Returns
    -------
    None
    """

    #TODO: cleanup this code:
    if isinstance(op_to_optimize, StaticDenseOp):
        return  # nothing to optimize

    if isinstance(op_to_optimize, FullDenseOp):
        if(target_op.dim != op_to_optimize.dim):  # special case: gates can have different overall dimension
            op_to_optimize.dim = target_op.dim  # this is a HACK to allow model selection code to work correctly
        op_to_optimize.set_value(target_op)  # just copy entire overall matrix since fully parameterized
        return

    assert(target_op.dim == op_to_optimize.dim)  # gates must have the same overall dimension
    targetMatrix = _np.asarray(target_op)

    def _objective_func(param_vec):
        op_to_optimize.from_vector(param_vec)
        return _mt.frobeniusnorm(op_to_optimize - targetMatrix)

    x0 = op_to_optimize.to_vector()
    minSol = _opt.minimize(_objective_func, x0, method='BFGS', maxiter=10000, maxfev=10000,
                           tol=1e-6, callback=None)

    op_to_optimize.from_vector(minSol.x)
    #print("DEBUG: optimized gate to min frobenius distance %g" %
    #      _mt.frobeniusnorm(op_to_optimize-targetMatrix))


def compose(op1, op2, basis, parameterization="auto"):
    """
    Returns a new LinearOperator that is the composition of op1 and op2.

    The resulting gate's matrix == dot(op1, op2),
     (so op1 acts *second* on an input) and the type of LinearOperator instance
     returned will depend on how much of the parameterization in op1
     and op2 can be preserved in the resulting gate.

    Parameters
    ----------
    op1 : LinearOperator
        LinearOperator to compose as left term of matrix product (applied second).

    op2 : LinearOperator
        LinearOperator to compose as right term of matrix product (applied first).

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    parameterization : {"auto","full","TP","linear","static"}, optional
        The parameterization of the resulting gates.  The default, "auto",
        attempts to convert to the most restrictive common parameterization.

    Returns
    -------
    LinearOperator
       The composed gate.
    """

    #Find the most restrictive common parameterization that both op1
    # and op2 can be cast/converted into. Utilized converstions are:
    #
    # Static => TP (sometimes)
    # Static => Linear
    # Static => Full
    # Linear => TP (sometimes)
    # Linear => Full
    # TP => Full

    if parameterization == "auto":
        if any([isinstance(g, FullDenseOp) for g in (op1, op2)]):
            paramType = "full"
        elif any([isinstance(g, TPDenseOp) for g in (op1, op2)]):
            paramType = "TP"  # update to "full" below if TP-conversion
            #not possible?
        elif any([isinstance(g, LinearlyParamDenseOp)
                  for g in (op1, op2)]):
            paramType = "linear"
        else:
            assert(isinstance(op1, StaticDenseOp)
                   and isinstance(op2, StaticDenseOp))
            paramType = "static"
    else:
        paramType = parameterization  # user-specified final parameterization

    #Convert to paramType as necessary
    cop1 = convert(op1, paramType, basis)
    cop2 = convert(op2, paramType, basis)

    # cop1 and cop2 are the same type, so can invoke the gate's compose method
    return cop1.compose(cop2)


def convert(gate, to_type, basis, extra=None):
    """
    Convert gate to a new type of parameterization, potentially creating
    a new LinearOperator object.  Raises ValueError for invalid conversions.

    Parameters
    ----------
    gate : LinearOperator
        LinearOperator to convert

    to_type : {"full","TP","static","static unitary","clifford",LINDBLAD}
        The type of parameterizaton to convert to.  "LINDBLAD" is a placeholder
        for the various Lindblad parameterization types.  See
        :method:`Model.set_all_parameterizations` for more details.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for `gate`.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).

    extra : object, optional
        Additional information for conversion.

    Returns
    -------
    LinearOperator
       The converted gate, usually a distinct
       object from the gate object passed as input.
    """
    if to_type == "full":
        if isinstance(gate, FullDenseOp):
            return gate  # no conversion necessary
        else:
            return FullDenseOp(gate.todense())

    elif to_type == "TP":
        if isinstance(gate, TPDenseOp):
            return gate  # no conversion necessary
        else:
            return TPDenseOp(gate.todense())
            # above will raise ValueError if conversion cannot be done

    elif to_type == "linear":
        if isinstance(gate, LinearlyParamDenseOp):
            return gate  # no conversion necessary
        elif isinstance(gate, StaticDenseOp):
            real = _np.isclose(_np.linalg.norm(gate.imag), 0)
            return LinearlyParamDenseOp(gate.todense(), _np.array([]), {}, real)
        else:
            raise ValueError("Cannot convert type %s to LinearlyParamDenseOp"
                             % type(gate))

    elif to_type == "static":
        if isinstance(gate, StaticDenseOp):
            return gate  # no conversion necessary
        else:
            return StaticDenseOp(gate.todense())

    elif to_type == "static unitary":
        op_std = _bt.change_basis(gate, basis, 'std')
        unitary = _gt.process_mx_to_unitary(op_std)
        return StaticDenseOp(unitary, "statevec")

    elif _gt.is_valid_lindblad_paramtype(to_type):
        # e.g. "H+S terms","H+S clifford terms"

        _, evotype = _gt.split_lindblad_paramtype(to_type)
        LindbladOpType = LindbladOp \
            if evotype in ("svterm", "cterm") else \
            LindbladDenseOp

        nQubits = _np.log2(gate.dim) / 2.0
        bQubits = bool(abs(nQubits - round(nQubits)) < 1e-10)  # integer # of qubits?
        proj_basis = "pp" if (basis == "pp" or bQubits) else basis

        #FUTURE: do something like this to get a guess for the post-op unitary factor
        #  (this commented code doesn't seem to work quite right).  Such intelligence should
        #  help scenarios where the assertion below fails.
        #if isinstance(gate, DenseOperator):
        #    J = _jt.jamiolkowski_iso(gate.todense(), opMxBasis=basis, choiMxBasis="std")
        #    ev, U = _np.linalg.eig(gate.todense())
        #    imax = _np.argmax(ev)
        #    J_unitary = _np.kron(U[:,imax:imax+1], U[:,imax:imax+1].T)
        #    postfactor = _jt.jamiolkowski_iso_inv(J_unitary, choiMxBasis="std", opMxBasis=basis)
        #    unitary = _gt.process_mx_to_unitary(postfactor)
        #else:
        postfactor = None

        ret = LindbladOpType.from_operation_obj(gate, to_type, postfactor, proj_basis,
                                                basis, truncate=True, lazy=True)
        if ret.dim <= 16:  # only do this for up to 2Q gates, otherwise todense is too expensive
            assert(_np.linalg.norm(gate.todense() - ret.todense()) < 1e-6), \
                "Failure to create CPTP gate (maybe due the complex log's branch cut?)"
        return ret

    elif to_type == "clifford":
        if isinstance(gate, CliffordOp):
            return gate  # no conversion necessary

        # assume gate represents a unitary op (otherwise
        #  would need to change Model dim, which isn't allowed)
        return CliffordOp(gate)

    else:
        raise ValueError("Invalid to_type argument: %s" % to_type)


def finite_difference_deriv_wrt_params(gate, wrt_filter, eps=1e-7):
    """
    Computes a finite-difference Jacobian for a LinearOperator object.

    The returned value is a matrix whose columns are the vectorized
    derivatives of the flattened operation matrix with respect to a single
    gate parameter, matching the format expected from the gate's
    `deriv_wrt_params` method.

    Parameters
    ----------
    gate : LinearOperator
        The gate object to compute a Jacobian for.

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    numpy.ndarray
        An M by N matrix where M is the number of gate elements and
        N is the number of gate parameters.
    """
    dim = gate.get_dimension()
    op2 = gate.copy()
    p = gate.to_vector()
    fd_deriv = _np.empty((dim, dim, gate.num_params()), gate.dtype)

    for i in range(gate.num_params()):
        p_plus_dp = p.copy()
        p_plus_dp[i] += eps
        op2.from_vector(p_plus_dp)
        fd_deriv[:, :, i] = (op2 - gate) / eps

    fd_deriv.shape = [dim**2, gate.num_params()]
    if wrt_filter is None:
        return fd_deriv
    else:
        return _np.take(fd_deriv, wrt_filter, axis=1)


def check_deriv_wrt_params(gate, deriv_to_check=None, wrt_filter=None, eps=1e-7):
    """
    Checks the `deriv_wrt_params` method of a LinearOperator object.

    This routine is meant to be used as an aid in testing and debugging
    gate classes by comparing the finite-difference Jacobian that
    *should* be returned by `gate.deriv_wrt_params` with the one that
    actually is.  A ValueError is raised if the two do not match.

    Parameters
    ----------
    gate : LinearOperator
        The gate object to test.

    deriv_to_check : numpy.ndarray or None, optional
        If not None, the Jacobian to compare against the finite difference
        result.  If None, `gate.deriv_wrt_parms()` is used.  Setting this
        argument can be useful when the function is called *within* a LinearOperator
        class's `deriv_wrt_params()` method itself as a part of testing.

    eps : float, optional
        The finite difference step to use.

    Returns
    -------
    None
    """
    fd_deriv = finite_difference_deriv_wrt_params(gate, wrt_filter, eps)
    if deriv_to_check is None:
        deriv_to_check = gate.deriv_wrt_params()

    #print("Deriv shapes = %s and %s" % (str(fd_deriv.shape),
    #                                    str(deriv_to_check.shape)))
    #print("finite difference deriv = \n",fd_deriv)
    #print("deriv_wrt_params deriv = \n",deriv_to_check)
    #print("deriv_wrt_params - finite diff deriv = \n",
    #      deriv_to_check - fd_deriv)
    for i in range(deriv_to_check.shape[0]):
        for j in range(deriv_to_check.shape[1]):
            diff = abs(deriv_to_check[i, j] - fd_deriv[i, j])
            if diff > 10 * eps:
                print("deriv_chk_mismatch: (%d,%d): %g (comp) - %g (fd) = %g" %
                      (i, j, deriv_to_check[i, j], fd_deriv[i, j], diff))  # pragma: no cover

    if _np.linalg.norm(fd_deriv - deriv_to_check) / fd_deriv.size > 10 * eps:
        raise ValueError("Failed check of deriv_wrt_params:\n"
                         " norm diff = %g" %
                         _np.linalg.norm(fd_deriv - deriv_to_check))  # pragma: no cover


#Note on initialization sequence of Gates within a Model:
# 1) a Model is constructed (empty)
# 2) a LinearOperator is constructed - apart from a Model if it's locally parameterized,
#    otherwise with explicit reference to an existing Model's labels/indices.
#    All gates (ModelMember objs in general) have a "gpindices" member which
#    can either be initialized upon construction or set to None, which signals
#    that the Model must initialize it.
# 3) the LinearOperator is assigned/added to a dict within the Model.  As a part of this
#    process, the LinearOperator's 'gpindices' member is set, if it isn't already, and the
#    Model's "global" parameter vector (and number of params) is updated as
#    needed to accomodate new parameters.
#
# Note: gpindices may be None (before initialization) or any valid index
#  into a 1D numpy array (e.g. a slice or integer array).  It may NOT have
#  any repeated elements.
#
# When a LinearOperator is removed from the Model, parameters only used by it can be
# removed from the Model, and the gpindices members of existing gates
# adjusted as needed.
#
# When derivatives are taken wrt. a model parameter (1 col of a jacobian)
# derivatives wrt each gate that includes that parameter in its gpindices
# must be processed.


class LinearOperator(_modelmember.ModelMember):
    """ Base class for all gate representations """

    def __init__(self, rep, evotype):
        """ Initialize a new LinearOperator """
        if isinstance(rep, int):  # For operators that have no representation themselves (term ops)
            dim = rep             # allow passing an integer as `rep`.
            rep = None
        else:
            dim = rep.dim
        super(LinearOperator, self).__init__(dim, evotype)
        self._rep = rep

    @property
    def size(self):
        """
        Return the number of independent elements in this gate (when viewed as a dense array)
        """
        return (self.dim)**2

    def set_value(self, m):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

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

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        raise NotImplementedError("todense(...) not implemented for %s objects!" % self.__class__.__name__)

    def acton(self, state):
        """
        Act with this operator upon `state`

        Parameters
        ----------
        state : SPAMVec
            The state to act on

        Returns
        -------
        SPAMVec
            The output state
        """
        from . import spamvec as _sv  # can we move this to top?
        assert(self._evotype in ('densitymx', 'statevec', 'stabilizer')), \
            "acton(...) cannot be used with the %s evolution type!" % self._evotype
        assert(self._rep is not None), "Internal Error: representation is None!"
        assert(state._evotype == self._evotype), "Evolution type mismatch: %s != %s" % (self._evotype, state._evotype)

        #Perform actual 'acton' operation
        output_rep = self._rep.acton(state._rep)

        #Build a SPAMVec around output_rep
        if self._evotype in ("densitymx", "statevec"):
            return _sv.StaticSPAMVec(output_rep.todense(), self._evotype, 'prep')
        else:  # self._evotype == "stabilizer"
            return _sv.StabilizerSPAMVec(sframe=_stabilizer.StabilizerFrame(
                output_rep.smatrix, output_rep.pvectors, output_rep.amps))

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "statevec":
    #        return replib.SVOpRepDense(_np.ascontiguousarray(self.todense(), complex))
    #    elif self._evotype == "densitymx":
    #        if LinearOperator.cache_reps:  # cache reps to avoid recomputation
    #            if self._cachedrep is None:
    #                self._cachedrep = replib.DMOpRepDense(_np.ascontiguousarray(self.todense(), 'd'))
    #            return self._cachedrep
    #        else:
    #            return replib.DMOpRepDense(_np.ascontiguousarray(self.todense(), 'd'))
    #    else:
    #        raise NotImplementedError("torep(%s) not implemented for %s objects!" %
    #                                  (self._evotype, self.__class__.__name__))

    @property
    def dirty(self):
        return _modelmember.ModelMember.dirty.fget(self)  # call base class

    @dirty.setter
    def dirty(self, value):
        if value:
            self._cachedrep = None  # clear cached rep
        _modelmember.ModelMember.dirty.fset(self, value)  # call base class setter

    def __getstate__(self):
        st = super(LinearOperator, self).__getstate__()
        st['_cachedrep'] = None  # can't pickle this!
        return st

    def copy(self, parent=None):
        self._cachedrep = None  # deepcopy in ModelMember.copy can't copy CReps!
        return _modelmember.ModelMember.copy(self, parent)

    def tosparse(self):
        """
        Return this operation as a sparse matrix.
        """
        raise NotImplementedError("tosparse(...) not implemented for %s objects!" % self.__class__.__name__)

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


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
        Get the terms (from a Taylor expansion of this operator) that have
        magnitude above `min_term_mag` (the magnitude of a term is taken to
        be the absolute value of its coefficient), considering only those
        terms up to some maximum Taylor expansion order, `max_taylor_order`.

        Note that this function also *sets* the magnitudes of the returned
        terms (by calling `term.set_magnitude(...)`) based on the current
        values of this operator's parameters.  This is an essential step
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
        #print("DB: OP get_high_magnitude_terms")
        v = self.to_vector()
        taylor_order = 0
        terms = []; last_len = -1; first_order_magmax = 1.0
        while len(terms) > last_len:  # while we keep adding something
            if taylor_order > 1 and first_order_magmax**taylor_order < min_term_mag:
                break  # there's no way any terms at this order reach min_term_mag - exit now!

            MAX_CACHED_TERM_ORDER = 1
            if taylor_order <= MAX_CACHED_TERM_ORDER:
                terms_at_order, cpolys = self.get_taylor_order_terms(taylor_order, max_poly_vars, True)
                coeffs = _bulk_eval_compact_polys_complex(
                    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
                terms_at_order = [t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order)]

                # CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs)
                # REMOVE later
                # for t in terms_at_order:
                #     vt, ct = t._rep.coeff.compact_complex()
                #     coeff_array = _bulk_eval_compact_polys_complex(vt, ct, self.parent.to_vector(), (1,))
                #     if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
                #         print(coeff_array[0], "vs.", t._rep.magnitude)
                #         import bpdb; bpdb.set_trace()

                if taylor_order == 1:
                    first_order_magmax = max([t.magnitude for t in terms_at_order])

                last_len = len(terms)
                for t in terms_at_order:
                    if t.magnitude >= min_term_mag or (taylor_order == 1 and force_firstorder):
                        terms.append((taylor_order, t))
            else:
                terms.extend(
                    [(taylor_order, t)
                     for t in self.get_taylor_order_terms_above_mag(taylor_order, max_poly_vars, min_term_mag)]
                )

            #print("order ", taylor_order, " : ", len(terms_at_order), " maxmag=",
            #      max([t.magnitude for t in terms_at_order]), len(terms), " running terms ",
            #      len(terms)-last_len, "added at this order")

            taylor_order += 1
            if taylor_order > max_taylor_order: break

        #Sort terms based on magnitude
        sorted_terms = sorted(terms, key=lambda t: t[1].magnitude, reverse=True)
        first_order_indices = [i for i, t in enumerate(sorted_terms) if t[0] == 1]

        #DEBUG TODO REMOVE
        #chk1 = sum([t[1].magnitude for t in sorted_terms])
        #chk2 = self.get_total_term_magnitude()
        #print("HIGHMAG ",self.__class__.__name__, len(sorted_terms), " maxorder=",max_taylor_order,
        #      " minmag=",min_term_mag)
        #print("  sum of magnitudes =",chk1, " <?= ", chk2)
        #if chk1 > chk2:
        #    print("Term magnitudes = ", [t[1].magnitude for t in sorted_terms])
        #    egterms = self.errorgen.get_taylor_order_terms(0)
        #    #vtape, ctape = self.errorgen.Lterm_coeffs
        #    #coeffs = [ abs(x) for x in _bulk_eval_compact_polys_complex(vtape, ctape, self.errorgen.to_vector(),
        #    #  (len(self.errorgen.Lterms),)) ]
        #    mags = [ abs(t.evaluate_coeff(self.errorgen.to_vector()).coeff) for t in egterms ]
        #    print("Errorgen ", self.errorgen.__class__.__name__, " term magnitudes (%d): " % len(egterms),
        #    "\n",list(sorted(mags, reverse=True)))
        #    print("Errorgen sum = ",sum(mags), " vs ", self.errorgen.get_total_term_magnitude())
        #assert(chk1 <= chk2)

        return [t[1] for t in sorted_terms], first_order_indices

    def get_taylor_order_terms_above_mag(self, order, max_poly_vars, min_term_mag):
        """ TODO: docstring """
        v = self.to_vector()
        terms_at_order, cpolys = self.get_taylor_order_terms(order, max_poly_vars, True)
        coeffs = _bulk_eval_compact_polys_complex(
            cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        terms_at_order = [t.copy_with_magnitude(abs(coeff)) for coeff, t in zip(coeffs, terms_at_order)]

        #CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs) REMOVE later
        #for t in terms_at_order:
        #    vt,ct = t._rep.coeff.compact_complex()
        #    coeff_array = _bulk_eval_compact_polys_complex(vt,ct,self.parent.to_vector(),(1,))
        #    if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
        #        print(coeff_array[0], "vs.", t._rep.magnitude)
        #        import bpdb; bpdb.set_trace()

        return [t for t in terms_at_order if t.magnitude >= min_term_mag]

    def frobeniusdist2(self, other_op, transform=None, inv_transform=None):
        """
        Return the squared frobenius difference between this gate and
        `other_op`, optionally transforming this gate first using matrices
        `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.frobeniusdist2(self.todense(), other_op.todense())
        else:
            return _gt.frobeniusdist2(_np.dot(
                inv_transform, _np.dot(self.todense(), transform)),
                other_op.todense())

    def frobeniusdist(self, other_op, transform=None, inv_transform=None):
        """
        Return the frobenius distance between this gate
        and `other_op`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        return _np.sqrt(self.frobeniusdist2(other_op, transform, inv_transform))

    def residuals(self, other_op, transform=None, inv_transform=None):
        """
        The per-element difference between this `DenseOperator` and `other_op`,
        possibly after transforming this operation as
        `G => inv_transform * G * transform`.

        Parameters
        ----------
        other_op : DenseOperator
            The gate to compare against.

        transform, inv_transform : numpy.ndarray, optional
            The transform and its inverse, respectively, to apply before
            taking the element-wise difference.

        Returns
        -------
        numpy.ndarray
            A 1D-array of size equal to that of the flattened operation matrix.
        """
        if transform is None and inv_transform is None:
            return _gt.residuals(self.todense(), other_op.todense())
        else:
            return _gt.residuals(_np.dot(
                inv_transform, _np.dot(self.todense(), transform)),
                other_op.todense())

    def jtracedist(self, other_op, transform=None, inv_transform=None):
        """
        Return the Jamiolkowski trace distance between this gate
        and `other_op`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.jtracedist(self.todense(), other_op.todense())
        else:
            return _gt.jtracedist(_np.dot(
                inv_transform, _np.dot(self.todense(), transform)),
                other_op.todense())

    def diamonddist(self, other_op, transform=None, inv_transform=None):
        """
        Return the diamon distance between this gate
        and `other_op`, optionally transforming this gate first
        using `transform` and `inv_transform`.
        """
        if transform is None and inv_transform is None:
            return _gt.diamonddist(self.todense(), other_op.todense())
        else:
            return _gt.diamonddist(_np.dot(
                inv_transform, _np.dot(self.todense(), transform)),
                other_op.todense())

    def transform(self, s):
        """
        Update operation matrix G with inv(s) * G * s,

        Generally, the transform function updates the *parameters* of
        the gate such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case *any* transform of the appropriate
        dimension is possible, since all operation matrix elements are parameters.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        Smx = s.get_transform_matrix()
        Si = s.get_transform_matrix_inverse()
        self.set_value(_np.dot(Si, _np.dot(self.todense(), Smx)))

    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the gate such that the resulting operation matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the operation matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        if isinstance(amount, float):
            D = _np.diag([1] + [1 - amount] * (self.dim - 1))
        else:
            assert(len(amount) == self.dim - 1)
            D = _np.diag([1] + list(1.0 - _np.array(amount, 'd')))
        self.set_value(_np.dot(D, self.todense()))

    def rotate(self, amount, mx_basis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of
        the gate such that the resulting operation matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        rotnMx = _gt.rotation_gate_mx(amount, mx_basis)
        self.set_value(_np.dot(rotnMx, self.todense()))

    def compose(self, other_op):
        """
        Create and return a new gate that is the composition of this operation
        followed by other_op of the same type.  (For more general compositions
        between different types of gates, use the module-level compose function.
        )  The returned gate's matrix is equal to dot(this, other_op).

        Parameters
        ----------
        other_op : DenseOperator
            The gate to compose to the right of this one.

        Returns
        -------
        DenseOperator
        """
        cpy = self.copy()
        cpy.set_value(_np.dot(self.todense(), other_op.todense()))
        return cpy

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        if(self.num_params() != 0):
            raise NotImplementedError("Default deriv_wrt_params is only for 0-parameter (default) case (%s)"
                                      % str(self.__class__.__name__))

        dtype = complex if self._evotype == 'statevec' else 'd'
        derivMx = _np.zeros((self.size, 0), dtype)
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        #Default: assume Hessian can be nonzero if there are any parameters
        return self.num_params() > 0

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this operation with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1, wrt_filter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        if not self.has_nonzero_hessian():
            return _np.zeros((self.size, self.num_params(), self.num_params()), 'd')

        # FUTURE: create a finite differencing hessian method?
        raise NotImplementedError("hessian_wrt_params(...) is not implemented for %s objects" % self.__class__.__name__)

    ##Pickle plumbing

    def __setstate__(self, state):
        self.__dict__.update(state)

    #Note: no __str__ fn

    @staticmethod
    def convert_to_matrix(m):
        """
        Static method that converts a matrix-like object to a 2D numpy array.

        Parameters
        ----------
        m : array_like

        Returns
        -------
        numpy array
        """
        if isinstance(m, LinearOperator):
            dim = m.dim
            matrix = _np.asarray(m).copy()
            # LinearOperator objs should also derive from ndarray
        elif isinstance(m, _np.ndarray):
            matrix = m.copy()
        else:
            try:
                dim = len(m)
                len(m[0])
                # XXX this is an abuse of exception handling
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % m)
            if any([len(row) != dim for row in m]):
                raise ValueError("%s is not a *square* 2D array" % m)

            ar = _np.array(m)
            if _np.all(_np.isreal(ar)):
                matrix = _np.array(ar.real, 'd')
            else:
                matrix = _np.array(ar, 'complex')

        if len(matrix.shape) != 2:
            raise ValueError("%s has %d dimensions when 2 are expected"
                             % (m, len(matrix.shape)))

        if matrix.shape[0] != matrix.shape[1]:  # checked above, but just to be safe
            raise ValueError("%s is not a *square* 2D array" % m)  # pragma: no cover

        return matrix


#class MapOperator(LinearOperator):
#    def __init__(self, dim, evotype):
#        """ Initialize a new LinearOperator """
#        super(MapOperator, self).__init__(dim, evotype)
#
#    #Maybe add an as_sparse_mx function and compute
#    # metrics using this?
#    #And perhaps a sparse-mode finite-difference deriv_wrt_params?
class DenseOperatorInterface(object):
    """
    Adds a numpy-array-mimicing interface onto an object whose ._rep
    is a *dense* representation (with a .base that is a numpy array).

    This class is distinct from DenseOperator because there are some
    operators, e.g. LindbladOp, that *can* but don't *always* have
    a dense representation.  With such types, a base class allows
    a 'dense_rep' argument to its constructor and a derived class
    sets this to True *and* inherits from DenseOperatorInterface.
    If would not be appropriate to inherit from DenseOperator because
    this is a standalone operator with it's own (dense) ._rep, etc.
    """

    def __init__(self):
        pass

    @property
    def _ptr(self):
        return self._rep.base

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        return finite_difference_deriv_wrt_params(self, wrt_filter, eps=1e-7)

    def todense(self):
        """
        Return this operation as a dense matrix.

        Note: for efficiency, this doesn't copy the underlying data, so
        the caller should copy this data before modifying it.
        """
        return _np.asarray(self._ptr)
        # *must* be a numpy array for Cython arg conversion

    def tosparse(self):
        """
        Return the operation as a sparse matrix.
        """
        return _sps.csr_matrix(self.todense())

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

    #Access to underlying ndarray
    def __getitem__(self, key):
        self.dirty = True
        return self._ptr.__getitem__(key)

    def __getslice__(self, i, j):
        self.dirty = True
        return self.__getitem__(slice(i, j))  # Called for A[:]

    def __setitem__(self, key, val):
        self.dirty = True
        return self._ptr.__setitem__(key, val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        ret = getattr(self.__dict__['_rep'].base, attr)
        self.dirty = True
        return ret

    #Mimic array behavior
    def __pos__(self): return self._ptr
    def __neg__(self): return -self._ptr
    def __abs__(self): return abs(self._ptr)
    def __add__(self, x): return self._ptr + x
    def __radd__(self, x): return x + self._ptr
    def __sub__(self, x): return self._ptr - x
    def __rsub__(self, x): return x - self._ptr
    def __mul__(self, x): return self._ptr * x
    def __rmul__(self, x): return x * self._ptr
    def __truediv__(self, x): return self._ptr / x
    def __rtruediv__(self, x): return x / self._ptr
    def __floordiv__(self, x): return self._ptr // x
    def __rfloordiv__(self, x): return x // self._ptr
    def __pow__(self, x): return self._ptr ** x
    def __eq__(self, x): return self._ptr == x
    def __len__(self): return len(self._ptr)
    def __int__(self): return int(self._ptr)
    def __long__(self): return int(self._ptr)
    def __float__(self): return float(self._ptr)
    def __complex__(self): return complex(self._ptr)


class BasedDenseOperatorInterface(DenseOperatorInterface):
    """
    A DenseOperatorInterface that uses self.base instead of
    self._rep.base as the "base pointer" to data.  This is
    used by the TPDenseOp class, for example, which has a .base
    that is different from its ._rep.base.
    """
    @property
    def _ptr(self):
        return self.base


class DenseOperator(BasedDenseOperatorInterface, LinearOperator):
    """
    Excapulates a parameterization of a operation matrix.  This class is the
    common base class for all specific parameterizations of a gate.
    """

    def __init__(self, mx, evotype):
        """ Initialize a new LinearOperator """
        dtype = complex if evotype == "statevec" else 'd'
        mx = _np.ascontiguousarray(mx, dtype)  # may not give mx it's own data
        mx = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])

        if evotype == "statevec":
            rep = replib.SVOpRepDense(mx)
        elif evotype == "densitymx":
            rep = replib.DMOpRepDense(mx)
        else:
            raise ValueError("Invalid evotype for a DenseOperator: %s" % evotype)

        LinearOperator.__init__(self, rep, evotype)
        BasedDenseOperatorInterface.__init__(self)
        # "Based" interface requires this and derived classes to have a .base attribute
        # or property that points to the data to interface with.  This gives derived classes
        # flexibility in defining something other than self._rep.base to be used (see TPDenseOp).

    @property
    def base(self):
        return self._rep.base

    def __str__(self):
        s = "%s with shape %s\n" % (self.__class__.__name__, str(self.base.shape))
        s += _mt.mx_to_string(self.base, width=4, prec=2)
        return s


class StaticDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is completely fixed, or "static", meaning
      that is contains no parameters.
    """

    def __init__(self, m, evotype="auto"):
        """
        Initialize a StaticDenseOp object.

        Parameters
        ----------
        m : array_like or LinearOperator
            a square 2D array-like or LinearOperator object representing the gate action.
            The shape of m sets the dimension of the gate.
        """
        m = LinearOperator.convert_to_matrix(m)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(m) else "densitymx"
        assert(evotype in ("statevec", "densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)
        DenseOperator.__init__(self, m, evotype)
        #(default DenseOperator/LinearOperator methods implement an object with no parameters)

        #if self._evotype == "svterm": # then we need to extract unitary
        #    op_std = _bt.change_basis(gate, basis, 'std')
        #    U = _gt.process_mx_to_unitary(self)

    def compose(self, other_op):
        """
        Create and return a new gate that is the composition of this operation
        followed by other_op, which *must be another StaticDenseOp*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, other_op).

        Parameters
        ----------
        other_op : StaticDenseOp
            The gate to compose to the right of this one.

        Returns
        -------
        StaticDenseOp
        """
        return StaticDenseOp(_np.dot(self.base, other_op.base), self._evotype)


class FullDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is fully parameterized, that is,
      each element of the operation matrix is an independent parameter.
    """

    def __init__(self, m, evotype="auto"):
        """
        Initialize a FullDenseOp object.

        Parameters
        ----------
        m : array_like or LinearOperator
            a square 2D array-like or LinearOperator object representing the gate action.
            The shape of m sets the dimension of the gate.
        """
        m = LinearOperator.convert_to_matrix(m)
        if evotype == "auto":
            evotype = "statevec" if _np.iscomplexobj(m) else "densitymx"
        assert(evotype in ("statevec", "densitymx")), \
            "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)
        DenseOperator.__init__(self, m, evotype)

    def set_value(self, m):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """
        mx = LinearOperator.convert_to_matrix(m)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim, self.dim))
        self.base[:, :] = _np.array(mx)
        self.dirty = True

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return 2 * self.size if self._evotype == "statevec" else self.size

    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        if self._evotype == "statevec":
            return _np.concatenate((self.base.real.flatten(), self.base.imag.flatten()), axis=0)
        else:
            return self.base.flatten()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.base.shape == (self.dim, self.dim))
        if self._evotype == "statevec":
            self.base[:, :] = v[0:self.dim**2].reshape((self.dim, self.dim)) + \
                1j * v[self.dim**2:].reshape((self.dim, self.dim))
        else:
            self.base[:, :] = v.reshape((self.dim, self.dim))
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        if self._evotype == "statevec":
            derivMx = _np.concatenate((_np.identity(self.dim**2, 'complex'),
                                       1j * _np.identity(self.dim**2, 'complex')),
                                      axis=1)
        else:
            derivMx = _np.identity(self.dim**2, self.base.dtype)

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class TPDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is fully parameterized except for
    the first row, which is frozen to be [1 0 ... 0] so that the action
    of the gate, when interpreted in the Pauli or Gell-Mann basis, is
    trace preserving (TP).
    """

    def __init__(self, m):
        """
        Initialize a TPDenseOp object.

        Parameters
        ----------
        m : array_like or LinearOperator
            a square 2D numpy array representing the gate action.  The
            shape of this array sets the dimension of the gate.
        """
        #LinearOperator.__init__(self, LinearOperator.convert_to_matrix(m))
        mx = LinearOperator.convert_to_matrix(m)
        assert(_np.isrealobj(mx)), "TPDenseOp must have *real* values!"
        if not (_np.isclose(mx[0, 0], 1.0)
                and _np.allclose(mx[0, 1:], 0.0)):
            raise ValueError("Cannot create TPDenseOp: "
                             "invalid form for 1st row!")
        raw = _np.require(mx, requirements=['OWNDATA', 'C_CONTIGUOUS'])

        DenseOperator.__init__(self, raw, "densitymx")
        assert(self._rep.base.flags['C_CONTIGUOUS'] and self._rep.base.flags['OWNDATA'])
        assert(isinstance(self.base, _ProtectedArray))

    @property
    def base(self):
        return _ProtectedArray(self._rep.base, indices_to_protect=(0, slice(None, None, None)))

    def set_value(self, m):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """
        mx = LinearOperator.convert_to_matrix(m)
        if(mx.shape != (self.dim, self.dim)):
            raise ValueError("Argument must be a (%d,%d) matrix!"
                             % (self.dim, self.dim))
        if not (_np.isclose(mx[0, 0], 1.0) and _np.allclose(mx[0, 1:], 0.0)):
            raise ValueError("Cannot set TPDenseOp: "
                             "invalid form for 1st row!")
            #For further debugging:  + "\n".join([str(e) for e in mx[0,:]])
        self.base[1:, :] = mx[1:, :]
        self.dirty = True

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.dim**2 - self.dim

    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.base.flatten()[self.dim:]  # .real in case of complex matrices?

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.base.shape == (self.dim, self.dim))
        self.base[1:, :] = v.reshape((self.dim - 1, self.dim))
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        derivMx = _np.identity(self.dim**2, 'd')  # TP gates are assumed to be real
        derivMx = derivMx[:, self.dim:]  # remove first op_dim cols ( <=> first-row parameters )

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class LinearlyParameterizedElementTerm(object):
    """
    Encapsulates a single term within a LinearlyParamDenseOp.
    """

    def __init__(self, coeff=1.0, param_indices=[]):
        """
        Create a new LinearlyParameterizedElementTerm

        Parameters
        ----------
        coeff : float
            The term's coefficient

        param_indices : list
            A list of integers, specifying which parameters are muliplied
            together (and finally, with `coeff`) to form this term.
        """
        self.coeff = coeff
        self.paramIndices = param_indices

    def copy(self, parent=None):
        """ Copy this term. """
        return LinearlyParameterizedElementTerm(self.coeff, self.paramIndices[:])


class LinearlyParamDenseOp(DenseOperator):
    """
    Encapsulates a operation matrix that is parameterized such that each
    element of the operation matrix depends only linearly on any parameter.
    """

    def __init__(self, base_matrix, parameter_array, parameter_to_base_indices_map,
                 left_transform=None, right_transform=None, real=False, evotype="auto"):
        """
        Initialize a LinearlyParamDenseOp object.

        Parameters
        ----------
        basematrix : numpy array
            a square 2D numpy array that acts as the starting point when
            constructin the gate's matrix.  The shape of this array sets
            the dimension of the gate.

        parameter_array : numpy array
            a 1D numpy array that holds the all the parameters for this
            gate.  The shape of this array sets is what is returned by
            value_dimension(...).

        parameter_to_base_indices_map : dict
            A dictionary with keys == index of a parameter
            (i.e. in parameter_array) and values == list of 2-tuples
            indexing potentially multiple operation matrix coordinates
            which should be set equal to this parameter.

        left_transform : numpy array or None, optional
            A 2D array of the same shape as basematrix which left-multiplies
            the base matrix after parameters have been evaluated.  Defaults to
            no transform.

        right_transform : numpy array or None, optional
            A 2D array of the same shape as basematrix which right-multiplies
            the base matrix after parameters have been evaluated.  Defaults to
            no transform.

        real : bool, optional
            Whether or not the resulting operation matrix, after all
            parameter evaluation and left & right transforms have
            been performed, should be real.  If True, ValueError will
            be raised if the matrix contains any complex or imaginary
            elements.
        """

        base_matrix = _np.array(LinearOperator.convert_to_matrix(base_matrix), 'complex')
        #complex, even if passed all real base matrix

        elementExpressions = {}
        for p, ij_tuples in list(parameter_to_base_indices_map.items()):
            for i, j in ij_tuples:
                assert((i, j) not in elementExpressions)  # only one parameter allowed per base index pair
                elementExpressions[(i, j)] = [LinearlyParameterizedElementTerm(1.0, [p])]

        typ = "d" if real else "complex"
        mx = _np.empty(base_matrix.shape, typ)
        self.baseMatrix = base_matrix
        self.parameterArray = parameter_array
        self.numParams = len(parameter_array)
        self.elementExpressions = elementExpressions
        assert(_np.isrealobj(self.parameterArray)), "Parameter array must be real-valued!"

        I = _np.identity(self.baseMatrix.shape[0], 'd')  # LinearlyParameterizedGates are currently assumed to be real
        self.leftTrans = left_transform if (left_transform is not None) else I
        self.rightTrans = right_transform if (right_transform is not None) else I
        self.enforceReal = real

        if evotype == "auto": evotype = "densitymx" if real else "statevec"
        assert(evotype in ("densitymx", "statevec")), \
            "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)

        #Note: dense op reps *always* own their own data so setting writeable flag is OK
        DenseOperator.__init__(self, mx, evotype)
        self.base.flags.writeable = False  # only _construct_matrix can change array
        self._construct_matrix()  # construct base from the parameters

    def _construct_matrix(self):
        """
        Build the internal operation matrix using the current parameters.
        """
        matrix = self.baseMatrix.copy()
        for (i, j), terms in self.elementExpressions.items():
            for term in terms:
                param_prod = _np.prod([self.parameterArray[p] for p in term.paramIndices])
                matrix[i, j] += term.coeff * param_prod
        matrix = _np.dot(self.leftTrans, _np.dot(matrix, self.rightTrans))

        if self.enforceReal:
            if _np.linalg.norm(_np.imag(matrix)) > IMAG_TOL:
                raise ValueError("Linearly parameterized matrix has non-zero"
                                 "imaginary part (%g)!" % _np.linalg.norm(_np.imag(matrix)))
            matrix = _np.real(matrix)

        #Note: dense op reps *always* own their own data so setting writeable flag is OK
        assert(matrix.shape == (self.dim, self.dim))
        self.base.flags.writeable = True
        self.base[:, :] = matrix
        self.base.flags.writeable = False

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.numParams

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.parameterArray

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
        self.parameterArray[:] = v
        self._construct_matrix()
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        derivMx = _np.zeros((self.numParams, self.dim, self.dim), 'complex')
        for (i, j), terms in self.elementExpressions.items():
            for term in terms:
                params_to_mult = [self.parameterArray[p] for p in term.paramIndices]
                for k, p in enumerate(term.paramIndices):
                    param_partial_prod = _np.prod(params_to_mult[0:k] + params_to_mult[k + 1:])  # exclude k-th factor
                    derivMx[p, i, j] += term.coeff * param_partial_prod

        derivMx = _np.dot(self.leftTrans, _np.dot(derivMx, self.rightTrans))  # (d,d) * (P,d,d) * (d,d) => (d,P,d)
        derivMx = _np.rollaxis(derivMx, 1, 3)  # now (d,d,P)
        derivMx = derivMx.reshape([self.dim**2, self.numParams])  # (d^2,P) == final shape

        if self.enforceReal:
            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)
            derivMx = _np.real(derivMx)

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False

    def compose(self, other_op):
        """
        Create and return a new gate that is the composition of this operation
        followed by other_op, which *must be another LinearlyParamDenseOp*.
        (For more general compositions between different types of gates, use
        the module-level compose function.)  The returned gate's matrix is
        equal to dot(this, other_op).

        Parameters
        ----------
        other_op : LinearlyParamDenseOp
            The gate to compose to the right of this one.

        Returns
        -------
        LinearlyParamDenseOp
        """
        assert(isinstance(other_op, LinearlyParamDenseOp))

        ### Implementation Notes ###
        #
        # Let self == L1 * A * R1, other == L2 * B * R2
        #   where  [A]_ij = a_ij + sum_l c^(ij)_l T^(ij)_l  so that
        #      a_ij == base matrix of self, c's are term coefficients, and T's are parameter products
        #   and similarly [B]_ij = b_ij + sum_l d^(ij)_l R^(ij)_l.
        #
        # We want in the end a gate with matrix:
        #   L1 * A * R1 * L2 * B * R2 == L1 * (A * W * B) * R2,  (where W := R1 * L2 )
        #   which is a linearly parameterized gate with leftTrans == L1, rightTrans == R2
        #   and a parameterized part == (A * W * B) which can be written with implied sum on k,n:
        #
        #  [A * W * B]_ij
        #   = (a_ik + sum_l c^(ik)_l T^(ik)_l) * W_kn * (b_nj + sum_m d^(nj)_m R^(nj)_m)
        #
        #   = a_ik * W_kn * b_nj +
        #     a_ik * W_kn * sum_m d^(nj)_m R^(nj)_m +
        #     sum_l c^(ik)_l T^(ik)_l * W_kn * b_nj +
        #     (sum_l c^(ik)_l T^(ik)_l) * W_kn * (sum_m d^(nj)_m R^(nj)_m)
        #
        #   = aWb_ij   # (new base matrix == a*W*b)
        #     aW_in * sum_m d^(nj)_m R^(nj)_m +   # coeffs w/params of other_op
        #     sum_l c^(ik)_l T^(ik)_l * Wb_kj +   # coeffs w/params of this gate
        #     sum_m,l c^(ik)_l W_kn d^(nj)_m T^(ik)_l R^(nj)_m) # coeffs w/params of both gates
        #

        W = _np.dot(self.rightTrans, other_op.leftTrans)
        baseMx = _np.dot(self.baseMatrix, _np.dot(W, other_op.baseMatrix))  # aWb above
        paramArray = _np.concatenate((self.parameterArray, other_op.parameterArray), axis=0)
        composedOp = LinearlyParamDenseOp(baseMx, paramArray, {},
                                          self.leftTrans, other_op.rightTrans,
                                          self.enforceReal and other_op.enforceReal,
                                          self._evotype)

        # Precompute what we can before the compute loop
        aW = _np.dot(self.baseMatrix, W)
        Wb = _np.dot(W, other_op.baseMatrix)

        kMax, nMax = (self.dim, self.dim)  # W.shape
        offset = len(self.parameterArray)  # amt to offset parameter indices of other_op

        # Compute  [A * W * B]_ij element expression as described above
        for i in range(self.baseMatrix.shape[0]):
            for j in range(other_op.baseMatrix.shape[1]):
                terms = []
                for n in range(nMax):
                    if (n, j) in other_op.elementExpressions:
                        for term in other_op.elementExpressions[(n, j)]:
                            coeff = aW[i, n] * term.coeff
                            paramIndices = [p + offset for p in term.paramIndices]
                            terms.append(LinearlyParameterizedElementTerm(coeff, paramIndices))

                for k in range(kMax):
                    if (i, k) in self.elementExpressions:
                        for term in self.elementExpressions[(i, k)]:
                            coeff = term.coeff * Wb[k, j]
                            terms.append(LinearlyParameterizedElementTerm(coeff, term.paramIndices))

                            for n in range(nMax):
                                if (n, j) in other_op.elementExpressions:
                                    for term2 in other_op.elementExpressions[(n, j)]:
                                        coeff = term.coeff * W[k, n] * term2.coeff
                                        paramIndices = term.paramIndices + [p + offset for p in term2.paramIndices]
                                        terms.append(LinearlyParameterizedElementTerm(coeff, paramIndices))

                composedOp.elementExpressions[(i, j)] = terms

        composedOp._construct_matrix()
        return composedOp

    def __str__(self):
        s = "Linearly Parameterized gate with shape %s, num params = %d\n" % \
            (str(self.base.shape), self.numParams)
        s += _mt.mx_to_string(self.base, width=5, prec=1)
        s += "\nParameterization:"
        for (i, j), terms in self.elementExpressions.items():
            tStr = ' + '.join(['*'.join(["p%d" % p for p in term.paramIndices])
                               for term in terms])
            s += "LinearOperator[%d,%d] = %s\n" % (i, j, tStr)
        return s


class EigenvalueParamDenseOp(DenseOperator):
    """
    Encapsulates a real operation matrix that is parameterized only by its
    eigenvalues, which are assumed to be either real or to occur in
    conjugate pairs.  Thus, the number of parameters is equal to the
    number of eigenvalues.
    """

    def __init__(self, matrix, include_off_diags_in_degen_2_blocks=False,
                 tp_constrained_and_unital=False):
        """
        Initialize an EigenvalueParamDenseOp object.

        Parameters
        ----------
        matrix : numpy array
            a square 2D numpy array that gives the raw operation matrix to
            paramterize.  The shape of this array sets the dimension
            of the gate.

        include_off_diags_in_degen_2_blocks : bool
            If True, include as parameters the (initially zero)
            off-diagonal elements in degenerate 2x2 blocks of the
            the diagonalized operation matrix (no off-diagonals are
            included in blocks larger than 2x2).  This is an option
            specifically used in the intelligent fiducial pair
            reduction (IFPR) algorithm.

        tp_constrained_and_unital : bool
            If True, assume the top row of the operation matrix is fixed
            to [1, 0, ... 0] and should not be parameterized, and verify
            that the matrix is unital.  In this case, "1" is always a
            fixed (not-paramterized0 eigenvalue with eigenvector
            [1,0,...0] and if include_off_diags_in_degen_2_blocks is True
            any off diagonal elements lying on the top row are *not*
            parameterized as implied by the TP constraint.
        """
        def cmplx_compare(ia, ib):
            return _mt.complex_compare(evals[ia], evals[ib])
        cmplx_compare_key = _functools.cmp_to_key(cmplx_compare)

        def isreal(a):
            """ b/c numpy's isreal tests for strict equality w/0 """
            return _np.isclose(_np.imag(a), 0.0)

        # Since matrix is real, eigenvalues must either be real or occur in
        #  conjugate pairs.  Find and sort by conjugate pairs.

        assert(_np.linalg.norm(_np.imag(matrix)) < IMAG_TOL)  # matrix should be real
        evals, B = _np.linalg.eig(matrix)  # matrix == B * diag(evals) * Bi
        dim = len(evals)

        #Sort eigenvalues & eigenvectors by:
        # 1) unit eigenvalues first (with TP eigenvalue first of all)
        # 2) non-unit real eigenvalues in order of magnitude
        # 3) complex eigenvalues in order of real then imaginary part

        unitInds = []; realInds = []; complexInds = []
        for i, ev in enumerate(evals):
            if _np.isclose(ev, 1.0): unitInds.append(i)
            elif isreal(ev): realInds.append(i)
            else: complexInds.append(i)

        if tp_constrained_and_unital:
            #check matrix is TP and unital
            unitRow = _np.zeros((len(evals)), 'd'); unitRow[0] = 1.0
            assert(_np.allclose(matrix[0, :], unitRow))
            assert(_np.allclose(matrix[:, 0], unitRow))

            #find the eigenvector with largest first element and make sure
            # this is the first index in unitInds
            k = _np.argmax([B[0, i] for i in unitInds])
            if k != 0:  # swap indices 0 <-> k in unitInds
                t = unitInds[0]; unitInds[0] = unitInds[k]; unitInds[k] = t

            #Assume we can recombine unit-eval eigenvectors so that the first
            # one (actually the closest-to-unit-row one) == unitRow and the
            # rest do not have any 0th component.
            iClose = _np.argmax([abs(B[0, ui]) for ui in unitInds])
            B[:, unitInds[iClose]] = unitRow
            for i, ui in enumerate(unitInds):
                if i == iClose: continue
                B[0, ui] = 0.0; B[:, ui] /= _np.linalg.norm(B[:, ui])

        realInds = sorted(realInds, key=lambda i: -abs(evals[i]))
        complexInds = sorted(complexInds, key=cmplx_compare_key)
        new_ordering = unitInds + realInds + complexInds

        #Re-order the eigenvalues & vectors
        sorted_evals = _np.zeros(evals.shape, 'complex')
        sorted_B = _np.zeros(B.shape, 'complex')
        for i, indx in enumerate(new_ordering):
            sorted_evals[i] = evals[indx]
            sorted_B[:, i] = B[:, indx]

        #Save the final list of (sorted) eigenvalues & eigenvectors
        self.evals = sorted_evals
        self.B = sorted_B
        self.Bi = _np.linalg.inv(sorted_B)

        self.options = {'includeOffDiags': include_off_diags_in_degen_2_blocks,
                        'TPandUnital': tp_constrained_and_unital}

        #Check that nothing has gone horribly wrong
        assert(_np.allclose(_np.dot(
            self.B, _np.dot(_np.diag(self.evals), self.Bi)), matrix))

        #Build a list of parameter descriptors.  Each element of self.params
        # is a list of (prefactor, (i,j)) tuples.
        self.params = []
        i = 0; N = len(self.evals); processed = [False] * N
        while i < N:
            if processed[i]:
                i += 1; continue

            # Find block (i -> j) of degenerate eigenvalues
            j = i + 1
            while j < N and _np.isclose(self.evals[i], self.evals[j]): j += 1
            blkSize = j - i

            #Add eigenvalues as parameters
            ev = self.evals[i]  # current eigenvalue being processed
            if isreal(ev):

                # Side task: for a *real* block of degenerate evals, we want
                # to ensure the eigenvectors are real, which numpy doesn't
                # always guarantee (could be conj. pairs for instance).

                # Solve or Cmx: [v1,v2,v3,v4]Cmx = [v1',v2',v3',v4'] ,
                # where ' qtys == real, so Im([v1,v2,v3,v4]Cmx) = 0
                # Let Cmx = Cr + i*Ci, v1 = v1.r + i*v1.i, etc.,
                #  then solve [v1.r, ...]Ci + [v1.i, ...]Cr = 0
                #  which can be cast as [Vr,Vi]*[Ci] = 0
                #                               [Cr]      (nullspace of V)
                # Note: only involve complex evecs (don't disturb TP evec!)
                evecIndsToMakeReal = []
                for k in range(i, j):
                    if _np.linalg.norm(self.B[:, k].imag) >= IMAG_TOL:
                        evecIndsToMakeReal.append(k)

                nToReal = len(evecIndsToMakeReal)
                if nToReal > 0:
                    vecs = _np.empty((dim, nToReal), 'complex')
                    for ik, k in enumerate(evecIndsToMakeReal):
                        vecs[:, ik] = self.B[:, k]
                    V = _np.concatenate((vecs.real, vecs.imag), axis=1)
                    nullsp = _mt.nullspace(V)
                    # if nullsp.shape[1] < nToReal: # DEBUG
                    #    raise ValueError("Nullspace only has dimension %d when %d was expected! "
                    #                     "(i=%d, j=%d, blkSize=%d)\nevals = %s" \
                    #                     % (nullsp.shape[1],nToReal, i,j,blkSize,str(self.evals)) )
                    assert(nullsp.shape[1] >= nToReal), "Cannot find enough real linear combos!"
                    nullsp = nullsp[:, 0:nToReal]  # truncate #cols if there are more than we need

                    Cmx = nullsp[nToReal:, :] + 1j * nullsp[0:nToReal, :]  # Cr + i*Ci
                    new_vecs = _np.dot(vecs, Cmx)
                    assert(_np.linalg.norm(new_vecs.imag) < IMAG_TOL), \
                        "Imaginary mag = %g!" % _np.linalg.norm(new_vecs.imag)
                    for ik, k in enumerate(evecIndsToMakeReal):
                        self.B[:, k] = new_vecs[:, ik]
                    self.Bi = _np.linalg.inv(self.B)

                #Now, back to constructing parameter descriptors...
                for k in range(i, j):
                    if tp_constrained_and_unital and k == 0: continue
                    prefactor = 1.0; mx_indx = (k, k)
                    self.params.append([(prefactor, mx_indx)])
                    processed[k] = True
            else:
                iConjugate = {}
                for k in range(i, j):
                    #Find conjugate eigenvalue to eval[k]
                    conj = _np.conj(self.evals[k])  # == conj(ev), indep of k
                    conjB = _np.conj(self.B[:, k])
                    for l in range(j, N):
                        # numpy normalizes but doesn't fix "phase" of evecs
                        if _np.isclose(conj, self.evals[l]) \
                           and (_np.allclose(conjB, self.B[:, l])
                                or _np.allclose(conjB, 1j * self.B[:, l])
                                or _np.allclose(conjB, -1j * self.B[:, l])
                                or _np.allclose(conjB, -1 * self.B[:, l])):
                            self.params.append([  # real-part param
                                (1.0, (k, k)),  # (prefactor, index)
                                (1.0, (l, l))])
                            self.params.append([  # imag-part param
                                (1j, (k, k)),  # (prefactor, index)
                                (-1j, (l, l))])
                            processed[k] = processed[l] = True
                            iConjugate[k] = l  # save conj. pair index for below
                            break
                    else:
                        # should be unreachable, since we ensure mx is real above - but
                        # this may fail when there are multiple degenerate complex evals
                        # since the evecs can get mixed (and we check for evec "match" above)
                        raise ValueError("Could not find conjugate pair "
                                         + " for %s" % self.evals[k])  # pragma: no cover

            if include_off_diags_in_degen_2_blocks and blkSize == 2:
                #Note: could remove blkSize == 2 condition or make this a
                # separate option.  This is useful currently so that we don't
                # add lots of off-diag elements in accidentally-degenerate
                # cases, but there's probabaly a better heuristic for this, such
                # as only including off-diag els for unit-eigenvalue blocks
                # of size 2 (?)
                for k1 in range(i, j - 1):
                    for k2 in range(k1 + 1, j):
                        if isreal(ev):
                            # k1,k2 element
                            if not tp_constrained_and_unital or k1 != 0:
                                self.params.append([(1.0, (k1, k2))])

                            # k2,k1 element
                            if not tp_constrained_and_unital or k2 != 0:
                                self.params.append([(1.0, (k2, k1))])
                        else:
                            k1c, k2c = iConjugate[k1], iConjugate[k2]

                            # k1,k2 element
                            self.params.append([  # real-part param
                                (1.0, (k1, k2)),
                                (1.0, (k1c, k2c))])
                            self.params.append([  # imag-part param
                                (1j, (k1, k2)),
                                (-1j, (k1c, k2c))])

                            # k2,k1 element
                            self.params.append([  # real-part param
                                (1.0, (k2, k1)),
                                (1.0, (k2c, k1c))])
                            self.params.append([  # imag-part param
                                (1j, (k2, k1)),
                                (-1j, (k2c, k1c))])

            i = j  # advance to next block

        #Allocate array of parameter values (all zero initially)
        self.paramvals = _np.zeros(len(self.params), 'd')

        #Finish LinearOperator construction
        mx = _np.empty(matrix.shape, "d")
        DenseOperator.__init__(self, mx, "densitymx")
        self.base.flags.writeable = False  # only _construct_matrix can change array
        self._construct_matrix()  # construct base from the parameters

    def _construct_matrix(self):
        """
        Build the internal operation matrix using the current parameters.
        """
        base_diag = _np.diag(self.evals)
        for pdesc, pval in zip(self.params, self.paramvals):
            for prefactor, (i, j) in pdesc:
                base_diag[i, j] += prefactor * pval
        matrix = _np.dot(self.B, _np.dot(base_diag, self.Bi))
        assert(_np.linalg.norm(matrix.imag) < IMAG_TOL)
        assert(matrix.shape == (self.dim, self.dim))
        self.base.flags.writeable = True
        self.base[:, :] = matrix.real
        self.base.flags.writeable = False

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.paramvals)

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.paramvals

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
        assert(len(v) == self.num_params())
        self.paramvals = v
        self._construct_matrix()
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """

        # matrix = B * diag * Bi, and only diag depends on parameters
        #  (and only linearly), so:
        # d(matrix)/d(param) = B * d(diag)/d(param) * Bi

        # EigenvalueParameterizedGates are assumed to be real
        derivMx = _np.zeros((self.dim**2, self.num_params()), 'd')

        # Compute d(diag)/d(param) for each params, then apply B & Bi
        for k, pdesc in enumerate(self.params):
            dMx = _np.zeros((self.dim, self.dim), 'complex')
            for prefactor, (i, j) in pdesc:
                dMx[i, j] = prefactor
            tmp = _np.dot(self.B, _np.dot(dMx, self.Bi))
            if _np.linalg.norm(tmp.imag) >= IMAG_TOL:  # just a warning until we figure this out.
                print("EigenvalueParamDenseOp deriv_wrt_params WARNING:"
                      " Imag part = ", _np.linalg.norm(tmp.imag), " pdesc = ", pdesc)  # pragma: no cover
            #assert(_np.linalg.norm(tmp.imag) < IMAG_TOL), \
            #       "Imaginary mag = %g!" % _np.linalg.norm(tmp.imag)
            derivMx[:, k] = tmp.real.flatten()

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False


class StochasticNoiseOp(LinearOperator):
    """
    A stocastic noise map:
    rho -> (1-sum(p_i))rho + sum_(i>0) p_i * B_i * rho * B_i^dagger
    where p_i > 0 and sum(p_i) < 1, and B_i is basis where B_0 == I
    """
    # Difficult to parameterize and maintain the p_i conditions - Initially just store positive p_i's
    # and don't bother restricting their sum to be < 1?

    def __init__(self, dim, basis="pp", evotype="densitymx", initial_rates=None):
        """
        Create a new StochasticNoiseOp, representing a stochastic noise
        channel with possibly asymmetric noise but only noise that is
        "diagonal" in a particular basis (e.g. Pauli-stochastic noise).

        Parameters
        ----------
        dim : int
            The dimension of this operator (4 for a single qubit).

        basis : Basis or {'pp','gm','qt'}, optional
            The basis to use, defining the "principle axes"
            along which there is stochastic noise.  We assume that
            the first element of `basis` is the identity.

        evotype : {"densitymx", "cterm", "svterm"}
            the evolution type being used.

        initial_rates : list or array
            if not None, a list of `dim-1` initial error rates along each of
            the directions corresponding to each basis element.  If None,
            then all initial rates are zero.
        """
        self.basis = _Basis.cast(basis, dim, sparse=False)  # sparse??
        assert(dim == self.basis.dim), "Dimension of `basis` must match the dimension (`dim`) of this op."

        self.stochastic_superops = []
        for b in self.basis.elements[1:]:
            std_superop = _lbt.nonham_lindbladian(b, b, sparse=False)
            self.stochastic_superops.append(_bt.change_basis(std_superop, 'std', self.basis))

        #Setup initial parameters
        self.params = _np.zeros(self.basis.size - 1, 'd')  # note that basis.dim can be < self.dim (OK)
        if initial_rates is not None:
            assert(len(initial_rates) == self.basis.size - 1), \
                "Expected %d initial rates but got %d!" % (len(initial_rates), self.basis.size - 1)
            self.params[:] = self._rates_to_params(initial_rates)
        assert(evotype in ("densitymx", "svterm", "cterm"))

        if evotype == "densitymx":  # for now just densitymx is supported
            rep = replib.DMOpRepDense(_np.ascontiguousarray(_np.identity(dim, 'd')))
        else:
            raise ValueError("Invalid evotype '%s' for %s" % (evotype, self.__class__.__name__))

        LinearOperator.__init__(self, rep, evotype)
        self._update_rep()  # initialize self._rep

    def _update_rep(self):
        # Create dense error superoperator from paramvec
        errormap = _np.identity(self.dim)
        for rate, ss in zip(self._params_to_rates(self.params), self.stochastic_superops):
            errormap += rate * ss
        self._rep.base[:, :] = errormap

    def _rates_to_params(self, rates):
        return _np.sqrt(_np.array(rates))

    def _params_to_rates(self, params):
        return params**2

    def _get_rate_poly_dicts(self):
        """ Return a list of dicts, one per rate, expressing the
            rate as a polynomial of the local parameters (tuple
            keys of dicts <=> poly terms, e.g. (1,1) <=> x1^2) """
        return [{(i, i): 1.0} for i in range(self.basis.size - 1)]  # rates are just parameters squared

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        StochasticNoiseOp
            A copy of this object.
        """
        copyOfMe = StochasticNoiseOp(self.dim, self.basis, self._evotype, self._params_to_rates(self.to_vector()))
        return self._copy_gpindices(copyOfMe, parent)

    #to_dense / to_sparse?
    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        return self._rep.base  # copy?

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        return replib.DMOpRepDense(_np.ascontiguousarray(self.todense(), 'd'))
    #    else:
    #        raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
    #                         (self._evotype, self.__class__.__name__))

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

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

        def _compose_poly_indices(terms):
            for term in terms:
                term.map_indices_inplace(lambda x: tuple(_modelmember._compose_gpindices(
                    self.gpindices, _np.array(x, _np.int64))))
            return terms

        IDENT = None  # sentinel for the do-nothing identity op
        mpv = max_poly_vars
        if order == 0:
            polydict = {(): 1.0}
            for pd in self._get_rate_poly_dicts():
                polydict.update({k: -v for k, v in pd.items()})  # subtracts the "rate" `pd` from `polydict`
            loc_terms = [_term.RankOnePolyOpTerm.simple_init(_Polynomial(polydict, mpv), IDENT, IDENT, self._evotype)]

        elif order == 1:
            loc_terms = [_term.RankOnePolyOpTerm.simple_init(_Polynomial(pd, mpv), bel, bel, self._evotype)
                         for i, (pd, bel) in enumerate(zip(self._get_rate_poly_dicts(), self.basis.elements[1:]))]
        else:
            loc_terms = []  # only first order "taylor terms"

        poly_coeffs = [t.coeff for t in loc_terms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)

        local_term_poly_coeffs = coeffs_as_compact_polys
        global_param_terms = _compose_poly_indices(loc_terms)

        if return_coeff_polys:
            return global_param_terms, local_term_poly_coeffs
        else:
            return global_param_terms

    def get_total_term_magnitude(self):
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
        # return exp( mag of errorgen ) = exp( sum of absvals of errgen term coeffs )
        # (unitary postfactor has weight == 1.0 so doesn't enter)
        rates = self._params_to_rates(self.to_vector())
        return _np.sum(_np.abs(rates))

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        # abs(rates) = rates = params**2
        # so d( sum(abs(rates)) )/dparam_i = 2*param_i
        return 2 * self.to_vector()

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.to_vector())

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.params

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
        self.params[:] = v
        self._update_rep()
        if not nodirty: self.dirty = True

    #Transform functions? (for gauge opt)

    def __str__(self):
        s = "Stochastic noise gate map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params())
        return s


class DepolarizeOp(StochasticNoiseOp):
    def __init__(self, dim, basis="pp", evotype="densitymx", initial_rate=0):
        """
        Create a new DepolarizeOp, representing a depolarizing channel.

        Parameters
        ----------
        dim : int
            The dimension of this operator (4 for a single qubit).

        basis : Basis or {'pp','gm','qt'}, optional
            The basis to use, defining the "principle axes"
            along which there is stochastic noise.  While strictly unnecessary
            since all complete bases yield the same operator, this affects the
            underlying :class:`StochasticNoiseOp` and so is given as an option
            to the user.

        evotype : {"densitymx", "cterm", "svterm"}
            the evolution type being used.

        initial_rate : float, optional
            the initial error rate.
        """
        num_rates = dim - 1
        initial_sto_rates = [initial_rate / num_rates] * num_rates
        StochasticNoiseOp.__init__(self, dim, basis, evotype, initial_sto_rates)

    def _rates_to_params(self, rates):
        """Note: requires rates to all be the same"""
        assert(all([rates[0] == r for r in rates[1:]]))
        return _np.array([_np.sqrt(rates[0])], 'd')

    def _params_to_rates(self, params):
        return _np.array([params[0]**2] * (self.basis.size - 1), 'd')

    def _get_rate_poly_dicts(self):
        """ Return a list of dicts, one per rate, expressing the
            rate as a polynomial of the local parameters (tuple
            keys of dicts <=> poly terms, e.g. (1,1) <=> x1^2) """
        return [{(0, 0): 1.0} for i in range(self.basis.size - 1)]  # rates are all just param0 squared

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        DepolarizeOp
            A copy of this object.
        """
        copyOfMe = DepolarizeOp(self.dim, self.basis, self.evotype, self._params_to_rates(self.to_vector())[0])
        return self._copy_gpindices(copyOfMe, parent)


class LindbladOp(LinearOperator):
    """
    A gate parameterized by the coefficients of Lindblad-like terms, which are
    exponentiated to give the gate action.
    """

    @classmethod
    def decomp_paramtype(cls, param_type):
        """
        A utility method for creating LindbladOp objects.

        Decomposes a high-level parameter-type `param_type` (e.g. `"H+S terms"`
        into a "base" type (specifies parameterization without evolution type,
        e.g. "H+S"), an evolution type (i.e. one of "densitymx", "svterm",
        "cterm", or "statevec").  Furthermore, from the base type two "modes"
        - one describing the number (and structure) of the non-Hamiltonian
        Lindblad coefficients and one describing how the Lindblad coefficients
        are converted to/from parameters - are derived.

        The "non-Hamiltonian mode" describes which non-Hamiltonian Lindblad
        coefficients are stored in a LindbladOp, and is one
        of `"diagonal"` (only the diagonal elements of the full coefficient
        matrix as a 1D array), `"diag_affine"` (a 2-by-d array of the diagonal
        coefficients on top of the affine projections), or `"all"` (the entire
        coefficient matrix).

        The "parameter mode" describes how the Lindblad coefficients/projections
        are converted into parameter values.  This can be:
        `"unconstrained"` (coefficients are independent unconstrained parameters),
        `"cptp"` (independent parameters but constrained so map is CPTP),
        `"depol"` (all non-Ham. diagonal coeffs are the *same, positive* value), or
        `"reldepol"` (same as `"depol"` but no positivity constraint).

        Parameters
        ----------
        param_type : str
            The high-level Lindblad parameter type to decompose.  E.g "H+S",
            "H+S+A terms", "CPTP clifford terms".

        Returns
        -------
        basetype : str
        evotype : str
        nonham_mode : str
        param_mode : str
        """
        bTyp, evotype = _gt.split_lindblad_paramtype(param_type)

        if bTyp == "CPTP":
            nonham_mode = "all"; param_mode = "cptp"
        elif bTyp == "H":
            nonham_mode = "all"; param_mode = "cptp"  # these don't matter since there's no non-ham errors
        elif bTyp in ("H+S", "S"):
            nonham_mode = "diagonal"; param_mode = "cptp"
        elif bTyp in ("H+s", "s"):
            nonham_mode = "diagonal"; param_mode = "unconstrained"
        elif bTyp in ("H+S+A", "S+A"):
            nonham_mode = "diag_affine"; param_mode = "cptp"
        elif bTyp in ("H+s+A", "s+A"):
            nonham_mode = "diag_affine"; param_mode = "unconstrained"
        elif bTyp in ("H+D", "D"):
            nonham_mode = "diagonal"; param_mode = "depol"
        elif bTyp in ("H+d", "d"):
            nonham_mode = "diagonal"; param_mode = "reldepol"
        elif bTyp in ("H+D+A", "D+A"):
            nonham_mode = "diag_affine"; param_mode = "depol"
        elif bTyp in ("H+d+A", "d+A"):
            nonham_mode = "diag_affine"; param_mode = "reldepol"

        elif bTyp == "GLND":
            nonham_mode = "all"; param_mode = "unconstrained"
        else:
            raise ValueError("Unrecognized base type in `param_type`=%s" % param_type)

        return bTyp, evotype, nonham_mode, param_mode

    @classmethod
    def from_operation_obj(cls, gate, param_type="GLND", unitary_postfactor=None,
                           proj_basis="pp", mx_basis="pp", truncate=True, lazy=False):
        """
        Creates a LindbladOp from an existing LinearOperator object and
        some additional information.

        This function is different from `from_operation_matrix` in that it assumes
        that `gate` is a :class:`LinearOperator`-derived object, and if `lazy=True` and
        if `gate` is already a matching LindbladOp, it is
        returned directly.  This routine is primarily used in gate conversion
        functions, where conversion is desired only when necessary.

        Parameters
        ----------
        gate : LinearOperator
            The gate object to "convert" to a `LindbladOp`.

        param_type : str
            The high-level "parameter type" of the gate to create.  This
            specifies both which Lindblad parameters are included and what
            type of evolution is used.  Examples of valid values are
            `"CPTP"`, `"H+S"`, `"S terms"`, and `"GLND clifford terms"`.

        unitary_postfactor : numpy array or SciPy sparse matrix, optional
            a square 2D array of the same dimension of `gate`.  This specifies
            a part of the gate action to remove before parameterization via
            Lindblad projections.  Typically, this is a target (desired) gate
            operation such that only the erroneous part of the gate (i.e. the
            gate relative to the target), which should be close to the identity,
            is parameterized.  If none, the identity is used by default.

        proj_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct the Lindblad-term error generators onto
            which the gate's error generator is projected.  Allowed values are
            Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `gate` cannot
            be realized by the specified set of Lindblad projections.

        lazy : bool, optional
            If True, then if `gate` is already a LindbladOp
            with the requested details (given by the other arguments), then
            `gate` is returned directly and no conversion/copying is performed.
            If False, then a new gate object is always created and returned.

        Returns
        -------
        LindbladOp
        """
        RANK_TOL = 1e-6

        if unitary_postfactor is None:
            #Try to obtain unitary_post by getting the closest unitary
            if isinstance(gate, LindbladDenseOp):
                unitary_postfactor = gate.unitary_postfactor
            elif isinstance(gate, LinearOperator) and gate._evotype == "densitymx":
                J = _jt.fast_jamiolkowski_iso_std(gate.todense(), mx_basis)  # Choi mx basis doesn't matter
                if _np.linalg.matrix_rank(J, RANK_TOL) == 1:
                    unitary_postfactor = gate  # when 'gate' is unitary
            # FUTURE: support other gate._evotypes?
            else:
                unitary_postfactor = None

        #Break param_type in to a "base" type and an evotype
        bTyp, evotype, nonham_mode, param_mode = cls.decomp_paramtype(param_type)

        ham_basis = proj_basis if (("H+" in bTyp) or bTyp in ("CPTP", "GLND")) else None
        nonham_basis = proj_basis

        def beq(b1, b2):
            """ Check if bases have equal names """
            if not isinstance(b1, _Basis):  # b1 may be a string, in which case create a Basis
                b1 = _BuiltinBasis(b1, b2.dim, b2.sparse)  # from b2, which *will* be a Basis
            return b1 == b2

        def normeq(a, b):
            if a is None and b is None: return True
            if a is None or b is None: return False
            return _mt.safenorm(a - b) < 1e-6  # what about possibility of Clifford gates?

        if lazy and isinstance(gate, LindbladOp) and \
           normeq(gate.unitary_postfactor, unitary_postfactor) and \
           isinstance(gate.errorgen, LindbladErrorgen) \
           and beq(ham_basis, gate.errorgen.ham_basis) and beq(nonham_basis, gate.errorgen.other_basis) \
           and param_mode == gate.errorgen.param_mode and nonham_mode == gate.errorgen.nonham_mode \
           and beq(mx_basis, gate.errorgen.matrix_basis) and gate._evotype == evotype:
            return gate  # no creation necessary!
        else:
            return cls.from_operation_matrix(
                gate, unitary_postfactor, ham_basis, nonham_basis, param_mode,
                nonham_mode, truncate, mx_basis, evotype)

    @classmethod
    def from_operation_matrix(cls, op_matrix, unitary_postfactor=None,
                              ham_basis="pp", nonham_basis="pp", param_mode="cptp",
                              nonham_mode="all", truncate=True, mx_basis="pp",
                              evotype="densitymx"):
        """
        Creates a Lindblad-parameterized gate from a matrix and a basis which
        specifies how to decompose (project) the gate's error generator.

        Parameters
        ----------
        op_matrix : numpy array or SciPy sparse matrix
            a square 2D array that gives the raw operation matrix, assumed to
            be in the `mx_basis` basis, to parameterize.  The shape of this
            array sets the dimension of the gate. If None, then it is assumed
            equal to `unitary_postfactor` (which cannot also be None). The
            quantity `op_matrix inv(unitary_postfactor)` is parameterized via
            projection onto the Lindblad terms.

        unitary_postfactor : numpy array or SciPy sparse matrix, optional
            a square 2D array of the same size of `op_matrix` (if
            not None).  This matrix specifies a part of the gate action
            to remove before parameterization via Lindblad projections.
            Typically, this is a target (desired) gate operation such
            that only the erroneous part of the gate (i.e. the gate
            relative to the target), which should be close to the identity,
            is parameterized.  If none, the identity is used by default.

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        other_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Stochastic-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            gate's parameter values.  Allowed values are:
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

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the gate being constructed.  `"densitymx"` is
            usual Lioville density-matrix-vector propagation via matrix-vector
            products.  `"svterm"` denotes state-vector term-based evolution
            (action of gate is obtained by evaluating the rank-1 terms up to
            some order).  `"cterm"` is similar but uses Clifford gate action
            on stabilizer states.

        Returns
        -------
        LindbladOp
        """

        #Compute a (errgen, unitary_postfactor) pair from the given
        # (op_matrix, unitary_postfactor) pair.  Works with both
        # dense and sparse matrices.

        if op_matrix is None:
            assert(unitary_postfactor is not None), "arguments cannot both be None"
            op_matrix = unitary_postfactor

        sparseOp = _sps.issparse(op_matrix)
        if unitary_postfactor is None:
            if sparseOp:
                upost = _sps.identity(op_matrix.shape[0], 'd', 'csr')
            else: upost = _np.identity(op_matrix.shape[0], 'd')
        else: upost = unitary_postfactor

        #Init base from error generator: sets basis members and ultimately
        # the parameters in self.paramvals
        if sparseOp:
            #Instead of making error_generator(...) compatible with sparse matrices
            # we require sparse matrices to have trivial initial error generators
            # or we convert to dense:
            if(_mt.safenorm(op_matrix - upost) < 1e-8):
                errgenMx = _sps.csr_matrix(op_matrix.shape, dtype='d')  # all zeros
            else:
                errgenMx = _sps.csr_matrix(
                    _gt.error_generator(op_matrix.toarray(), upost.toarray(),
                                        mx_basis, "logGTi"), dtype='d')
        else:
            #DB: assert(_np.linalg.norm(op_matrix.imag) < 1e-8)
            #DB: assert(_np.linalg.norm(upost.imag) < 1e-8)
            errgenMx = _gt.error_generator(op_matrix, upost, mx_basis, "logGTi")

        errgen = LindbladErrorgen.from_error_generator(errgenMx, ham_basis,
                                                       nonham_basis, param_mode, nonham_mode,
                                                       mx_basis, truncate, evotype)

        #Use "sparse" matrix exponentiation when given operation matrix was sparse.
        return cls(unitary_postfactor, errgen, dense_rep=not sparseOp)

    def __init__(self, unitary_postfactor, errorgen, dense_rep=False):
        """
        Create a new `LinbladOp` based on an error generator and postfactor.

        Note that if you want to construct a `LinbladOp` from an operation
        matrix, you can use the :method:`from_operation_matrix` class
        method and save youself some time and effort.

        Parameters
        ----------
        unitary_postfactor : numpy array or SciPy sparse matrix or int
            a square 2D array which specifies a part of the gate action
            to remove before parameterization via Lindblad projections.
            While this is termed a "post-factor" because it occurs to the
            right of the exponentiated Lindblad terms, this means it is applied
            to a state *before* the Lindblad terms (which usually represent
            gate errors).  Typically, this is a target (desired) gate operation.
            If this post-factor is just the identity you can simply pass the
            integer dimension as `unitary_postfactor` instead of a matrix, or
            you can pass `None` and the dimension will be inferred from
            `errorgen`.

        errorgen : LinearOperator
            The error generator for this operator.  That is, the `L` if this
            operator is `exp(L)*unitary_postfactor`.

        dense_rep : bool, optional
            Whether to internally implement this operation as a dense matrix.
            If `True` the error generator is rendered as a dense matrix and
            exponentiation is "exact".  If `False`, then this operation
            implements exponentiation in an approximate way that treats the
            error generator as a sparse matrix and only uses its action (and
            its adjoint's action) on a state.  Setting `dense_rep=False` is
            typically more efficient when `errorgen` has a large dimension,
            say greater than 100.
        """

        # Extract superop dimension from 'errorgen'
        d2 = errorgen.dim
        d = int(round(_np.sqrt(d2)))
        assert(d * d == d2), "LinearOperator dim must be a perfect square"

        self.errorgen = errorgen  # don't copy (allow object reuse)

        evotype = self.errorgen._evotype
        if evotype in ("svterm", "cterm"):
            dense_rep = True  # we need *dense* unitary postfactors for the term-based processing below
        self.dense_rep = dense_rep

        # make unitary postfactor sparse when dense_rep == False and vice versa.
        # (This doens't have to be the case, but we link these two "sparseness" notions:
        #  when we perform matrix exponentiation in a "sparse" way we assume the matrices
        #  are large and so the unitary postfactor (if present) should be sparse).
        # FUTURE: warn if there is a sparsity mismatch btwn basis and postfactor?
        if unitary_postfactor is not None:
            if self.dense_rep and _sps.issparse(unitary_postfactor):
                unitary_postfactor = unitary_postfactor.toarray()  # sparse -> dense
            elif not self.dense_rep and not _sps.issparse(unitary_postfactor):
                unitary_postfactor = _sps.csr_matrix(_np.asarray(unitary_postfactor))  # dense -> sparse

        #Finish initialization based on evolution type
        if evotype == "densitymx":
            self.unitary_postfactor = unitary_postfactor  # can be None
            #self.err_gen_prep = None REMOVE

            #Pre-compute the exponential of the error generator if dense matrices
            # are used, otherwise cache prepwork for sparse expm calls
            if self.dense_rep:
                rep = replib.DMOpRepDense(_np.ascontiguousarray(_np.identity(d2, 'd'), 'd'))
            else:
                # "sparse mode" => don't ever compute matrix-exponential explicitly

                #Allocate sparse matrix arrays for rep
                if self.unitary_postfactor is None:
                    Udata = _np.empty(0, 'd')
                    Uindices = _np.empty(0, _np.int64)
                    Uindptr = _np.zeros(1, _np.int64)
                else:
                    assert(_sps.isspmatrix_csr(self.unitary_postfactor)), \
                        "Internal error! Unitary postfactor should be a *sparse* CSR matrix!"
                    Udata = self.unitary_postfactor.data
                    Uindptr = _np.ascontiguousarray(self.unitary_postfactor.indptr, _np.int64)
                    Uindices = _np.ascontiguousarray(self.unitary_postfactor.indices, _np.int64)

                mu, m_star, s, eta = 1.0, 0, 0, 1.0  # initial values - will be updated by call to _update_rep below
                rep = replib.DMOpRepLindblad(self.errorgen._rep,
                                             mu, eta, m_star, s,
                                             Udata, Uindices, Uindptr)

        else:  # Term-based evolution

            assert(self.dense_rep), "Sparse unitary postfactors are not supported for term-based evolution"
            #TODO: make terms init-able from sparse elements, and below code work with a *sparse* unitary_postfactor
            termtype = "dense" if evotype == "svterm" else "clifford"

            # Store *unitary* as self.unitary_postfactor - NOT a superop
            if unitary_postfactor is not None:  # can be None
                op_std = _bt.change_basis(unitary_postfactor, self.errorgen.matrix_basis, 'std')
                self.unitary_postfactor = _gt.process_mx_to_unitary(op_std)

                # automatically "up-convert" gate to CliffordOp if needed
                if termtype == "clifford":
                    self.unitary_postfactor = CliffordOp(self.unitary_postfactor)
            else:
                self.unitary_postfactor = None

            rep = d2  # no representation object in term-mode (will be set to None by LinearOperator)

        #Cache values
        self.terms = {}
        self.exp_terms_cache = {}  # used for repeated calls to the exp_terms function
        self.local_term_poly_coeffs = {}
        self.exp_err_gen = None   # used for dense_rep=True mode to cache qty needed in deriv_wrt_params
        self.base_deriv = None
        self.base_hessian = None
        # TODO REMOVE self.direct_terms = {}
        # TODO REMOVE self.direct_term_poly_coeffs = {}

        LinearOperator.__init__(self, rep, evotype)
        self._update_rep()  # updates self._rep
        #Done with __init__(...)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.errorgen]

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that error map has its
        # parent reset correctly.
        if self.unitary_postfactor is None:
            upost = None
        elif self._evotype == "densitymx":
            upost = self.unitary_postfactor
        else:
            #self.unitary_postfactor is actually the *unitary* not the postfactor
            termtype = "dense" if self._evotype == "svterm" else "clifford"

            # automatically "up-convert" gate to CliffordOp if needed
            if termtype == "clifford":
                assert(isinstance(self.unitary_postfactor, CliffordOp))  # see __init__
                U = self.unitary_postfactor.unitary
            else: U = self.unitary_postfactor
            op_std = _gt.unitary_to_process_mx(U)
            upost = _bt.change_basis(op_std, 'std', self.errorgen.matrix_basis)

        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(upost, self.errorgen.copy(parent), self.dense_rep)
        return self._copy_gpindices(copyOfMe, parent)

    def _update_rep(self, close=False):
        """
        Updates self._rep as needed after parameters have changed.
        """
        if self._evotype == "densitymx":
            if self.dense_rep:  # "sparse mode" => don't ever compute matrix-exponential explicitly
                self.exp_err_gen = _spl.expm(self.errorgen.todense())  # used in deriv_wrt_params
                if self.unitary_postfactor is not None:
                    dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
                else: dense = self.exp_err_gen
                self._rep.base.flags.writeable = True
                self._rep.base[:, :] = dense
                self._rep.base.flags.writeable = False
                self.base_deriv = None
                self.base_hessian = None
            elif not close:
                # don't reset matrix exponential params (based on operator norm) when vector hasn't changed much
                mu, m_star, s, eta = _mt.expop_multiply_prep(
                    self.errorgen._rep.aslinearoperator(),
                    a_1_norm=self.errorgen.onenorm_upperbound())
                self._rep.set_exp_params(mu, eta, m_star, s)

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
        self.exp_terms_cache = {}
        self.local_term_poly_coeffs = {}
        #TODO REMOVE self.direct_terms = {}
        #TODO REMOVE self.direct_term_poly_coeffs = {}
        _modelmember.ModelMember.set_gpindices(self, gpindices, parent, memo)

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        if self._evotype == "densitymx" and self.dense_rep:
            # Then self._rep contains a dense version already
            return self._rep.base  # copy() unnecessary since we set to readonly

        else:
            # Construct a dense version from scratch (more time consuming)
            exp_errgen = _spl.expm(self.errorgen.todense())

            if self.unitary_postfactor is not None:
                if self._evotype in ("svterm", "cterm"):
                    if self._evotype == "cterm":
                        assert(isinstance(self.unitary_postfactor, CliffordOp))  # see __init__
                        U = self.unitary_postfactor.unitary
                    else: U = self.unitary_postfactor
                    op_std = _gt.unitary_to_process_mx(U)
                    upost = _bt.change_basis(op_std, 'std', self.errorgen.matrix_basis)
                else:
                    upost = self.unitary_postfactor

                dense = _mt.safedot(exp_errgen, upost)
            else:
                dense = exp_errgen
            return dense

    #FUTURE: maybe remove this function altogether, as it really shouldn't be called
    def tosparse(self):
        """
        Return the operation as a sparse matrix.
        """
        _warnings.warn(("Constructing the sparse matrix of a LindbladDenseOp."
                        "  Usually this is *NOT* acutally sparse (the exponential of a"
                        " sparse matrix isn't generally sparse)!"))
        if self.dense_rep:
            return _sps.csr_matrix(self.todense())
        else:
            exp_err_gen = _spsl.expm(self.errorgen.tosparse().tocsc()).tocsr()
            if self.unitary_postfactor is not None:
                return exp_err_gen.dot(self.unitary_postfactor)
            else:
                return exp_err_gen

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        if self.sparse_expm:
    #            if self.unitary_postfactor is None:
    #                Udata = _np.empty(0, 'd')
    #                Uindices = Uindptr = _np.empty(0, _np.int64)
    #            else:
    #                assert(_sps.isspmatrix_csr(self.unitary_postfactor)), \
    #                    "Internal error! Unitary postfactor should be a *sparse* CSR matrix!"
    #                Udata = self.unitary_postfactor.data
    #                Uindptr = _np.ascontiguousarray(self.unitary_postfactor.indptr, _np.int64)
    #                Uindices = _np.ascontiguousarray(self.unitary_postfactor.indices, _np.int64)
    #
    #            mu, m_star, s, eta = self.err_gen_prep
    #            errorgen_rep = self.errorgen.torep()
    #            return replib.DMOpRepLindblad(errorgen_rep,
    #                                           mu, eta, m_star, s,
    #                                           Udata, Uindices, Uindptr) # HERE
    #        else:
    #            if self.unitary_postfactor is not None:
    #                dense = _np.dot(self.exp_err_gen, self.unitary_postfactor)
    #            else: dense = self.exp_err_gen
    #            return replib.DMOpRepDense(_np.ascontiguousarray(dense, 'd'))
    #    else:
    #        raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
    #                         (self._evotype, self.__class__.__name__))

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        if not self.dense_rep:
            raise NotImplementedError("deriv_wrt_params is only implemented for *dense-rep* LindbladOps")
            # because we need self.unitary_postfactor to be a dense operation below (and it helps to
            # have self.exp_err_gen cached)

        if self.base_deriv is None:
            d2 = self.dim

            #Deriv wrt hamiltonian params
            derrgen = self.errorgen.deriv_wrt_params(None)  # apply filter below; cache *full* deriv
            derrgen.shape = (d2, d2, -1)  # separate 1st d2**2 dim to (d2,d2)
            dexpL = _d_exp_x(self.errorgen.todense(), derrgen, self.exp_err_gen,
                             self.unitary_postfactor)
            derivMx = dexpL.reshape(d2**2, self.num_params())  # [iFlattenedOp,iParam]

            assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL), \
                ("Deriv matrix has imaginary part = %s.  This can result from "
                 "evaluating a Model derivative at a 'bad' point where the "
                 "error generator is large.  This often occurs when GST's "
                 "starting Model has *no* stochastic error and all such "
                 "parameters affect error rates at 2nd order.  Try "
                 "depolarizing the seed Model.") % str(_np.linalg.norm(_np.imag(derivMx)))
            # if this fails, uncomment around "DB COMMUTANT NORM" for further debugging.
            derivMx = _np.real(derivMx)
            self.base_deriv = derivMx

            #check_deriv_wrt_params(self, derivMx, eps=1e-7)
            #fd_deriv = finite_difference_deriv_wrt_params(self, wrt_filter, eps=1e-7)
            #derivMx = fd_deriv

        if wrt_filter is None:
            return self.base_deriv.view()
            #view because later setting of .shape by caller can mess with self.base_deriv!
        else:
            return _np.take(self.base_deriv, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return True

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this operation with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1, wrt_filter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        if not self.dense_rep:
            raise NotImplementedError("hessian_wrt_params is only implemented for *dense-rep* LindbladOps")
            # because we need self.unitary_postfactor to be a dense operation below (and it helps to
            # have self.exp_err_gen cached)

        if self.base_hessian is None:
            d2 = self.dim
            nP = self.num_params()
            hessianMx = _np.zeros((d2**2, nP, nP), 'd')

            #Deriv wrt other params
            dEdp = self.errorgen.deriv_wrt_params(None)  # filter later, cache *full*
            d2Edp2 = self.errorgen.hessian_wrt_params(None, None)  # hessian
            dEdp.shape = (d2, d2, nP)  # separate 1st d2**2 dim to (d2,d2)
            d2Edp2.shape = (d2, d2, nP, nP)  # ditto

            series, series2 = _d2_exp_series(self.errorgen.todense(), dEdp, d2Edp2)
            term1 = series2
            term2 = _np.einsum("ija,jkq->ikaq", series, series)
            if self.unitary_postfactor is None:
                d2expL = _np.einsum("ikaq,kj->ijaq", term1 + term2,
                                    self.exp_err_gen)
            else:
                d2expL = _np.einsum("ikaq,kl,lj->ijaq", term1 + term2,
                                    self.exp_err_gen, self.unitary_postfactor)
            hessianMx = d2expL.reshape((d2**2, nP, nP))

            #hessian has been made so index as [iFlattenedOp,iDeriv1,iDeriv2]
            assert(_np.linalg.norm(_np.imag(hessianMx)) < IMAG_TOL)
            hessianMx = _np.real(hessianMx)  # d2O block of hessian

            self.base_hessian = hessianMx

            #TODO: check hessian with finite difference here?

        if wrt_filter1 is None:
            if wrt_filter2 is None:
                return self.base_hessian.view()
                #view because later setting of .shape by caller can mess with self.base_hessian!
            else:
                return _np.take(self.base_hessian, wrt_filter2, axis=2)
        else:
            if wrt_filter2 is None:
                return _np.take(self.base_hessian, wrt_filter1, axis=1)
            else:
                return _np.take(_np.take(self.base_hessian, wrt_filter1, axis=1),
                                wrt_filter2, axis=2)

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


        Parameters
        ----------
        order : int
            Which order terms (in a Taylor expansion of this :class:`LindbladOp`)
            to retrieve.

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
            self._compute_taylor_order_terms(order, max_poly_vars)

        if return_coeff_polys:
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_poly_vars):  # separated for profiling

        mapvec = _np.ascontiguousarray(_np.zeros(max_poly_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[ii] = i

        def _compose_poly_indices(terms):
            for term in terms:
                #term.map_indices_inplace(lambda x: tuple(_modelmember._compose_gpindices(
                #    self.gpindices, _np.array(x, _np.int64))))
                term.mapvec_indices_inplace(mapvec)
            return terms

        assert(self.gpindices is not None), "LindbladOp must be added to a Model before use!"
        assert(not _sps.issparse(self.unitary_postfactor)
               ), "Unitary post-factor needs to be dense for term-based evotypes"
        # for now - until StaticDenseOp and CliffordOp can init themselves from a *sparse* matrix
        mpv = max_poly_vars
        postTerm = _term.RankOnePolyOpTerm.simple_init(_Polynomial({(): 1.0}, mpv), self.unitary_postfactor,
                                                       self.unitary_postfactor, self._evotype)
        #Note: for now, *all* of an error generator's terms are considered 0-th order,
        # so the below call to get_taylor_order_terms just gets all of them.  In the FUTURE
        # we might want to allow a distinction among the error generator terms, in which
        # case this term-exponentiation step will need to become more complicated...
        loc_terms = _term.exp_terms(self.errorgen.get_taylor_order_terms(0, max_poly_vars),
                                    order, postTerm, self.exp_terms_cache)
        #OLD: loc_terms = [ t.collapse() for t in loc_terms ] # collapse terms for speed

        poly_coeffs = [t.coeff for t in loc_terms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)
        self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

        # only cache terms with *global* indices to avoid confusion...
        self.terms[order] = _compose_poly_indices(loc_terms)

    def get_taylor_order_terms_above_mag(self, order, max_poly_vars, min_term_mag):

        mapvec = _np.ascontiguousarray(_np.zeros(max_poly_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[ii] = i

        assert(self.gpindices is not None), "LindbladOp must be added to a Model before use!"
        assert(not _sps.issparse(self.unitary_postfactor)
               ), "Unitary post-factor needs to be dense for term-based evotypes"
        # for now - until StaticDenseOp and CliffordOp can init themselves from a *sparse* matrix
        mpv = max_poly_vars
        postTerm = _term.RankOnePolyOpTerm.simple_init(_Polynomial({(): 1.0}, mpv), self.unitary_postfactor,
                                                       self.unitary_postfactor, self._evotype)
        postTerm = postTerm.copy_with_magnitude(1.0)
        #Note: for now, *all* of an error generator's terms are considered 0-th order,
        # so the below call to get_taylor_order_terms just gets all of them.  In the FUTURE
        # we might want to allow a distinction among the error generator terms, in which
        # case this term-exponentiation step will need to become more complicated...
        errgen_terms = self.errorgen.get_taylor_order_terms(0, max_poly_vars)

        #DEBUG: CHECK MAGS OF ERRGEN COEFFS
        #poly_coeffs = [t.coeff for t in errgen_terms]
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #v = self.to_vector()
        #errgen_coeffs = _bulk_eval_compact_polys_complex(
        #    vtape, ctape, v, (len(errgen_terms),))  # an array of coeffs
        #for coeff, t in zip(errgen_coeffs, errgen_terms):
        #    coeff2 = t.coeff.evaluate(v)
        #    if not _np.isclose(coeff,coeff2):
        #        assert(False), "STOP"
        #    t.set_magnitude(abs(coeff))

        #evaluate errgen_terms' coefficients using their local vector of parameters
        # (which happends to be the same as our paramvec in this case)
        egvec = self.errorgen.to_vector()
        errgen_terms = [egt.copy_with_magnitude(abs(egt.coeff.evaluate(egvec))) for egt in errgen_terms]

        #DEBUG!!!
        #import bpdb; bpdb.set_trace()
        #loc_terms = _term.exp_terms_above_mag(errgen_terms, order, postTerm, min_term_mag=-1)
        #loc_terms_chk = _term.exp_terms(errgen_terms, order, postTerm)
        #assert(len(loc_terms) == len(loc_terms2))
        #poly_coeffs = [t.coeff for t in loc_terms_chk]
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #v = self.to_vector()
        #coeffs = _bulk_eval_compact_polys_complex(
        #    vtape, ctape, v, (len(loc_terms_chk),))  # an array of coeffs
        #for coeff, t, t2 in zip(coeffs, loc_terms, loc_terms_chk):
        #    coeff2 = t.coeff.evaluate(v)
        #    if not _np.isclose(coeff,coeff2):
        #        assert(False), "STOP"
        #    t.set_magnitude(abs(coeff))

        #for ii,t in enumerate(loc_terms):
        #    coeff1 = t.coeff.evaluate(egvec)
        #    if not _np.isclose(abs(coeff1), t.magnitude):
        #        assert(False),"STOP"
        #    #t.set_magnitude(abs(t.coeff.evaluate(egvec)))

        #FUTURE:  maybe use bulk eval of compact polys? Something like this:
        #coeffs = _bulk_eval_compact_polys_complex(
        #    cpolys[0], cpolys[1], v, (len(terms_at_order),))  # an array of coeffs
        #for coeff, t in zip(coeffs, terms_at_order):
        #    t.set_magnitude(abs(coeff))

        terms = []
        for term in _term.exp_terms_above_mag(errgen_terms, order,
                                              postTerm, min_term_mag=min_term_mag):
            #poly_coeff = term.coeff
            #compact_poly_coeff = poly_coeff.compact(complex_coeff_tape=True)
            term.mapvec_indices_inplace(mapvec)  # local -> global indices

            # CHECK - to ensure term magnitudes are being set correctly (i.e. are in sync with evaluated coeffs)
            # REMOVE later
            # t = term
            # vt, ct = t._rep.coeff.compact_complex()
            # coeff_array = _bulk_eval_compact_polys_complex(vt, ct, self.parent.to_vector(), (1,))
            # if not _np.isclose(abs(coeff_array[0]), t._rep.magnitude):  # DEBUG!!!
            #     print(coeff_array[0], "vs.", t._rep.magnitude)
            #     import bpdb; bpdb.set_trace()
            #     c1 = _Polynomial.fromrep(t._rep.coeff)

            terms.append(term)
        return terms

    def get_total_term_magnitude(self):
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
        # return exp( mag of errorgen ) = exp( sum of absvals of errgen term coeffs )
        # (unitary postfactor has weight == 1.0 so doesn't enter)
        #TODO REMOVE:
        #print("  DB: LindbladOp.get_totat_term_magnitude: (errgen type =",self.errorgen.__class__.__name__)
        #egttm = self.errorgen.get_total_term_magnitude()
        #print("  DB: exp(", egttm, ") = ",_np.exp(egttm))
        #return _np.exp(egttm)
        return _np.exp(self.errorgen.get_total_term_magnitude())

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        return _np.exp(self.errorgen.get_total_term_magnitude()) * self.errorgen.get_total_term_magnitude_deriv()

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.errorgen.num_params()

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.errorgen.to_vector()

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
        self.errorgen.from_vector(v, close, nodirty)
        self._update_rep(close)
        if not nodirty: self.dirty = True

    def get_errgen_coeffs(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients
        of this operation.  Note that these are not  necessarily the parameter
        values, as these coefficients are generally functions of the parameters
        (so as to keep the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool, optional
            Whether to also return a :class:`Basis` containing the elements
            with which the error generator terms were constructed.

        logscale_nonham : bool, optional
            Whether or not the non-hamiltonian error generator coefficients
            should be scaled so that the returned dict contains:
            `(1 - exp(-d^2 * coeff)) / d^2` instead of `coeff`.  This
            essentially converts the coefficient into a rate that is
            the contribution this term would have within a depolarizing
            channel where all stochastic generators had this same coefficient.
            This is the value returned by :method:`get_error_rates`.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients.

        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `lindblad_term_dict` to basis matrices.
        """
        return self.errorgen.get_coeffs(return_basis, logscale_nonham)

    def get_error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this
        operation.

        The "error rate" for an individual Hamiltonian error is the angle
        about the "axis" (generalized in the multi-qubit case)
        corresponding to a particular basis element, i.e. `theta` in
        the unitary channel `U = exp(i * theta/2 * BasisElement)`.

        The "error rate" for an individual Stochastic error is the
        contribution that basis element's term would have to the
        error rate of a depolarization channel.  For example, if
        the rate corresponding to the term ('S','X') is 0.01 this
        means that the coefficient of the rho -> X*rho*X-rho error
        generator is set such that if this coefficient were used
        for all 3 (X,Y, and Z) terms the resulting depolarizing
        channel would have error rate 3*0.01 = 0.03.

        Note that because error generator terms do not necessarily
        commute with one another, the sum of the returned error
        rates is not necessarily the error rate of the overall
        channel.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case.
        """
        return self.get_errgen_coeffs(return_basis=False, logscale_nonham=True)

    def set_errgen_coeffs(self, lindblad_term_dict, action="update", logscale_nonham=False):
        """
        Sets the coefficients of terms in the error generator of this :class:`LindbladOp`.
        The dictionary `lindblad_term_dict` has tuple-keys describing the type of term and the basis
        elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are the coefficients of these error generators,
            and should be real except for the 2-basis-label case.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :method:`get_errgen_coeffs`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :method:`set_error_rates`.

        Returns
        -------
        None
        """
        self.errorgen.set_coeffs(lindblad_term_dict, action, logscale_nonham)
        self._update_rep()
        self.dirty = True

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in the error generator of this :class:`LindbladOp`
        so that the contributions of the resulting channel's error rate are given by
        the values in `lindblad_term_dict`.  See :method:`get_error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case, when they may be complex.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_errgen_coeffs(lindblad_term_dict, action, logscale_nonham=True)

    def set_value(self, m):
        """
        Attempts to modify gate parameters so that the specified raw
        operation matrix becomes mx.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        m : array_like or LinearOperator
            An array of shape (dim, dim) or LinearOperator representing the gate action.

        Returns
        -------
        None
        """

        #TODO: move this function to errorgen?
        if not isinstance(self.errorgen, LindbladErrorgen):
            raise NotImplementedError(("Can only set the value of a LindbladDenseOp that "
                                       "contains a single LindbladErrorgen error generator"))

        tOp = LindbladOp.from_operation_matrix(
            m, self.unitary_postfactor,
            self.errorgen.ham_basis, self.errorgen.other_basis,
            self.errorgen.param_mode, self.errorgen.nonham_mode,
            True, self.errorgen.matrix_basis, self._evotype)

        #Note: truncate=True to be safe
        self.errorgen.from_vector(tOp.errorgen.to_vector())
        self._update_rep()
        self.dirty = True

    def transform(self, s):
        """
        Update operation matrix G with inv(s) * G * s,

        Generally, the transform function updates the *parameters* of
        the gate such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.get_transform_matrix()
            Uinv = s.get_transform_matrix_inverse()
            #assert(_np.allclose(U, _np.linalg.inv(Uinv)))

            #just conjugate postfactor and Lindbladian exponent by U:
            if self.unitary_postfactor is not None:
                self.unitary_postfactor = _mt.safedot(Uinv, _mt.safedot(self.unitary_postfactor, U))
            self.errorgen.transform(s)
            self._update_rep()  # needed to rebuild exponentiated error gen
            self.dirty = True

            #CHECK WITH OLD (passes) TODO move to unit tests?
            #tMx = _np.dot(Uinv,_np.dot(self.base, U)) #Move above for checking
            #tOp = LindbladDenseOp(tMx,self.unitary_postfactor,
            #                                self.ham_basis, self.other_basis,
            #                                self.cptp,self.nonham_diagonal_only,
            #                                True, self.matrix_basis)
            #assert(_np.linalg.norm(tOp.paramvals - self.paramvals) < 1e-6)
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(s)))

    def spam_transform(self, s, typ):
        """
        Update operation matrix G with inv(s) * G OR G * s,
        depending on the value of `typ`.

        This functions as `transform(...)` but is used when this
        Lindblad-parameterized gate is used as a part of a SPAM
        vector.  When `typ == "prep"`, the spam vector is assumed
        to be `rho = dot(self, <spamvec>)`, which transforms as
        `rho -> inv(s) * rho`, so `self -> inv(s) * self`. When
        `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
        `self` is NOT `self.dag` here), and `e.dag -> e.dag * s`
        so that `self -> self * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        assert(typ in ('prep', 'effect')), "Invalid `typ` argument: %s" % typ

        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.get_transform_matrix()
            Uinv = s.get_transform_matrix_inverse()

            #Note: this code may need to be tweaked to work with sparse matrices
            if typ == "prep":
                tMx = _mt.safedot(Uinv, self.todense())
            else:
                tMx = _mt.safedot(self.todense(), U)
            trunc = bool(isinstance(s, _gaugegroup.UnitaryGaugeGroupElement))
            tOp = LindbladOp.from_operation_matrix(tMx, self.unitary_postfactor,
                                                   self.errorgen.ham_basis, self.errorgen.other_basis,
                                                   self.errorgen.param_mode, self.errorgen.nonham_mode,
                                                   trunc, self.errorgen.matrix_basis)
            self.from_vector(tOp.to_vector())
            #Note: truncate=True above for unitary transformations because
            # while this trunctation should never be necessary (unitaries map CPTP -> CPTP)
            # sometimes a unitary transform can modify eigenvalues to be negative beyond
            # the tight tolerances checked when truncate == False. Maybe we should be able
            # to give a tolerance as `truncate` in the future?

            #NOTE: This *doesn't* work as it does in the 'gate' case b/c this isn't a
            # similarity transformation!
            ##just act on postfactor and Lindbladian exponent:
            #if typ == "prep":
            #    if self.unitary_postfactor is not None:
            #        self.unitary_postfactor = _mt.safedot(Uinv, self.unitary_postfactor)
            #else:
            #    if self.unitary_postfactor is not None:
            #        self.unitary_postfactor = _mt.safedot(self.unitary_postfactor, U)
            #
            #self.errorgen.spam_transform(s, typ)
            #self._update_rep()  # needed to rebuild exponentiated error gen
            #self.dirty = True
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(s)))

    def __str__(self):
        s = "Lindblad Parameterized gate map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params())
        return s


class LindbladDenseOp(LindbladOp, DenseOperatorInterface):
    """
    Encapsulates a operation matrix that is parameterized by a Lindblad-form
    expression, such that each parameter multiplies a particular term in
    the Lindblad form that is exponentiated to give the operation matrix up
    to an optional unitary prefactor).  The basis used by the Lindblad
    form is referred to as the "projection basis".
    """

    def __init__(self, unitary_postfactor, errorgen, dense_rep=True):
        """
        Create a new LinbladDenseOp based on an error generator and postfactor.

        Note that if you want to construct a `LinbladDenseOp` from an operation
        matrix, you can use the :method:`from_operation_matrix` class method
        and save youself some time and effort.

        Parameters
        ----------
        unitary_postfactor : numpy array or SciPy sparse matrix or int
            a square 2D array which specifies a part of the gate action
            to remove before parameterization via Lindblad projections.
            While this is termed a "post-factor" because it occurs to the
            right of the exponentiated Lindblad terms, this means it is applied
            to a state *before* the Lindblad terms (which usually represent
            gate errors).  Typically, this is a target (desired) gate operation.
            If this post-factor is just the identity you can simply pass the
            integer dimension as `unitary_postfactor` instead of a matrix, or
            you can pass `None` and the dimension will be inferred from
            `errorgen`.

        errorgen : LinearOperator
            The error generator for this operator.  That is, the `L` if this
            operator is `exp(L)*unitary_postfactor`.
        """
        assert(dense_rep), "LindbladDenseOp must be created with `dense_rep == True`"
        assert(errorgen._evotype == "densitymx"), \
            "LindbladDenseOp objects can only be used for the 'densitymx' evolution type"
        #Note: cannot remove the evotype argument b/c we need to maintain the same __init__
        # signature as LindbladOp so its @classmethods will work on us.

        #Start with base class construction
        LindbladOp.__init__(self, unitary_postfactor, errorgen, dense_rep=True)
        DenseOperatorInterface.__init__(self)


def _d_exp_series(x, dx):
    TERM_TOL = 1e-12
    tr = len(dx.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    assert((tr - 2) in (1, 2)), "Currently, dx can only have 1 or 2 derivative dimensions"
    #assert( len( (_np.isnan(dx)).nonzero()[0] ) == 0 ) # NaN debugging
    #assert( len( (_np.isnan(x)).nonzero()[0] ) == 0 ) # NaN debugging
    series = dx.copy()  # accumulates results, so *need* a separate copy
    last_commutant = term = dx; i = 2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL:  # _np.linalg.norm(term)
        if tr == 3:
            #commutant = _np.einsum("ik,kja->ija",x,last_commutant) - \
            #            _np.einsum("ika,kj->ija",last_commutant,x)
            commutant = _np.tensordot(x, last_commutant, (1, 0)) - \
                _np.transpose(_np.tensordot(last_commutant, x, (1, 0)), (0, 2, 1))
        elif tr == 4:
            #commutant = _np.einsum("ik,kjab->ijab",x,last_commutant) - \
            #        _np.einsum("ikab,kj->ijab",last_commutant,x)
            commutant = _np.tensordot(x, last_commutant, (1, 0)) - \
                _np.transpose(_np.tensordot(last_commutant, x, (1, 0)), (0, 3, 1, 2))

        term = 1 / _np.math.factorial(i) * commutant

        #Uncomment some/all of this when you suspect an overflow due to x having large norm.
        #print("DB COMMUTANT NORM = ",_np.linalg.norm(commutant)) # sometimes this increases w/iter -> divergence => NaN
        #assert(not _np.isnan(_np.linalg.norm(term))), \
        #    ("Haddamard series = NaN! Probably due to trying to differentiate "
        #     "exp(x) where x has a large norm!")

        #DEBUG
        #if not _np.isfinite(_np.linalg.norm(term)): break # DEBUG high values -> overflow for nqubit gates
        #if len( (_np.isnan(term)).nonzero()[0] ) > 0: # NaN debugging
        #    #WARNING: stopping early b/c of NaNs!!! - usually caused by infs
        #    break

        series += term  # 1/_np.math.factorial(i) * commutant
        last_commutant = commutant; i += 1
    return series


def _d2_exp_series(x, dx, d2x):
    TERM_TOL = 1e-12
    tr = len(dx.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    tr2 = len(d2x.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    assert((tr - 2, tr2 - 2) in [(1, 2), (2, 4)]), "Current support for only 1 or 2 derivative dimensions"

    series = dx.copy()  # accumulates results, so *need* a separate copy
    series2 = d2x.copy()  # accumulates results, so *need* a separate copy
    term = last_commutant = dx
    last_commutant2 = term2 = d2x
    i = 2

    #take d(matrix-exp) using series approximation
    while _np.amax(_np.abs(term)) > TERM_TOL or _np.amax(_np.abs(term2)) > TERM_TOL:
        if tr == 3:
            commutant = _np.einsum("ik,kja->ija", x, last_commutant) - \
                _np.einsum("ika,kj->ija", last_commutant, x)
            commutant2A = _np.einsum("ikq,kja->ijaq", dx, last_commutant) - \
                _np.einsum("ika,kjq->ijaq", last_commutant, dx)
            commutant2B = _np.einsum("ik,kjaq->ijaq", x, last_commutant2) - \
                _np.einsum("ikaq,kj->ijaq", last_commutant2, x)

        elif tr == 4:
            commutant = _np.einsum("ik,kjab->ijab", x, last_commutant) - \
                _np.einsum("ikab,kj->ijab", last_commutant, x)
            commutant2A = _np.einsum("ikqr,kjab->ijabqr", dx, last_commutant) - \
                _np.einsum("ikab,kjqr->ijabqr", last_commutant, dx)
            commutant2B = _np.einsum("ik,kjabqr->ijabqr", x, last_commutant2) - \
                _np.einsum("ikabqr,kj->ijabqr", last_commutant2, x)

        term = 1 / _np.math.factorial(i) * commutant
        term2 = 1 / _np.math.factorial(i) * (commutant2A + commutant2B)
        series += term
        series2 += term2
        last_commutant = commutant
        last_commutant2 = (commutant2A + commutant2B)
        i += 1
    return series, series2


def _d_exp_x(x, dx, exp_x=None, postfactor=None):
    """
    Computes the derivative of the exponential of x(t) using
    the Haddamard lemma series expansion.

    Parameters
    ----------
    x : ndarray
        The 2-tensor being exponentiated

    dx : ndarray
        The derivative of x; can be either a 3- or 4-tensor where the
        3rd+ dimensions are for (multi-)indexing the parameters which
        are differentiated w.r.t.  For example, in the simplest case
        dx is a 3-tensor s.t. dx[i,j,p] == d(x[i,j])/dp.

    exp_x : ndarray, optional
        The value of `exp(x)`, which can be specified in order to save
        a call to `scipy.linalg.expm`.  If None, then the value is
        computed internally.

    postfactor : ndarray, optional
        A 2-tensor of the same shape as x that post-multiplies the
        result.

    Returns
    -------
    ndarray
        The derivative of `exp(x)*postfactor` given as a tensor with the
        same shape and axes as `dx`.
    """
    tr = len(dx.shape)  # tensor rank of dx; tr-2 == # of derivative dimensions
    assert((tr - 2) in (1, 2)), "Currently, dx can only have 1 or 2 derivative dimensions"

    series = _d_exp_series(x, dx)
    if exp_x is None: exp_x = _spl.expm(x)

    if tr == 3:
        #dExpX = _np.einsum('ika,kj->ija', series, exp_x)
        dExpX = _np.transpose(_np.tensordot(series, exp_x, (1, 0)), (0, 2, 1))
        if postfactor is not None:
            #dExpX = _np.einsum('ila,lj->ija', dExpX, postfactor)
            dExpX = _np.transpose(_np.tensordot(dExpX, postfactor, (1, 0)), (0, 2, 1))
    elif tr == 4:
        #dExpX = _np.einsum('ikab,kj->ijab', series, exp_x)
        dExpX = _np.transpose(_np.tensordot(series, exp_x, (1, 0)), (0, 3, 1, 2))
        if postfactor is not None:
            #dExpX = _np.einsum('ilab,lj->ijab', dExpX, postfactor)
            dExpX = _np.transpose(_np.tensordot(dExpX, postfactor, (1, 0)), (0, 3, 1, 2))

    return dExpX


class TPInstrumentOp(DenseOperator):
    """
    A partial implementation of :class:`LinearOperator` which encapsulates an element of a
    :class:`TPInstrument`.  Instances rely on their parent being a
    `TPInstrument`.
    """

    def __init__(self, param_ops, index):
        """
        Initialize a TPInstrumentOp object.

        Parameters
        ----------
        param_ops : list of LinearOperator objects
            A list of the underlying gate objects which constitute a simple
            parameterization of a :class:`TPInstrument`.  Namely, this is
            the list of [MT,D1,D2,...Dn] gates which parameterize *all* of the
            `TPInstrument`'s elements.

        index : int
            The index indicating which element of the `TPInstrument` the
            constructed object is.  Must be in the range
            `[0,len(param_ops)-1]`.
        """
        self.param_ops = param_ops
        self.index = index
        DenseOperator.__init__(self, _np.identity(param_ops[0].dim, 'd'),
                               "densitymx")  # Note: sets self.gpindices; TP assumed real
        self._construct_matrix()

        #Set our own parent and gpindices based on param_ops
        # (this breaks the usual paradigm of having the parent object set these,
        #  but the exception is justified b/c the parent has set these members
        #  of the underlying 'param_ops' gates)
        self.dependents = [0, index + 1] if index < len(param_ops) - 1 \
            else list(range(len(param_ops)))
        #indices into self.param_ops of the gates this gate depends on
        self.set_gpindices(_slct.list_to_slice(
            _np.concatenate([param_ops[i].gpindices_as_array()
                             for i in self.dependents], axis=0), True, False),
                           param_ops[0].parent)  # use parent of first param gate
        # (they should all be the same)

    def _construct_matrix(self):
        """
        Mi = Di + MT for i = 1...(n-1)
           = -(n-2)*MT-sum(Di) = -(n-2)*MT-[(MT-Mi)-n*MT] for i == (n-1)
        """
        nEls = len(self.param_ops)
        self.base.flags.writeable = True
        if self.index < nEls - 1:
            self.base[:, :] = _np.asarray(self.param_ops[self.index + 1]
                                          + self.param_ops[0])
        else:
            assert(self.index == nEls - 1), \
                "Invalid index %d > %d" % (self.index, nEls - 1)
            self.base[:, :] = _np.asarray(-sum(self.param_ops)
                                          - (nEls - 3) * self.param_ops[0])

        assert(self.base.shape == (self.dim, self.dim))
        self.base.flags.writeable = False

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter. An
        empty 2D array in the StaticDenseOp case (num_params == 0).

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        Np = self.num_params()
        derivMx = _np.zeros((self.dim**2, Np), 'd')
        Nels = len(self.param_ops)

        off = 0
        if self.index < Nels - 1:  # matrix = Di + MT = param_ops[index+1] + param_ops[0]
            for i in [0, self.index + 1]:
                Np = self.param_ops[i].num_params()
                derivMx[:, off:off + Np] = self.param_ops[i].deriv_wrt_params()
                off += Np

        else:  # matrix = -(nEls-2)*MT-sum(Di)
            Np = self.param_ops[0].num_params()
            derivMx[:, off:off + Np] = -(Nels - 2) * self.param_ops[0].deriv_wrt_params()
            off += Np

            for i in range(1, Nels):
                Np = self.param_ops[i].num_params()
                derivMx[:, off:off + Np] = -self.param_ops[i].deriv_wrt_params()
                off += Np

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return False

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        raise ValueError(("TPInstrumentOp.to_vector() should never be called"
                          " - use TPInstrument.to_vector() instead"))

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize this partially-implemented gate using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of parameters.

        Returns
        -------
        None
        """
        #Rely on the Instrument ordering of it's elements: if we're being called
        # to init from v then this is within the context of a TPInstrument's gates
        # having been simplified and now being initialized from a vector (within a
        # calculator).  We rely on the Instrument elements having their
        # from_vector() methods called in self.index order.

        if self.index < len(self.param_ops) - 1:  # final element doesn't need to init any param gates
            for i in self.dependents:  # re-init all my dependents (may be redundant)
                if i == 0 and self.index > 0: continue  # 0th param-gate already init by index==0 element
                paramop_local_inds = _modelmember._decompose_gpindices(
                    self.gpindices, self.param_ops[i].gpindices)
                self.param_ops[i].from_vector(v[paramop_local_inds], close, nodirty)

        self._construct_matrix()


class ComposedOp(LinearOperator):
    """
    A gate map that is the composition of a number of map-like factors (possibly
    other `LinearOperator`s)
    """

    def __init__(self, ops_to_compose, dim="auto", evotype="auto", dense_rep=False):
        """
        Creates a new ComposedOp.

        Parameters
        ----------
        ops_to_compose : list
            List of `LinearOperator`-derived objects
            that are composed to form this gate map.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.

        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one gate being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one gate being composed.

        dense_rep : bool, optional
            Whether this operator should be internally represented using a dense
            matrix.  This is expert-level functionality, and you should leave their
            the default value unless you know what you're doing.
        """
        assert(len(ops_to_compose) > 0 or dim != "auto"), \
            "Must compose at least one gate when dim='auto'!"
        self.factorops = list(ops_to_compose)
        self.dense_rep = dense_rep

        if dim == "auto":
            dim = ops_to_compose[0].dim
        assert(all([dim == gate.dim for gate in ops_to_compose])), \
            "All gates must have the same dimension (%d expected)!" % dim

        if evotype == "auto":
            evotype = ops_to_compose[0]._evotype
        assert(all([evotype == gate._evotype for gate in ops_to_compose])), \
            "All gates must have the same evolution type (%s expected)!" % evotype

        #Term cache dicts (only used for "svterm" and "cterm" evotypes)
        self.terms = {}
        self.local_term_poly_coeffs = {}

        #Create representation object
        factor_op_reps = [op._rep for op in self.factorops]
        if evotype == "densitymx":
            if dense_rep:
                rep = replib.DMOpRepDense(_np.require(_np.identity(dim, 'd'),
                                                      requirements=['OWNDATA', 'C_CONTIGUOUS']))
            else:
                rep = replib.DMOpRepComposed(factor_op_reps, dim)
        elif evotype == "statevec":
            if dense_rep:
                rep = replib.SVOpRepDense(_np.require(_np.identity(dim, complex),
                                                      requirements=['OWNDATA', 'C_CONTIGUOUS']))
            else:
                rep = replib.SVOpRepComposed(factor_op_reps, dim)
        elif evotype == "stabilizer":
            assert(not dense_rep), "Cannot require a dense representation with stabilizer evotype!"
            nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
            rep = replib.SBOpRepComposed(factor_op_reps, nQubits)
        else:
            assert(not dense_rep), "Cannot require a dense representation with %s evotype!" % evotype
            rep = dim  # no proper representation (_rep will be set to None by LinearOperator)

        LinearOperator.__init__(self, rep, evotype)
        if self.dense_rep: self._update_denserep()  # update dense rep if needed

    def _update_denserep(self):
        if len(self.factorops) == 0:
            mx = _np.identity(self.dim, 'd')
        else:
            mx = self.factorops[0].todense()
            for op in self.factorops[1:]:
                mx = _np.dot(op.todense(), mx)

        self._rep.base.flags.writeable = True
        self._rep.base[:, :] = mx
        self._rep.base.flags.writeable = False

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factorops

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
        _modelmember.ModelMember.set_gpindices(self, gpindices, parent, memo)

    def append(self, *factorops_to_add):
        """
        Add one or more factors to this operator.

        Parameters
        ----------
        *factors_to_add : LinearOperator
            One or multiple factor operators to add on at the *end* (evaluated
            last) of this operator.

        Returns
        -------
        None
        """
        self.factorops.extend(factorops_to_add)
        if self.dense_rep:
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed

    def remove(self, *factorop_indices):
        """
        Remove one or more factors from this operator.

        Parameters
        ----------
        *factorop_indices : int
            One or multiple factor indices to remove from this operator.

        Returns
        -------
        None
        """
        for i in sorted(factorop_indices, reverse=True):
            del self.factorops[i]
        if self.dense_rep:
            self._update_denserep()
        elif self._rep is not None:
            self._rep.reinit_factor_op_reps([op._rep for op in self.factorops])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factor gates have their
        # parent reset correctly.
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls([g.copy(parent) for g in self.factorops], self.dim, self._evotype)
        return self._copy_gpindices(copyOfMe, parent)

    def tosparse(self):
        """ Return the operation as a sparse matrix """
        mx = self.factorops[0].tosparse()
        for op in self.factorops[1:]:
            mx = op.tosparse().dot(mx)
        return mx

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        if self.dense_rep:
            #We already have a dense version stored
            return self._rep.base
        elif len(self.factorops) == 0:
            return _np.identity(self.dim, 'd')
        else:
            mx = self.factorops[0].todense()
            for op in self.factorops[1:]:
                mx = _np.dot(op.todense(), mx)
            return mx

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    factor_op_reps = [gate.torep() for gate in self.factorops]
    #    #FUTURE? factor_op_reps = [ repmemo.get(id(gate), gate.torep(debug_time_dict)) for gate in self.factorops ] #something like this? # noqa
    #
    #    if self._evotype == "densitymx":
    #        return replib.DMOpRepComposed(factor_op_reps, self.dim)
    #    elif self._evotype == "statevec":
    #        return replib.SVOpRepComposed(factor_op_reps, self.dim)
    #    elif self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))  # "stabilizer" is a unitary-evolution type mode
    #        return replib.SBOpRepComposed(factor_op_reps, nQubits)
    #
    #    assert(False), "Invalid internal _evotype: %s" % self._evotype

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


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
            self._compute_taylor_order_terms(order, max_poly_vars)

        if return_coeff_polys:
            #Return coefficient polys in terms of *local* parameters (get_taylor_terms
            #  and above composition gives polys in terms of *global*, model params)
            return self.terms[order], self.local_term_poly_coeffs[order]
        else:
            return self.terms[order]

    def _compute_taylor_order_terms(self, order, max_poly_vars):  # separated for profiling
        terms = []

        #DEBUG TODO REMOVE
        #print("Composed op getting order",order,"terms:")
        #for i,fop in enumerate(self.factorops):
        #    print(" ",i,fop.__class__.__name__,"totalmag = ",fop.get_total_term_magnitude())
        #    hmdebug,_ = fop.get_highmagnitude_terms(0.00001, True, order)
        #    print("  hmterms w/max order=",order," have magnitude ",sum([t.magnitude for t in hmdebug]))

        for p in _lt.partition_into(order, len(self.factorops)):
            factor_lists = [self.factorops[i].get_taylor_order_terms(pi, max_poly_vars) for i, pi in enumerate(p)]
            for factors in _itertools.product(*factor_lists):
                terms.append(_term.compose_terms(factors))
        self.terms[order] = terms

        #def _decompose_indices(x):
        #    return tuple(_modelmember._decompose_gpindices(
        #        self.gpindices, _np.array(x, _np.int64)))

        mapvec = _np.ascontiguousarray(_np.zeros(max_poly_vars, _np.int64))
        for ii, i in enumerate(self.gpindices_as_array()):
            mapvec[i] = ii

        #poly_coeffs = [t.coeff.map_indices(_decompose_indices) for t in terms]  # with *local* indices
        poly_coeffs = [t.coeff.mapvec_indices(mapvec) for t in terms]  # with *local* indices
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)
        self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

    def get_taylor_order_terms_above_mag(self, order, max_poly_vars, min_term_mag):
        terms = []
        factor_lists_cache = [
            [ops.get_taylor_order_terms_above_mag(i, max_poly_vars, min_term_mag) for i in range(order + 1)]
            for ops in self.factorops
        ]
        for p in _lt.partition_into(order, len(self.factorops)):
            # factor_lists = [self.factorops[i].get_taylor_order_terms_above_mag(pi, max_poly_vars, min_term_mag)
            #                 for i, pi in enumerate(p)]
            factor_lists = [factor_lists_cache[i][pi] for i, pi in enumerate(p)]
            for factors in _itertools.product(*factor_lists):
                mag = _np.product([factor.magnitude for factor in factors])
                if mag >= min_term_mag:
                    terms.append(_term.compose_terms_with_mag(factors, mag))
        return terms
        #def _decompose_indices(x):
        #    return tuple(_modelmember._decompose_gpindices(
        #        self.gpindices, _np.array(x, _np.int64)))
        #
        #mapvec = _np.ascontiguousarray(_np.zeros(max_poly_vars,_np.int64))
        #for ii,i in enumerate(self.gpindices_as_array()):
        #    mapvec[i] = ii
        #
        ##poly_coeffs = [t.coeff.map_indices(_decompose_indices) for t in terms]  # with *local* indices
        #poly_coeffs = [t.coeff.mapvec_indices(mapvec) for t in terms]  # with *local* indices
        #tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        #if len(tapes) > 0:
        #    vtape = _np.concatenate([t[0] for t in tapes])
        #    ctape = _np.concatenate([t[1] for t in tapes])
        #else:
        #    vtape = _np.empty(0, _np.int64)
        #    ctape = _np.empty(0, complex)
        #coeffs_as_compact_polys = (vtape, ctape)
        #self.local_term_poly_coeffs[order] = coeffs_as_compact_polys

    def get_total_term_magnitude(self):
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
        # In general total term mag == sum of the coefficients of all the terms (taylor expansion)
        #  of an errorgen or operator.
        # In this case, since the taylor expansions are composed (~multiplied),
        # the total term magnitude is just the product of those of the components.
        return _np.product([f.get_total_term_magnitude() for f in self.factorops])

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        opmags = [f.get_total_term_magnitude() for f in self.factorops]
        product = _np.product(opmags)
        ret = _np.zeros(self.num_params(), 'd')
        for opmag, f in zip(opmags, self.factorops):
            f_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, f.gpindices)
            local_deriv = product / opmag * f.get_total_term_magnitude_deriv()
            ret[f_local_inds] += local_deriv
        return ret

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        assert(self.gpindices is not None), "Must set a ComposedOp's .gpindices before calling to_vector"
        v = _np.empty(self.num_params(), 'd')
        for gate in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, gate.gpindices)
            v[factorgate_local_inds] = gate.to_vector()
        return v

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.gpindices is not None), "Must set a ComposedOp's .gpindices before calling from_vector"
        for gate in self.factorops:
            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, gate.gpindices)
            gate.from_vector(v[factorgate_local_inds], close, nodirty)
        if self.dense_rep: self._update_denserep()
        self.dirty = True

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        typ = complex if any([_np.iscomplexobj(op.todense()) for op in self.factorops]) else 'd'
        derivMx = _np.zeros((self.dim, self.dim, self.num_params()), typ)

        #Product rule to compute jacobian
        for i, op in enumerate(self.factorops):  # loop over the gate we differentiate wrt
            if op.num_params() == 0: continue  # no contribution
            deriv = op.deriv_wrt_params(None)  # TODO: use filter?? / make relative to this gate...
            deriv.shape = (self.dim, self.dim, op.num_params())

            if i > 0:  # factors before ith
                pre = self.factorops[0].todense()
                for opA in self.factorops[1:i]:
                    pre = _np.dot(opA.todense(), pre)
                #deriv = _np.einsum("ija,jk->ika", deriv, pre )
                deriv = _np.transpose(_np.tensordot(deriv, pre, (1, 0)), (0, 2, 1))

            if i + 1 < len(self.factorops):  # factors after ith
                post = self.factorops[i + 1].todense()
                for opA in self.factorops[i + 2:]:
                    post = _np.dot(opA.todense(), post)
                #deriv = _np.einsum("ij,jka->ika", post, deriv )
                deriv = _np.tensordot(post, deriv, (1, 0))

            factorgate_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, op.gpindices)
            derivMx[:, :, factorgate_local_inds] += deriv

        derivMx.shape = (self.dim**2, self.num_params())
        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return any([op.has_nonzero_hessian() for op in self.factorops])

    def transform(self, s):
        """
        Update operation matrix G with inv(s) * G * s,

        Generally, the transform function updates the *parameters* of
        the gate such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `s` is an instance of `TPGaugeGroupElement` or
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        for gate in self.factorops:
            gate.transform(s)

    def __str__(self):
        """ Return string representation """
        s = "Composed gate of %d factors:\n" % len(self.factorops)
        for i, gate in enumerate(self.factorops):
            s += "Factor %d:\n" % i
            s += str(gate)
        return s


class ComposedDenseOp(ComposedOp, DenseOperatorInterface):
    """
    A gate that is the composition of a number of matrix factors (possibly other gates).
    """

    def __init__(self, ops_to_compose, dim="auto", evotype="auto"):
        """
        Creates a new ComposedDenseOp.

        Parameters
        ----------
        ops_to_compose : list
            A list of 2D numpy arrays (matrices) and/or `DenseOperator`-derived
            objects that are composed to form this gate.  Elements are composed
            with vectors  in  *left-to-right* ordering, maintaining the same
            convention as operation sequences in pyGSTi.  Note that this is
            *opposite* from standard matrix multiplication order.

        dim : int or "auto"
            Dimension of this operation.  Can be set to `"auto"` to take dimension
            from `ops_to_compose[0]` *if* there's at least one gate being
            composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this operation.  Can be set to `"auto"` to take
            the evolution type of `ops_to_compose[0]` *if* there's at least
            one gate being composed.
        """
        ComposedOp.__init__(self, ops_to_compose, dim, evotype, dense_rep=True)
        DenseOperatorInterface.__init__(self)


class ExponentiatedOp(LinearOperator):
    """
    A gate map that is the composition of a number of map-like factors (possibly
    other `LinearOperator`s)
    """

    def __init__(self, op_to_exponentiate, power, evotype="auto"):
        """
        Creates a new ExponentiatedOp.

        Parameters
        ----------
        op_to_exponentiate : list
            A `LinearOperator`-derived object that is exponentiated to
            some integer power to produce this operator.

        power : int
            the power to exponentiate `op_to_exponentiate` to.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            the evolution type.  `"auto"` uses the evolution type of
            `op_to_exponentiate`.
        """
        #We may not actually need to save these, since they can be inferred easily
        self.exponentiated_op = op_to_exponentiate
        self.power = power

        dim = op_to_exponentiate.dim

        if evotype == "auto":
            evotype = op_to_exponentiate._evotype

        if evotype == "densitymx":
            rep = replib.DMOpRepExponentiated(self.exponentiated_op._rep, self.power, dim)
        elif evotype == "statevec":
            rep = replib.SVOpRepExponentiated(self.exponentiated_op._rep, self.power, dim)
        elif evotype == "stabilizer":
            nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
            rep = replib.SVOpRepExponentiated(self.exponentiated_op._rep, self.power, nQubits)
        else:
            raise ValueError("Invalid evotype: %s for ExponentiatedOp object" % evotype)

        LinearOperator.__init__(self, rep, evotype)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.exponentiated_op]

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factor gates have their
        # parent reset correctly.
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls(self.exponentiated_op.copy(parent), self.power, self._evotype)
        return self._copy_gpindices(copyOfMe, parent)

    def tosparse(self):
        """ Return the operation as a sparse matrix """
        if self.power == 0:
            return _sps.identity(self.dim, dtype=_np.dtype('d'), format='csr')

        op = self.exponentiated_op.tosparse()
        mx = op.copy()
        for i in range(self.power - 1):
            mx = mx.dot(op)
        return mx

    def todense(self):
        """
        Return this operation as a dense matrix.
        """
        op = self.exponentiated_op.todense()
        return _np.linalg.matrix_power(op, self.power)

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        return replib.DMOpRepExponentiated(self.exponentiated_op.torep(), self.power, self.dim)
    #    elif self._evotype == "statevec":
    #        return replib.SVOpRepExponentiated(self.exponentiated_op.torep(), self.power, self.dim)
    #    elif self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))  # "stabilizer" is a unitary-evolution type mode
    #        return replib.SVOpRepExponentiated(self.exponentiated_op.torep(), self.power, nQubits)
    #    assert(False), "Invalid internal _evotype: %s" % self._evotype

    #FUTURE: term-related functions (maybe base off of ComposedOp or use a composedop to generate them?)
    # e.g. ComposedOp([self.exponentiated_op] * power, dim, evotype)

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.exponentiated_op.num_params()

    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.exponentiated_op.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of parameters.

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
        self.exponentiated_op.from_vector(v, close, nodirty)
        if not nodirty: self.dirty = True

    def __str__(self):
        """ Return string representation """
        s = "Exponentiated gate that raise the below op to the %d power\n" % self.power
        s += str(self.exponentiated_op)
        return s


class EmbeddedOp(LinearOperator):
    """
    A gate map containing a single lower (or equal) dimensional gate within it.
    An EmbeddedOp acts as the identity on all of its domain except the
    subspace of its contained gate, where it acts as the contained gate does.
    """

    def __init__(self, state_space_labels, target_labels, gate_to_embed, dense_rep=False):
        """
        Initialize an EmbeddedOp object.

        Parameters
        ----------
        state_space_labels : a list of tuples
            This argument specifies the density matrix space upon which this
            gate acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        target_labels : list of strs
            The labels contained in `state_space_labels` which demarcate the
            portions of the state space acted on by `gate_to_embed` (the
            "contained" gate).

        gate_to_embed : LinearOperator
            The gate object that is to be contained within this gate, and
            that specifies the only non-trivial action of the EmbeddedOp.

        dense_rep : bool, optional
            Whether this operator should be internally represented using a dense
            matrix.  This is expert-level functionality, and you should leave their
            the default value unless you know what you're doing.
        """
        from .labeldicts import StateSpaceLabels as _StateSpaceLabels
        self.state_space_labels = _StateSpaceLabels(state_space_labels,
                                                    evotype=gate_to_embed._evotype)
        self.targetLabels = target_labels
        self.embedded_op = gate_to_embed
        self.dense_rep = dense_rep
        self._iter_elements_cache = None  # speeds up _iter_matrix_elements significantly

        evotype = gate_to_embed._evotype
        opDim = self.state_space_labels.dim

        #Create representation
        if evotype == "stabilizer":
            # assert that all state space labels == qubits, since we only know
            # how to embed cliffords on qubits...
            assert(len(self.state_space_labels.labels) == 1
                   and all([ld == 2 for ld in self.state_space_labels.labeldims.values()])), \
                "All state space labels must correspond to *qubits*"
            if isinstance(self.embedded_op, CliffordOp):
                assert(len(target_labels) == len(self.embedded_op.svector) // 2), \
                    "Inconsistent number of qubits in `target_labels` and Clifford `embedded_op`"
            assert(not self.dense_rep), "`dense_rep` can only be set to True for densitymx and statevec evotypes"

            #Cache info to speedup representation's acton(...) methods:
            # Note: ...labels[0] is the *only* tensor-prod-block, asserted above
            qubitLabels = self.state_space_labels.labels[0]
            qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                       for targetLbl in target_labels], _np.int64)

            nQubits = int(round(_np.log2(opDim)))
            rep = replib.SBOpRepEmbedded(self.embedded_op._rep,
                                         nQubits, qubit_indices)

        elif evotype in ("statevec", "densitymx"):

            iTensorProdBlks = [self.state_space_labels.tpb_index[label] for label in target_labels]
            # index of tensor product block (of state space) a bit label is part of
            if len(set(iTensorProdBlks)) != 1:
                raise ValueError("All qubit labels of a multi-qubit gate must correspond to the"
                                 " same tensor-product-block of the state space -- checked previously")  # pragma: no cover # noqa

            iTensorProdBlk = iTensorProdBlks[0]  # because they're all the same (tested above) - this is "active" block
            tensorProdBlkLabels = self.state_space_labels.labels[iTensorProdBlk]
            # count possible *density-matrix-space* indices of each component of the tensor product block
            numBasisEls = _np.array([self.state_space_labels.labeldims[l] for l in tensorProdBlkLabels], _np.int64)

            # Separate the components of the tensor product that are not operated on, i.e. that our
            # final map just acts as identity w.r.t.
            labelIndices = [tensorProdBlkLabels.index(label) for label in target_labels]
            actionInds = _np.array(labelIndices, _np.int64)
            assert(_np.product([numBasisEls[i] for i in actionInds]) == self.embedded_op.dim), \
                "Embedded gate has dimension (%d) inconsistent with the given target labels (%s)" % (
                    self.embedded_op.dim, str(target_labels))

            if self.dense_rep:
                #maybe cache items to speed up _iter_matrix_elements in FUTURE here?
                if evotype == "statevec":
                    rep = replib.SVOpRepDense(_np.require(_np.identity(opDim, complex),
                                                          requirements=['OWNDATA', 'C_CONTIGUOUS']))
                else:  # "densitymx"
                    rep = replib.DMOpRepDense(_np.require(_np.identity(opDim, 'd'),
                                                          requirements=['OWNDATA', 'C_CONTIGUOUS']))
            else:
                nBlocks = self.state_space_labels.num_tensor_prod_blocks()
                iActiveBlock = iTensorProdBlk
                nComponents = len(self.state_space_labels.labels[iActiveBlock])
                embeddedDim = self.embedded_op.dim
                blocksizes = _np.array([_np.product(self.state_space_labels.tensor_product_block_dims(k))
                                        for k in range(nBlocks)], _np.int64)
                if evotype == "statevec":
                    rep = replib.SVOpRepEmbedded(self.embedded_op._rep,
                                                 numBasisEls, actionInds, blocksizes, embeddedDim,
                                                 nComponents, iActiveBlock, nBlocks, opDim)
                else:  # "densitymx"
                    rep = replib.DMOpRepEmbedded(self.embedded_op._rep,
                                                 numBasisEls, actionInds, blocksizes, embeddedDim,
                                                 nComponents, iActiveBlock, nBlocks, opDim)

        elif evotype in ("svterm", "cterm"):
            assert(not self.dense_rep), "`dense_rep` can only be set to True for densitymx and statevec evotypes"
            rep = opDim  # these evotypes don't have representations (LinearOperator will set _rep to None)
        else:
            raise ValueError("Invalid evotype `%s` for %s" % (evotype, self.__class__.__name__))

        LinearOperator.__init__(self, rep, evotype)
        if self.dense_rep: self._update_denserep()

    def _update_denserep(self):
        self._rep.base.flags.writeable = True
        self._rep.base[:, :] = self.todense()
        self._rep.base.flags.writeable = False

    def __getstate__(self):
        # Don't pickle 'instancemethod' or parent (see modelmember implementation)
        return _modelmember.ModelMember.__getstate__(self)

    def __setstate__(self, d):
        if "dirty" in d:  # backward compat: .dirty was replaced with ._dirty in ModelMember
            d['_dirty'] = d['dirty']; del d['dirty']
        self.__dict__.update(d)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return [self.embedded_op]

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
        copyOfMe = cls(self.state_space_labels, self.targetLabels,
                       self.embedded_op.copy(parent))
        return self._copy_gpindices(copyOfMe, parent)

    def _iter_matrix_elements_precalc(self):
        divisor = 1; divisors = []
        for l in self.targetLabels:
            divisors.append(divisor)
            divisor *= self.state_space_labels.labeldims[l]  # e.g. 4 or 2 for qubits (depending on evotype)

        iTensorProdBlk = [self.state_space_labels.tpb_index[label] for label in self.targetLabels][0]
        tensorProdBlkLabels = self.state_space_labels.labels[iTensorProdBlk]
        basisInds = [list(range(self.state_space_labels.labeldims[l])) for l in tensorProdBlkLabels]
        # e.g. [0,1,2,3] for densitymx qubits (I, X, Y, Z) OR [0,1] for statevec qubits (std *complex* basis)

        basisInds_noop = basisInds[:]
        basisInds_noop_blankaction = basisInds[:]
        labelIndices = [tensorProdBlkLabels.index(label) for label in self.targetLabels]
        for labelIndex in sorted(labelIndices, reverse=True):
            del basisInds_noop[labelIndex]
            basisInds_noop_blankaction[labelIndex] = [0]

        sorted_bili = sorted(list(enumerate(labelIndices)), key=lambda x: x[1])
        # for inserting target-qubit basis indices into list of noop-qubit indices

        # multipliers to go from per-label indices to tensor-product-block index
        # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
        multipliers = _np.array(_np.flipud(_np.cumprod([1] + list(
            reversed(list(map(len, basisInds[1:])))))), _np.int64)

        # number of basis elements preceding our block's elements
        blockDims = self.state_space_labels.tpb_dims
        offset = sum([blockDims[i] for i in range(0, iTensorProdBlk)])

        return divisors, multipliers, sorted_bili, basisInds_noop, offset

    def _iter_matrix_elements(self, rel_to_block=False):
        """ Iterates of (op_i,op_j,embedded_op_i,embedded_op_j) tuples giving mapping
            between nonzero elements of operation matrix and elements of the embedded gate matrx """
        if self._iter_elements_cache is not None:
            for item in self._iter_elements_cache:
                yield item
            return

        def _merge_op_and_noop_bases(op_b, noop_b, sorted_bili):
            """
            Merge the Pauli basis indices for the "gate"-parts of the total
            basis contained in op_b (i.e. of the components of the tensor
            product space that are operated on) and the "noop"-parts contained
            in noop_b.  Thus, len(op_b) + len(noop_b) == len(basisInds), and
            this function merges together basis indices for the operated-on and
            not-operated-on tensor product components.
            Note: return value always have length == len(basisInds) == number
            of components
            """
            ret = list(noop_b[:])  # start with noop part...
            for bi, li in sorted_bili:
                ret.insert(li, op_b[bi])  # ... and insert gate parts at proper points
            return ret

        def _decomp_op_index(indx, divisors):
            """ Decompose index of a Pauli-product matrix into indices of each
                Pauli in the product """
            ret = []
            for d in reversed(divisors):
                ret.append(indx // d)
                indx = indx % d
            return ret

        divisors, multipliers, sorted_bili, basisInds_noop, nonrel_offset = self._iter_matrix_elements_precalc()
        offset = 0 if rel_to_block else nonrel_offset

        #Begin iteration loop
        self._iter_elements_cache = []
        for op_i in range(self.embedded_op.dim):     # rows ~ "output" of the gate map
            for op_j in range(self.embedded_op.dim):  # cols ~ "input"  of the gate map
                op_b1 = _decomp_op_index(op_i, divisors)  # op_b? are lists of dm basis indices, one index per
                # tensor product component that the gate operates on (2 components for a 2-qubit gate)
                op_b2 = _decomp_op_index(op_j, divisors)

                # loop over all state configurations we don't operate on
                for b_noop in _itertools.product(*basisInds_noop):
                    # - so really a loop over diagonal dm elements
                    # using same b_noop for in and out says we're acting
                    b_out = _merge_op_and_noop_bases(op_b1, b_noop, sorted_bili)
                    # as the identity on the no-op state space
                    b_in = _merge_op_and_noop_bases(op_b2, b_noop, sorted_bili)
                    # index of output dm basis el within vec(tensor block basis)
                    out_vec_index = _np.dot(multipliers, tuple(b_out))
                    # index of input dm basis el within vec(tensor block basis)
                    in_vec_index = _np.dot(multipliers, tuple(b_in))

                    item = (out_vec_index + offset, in_vec_index + offset, op_i, op_j)
                    self._iter_elements_cache.append(item)
                    yield item

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))
    #        return replib.SBOpRepEmbedded(self.embedded_op.torep(),
    #                                       nQubits, self.qubit_indices)
    #
    #    if self._evotype not in ("statevec", "densitymx"):
    #        raise ValueError("Invalid evotype '%s' for %s.torep(...)" %
    #                         (self._evotype, self.__class__.__name__))
    #
    #    nBlocks = self.state_space_labels.num_tensor_prod_blocks()
    #    iActiveBlock = self.iTensorProdBlk
    #    nComponents = len(self.state_space_labels.labels[iActiveBlock])
    #    embeddedDim = self.embedded_op.dim
    #    blocksizes = _np.array([_np.product(self.state_space_labels.tensor_product_block_dims(k))
    #                            for k in range(nBlocks)], _np.int64)
    #
    #    if self._evotype == "statevec":
    #        return replib.SVOpRepEmbedded(self.embedded_op.torep(),
    #                                       self.numBasisEls, self.actionInds, blocksizes,
    #                                       embeddedDim, nComponents, iActiveBlock, nBlocks,
    #                                       self.dim)
    #    else:
    #        return replib.DMOpRepEmbedded(self.embedded_op.torep(),
    #                                       self.numBasisEls, self.actionInds, blocksizes,
    #                                       embeddedDim, nComponents, iActiveBlock, nBlocks,
    #                                       self.dim)

    def tosparse(self):
        """ Return the operation as a sparse matrix """
        embedded_sparse = self.embedded_op.tosparse().tolil()
        finalOp = _sps.identity(self.dim, embedded_sparse.dtype, format='lil')

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements():
            finalOp[i, j] = embedded_sparse[gi, gj]
        return finalOp.tocsr()

    def todense(self):
        """ Return the operation as a dense matrix """

        #FUTURE: maybe here or in a new "tosymplectic" method, could
        # create an embeded clifford symplectic rep as follows (when
        # evotype == "stabilizer"):
        #def tosymplectic(self):
        #    #Embed gate's symplectic rep in larger "full" symplectic rep
        #    #Note: (qubit) labels are in first (and only) tensor-product-block
        #    qubitLabels = self.state_space_labels.labels[0]
        #    smatrix, svector = _symp.embed_clifford(self.embedded_op.smatrix,
        #                                            self.embedded_op.svector,
        #                                            self.qubit_indices,len(qubitLabels))

        embedded_dense = self.embedded_op.todense()
        # operates on entire state space (direct sum of tensor prod. blocks)
        finalOp = _np.identity(self.dim, embedded_dense.dtype)

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements():
            finalOp[i, j] = embedded_dense[gi, gj]
        return finalOp

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


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
        #Reduce labeldims b/c now working on *state-space* instead of density mx:
        sslbls = self.state_space_labels.copy()
        sslbls.reduce_dims_densitymx_to_state()
        if return_coeff_polys:
            terms, coeffs = self.embedded_op.get_taylor_order_terms(order, max_poly_vars, True)
            embedded_terms = [t.embed(sslbls, self.targetLabels) for t in terms]
            return embedded_terms, coeffs
        else:
            return [t.embed(sslbls, self.targetLabels)
                    for t in self.embedded_op.get_taylor_order_terms(order, max_poly_vars, False)]

    def get_taylor_order_terms_above_mag(self, order, max_poly_vars, min_term_mag):
        """TODO: docstring """
        sslbls = self.state_space_labels.copy()
        sslbls.reduce_dims_densitymx_to_state()
        return [t.embed(sslbls, self.targetLabels)
                for t in self.embedded_op.get_taylor_order_terms_above_mag(order, max_poly_vars, min_term_mag)]

    def get_total_term_magnitude(self):
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
        # In general total term mag == sum of the coefficients of all the terms (taylor expansion)
        #  of an errorgen or operator.
        # In this case, since the coeffs of the terms of an EmbeddedOp are the same as those
        # of the operator being embedded, the total term magnitude is the same:

        #DEBUG TODO REMOVE
        #print("DB: Embedded.total_term_magnitude = ",self.embedded_op.get_total_term_magnitude()," -- ",
        #   self.embedded_op.__class__.__name__)
        #ret = self.embedded_op.get_total_term_magnitude()
        #egterms = self.get_taylor_order_terms(0)
        #mags = [ abs(t.evaluate_coeff(self.to_vector()).coeff) for t in egterms ]
        #print("EmbeddedErrorgen CHECK = ",sum(mags), " vs ", ret)
        #assert(sum(mags) <= ret+1e-4)

        return self.embedded_op.get_total_term_magnitude()

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        return self.embedded_op.get_total_term_magnitude_deriv()

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return self.embedded_op.num_params()

    def to_vector(self):
        """
        Get the gate parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        return self.embedded_op.to_vector()

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of parameters.

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
        self.embedded_op.from_vector(v, close, nodirty)
        if self.dense_rep: self._update_denserep()
        if not nodirty: self.dirty = True

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single gate parameter.  Thus, each column is of length
        op_dim^2 and there is one column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives with shape (dimension^2, num_params)
        """
        # Note: this function exploits knowledge of EmbeddedOp internals!!
        embedded_deriv = self.embedded_op.deriv_wrt_params(wrt_filter)
        derivMx = _np.zeros((self.dim**2, embedded_deriv.shape[1]), embedded_deriv.dtype)
        M = self.embedded_op.dim

        #fill in embedded_op contributions (always overwrites the diagonal
        # of finalOp where appropriate, so OK it starts as identity)
        for i, j, gi, gj in self._iter_matrix_elements():
            derivMx[i * self.dim + j, :] = embedded_deriv[gi * M + gj, :]  # fill row of jacobian
        return derivMx  # Note: wrt_filter has already been applied above

    def transform(self, s):
        """
        Update operation matrix G with inv(s) * G * s,

        Generally, the transform function updates the *parameters* of
        the gate such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `s` is an instance of `TPGaugeGroupElement` or
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        # I think we could do this but extracting the approprate parts of the
        # s and Sinv matrices... but haven't needed it yet.
        raise NotImplementedError("Cannot transform an EmbeddedDenseOp yet...")

    def depolarize(self, amount):
        """
        Depolarize this gate by the given `amount`.

        Generally, the depolarize function updates the *parameters* of
        the gate such that the resulting operation matrix is depolarized.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : float or tuple
            The amount to depolarize by.  If a tuple, it must have length
            equal to one less than the dimension of the gate. In standard
            bases, depolarization corresponds to multiplying the operation matrix
            by a diagonal matrix whose first diagonal element (corresponding
            to the identity) equals 1.0 and whose subsequent elements
            (corresponding to non-identity basis elements) equal
            `1.0 - amount[i]` (or just `1.0 - amount` if `amount` is a
            float).

        Returns
        -------
        None
        """
        self.embedded_op.depolarize(amount)
        if self.dense_rep: self._update_denserep()

    def rotate(self, amount, mx_basis="gm"):
        """
        Rotate this gate by the given `amount`.

        Generally, the rotate function updates the *parameters* of
        the gate such that the resulting operation matrix is rotated.  If
        such an update cannot be done (because the gate parameters do not
        allow for it), ValueError is raised.

        Parameters
        ----------
        amount : tuple of floats, optional
            Specifies the rotation "coefficients" along each of the non-identity
            Pauli-product axes.  The gate's matrix `G` is composed with a
            rotation operation `R`  (so `G` -> `dot(R, G)` ) where `R` is the
            unitary superoperator corresponding to the unitary operator
            `U = exp( sum_k( i * rotate[k] / 2.0 * Pauli_k ) )`.  Here `Pauli_k`
            ranges over all of the non-identity un-normalized Pauli operators.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        Returns
        -------
        None
        """
        self.embedded_op.rotate(amount, mx_basis)
        if self.dense_rep: self._update_denserep()

    #def compose(self, other_op):
    #    """
    #    Create and return a new gate that is the composition of this gate
    #    followed by other_op, which *must be another EmbeddedDenseOp*.
    #    (For more general compositions between different types of gates, use
    #    the module-level compose function.)  The returned gate's matrix is
    #    equal to dot(this, other_op).
    #
    #    Parameters
    #    ----------
    #    other_op : EmbeddedDenseOp
    #        The gate to compose to the right of this one.
    #
    #    Returns
    #    -------
    #    EmbeddedDenseOp
    #    """
    #    raise NotImplementedError("Can't compose an EmbeddedDenseOp yet")

    def has_nonzero_hessian(self):
        """
        Returns whether this gate has a non-zero Hessian with
        respect to its parameters, i.e. whether it only depends
        linearly on its parameters or not.

        Returns
        -------
        bool
        """
        return self.embedded_op.has_nonzero_hessian()

    def __str__(self):
        """ Return string representation """
        s = "Embedded gate with full dimension %d and state space %s\n" % (self.dim, self.state_space_labels)
        s += " that embeds the following %d-dimensional gate into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.targetLabels))
        s += str(self.embedded_op)
        return s


class EmbeddedDenseOp(EmbeddedOp, DenseOperatorInterface):
    """
    A gate containing a single lower (or equal) dimensional gate within it.
    An EmbeddedDenseOp acts as the identity on all of its domain except the
    subspace of its contained gate, where it acts as the contained gate does.
    """

    def __init__(self, state_space_labels, target_labels, gate_to_embed):
        """
        Initialize a EmbeddedDenseOp object.

        Parameters
        ----------
        state_space_labels : a list of tuples
            This argument specifies the density matrix space upon which this
            gate acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        target_labels : list of strs
            The labels contained in `state_space_labels` which demarcate the
            portions of the state space acted on by `gate_to_embed` (the
            "contained" gate).

        gate_to_embed : DenseOperator
            The gate object that is to be contained within this gate, and
            that specifies the only non-trivial action of the EmbeddedDenseOp.
        """
        EmbeddedOp.__init__(self, state_space_labels, target_labels,
                            gate_to_embed, dense_rep=True)
        DenseOperatorInterface.__init__(self)


class CliffordOp(LinearOperator):
    """
    A Clifford gate, represented via a symplectic
    """

    def __init__(self, unitary, symplecticrep=None):
        """
        Creates a new CliffordOp from a unitary operation.

        Note: while the clifford gate is held internally in a symplectic
        representation, it is also be stored as a unitary (so the `unitary`
        argument is required) for keeping track of global phases when updating
        stabilizer frames.

        If a non-Clifford unitary is specified, then a ValueError is raised.

        Parameters
        ----------
        unitary : numpy.ndarray
            The unitary action of the clifford gate.

        symplecticrep : tuple, optional
            A (symplectic matrix, phase vector) 2-tuple specifying the pre-
            computed symplectic representation of `unitary`.  If None, then
            this representation is computed automatically from `unitary`.

        """
        #self.superop = superop
        self.unitary = unitary
        assert(self.unitary is not None), "Must supply `unitary` argument!"

        #if self.superop is not None:
        #    assert(unitary is None and symplecticrep is None),"Only supply one argument to __init__"
        #    raise NotImplementedError("Superop -> Unitary calc not implemented yet")

        if symplecticrep is not None:
            self.smatrix, self.svector = symplecticrep
        else:
            # compute symplectic rep from unitary
            self.smatrix, self.svector = _symp.unitary_to_symplectic(self.unitary, flagnonclifford=True)

        self.inv_smatrix, self.inv_svector = _symp.inverse_clifford(
            self.smatrix, self.svector)  # cache inverse since it's expensive

        #nQubits = len(self.svector) // 2
        #dim = 2**nQubits  # "stabilizer" is a "unitary evolution"-type mode

        #Update members so they reference the same (contiguous) memory as the rep
        U = self.unitary.todense() if isinstance(self.unitary, LinearOperator) else self.unitary
        self._dense_unitary = _np.ascontiguousarray(U, complex)
        self.smatrix = _np.ascontiguousarray(self.smatrix, _np.int64)
        self.svector = _np.ascontiguousarray(self.svector, _np.int64)
        self.inv_smatrix = _np.ascontiguousarray(self.inv_smatrix, _np.int64)
        self.inv_svector = _np.ascontiguousarray(self.inv_svector, _np.int64)

        #Create representation
        rep = replib.SBOpRepClifford(self.smatrix, self.svector,
                                     self.inv_smatrix, self.inv_svector,
                                     self._dense_unitary)
        LinearOperator.__init__(self, rep, "stabilizer")

    #NOTE: if this gate had parameters, we'd need to clear inv_smatrix & inv_svector
    # whenever the smatrix or svector changed, respectively (probably in from_vector?)

    #def torep(self):
    #    """
    #    Return a "representation" object for this gate.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self.inv_smatrix is None or self.inv_svector is None:
    #        self.inv_smatrix, self.inv_svector = _symp.inverse_clifford(
    #            self.smatrix, self.svector)  # cache inverse since it's expensive
    #
    #    invs, invp = self.inv_smatrix, self.inv_svector
    #    U = self.unitary.todense() if isinstance(self.unitary, LinearOperator) else self.unitary
    #    return replib.SBOpRepClifford(_np.ascontiguousarray(self.smatrix, _np.int64),
    #                                   _np.ascontiguousarray(self.svector, _np.int64),
    #                                   _np.ascontiguousarray(invs, _np.int64),
    #                                   _np.ascontiguousarray(invp, _np.int64),
    #                                   _np.ascontiguousarray(U, complex))

    def __str__(self):
        """ Return string representation """
        s = "Clifford gate with matrix:\n"
        s += _mt.mx_to_string(self.smatrix, width=2, prec=0)
        s += " and vector " + _mt.mx_to_string(self.svector, width=2, prec=0)
        return s


# STRATEGY:
# - maybe create an abstract base TermGate class w/get_taylor_order_terms(...) function?
# - Note: if/when terms return a *polynomial* coefficient the poly's 'variables' should
#    reference the *global* Model-level parameters, not just the local gate ones.
# - create an EmbeddedTermGate class to handle embeddings, which holds a
#    LindbladDenseOp (or other in the future?) and essentially wraps it's
#    terms in EmbeddedOp or EmbeddedClifford objects.
# - similarly create an ComposedTermGate class...
# - so LindbladDenseOp doesn't need to deal w/"kite-structure" bases of terms;
#    leave this to some higher level constructor which can create compositions
#    of multiple LindbladOps based on kite structure (one per kite block).


class ComposedErrorgen(LinearOperator):
    """
    A composition (sum!) of several Lindbladian exponent operators, that is, a
    *sum* (not product) of other error generators.
    """

    def __init__(self, errgens_to_compose, dim="auto", evotype="auto"):
        """
        Creates a new ComposedErrorgen.

        Parameters
        ----------
        errgens_to_compose : list
            List of `LinearOperator`-derived objects that are summed together (composed)
            to form this error generator.

        dim : int or "auto"
            Dimension of this error generator.  Can be set to `"auto"` to take
            the dimension from `errgens_to_compose[0]` *if* there's at least one
            error generator being composed.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm","auto"}
            The evolution type of this error generator.  Can be set to `"auto"`
            to take the evolution type of `errgens_to_compose[0]` *if* there's
            at least one error generator being composed.
        """
        assert(len(errgens_to_compose) > 0 or dim != "auto"), \
            "Must compose at least one error generator when dim='auto'!"
        self.factors = errgens_to_compose

        if dim == "auto":
            dim = errgens_to_compose[0].dim
        assert(all([dim == eg.dim for eg in errgens_to_compose])), \
            "All error generators must have the same dimension (%d expected)!" % dim

        if evotype == "auto":
            evotype = errgens_to_compose[0]._evotype
        assert(all([evotype == eg._evotype for eg in errgens_to_compose])), \
            "All error generators must have the same evolution type (%s expected)!" % evotype

        # set "API" error-generator members (to interface properly w/other objects)
        # FUTURE: create a base class that defines this interface (maybe w/properties?)
        #self.sparse = errgens_to_compose[0].sparse \
        #    if len(errgens_to_compose) > 0 else False
        #assert(all([self.sparse == eg.sparse for eg in errgens_to_compose])), \
        #    "All error generators must have the same sparsity (%s expected)!" % self.sparse

        self.matrix_basis = errgens_to_compose[0].matrix_basis \
            if len(errgens_to_compose) > 0 else None
        assert(all([self.matrix_basis == eg.matrix_basis for eg in errgens_to_compose])), \
            "All error generators must have the same matrix basis (%s expected)!" % self.matrix_basis

        #Create representation object
        factor_reps = [op._rep for op in self.factors]
        if evotype == "densitymx":
            rep = replib.DMOpRepSum(factor_reps, dim)
        elif evotype == "statevec":
            rep = replib.SVOpRepSum(factor_reps, dim)
        elif evotype == "stabilizer":
            nQubits = int(round(_np.log2(dim)))  # "stabilizer" is a unitary-evolution type mode
            rep = replib.SBOpRepSum(factor_reps, nQubits)
        else:
            rep = dim  # no representations for term-based evotypes

        LinearOperator.__init__(self, rep, evotype)

    def get_coeffs(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients
        of this error generator.  Note that these are not necessarily the
        parameter values, as these coefficients are generally functions of
        the parameters (so as to keep the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool
            Whether to also return a :class:`Basis` containing the elements
            with which the error generator terms were constructed.

        logscale_nonham : bool, optional
            Whether or not the non-hamiltonian error generator coefficients
            should be scaled so that the returned dict contains:
            `(1 - exp(-d^2 * coeff)) / d^2` instead of `coeff`.  This
            essentially converts the coefficient into a rate that is
            the contribution this term would have within a depolarizing
            channel where all stochastic generators had this same coefficient.
            This is the value returned by :method:`get_error_rates`.

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients.

        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `Ltermdict` to basis matrices.
        """
        Ltermdict = _collections.OrderedDict()
        basisdict = _collections.OrderedDict()
        first_nonempty_basis = None
        constant_basis = None  # the single same Basis used for every factor with a nonempty basis

        for eg in self.factors:
            factor_coeffs = eg.get_coeffs(return_basis, logscale_nonham)

            if return_basis:
                ltdict, factor_basis = factor_coeffs
                if len(factor_basis) > 0:
                    if first_nonempty_basis is None:
                        first_nonempty_basis = factor_basis
                        constant_basis = factor_basis  # seed constant_basis
                    elif factor_basis != constant_basis:
                        constant_basis = None  # factors have different bases - no constant_basis!

                # see if we need to update basisdict and ensure we do so in a consistent
                # way - if factors use the same basis labels these must refer to the same
                # basis elements.
                #FUTURE: maybe a way to do this without always accessing basis *elements*?
                #  (maybe do a pass to check for a constant_basis without any .elements refs?)
                for lbl, basisEl in zip(factor_basis.labels, factor_basis.elements):
                    if lbl in basisdict:
                        assert(_mt.safenorm(basisEl - basisdict[lbl]) < 1e-6), "Ambiguous basis label %s" % lbl
                    else:
                        basisdict[lbl] = basisEl
            else:
                ltdict = factor_coeffs

            for key, coeff in ltdict.items():
                if key in Ltermdict:
                    Ltermdict[key] += coeff
                else:
                    Ltermdict[key] = coeff

        if return_basis:
            #Use constant_basis or turn basisdict into a Basis to return
            if constant_basis is not None:
                basis = constant_basis
            elif first_nonempty_basis is not None:
                #Create an ExplictBasis using the matrices in basisdict plus the identity
                lbls = ['I'] + list(basisdict.keys())
                mxs = [first_nonempty_basis[0]] + list(basisdict.values())
                basis = _ExplicitBasis(mxs, lbls, name=None,
                                       real=first_nonempty_basis.real,
                                       sparse=first_nonempty_basis.sparse)
            return Ltermdict, basis
        else:
            return Ltermdict

    def get_error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this
        error generator (pertaining to the *channel* formed by
        exponentiating this object).

        The "error rate" for an individual Hamiltonian error is the angle
        about the "axis" (generalized in the multi-qubit case)
        corresponding to a particular basis element, i.e. `theta` in
        the unitary channel `U = exp(i * theta/2 * BasisElement)`.

        The "error rate" for an individual Stochastic error is the
        contribution that basis element's term would have to the
        error rate of a depolarization channel.  For example, if
        the rate corresponding to the term ('S','X') is 0.01 this
        means that the coefficient of the rho -> X*rho*X-rho error
        generator is set such that if this coefficient were used
        for all 3 (X,Y, and Z) terms the resulting depolarizing
        channel would have error rate 3*0.01 = 0.03.

        Note that because error generator terms do not necessarily
        commute with one another, the sum of the returned error
        rates is not necessarily the error rate of the overall
        channel.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case.
        """
        return self.get_coeffs(return_basis=False, logscale_nonham=True)

    def set_coeffs(self, lindblad_term_dict, action="update", logscale_nonham=False):
        """
        Sets the coefficients of terms in this error generator.  The dictionary
        `lindblad_term_dict` has tuple-keys describing the type of term and the basis
        elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are the coefficients of these error generators,
            and should be real except for the 2-basis-label case.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :method:`get_errgen_coeffs`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :method:`set_error_rates`.

        Returns
        -------
        None
        """
        factor_coeffs_list = [eg.get_coeffs(False, logscale_nonham) for eg in self.factors]
        perfactor_Ltermdicts = [_collections.OrderedDict() for eg in self.factors]
        unused_Lterm_keys = set(lindblad_term_dict.keys())

        #Divide lindblad_term_dict in per-factor Ltermdicts
        for k, val in lindblad_term_dict.items():
            for d, coeffs in zip(perfactor_Ltermdicts, factor_coeffs_list):
                if k in coeffs:
                    d[k] = val; unused_Lterm_keys.remove(k)
                    # only apply a given lindblad_term_dict entry once,
                    # even if it can be applied to multiple factors
                    break

        if len(unused_Lterm_keys) > 0:
            raise KeyError("Invalid L-term descriptor key(s): %s" % str(unused_Lterm_keys))

        #Set the L-term coefficients of each factor separately
        for d, eg in zip(perfactor_Ltermdicts, self.factors):
            eg.set_coeffs(d, action, logscale_nonham)

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in this error generator so that the
        contributions of the resulting channel's error rate are given by
        the values in `lindblad_term_dict`.  See :method:`get_error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case, when they may be complex.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_coeffs(lindblad_term_dict, action, logscale_nonham=True)

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        #TODO: in the furture could do this more cleverly so
        # each factor gets an appropriate wrt_filter instead of
        # doing all filtering at the end

        d2 = self.dim
        derivMx = _np.zeros((d2**2, self.num_params()), 'd')
        for eg in self.factors:
            factor_deriv = eg.deriv_wrt_params(None)  # do filtering at end
            rel_gpindices = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            derivMx[:, rel_gpindices] += factor_deriv[:, :]

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

        return derivMx

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this error generator with respect to
        its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1, wrt_filter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        #TODO: in the furture could do this more cleverly so
        # each factor gets an appropriate wrt_filter instead of
        # doing all filtering at the end

        d2 = self.dim
        nP = self.num_params()
        hessianMx = _np.zeros((d2**2, nP, nP), 'd')
        for eg in self.factors:
            factor_hessian = eg.hessian_wrt_params(None, None)  # do filtering at end
            rel_gpindices = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            hessianMx[:, rel_gpindices, rel_gpindices] += factor_hessian[:, :, :]

        if wrt_filter1 is None:
            if wrt_filter2 is None:
                return hessianMx
            else:
                return _np.take(hessianMx, wrt_filter2, axis=2)
        else:
            if wrt_filter2 is None:
                return _np.take(hessianMx, wrt_filter1, axis=1)
            else:
                return _np.take(_np.take(hessianMx, wrt_filter1, axis=1),
                                wrt_filter2, axis=2)

    def submembers(self):
        """
        Get the ModelMember-derived objects contained in this one.

        Returns
        -------
        list
        """
        return self.factors

    def append(self, *factors_to_add):
        """
        Add one or more factors to this operator.

        Parameters
        ----------
        *factors_to_add : LinearOperator
            One or multiple factor operators to add on at the *end* (summed
            last) of this operator.

        Returns
        -------
        None
        """
        self.factors.extend(factors_to_add)
        if self._rep is not None:
            self._rep.reinit_factor_reps([op._rep for op in self.factors])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed

    def remove(self, *factor_indices):
        """
        Remove one or more factors from this operator.

        Parameters
        ----------
        *factorop_indices : int
            One or multiple factor indices to remove from this operator.

        Returns
        -------
        None
        """
        for i in sorted(factor_indices, reverse=True):
            del self.factors[i]
        if self._rep is not None:
            self._rep.reinit_factor_reps([op._rep for op in self.factors])
        if self.parent:  # need to alert parent that *number* (not just value)
            self.parent._mark_for_rebuild(self)  # of our params may have changed

    def copy(self, parent=None):
        """
        Copy this object.

        Returns
        -------
        LinearOperator
            A copy of this object.
        """
        # We need to override this method so that factors have their
        # parent reset correctly.
        cls = self.__class__  # so that this method works for derived classes too
        copyOfMe = cls([f.copy(parent) for f in self.factors], self.dim, self._evotype)
        return self._copy_gpindices(copyOfMe, parent)

    def tosparse(self):
        """ Return this error generator as a sparse matrix """
        if len(self.factors) == 0:
            return _sps.csr_matrix((self.dim, self.dim), dtype='d')
        mx = self.factors[0].tosparse()
        for eg in self.factors[1:]:
            mx += eg.tosparse()
        return mx

    def todense(self):
        """ Return this error generator as a dense matrix """
        if len(self.factors) == 0:
            return _np.zeros((self.dim, self.dim), 'd')
        mx = self.factors[0].todense()
        for eg in self.factors[1:]:
            mx += eg.todense()
        return mx

    #OLD: UNUSED - now use tosparse/todense
    #def _construct_errgen_matrix(self):
    #    self.factors[0]._construct_errgen_matrix()
    #    mx = self.factors[0].err_gen_mx
    #    for eg in self.factors[1:]:
    #        eg._construct_errgen_matrix()
    #        mx += eg.err_gen_mx
    #    self.err_gen_mx = mx

    #def torep(self):
    #    """
    #    Return a "representation" object for this error generator.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    factor_reps = [factor.torep() for factor in self.factors]
    #    if self._evotype == "densitymx":
    #        return replib.DMOpRepSum(factor_reps, self.dim)
    #    elif self._evotype == "statevec":
    #        return replib.SVOpRepSum(factor_reps, self.dim)
    #    elif self._evotype == "stabilizer":
    #        nQubits = int(round(_np.log2(self.dim)))  # "stabilizer" is a unitary-evolution type mode
    #        return replib.SBOpRepSum(factor_reps, nQubits)
    #    assert(False), "Invalid internal _evotype: %s" % self._evotype

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this error generator..

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


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
        assert(order == 0), \
            "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        assert(return_coeff_polys is False)

        #Need to adjust indices b/c in error generators we (currently) expect terms to have local indices
        ret = []
        for eg in self.factors:
            eg_terms = [t.copy() for t in eg.get_taylor_order_terms(order, max_poly_vars, return_coeff_polys)]
            mapvec = _np.ascontiguousarray(_modelmember._decompose_gpindices(
                self.gpindices, _modelmember._compose_gpindices(eg.gpindices, _np.arange(eg.num_params()))))
            for t in eg_terms:
                # t.map_indices_inplace(lambda x: tuple(_modelmember._decompose_gpindices(
                #     # map global to *local* indices
                #     self.gpindices, _modelmember._compose_gpindices(eg.gpindices, _np.array(x, _np.int64)))))
                t.mapvec_indices_inplace(mapvec)
            ret.extend(eg_terms)
        return ret
        # return list(_itertools.chain(
        #     *[eg.get_taylor_order_terms(order, max_poly_vars, return_coeff_polys) for eg in self.factors]
        # ))

    def get_total_term_magnitude(self):
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
        # In general total term mag == sum of the coefficients of all the terms (taylor expansion)
        #  of an errorgen or operator.
        # In this case, since composed error generators are just summed, the total term
        # magnitude is just the sum of the components

        #DEBUG TODO REMOVE
        #factor_ttms = [eg.get_total_term_magnitude() for eg in self.factors]
        #print("DB: ComposedErrorgen.total_term_magnitude = sum(",factor_ttms,") -- ",
        #      [eg.__class__.__name__ for eg in self.factors])
        #for k,eg in enumerate(self.factors):
        #    sub_egterms = eg.get_taylor_order_terms(0)
        #    sub_mags = [ abs(t.evaluate_coeff(eg.to_vector()).coeff) for t in sub_egterms ]
        #    print(" -> ",k,": total terms mag = ",sum(sub_mags), "(%d)" % len(sub_mags),"\n", sub_mags)
        #    print("     gpindices = ",eg.gpindices)
        #
        #ret = sum(factor_ttms)
        #egterms = self.get_taylor_order_terms(0)
        #mags = [ abs(t.evaluate_coeff(self.to_vector()).coeff) for t in egterms ]
        #print("ComposedErrgen term mags (should concat above) ",len(egterms),":\n",mags)
        #print("gpindices = ",self.gpindices)
        #print("ComposedErrorgen CHECK = ",sum(mags), " vs ", ret)
        #assert(sum(mags) <= ret+1e-4)

        return sum([eg.get_total_term_magnitude() for eg in self.factors])

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        ret = _np.zeros(self.num_params(), 'd')
        for eg in self.factors:
            eg_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            ret[eg_local_inds] += eg.get_total_term_magnitude_deriv()
        return ret

    def num_params(self):
        """
        Get the number of independent parameters which specify this error generator.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())

    def to_vector(self):
        """
        Get the error generator parameters as an array of values.

        Returns
        -------
        numpy array
            The gate parameters as a 1D array with length num_params().
        """
        assert(self.gpindices is not None), "Must set a ComposedErrorgen's .gpindices before calling to_vector"
        v = _np.empty(self.num_params(), 'd')
        for eg in self.factors:
            factor_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            v[factor_local_inds] = eg.to_vector()
        return v

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the error generator using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        assert(self.gpindices is not None), "Must set a ComposedErrorgen's .gpindices before calling from_vector"
        for eg in self.factors:
            factor_local_inds = _modelmember._decompose_gpindices(
                self.gpindices, eg.gpindices)
            eg.from_vector(v[factor_local_inds], close, nodirty)
        if not nodirty: self.dirty = True

    def transform(self, s):
        """
        Update operation matrix G with inv(s) * G * s,

        Generally, the transform function updates the *parameters* of
        the gate such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        In this particular case any TP gauge transformation is possible,
        i.e. when `s` is an instance of `TPGaugeGroupElement` or
        corresponds to a TP-like transform matrix.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        for eg in self.factors:
            eg.transform(s)

    def onenorm_upperbound(self):
        """
        Returns an upper bound on the 1-norm for this error generator (
        viewed as a matrix).

        Returns
        -------
        float
        """
        # b/c ||A + B|| <= ||A|| + ||B||
        return sum([eg.onenorm_upperbound() for eg in self.factors])

    def __str__(self):
        """ Return string representation """
        s = "Composed error generator of %d factors:\n" % len(self.factors)
        for i, eg in enumerate(self.factors):
            s += "Factor %d:\n" % i
            s += str(eg)
        return s


# Idea:
# Op = exp(Errgen); Errgen is an error just on 2nd qubit (and say we have 3 qubits)
# so Op = I x (I+eps*A) x I (small eps limit); eps*A is 1-qubit error generator
# also Op ~= I+Errgen in small eps limit, so
# Errgen = I x (I+eps*A) x I - I x I x I
#        = I x I x I + eps I x A x I - I x I x I
#        = eps I x A x I = I x eps*A x I
# --> we embed error generators by tensoring with I's on non-target sectors.
#  (identical to how be embed gates)

class EmbeddedErrorgen(EmbeddedOp):
    """
    An error generator containing a single lower (or equal) dimensional gate within it.
    An EmbeddedErrorGen acts as the null map (zero) on all of its domain except the
    subspace of its contained error generator, where it acts as the contained item does.
    """

    def __init__(self, state_space_labels, target_labels, errgen_to_embed):
        """
        Initialize an EmbeddedErrorgen object.

        Parameters
        ----------
        state_space_labels : a list of tuples
            This argument specifies the density matrix space upon which this
            generator acts.  Each tuple corresponds to a block of a density matrix
            in the standard basis (and therefore a component of the direct-sum
            density matrix space). Elements of a tuple are user-defined labels
            beginning with "L" (single Level) or "Q" (two-level; Qubit) which
            interpret the d-dimensional state space corresponding to a d x d
            block as a tensor product between qubit and single level systems.
            (E.g. a 2-qubit space might be labelled `[('Q0','Q1')]`).

        target_labels : list of strs
            The labels contained in `state_space_labels` which demarcate the
            portions of the state space acted on by `errgen_to_embed` (the
            "contained" error generator).

        errgen_to_embed : LinearOperator
            The error generator object that is to be contained within this
            error generator, and that specifies the only non-trivial action
            of the EmbeddedErrorgen.
        """
        EmbeddedOp.__init__(self, state_space_labels, target_labels, errgen_to_embed)

        # set "API" error-generator members (to interface properly w/other objects)
        # FUTURE: create a base class that defines this interface (maybe w/properties?)
        #self.sparse = True # Embedded error generators are *always* sparse (pointless to
        #                   # have dense versions of these)

        embedded_matrix_basis = errgen_to_embed.matrix_basis
        if isinstance(embedded_matrix_basis, str):
            self.matrix_basis = embedded_matrix_basis
        else:  # assume a Basis object
            my_basis_dim = self.state_space_labels.dim
            self.matrix_basis = _Basis.cast(embedded_matrix_basis.name, my_basis_dim, sparse=True)

            #OLD: constructs a subset of this errorgen's full mxbasis, but not the whole thing:
            #self.matrix_basis = _Basis(
            #    name="embedded_" + embedded_matrix_basis.name,
            #    matrices=[self._embed_basis_mx(mx) for mx in
            #              embedded_matrix_basis.get_composite_matrices()],
            #    sparse=True)

        #OLD: when a generators "rep" was self.err_gen_mx
        #if errgen_to_embed._evotype == "densitymx":
        #    self._construct_errgen_matrix()
        #else:
        #    self.err_gen_mx = None

    #TODO REMOVE (UNUSED)
    #def _construct_errgen_matrix(self):
    #    #Always construct a sparse errgen matrix, so just use
    #    # base class's .tosparse() (which calls embedded errorgen's
    #    # .tosparse(), which will convert a dense->sparse embedded
    #    # error generator, but this is fine).
    #    self.err_gen_mx = self.tosparse()

    #TODO REMOVE (UNUSED)
    #def _embed_basis_mx(self, mx):
    #    """ Take a dense or sparse basis matrix and embed it. """
    #    mxAsGate = StaticDenseOp(mx) if isinstance(mx, _np.ndarray) \
    #        else StaticDenseOp(mx.todense())  # assume mx is a sparse matrix
    #    return EmbeddedOp(self.state_space_labels, self.targetLabels,
    #                      mxAsGate).tosparse()  # always convert to *sparse* basis els

    def from_vector(self, v, close=False, nodirty=False):
        """
        Initialize the gate using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of gate parameters.  Length
            must == num_params()

        Returns
        -------
        None
        """
        EmbeddedOp.from_vector(self, v, close, nodirty)
        if not nodirty: self.dirty = True

    def get_coeffs(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients
        of this operation.  Note that these are not necessarily the parameter
        values, as these coefficients are generally functions of the parameters
        (so as to keep the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool
            Whether to also return a :class:`Basis` containing the elements
            with which the error generator terms were constructed.

        logscale_nonham : bool, optional
            Whether or not the non-hamiltonian error generator coefficients
            should be scaled so that the returned dict contains:
            `(1 - exp(-d^2 * coeff)) / d^2` instead of `coeff`.  This
            essentially converts the coefficient into a rate that is
            the contribution this term would have within a depolarizing
            channel where all stochastic generators had this same coefficient.
            This is the value returned by :method:`get_error_rates`.

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients.

        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `Ltermdict` to basis matrices.
        """
        embedded_coeffs = self.embedded_op.get_coeffs(return_basis, logscale_nonham)
        embedded_Ltermdict = _collections.OrderedDict()

        if return_basis:
            # embed basis
            Ltermdict, basis = embedded_coeffs
            embedded_basis = _EmbeddedBasis(basis, self.state_space_labels, self.targetLabels)
            bel_map = {lbl: embedded_lbl for lbl, embedded_lbl in zip(basis.labels, embedded_basis.labels)}

            #go through and embed Ltermdict labels
            for k, val in Ltermdict.items():
                embedded_key = (k[0],) + tuple([bel_map[x] for x in k[1:]])
                embedded_Ltermdict[embedded_key] = val
            return embedded_Ltermdict, embedded_basis
        else:
            #go through and embed Ltermdict labels
            Ltermdict = embedded_coeffs
            for k, val in Ltermdict.items():
                embedded_key = (k[0],) + tuple([_EmbeddedBasis.embed_label(x, self.targetLabels) for x in k[1:]])
                embedded_Ltermdict[embedded_key] = val
            return embedded_Ltermdict

    def get_error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this
        error generator (pertaining to the *channel* formed by
        exponentiating this object).

        The "error rate" for an individual Hamiltonian error is the angle
        about the "axis" (generalized in the multi-qubit case)
        corresponding to a particular basis element, i.e. `theta` in
        the unitary channel `U = exp(i * theta/2 * BasisElement)`.

        The "error rate" for an individual Stochastic error is the
        contribution that basis element's term would have to the
        error rate of a depolarization channel.  For example, if
        the rate corresponding to the term ('S','X') is 0.01 this
        means that the coefficient of the rho -> X*rho*X-rho error
        generator is set such that if this coefficient were used
        for all 3 (X,Y, and Z) terms the resulting depolarizing
        channel would have error rate 3*0.01 = 0.03.

        Note that because error generator terms do not necessarily
        commute with one another, the sum of the returned error
        rates is not necessarily the error rate of the overall
        channel.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case.
        """
        return self.get_coeffs(return_basis=False, logscale_nonham=True)

    def set_coeffs(self, lindblad_term_dict, action="update", logscale_nonham=False):
        """
        Sets the coefficients of terms in this error generator.  The dictionary
        `lindblad_term_dict` has tuple-keys describing the type of term and the basis
        elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are the coefficients of these error generators,
            and should be real except for the 2-basis-label case.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :method:`get_errgen_coeffs`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :method:`set_error_rates`.

        Returns
        -------
        None
        """
        unembedded_Ltermdict = _collections.OrderedDict()
        for k, val in lindblad_term_dict.items():
            unembedded_key = (k[0],) + tuple([_EmbeddedBasis.unembed_label(x, self.targetLabels) for x in k[1:]])
            unembedded_Ltermdict[unembedded_key] = val
        self.embedded_op.set_coeffs(unembedded_Ltermdict, action, logscale_nonham)

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in this error generator so that the
        contributions of the resulting channel's error rate are given by
        the values in `lindblad_term_dict`.  See :method:`get_error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case, when they may be complex.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_coeffs(lindblad_term_dict, action, logscale_nonham=True)

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        raise NotImplementedError("deriv_wrt_params is not implemented for EmbeddedErrorGen objects")

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this error generator with respect to
        its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1, wrt_filter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        raise NotImplementedError("hessian_wrt_params is not implemented for EmbeddedErrorGen objects")

    def onenorm_upperbound(self):
        """
        Returns an upper bound on the 1-norm for this error generator (
        viewed as a matrix).

        Returns
        -------
        float
        """
        return self.embedded_op.onenorm_upperbound()
        # b/c ||A x B|| == ||A|| ||B|| and ||I|| == 1.0

    def __str__(self):
        """ Return string representation """
        s = "Embedded error generator with full dimension %d and state space %s\n" % (self.dim, self.state_space_labels)
        s += " that embeds the following %d-dimensional gate into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.targetLabels))
        s += str(self.embedded_op)
        return s


class LindbladErrorgen(LinearOperator):
    """
    An Lindblad-form error generator consisting of terms that, with appropriate
    constraints ensurse that the resulting (after exponentiation) gate/layer
    operation is CPTP.  These terms can be divided into "Hamiltonian"-type
    terms, which map rho -> i[H,rho] and "non-Hamiltonian"/"other"-type terms,
    which map rho -> A rho B + 0.5*(ABrho + rhoAB).
    """

    @classmethod
    def from_error_generator(cls, errgen, ham_basis="pp", nonham_basis="pp",
                             param_mode="cptp", nonham_mode="all",
                             mx_basis="pp", truncate=True, evotype="densitymx"):
        """
        Create a Lindblad-form error generator from an error generator matrix
        and a basis which specifies how to decompose (project) the error
        generator.

        errgen : numpy array or SciPy sparse matrix
            a square 2D array that gives the full error generator. The shape of
            this array sets the dimension of the operator. The projections of
            this quantity onto the `ham_basis` and `nonham_basis` are closely
            related to the parameters of the error generator (they may not be
            exactly equal if, e.g `cptp=True`).

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        other_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the non-Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            gate's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `errgen` cannot
            be realized by the specified set of Lindblad projections.

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the error generator being constructed.
            `"densitymx"` means usual Lioville density-matrix-vector propagation
            via matrix-vector products.  `"svterm"` denotes state-vector term-
            based evolution (action of gate is obtained by evaluating the rank-1
            terms up to some order).  `"cterm"` is similar but uses Clifford gate
            action on stabilizer states.

        Returns
        -------
        LindbladErrorgen
        """

        d2 = errgen.shape[0]
        #d = int(round(_np.sqrt(d2)))  # OLD TODO REMOVE
        #if d*d != d2: raise ValueError("Error generator dim must be a perfect square")

        #Determine whether we're using sparse bases or not
        sparse = None
        if ham_basis is not None:
            if isinstance(ham_basis, _Basis): sparse = ham_basis.sparse
            elif isinstance(ham_basis, str): sparse = _sps.issparse(errgen)
            elif len(ham_basis) > 0: sparse = _sps.issparse(ham_basis[0])
        if sparse is None and nonham_basis is not None:
            if isinstance(nonham_basis, _Basis): sparse = nonham_basis.sparse
            elif isinstance(nonham_basis, str): sparse = _sps.issparse(errgen)
            elif len(nonham_basis) > 0: sparse = _sps.issparse(nonham_basis[0])
        if sparse is None: sparse = False  # the default

        #Create or convert bases to appropriate sparsity
        if not isinstance(ham_basis, _Basis):
            # needed b/c ham_basis could be a Basis w/dim=0 which can't be cast as dim=d2
            ham_basis = _Basis.cast(ham_basis, d2, sparse=sparse)
        if not isinstance(nonham_basis, _Basis):
            nonham_basis = _Basis.cast(nonham_basis, d2, sparse=sparse)
        if not isinstance(mx_basis, _Basis):
            matrix_basis = _Basis.cast(mx_basis, d2, sparse=sparse)
        else: matrix_basis = mx_basis

        # errgen + bases => coeffs
        hamC, otherC = \
            _gt.lindblad_errgen_projections(
                errgen, ham_basis, nonham_basis, matrix_basis, normalize=False,
                return_generators=False, other_mode=nonham_mode, sparse=sparse)

        # coeffs + bases => Ltermdict, basis
        Ltermdict, basis = _gt.projections_to_lindblad_terms(
            hamC, otherC, ham_basis, nonham_basis, nonham_mode)

        return cls(d2, Ltermdict, basis,
                   param_mode, nonham_mode, truncate,
                   matrix_basis, evotype)

    def __init__(self, dim, lindblad_term_dict, basis=None,
                 param_mode="cptp", nonham_mode="all", truncate=True,
                 mx_basis="pp", evotype="densitymx"):
        """
        Create a new LinbladErrorgen based on a set of Lindblad terms.

        Note that if you want to construct a LinbladErrorgen from a
        error generator matrix, you can use the :method:`from_error_generator`
        class method.

        Parameters
        ----------
        dim : int
            The Hilbert-Schmidt (superoperator) dimension, which will be the
            dimension of the created operator.

        lindblad_term_dict : dict
            A dictionary specifying which Linblad terms are present in the
            parameteriztion.  Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
            (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
            have a single basis label (so key is a 2-tuple) whereas Stochastic
            tuples with 1 basis label indicate a *diagonal* term, and are the
            only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
            Stochastic term tuples can include 2 basis labels to specify
            "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
            strings or integers.  Values are complex coefficients.

        basis : Basis, optional
            A basis mapping the labels used in the keys of `lindblad_term_dict` to
            basis matrices (e.g. numpy arrays or Scipy sparse matrices).

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            error generator's parameter values.  Allowed values are:
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

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The basis for this error generator's linear mapping. Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the error generator being constructed.
            `"densitymx"` means the usual Lioville density-matrix-vector
            propagation via matrix-vector products.  `"svterm"` denotes
            state-vector term-based evolution (action of gate is obtained by
            evaluating the rank-1 terms up to some order).  `"cterm"` is similar
            but uses Clifford gate action on stabilizer states.
        """

        #FUTURE:
        # - maybe allow basisdict values to specify an "embedded matrix" w/a tuple like
        #  e.g. a *list* of (matrix, state_space_label) elements -- e.g. [(sigmaX,'Q1'), (sigmaY,'Q4')]
        # - maybe let keys be tuples of (basisname, state_space_label) e.g. (('X','Q1'),('Y','Q4')) -- and
        # maybe allow ('XY','Q1','Q4')? format when can assume single-letter labels.
        # - could add standard basis dict items so labels like "X", "XY", etc. are understood?

        # Store superop dimension
        d2 = dim
        #d = int(round(_np.sqrt(d2))) #OLD TODO REMOVE
        #assert(d*d == d2), "Dimension must be a perfect square"

        self.nonham_mode = nonham_mode
        self.param_mode = param_mode

        # lindblad_term_dict, basis => bases + parameter values
        # but maybe we want lindblad_term_dict, basisdict => basis + projections/coeffs,
        #  then projections/coeffs => paramvals? since the latter is what set_errgen needs
        hamC, otherC, self.ham_basis, self.other_basis = \
            _gt.lindblad_terms_to_projections(lindblad_term_dict, basis, self.nonham_mode)

        self.ham_basis_size = len(self.ham_basis)
        self.other_basis_size = len(self.other_basis)

        if self.ham_basis_size > 0: self.sparse = _sps.issparse(self.ham_basis[0])
        elif self.other_basis_size > 0: self.sparse = _sps.issparse(self.other_basis[0])
        else: self.sparse = False

        self.matrix_basis = _Basis.cast(mx_basis, d2, sparse=self.sparse)

        self.paramvals = _gt.lindblad_projections_to_paramvals(
            hamC, otherC, self.param_mode, self.nonham_mode, truncate)

        #Finish initialization based on evolution type
        assert(evotype in ("densitymx", "svterm", "cterm")), \
            "Invalid evotype: %s for %s" % (evotype, self.__class__.__name__)

        #Fast CSR-matrix summing variables: N/A if not sparse or using terms
        self._CSRSumIndices = self._CSRSumData = self._CSRSumPtr = None
        #self.hamCSRSumIndices = None  #REMOVE
        #self.otherCSRSumIndices = None #REMOVE
        self.sparse_err_gen_template = None

        # Generator matrices & cache qtys: N/A for term-based evotypes
        self.hamGens = self.otherGens = None
        self.hamGens_1norms = self.otherGens_1norms = None
        self._onenorm_upbound = None
        self.Lmx = None

        if evotype == "densitymx":
            self.hamGens, self.otherGens = self._init_generators(dim)

            if self.hamGens is not None:
                self.hamGens_1norms = _np.array([_mt.safe_onenorm(mx) for mx in self.hamGens], 'd')
            if self.otherGens is not None:
                if self.nonham_mode == "diagonal":
                    self.otherGens_1norms = _np.array([_mt.safe_onenorm(mx) for mx in self.otherGens], 'd')
                else:
                    self.otherGens_1norms = _np.array([_mt.safe_onenorm(mx)
                                                       for oGenRow in self.otherGens for mx in oGenRow], 'd')

            #Allocate space fo Cholesky mx (used in _construct_errgen_matrix)
            # (intermediate storage for matrix and for deriv computation)
            bsO = self.other_basis_size
            self.Lmx = _np.zeros((bsO - 1, bsO - 1), 'complex') if bsO > 0 else None

            if self.sparse:
                #Precompute for faster CSR sums in _construct_errgen
                all_csr_matrices = []
                if self.hamGens is not None:
                    all_csr_matrices.extend(self.hamGens)

                if self.otherGens is not None:
                    if self.nonham_mode == "diagonal":
                        oList = self.otherGens
                    else:  # nonham_mode in ("diag_affine", "all")
                        oList = [mx for mxRow in self.otherGens for mx in mxRow]
                    all_csr_matrices.extend(oList)

                #OLD REMOVE
                # csr_sum_array, indptr, indices, N = \
                #     _mt.get_csr_sum_indices(all_csr_matrices)
                #self.hamCSRSumIndices = csr_sum_array[0:len(self.hamGens)]
                #self.otherCSRSumIndices = csr_sum_array[len(self.hamGens):]

                flat_dest_indices, flat_src_data, flat_nnzptr, indptr, indices, N = \
                    _mt.get_csr_sum_flat_indices(all_csr_matrices)
                self._CSRSumIndices = flat_dest_indices
                self._CSRSumData = flat_src_data
                self._CSRSumPtr = flat_nnzptr

                self._data_scratch = _np.zeros(len(indices), complex)  # *complex* scratch space for updating rep
                rep = replib.DMOpRepSparse(_np.ascontiguousarray(_np.zeros(len(indices), 'd')),
                                           _np.ascontiguousarray(indices, _np.int64),
                                           _np.ascontiguousarray(indptr, _np.int64))
            else:
                rep = replib.DMOpRepDense(_np.ascontiguousarray(_np.zeros((dim, dim), 'd')))

        else:  # Term-based evolution

            assert(not self.sparse), "Sparse bases are not supported for term-based evolution"
            #TODO: make terms init-able from sparse elements, and below code  work with a *sparse* unitary_postfactor

            self.LtermdictAndBasis = (lindblad_term_dict, basis)  # HACK
            self.Lterms, self.Lterm_coeffs = None, None
            # # OLD: do this lazily now that we need max_poly_vars...
            # self._init_terms(lindblad_term_dict, basis, evotype, dim, max_poly_vars)
            rep = dim  # rep = None for term-based evotypes

        LinearOperator.__init__(self, rep, evotype)  # sets self.dim
        if self._rep is not None: self._update_rep()  # updates _rep whether it's a dense or sparse matrix
        #Done with __init__(...)

    def _init_generators(self, dim):
        #assumes self.dim, self.ham_basis, self.other_basis, and self.matrix_basis are setup...

        d2 = dim
        d = int(round(_np.sqrt(d2)))
        assert(d * d == d2), "Errorgen dim must be a perfect square"

        # Get basis transfer matrix
        mxBasisToStd = self.matrix_basis.transform_matrix(_BuiltinBasis("std", self.matrix_basis.dim, self.sparse))
        # use BuiltinBasis("std") instead of just "std" in case matrix_basis is a TensorProdBasis
        leftTrans = _spsl.inv(mxBasisToStd.tocsc()).tocsr() if _sps.issparse(mxBasisToStd) \
            else _np.linalg.inv(mxBasisToStd)
        rightTrans = mxBasisToStd

        hamBasisMxs = self.ham_basis.elements
        otherBasisMxs = self.other_basis.elements

        hamGens, otherGens = _gt.lindblad_error_generators(
            hamBasisMxs, otherBasisMxs, normalize=False,
            other_mode=self.nonham_mode)  # in std basis

        # Note: lindblad_error_generators will return sparse generators when
        #  given a sparse basis (or basis matrices)

        if hamGens is not None:
            bsH = len(hamGens) + 1  # projection-basis size (not nec. == d2)
            _gt._assert_shape(hamGens, (bsH - 1, d2, d2), self.sparse)

            # apply basis change now, so we don't need to do so repeatedly later
            if self.sparse:
                hamGens = [_mt.safereal(_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans)),
                                        inplace=True, check=True) for mx in hamGens]
                for mx in hamGens: mx.sort_indices()
                # for faster addition ops in _construct_errgen_matrix
            else:
                #hamGens = _np.einsum("ik,akl,lj->aij", leftTrans, hamGens, rightTrans)
                hamGens = _np.transpose(_np.tensordot(
                    _np.tensordot(leftTrans, hamGens, (1, 1)), rightTrans, (2, 0)), (1, 0, 2))
        else:
            bsH = 0
        assert(bsH == self.ham_basis_size)

        if otherGens is not None:

            if self.nonham_mode == "diagonal":
                bsO = len(otherGens) + 1  # projection-basis size (not nec. == d2)
                _gt._assert_shape(otherGens, (bsO - 1, d2, d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    otherGens = [_mt.safereal(_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans)),
                                              inplace=True, check=True) for mx in otherGens]
                    for mx in hamGens: mx.sort_indices()
                    # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,akl,lj->aij", leftTrans, otherGens, rightTrans)
                    otherGens = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, otherGens, (1, 1)), rightTrans, (2, 0)), (1, 0, 2))

            elif self.nonham_mode == "diag_affine":
                # projection-basis size (not nec. == d2) [~shape[1] but works for lists too]
                bsO = len(otherGens[0]) + 1
                _gt._assert_shape(otherGens, (2, bsO - 1, d2, d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    otherGens = [[_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans))
                                  for mx in mxRow] for mxRow in otherGens]

                    for mxRow in otherGens:
                        for mx in mxRow: mx.sort_indices()
                        # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,abkl,lj->abij", leftTrans,
                    #                          otherGens, rightTrans)
                    otherGens = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, otherGens, (1, 2)), rightTrans, (3, 0)), (1, 2, 0, 3))

            else:
                bsO = len(otherGens) + 1  # projection-basis size (not nec. == d2)
                _gt._assert_shape(otherGens, (bsO - 1, bsO - 1, d2, d2), self.sparse)

                # apply basis change now, so we don't need to do so repeatedly later
                if self.sparse:
                    otherGens = [[_mt.safedot(leftTrans, _mt.safedot(mx, rightTrans))
                                  for mx in mxRow] for mxRow in otherGens]
                    #Note: complex OK here, as only linear combos of otherGens (like (i,j) + (j,i)
                    # terms) need to be real

                    for mxRow in otherGens:
                        for mx in mxRow: mx.sort_indices()
                        # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,abkl,lj->abij", leftTrans,
                    #                            otherGens, rightTrans)
                    otherGens = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, otherGens, (1, 2)), rightTrans, (3, 0)), (1, 2, 0, 3))

        else:
            bsO = 0
        assert(bsO == self.other_basis_size)
        return hamGens, otherGens

    def _init_terms(self, lindblad_term_dict, basis, evotype, dim, max_poly_vars):

        d2 = dim
        # needed b/c operators produced by lindblad_error_generators have an extra 'd' scaling
        d = int(round(_np.sqrt(d2)))
        mpv = max_poly_vars

        # Lookup dictionaries for getting the *parameter* index associated
        # with a particlar basis label.  The -1 to compensates for the fact
        # that the identity element is the first element of each non-empty basis
        # and this does not have a correponding parameter/projection.
        hamBasisIndices = {lbl: i - 1 for i, lbl in enumerate(self.ham_basis.labels)}
        otherBasisIndices = {lbl: i - 1 for i, lbl in enumerate(self.other_basis.labels)}

        # as we expect `basis` will contain *dense* basis
        # matrices (maybe change in FUTURE?)
        numHamParams = max(len(hamBasisIndices) - 1, 0)  # compensate for first basis el,
        numOtherBasisEls = max(len(otherBasisIndices) - 1, 0)  # being the identity. (if there are any els at all)

        # Create Lindbladian terms - rank1 terms in the *exponent* with polynomial
        # coeffs (w/ *local* variable indices) that get converted to per-order
        # terms later.
        IDENT = None  # sentinel for the do-nothing identity op
        Lterms = []
        for termLbl in lindblad_term_dict:
            termType = termLbl[0]
            if termType == "H":  # Hamiltonian
                k = hamBasisIndices[termLbl[1]]  # index of parameter
                # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
                scale, U = _mt.to_unitary(basis[termLbl[1]])
                scale *= _np.sqrt(d) / 2  # mimics rho1's _np.sqrt(d) / 2 scaling in `hamiltonian_to_lindbladian`
                Lterms.append(_term.RankOnePolyOpTerm.simple_init(
                    _Polynomial({(k,): -1j * scale}, mpv), U, IDENT, evotype))
                Lterms.append(_term.RankOnePolyOpTerm.simple_init(_Polynomial({(k,): +1j * scale}, mpv),
                                                                  IDENT, U.conjugate().T, evotype))

            elif termType == "S":  # Stochastic
                if self.nonham_mode in ("diagonal", "diag_affine"):
                    if self.param_mode in ("depol", "reldepol"):  # => same single param for all stochastic terms
                        k = numHamParams + 0  # index of parameter
                    else:
                        k = numHamParams + otherBasisIndices[termLbl[1]]  # index of parameter
                    scale, U = _mt.to_unitary(basis[termLbl[1]])  # ensure all Rank1Term operators are *unitary*
                    scale *= _np.sqrt(d)  # mimics "rho1 *= d" scaling in `nonham_lindbladian`
                    Lm = Ln = U
                    # power to raise parameter to in order to get coeff
                    pw = 2 if self.param_mode in ("cptp", "depol") else 1

                    Lm_dag = Lm.conjugate().T
                    # assumes basis is dense (TODO: make sure works for sparse case too - and np.dots below!)
                    Ln_dag = Ln.conjugate().T
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(
                        _Polynomial({(k,) * pw: 1.0 * scale**2}, mpv), Ln, Lm_dag, evotype
                    ))
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(
                        _Polynomial({(k,) * pw: -0.5 * scale**2}, mpv), IDENT, _np.dot(Ln_dag, Lm), evotype
                    ))
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(
                        _Polynomial({(k,) * pw: -0.5 * scale**2}, mpv), _np.dot(Lm_dag, Ln), IDENT, evotype
                    ))

                else:
                    i = otherBasisIndices[termLbl[1]]  # index of row in "other" coefficient matrix
                    j = otherBasisIndices[termLbl[2]]  # index of col in "other" coefficient matrix
                    scalem, Um = _mt.to_unitary(basis[termLbl[1]])  # ensure all Rank1Term operators are *unitary*
                    scalen, Un = _mt.to_unitary(basis[termLbl[2]])  # ensure all Rank1Term operators are *unitary*
                    Lm, Ln = Um, Un
                    scale = scalem * scalen
                    scale *= d  # mimics "rho1 *= d" scaling in `nonham_lindbladian`

                    # TODO: create these polys and place below...
                    polyTerms = {}
                    assert(self.param_mode != "depol"), "`depol` mode not supported when nonham_mode=='all'"
                    assert(self.param_mode != "reldepol"), "`reldepol` mode not supported when nonham_mode=='all'"
                    if self.param_mode == "cptp":
                        # otherCoeffs = _np.dot(self.Lmx,self.Lmx.T.conjugate())
                        # coeff_ij = sum_k Lik * Ladj_kj = sum_k Lik * conjugate(L_jk)
                        #          = sum_k (Re(Lik) + 1j*Im(Lik)) * (Re(L_jk) - 1j*Im(Ljk))
                        def i_re(a, b): return numHamParams + (a * numOtherBasisEls + b)
                        def i_im(a, b): return numHamParams + (b * numOtherBasisEls + a)
                        for k in range(0, min(i, j) + 1):
                            if k <= i and k <= j:
                                polyTerms[(i_re(i, k), i_re(j, k))] = 1.0
                            if k <= i and k < j:
                                polyTerms[(i_re(i, k), i_im(j, k))] = -1.0j
                            if k < i and k <= j:
                                polyTerms[(i_im(i, k), i_re(j, k))] = 1.0j
                            if k < i and k < j:
                                polyTerms[(i_im(i, k), i_im(j, k))] = 1.0
                    else:  # param_mode == "unconstrained"
                        # coeff_ij = otherParam[i,j] + 1j*otherParam[j,i] (otherCoeffs is Hermitian)
                        ijIndx = numHamParams + (i * numOtherBasisEls + j)
                        jiIndx = numHamParams + (j * numOtherBasisEls + i)
                        polyTerms = {(ijIndx,): 1.0, (jiIndx,): 1.0j}

                    base_poly = _Polynomial(polyTerms, mpv)
                    Lm_dag = Lm.conjugate().T; Ln_dag = Ln.conjugate().T
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(1.0 * base_poly * scale, Ln, Lm, evotype))
                    # adjoint(_np.dot(Lm_dag,Ln))
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(
                        -0.5 * base_poly * scale, IDENT, _np.dot(Ln_dag, Lm), evotype
                    ))
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(
                        -0.5 * base_poly * scale, _np.dot(Lm_dag, Ln), IDENT, evotype
                    ))

            elif termType == "A":  # Affine
                assert(self.nonham_mode == "diag_affine")
                if self.param_mode in ("depol", "reldepol"):  # => same single param for all stochastic terms
                    k = numHamParams + 1 + otherBasisIndices[termLbl[1]]  # index of parameter
                else:
                    k = numHamParams + numOtherBasisEls + otherBasisIndices[termLbl[1]]  # index of parameter

                # rho -> basis[termLbl[1]] * I = basis[termLbl[1]] * sum{ P_i rho P_i } where Pi's
                #  are the normalized paulis (including the identity), and rho has trace == 1
                #  (all but "I/d" component of rho are annihilated by pauli sum; for the I/d component, all
                #   d^2 of the terms in the sum is P/sqrt(d) * I/d * P/sqrt(d) == I/d^2, so the result is just "I")
                scale, U = _mt.to_unitary(basis[termLbl[1]])  # ensure all Rank1Term operators are *unitary*
                L = U
                # Note: only works when `d` corresponds to integral # of qubits!
                Bmxs = _bt.basis_matrices("pp", d2, sparse=False)
                # scaling to make Bmxs unitary (reverse of `scale` above, where scale * U == basis[.])
                Bscale = d2**0.25

                for B in Bmxs:  # Note: *include* identity! (see pauli scratch notebook for details)
                    UB = Bscale * B  # UB is unitary
                    Lterms.append(_term.RankOnePolyOpTerm.simple_init(_Polynomial({(k,): 1.0 * scale / Bscale**2}, mpv),
                                                                      _np.dot(L, UB), UB, evotype))  # /(d2-1.)

                #TODO: check normalization of these terms vs those used in projections.

        #DEBUG
        #print("DB: params = ", list(enumerate(self.paramvals)))
        #print("DB: Lterms = ")
        #for i,lt in enumerate(Lterms):
        #    print("Term %d:" % i)
        #    print("  coeff: ", str(lt.coeff)) # list(lt.coeff.keys()) )
        #    print("  pre:\n", lt.pre_ops[0] if len(lt.pre_ops) else "IDENT")
        #    print("  post:\n",lt.post_ops[0] if len(lt.post_ops) else "IDENT")

        #Make compact polys that are ready to (repeatedly) evaluate (useful
        # for term-based calcs which call get_total_term_magnitude() a lot)
        poly_coeffs = [t.coeff for t in Lterms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)

        #DEBUG TODO REMOVE (and make into test) - check norm of rank-1 terms
        # (Note: doesn't work for Clifford terms, which have no .base):
        # rho =OP=> coeff * A rho B
        # want to bound | coeff * Tr(E Op rho) | = | coeff | * | <e|A|psi><psi|B|e> |
        # so A and B should be unitary so that | <e|A|psi><psi|B|e> | <= 1
        # but typically these are unitaries / (sqrt(2)*nqubits)
        #import bpdb; bpdb.set_trace()
        #scale = 1.0
        #for t in Lterms:
        #    for op in t._rep.pre_ops:
        #        test = _np.dot(_np.conjugate(scale * op.base.T), scale * op.base)
        #        assert(_np.allclose(test, _np.identity(test.shape[0], 'd')))
        #    for op in t._rep.post_ops:
        #        test = _np.dot(_np.conjugate(scale * op.base.T), scale * op.base)
        #        assert(_np.allclose(test, _np.identity(test.shape[0], 'd')))

        return Lterms, coeffs_as_compact_polys

    def _set_params_from_matrix(self, errgen, truncate):
        """ Sets self.paramvals based on `errgen` """
        hamC, otherC = \
            _gt.lindblad_errgen_projections(
                errgen, self.ham_basis, self.other_basis, self.matrix_basis, normalize=False,
                return_generators=False, other_mode=self.nonham_mode,
                sparse=self.sparse)  # in std basis

        self.paramvals = _gt.lindblad_projections_to_paramvals(
            hamC, otherC, self.param_mode, self.nonham_mode, truncate)
        if self._evotype == "densitymx": self._update_rep()

    def _update_rep(self):
        """
        Updates self._rep, which contains a representation of this error generator
        as either a dense or sparse matrix.  This routine essentially builds the
        error generator matrix using the current parameters and updates self._rep
        accordingly (by rewriting its data).
        """
        d2 = self.dim
        hamCoeffs, otherCoeffs = _gt.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.param_mode, self.nonham_mode, self.Lmx)
        onenorm = 0.0

        #Finally, build operation matrix from generators and coefficients:
        if self.sparse:
            coeffs = None
            data = self._data_scratch
            data.fill(0.0)  # data starts at zero

            if hamCoeffs is not None:
                onenorm += _np.dot(self.hamGens_1norms, _np.abs(hamCoeffs))
                if otherCoeffs is not None:
                    coeffs = _np.concatenate((hamCoeffs, otherCoeffs.flat), axis=0)
                else:
                    coeffs = hamCoeffs
            elif otherCoeffs is not None:
                onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))
                coeffs = otherCoeffs.flatten()

            if coeffs is not None:
                _mt.csr_sum_flat(data, coeffs, self._CSRSumIndices, self._CSRSumData, self._CSRSumPtr)

            #TODO: REMOVE
            # data.fill(0.0)  # data starts at zero
            #
            # if hamCoeffs is not None:
            #     # lnd_error_gen = sum([c*gen for c,gen in zip(hamCoeffs, self.hamGens)])
            #     _mt.csr_sum(data, hamCoeffs, self.hamGens, self.hamCSRSumIndices)
            #     onenorm += _np.dot(self.hamGens_1norms, _np.abs(hamCoeffs))
            #
            # if otherCoeffs is not None:
            #     if self.nonham_mode == "diagonal":
            #         # lnd_error_gen += sum([c*gen for c,gen in zip(otherCoeffs, self.otherGens)])
            #         _mt.csr_sum(data, otherCoeffs, self.otherGens, self.otherCSRSumIndices)
            #         onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs))
            #
            #     else:  # nonham_mode in ("diag_affine", "all")
            #         # lnd_error_gen += sum([c*gen for cRow,genRow in zip(otherCoeffs, self.otherGens)
            #         #                      for c,gen in zip(cRow,genRow)])
            #         _mt.csr_sum(data, otherCoeffs.flat,
            #                     [oGen for oGenRow in self.otherGens for oGen in oGenRow],
            #                     self.otherCSRSumIndices)
            #         onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))

            #Don't perform this check as this function is called a *lot* and it
            # could adversely impact performance
            #assert(_np.isclose(_np.linalg.norm(data.imag), 0)), \
            #    "Imaginary error gen norm: %g" % _np.linalg.norm(data.imag)

            #Update the rep's sparse matrix data stored in self._rep_data (the rep already
            # has the correct sparse matrix structure, as given by indices and indptr in
            # __init__, so we just update the *data* array).
            self._rep.data[:] = data.real

        else:  # dense matrices
            if hamCoeffs is not None:
                #lnd_error_gen = _np.einsum('i,ijk', hamCoeffs, self.hamGens)
                lnd_error_gen = _np.tensordot(hamCoeffs, self.hamGens, (0, 0))
                onenorm += _np.dot(self.hamGens_1norms, _np.abs(hamCoeffs))
            else:
                lnd_error_gen = _np.zeros((d2, d2), 'complex')

            if otherCoeffs is not None:
                if self.nonham_mode == "diagonal":
                    #lnd_error_gen += _np.einsum('i,ikl', otherCoeffs, self.otherGens)
                    lnd_error_gen += _np.tensordot(otherCoeffs, self.otherGens, (0, 0))
                    onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs))

                else:  # nonham_mode in ("diag_affine", "all")
                    #lnd_error_gen += _np.einsum('ij,ijkl', otherCoeffs,
                    #                            self.otherGens)
                    lnd_error_gen += _np.tensordot(otherCoeffs, self.otherGens, ((0, 1), (0, 1)))
                    onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))

            assert(_np.isclose(_np.linalg.norm(lnd_error_gen.imag), 0)), \
                "Imaginary error gen norm: %g" % _np.linalg.norm(lnd_error_gen.imag)
            #print("errgen pre-real = \n"); _mt.print_mx(lnd_error_gen,width=4,prec=1)
            self._rep.base[:, :] = lnd_error_gen.real
        self._onenorm_upbound = onenorm

    def todense(self):
        """
        Return this error generator as a dense matrix.
        """
        if self.sparse:
            return self.tosparse().toarray()
        else:
            if self._evotype in ("svterm", "cterm"):
                #Need to do similar things to __init__ - maybe consolidate?
                hamCoeffs, otherCoeffs = _gt.paramvals_to_lindblad_projections(
                    self.paramvals, self.ham_basis_size, self.other_basis_size,
                    self.param_mode, self.nonham_mode)

                hamGens, otherGens = self._init_generators(self.dim)

                if hamCoeffs is not None:
                    lnd_error_gen = _np.tensordot(hamCoeffs, hamGens, (0, 0))
                else:
                    lnd_error_gen = _np.zeros((self.dim, self.dim), 'complex')

                if otherCoeffs is not None:
                    if self.nonham_mode == "diagonal":
                        lnd_error_gen += _np.tensordot(otherCoeffs, otherGens, (0, 0))
                    else:  # nonham_mode in ("diag_affine", "all")
                        lnd_error_gen += _np.tensordot(otherCoeffs, otherGens, ((0, 1), (0, 1)))

                assert(_np.isclose(_np.linalg.norm(lnd_error_gen.imag), 0)), \
                    "Imaginary error gen norm: %g" % _np.linalg.norm(lnd_error_gen.imag)
                return lnd_error_gen.real

            else:  # dense rep
                return self._rep.base

    def tosparse(self):
        """
        Return the error generator as a sparse matrix.
        """
        _warnings.warn(("Constructing the sparse matrix of a LindbladDenseOp."
                        "  Usually this is *NOT* a sparse matrix (the exponential of a"
                        " sparse matrix isn't generally sparse)!"))
        if self.sparse:
            if self._evotype in ("svterm", "cterm"):
                #Need to do similar things to __init__ - maybe consolidate?
                hamCoeffs, otherCoeffs = _gt.paramvals_to_lindblad_projections(
                    self.paramvals, self.ham_basis_size, self.other_basis_size,
                    self.param_mode, self.nonham_mode)

                hamGens, otherGens = self._init_generators(self.dim)

                if hamCoeffs is not None:
                    lnd_error_gen = sum([c * gen for c, gen in zip(hamCoeffs, hamGens)])
                else:
                    lnd_error_gen = _sps.csr_matrix((self.dim, self.dim))

                if otherCoeffs is not None:
                    if self.nonham_mode == "diagonal":
                        lnd_error_gen += sum([c * gen for c, gen in zip(otherCoeffs, otherGens)])
                    else:  # nonham_mode in ("diag_affine", "all")
                        lnd_error_gen += sum([c * gen for cRow, genRow in zip(otherCoeffs, otherGens)
                                              for c, gen in zip(cRow, genRow)])

                return lnd_error_gen
            else:
                return _sps.csr_matrix((self._rep.data, self._rep.indices, self._rep.indptr),
                                       shape=(self.dim, self.dim))

        else:
            return _sps.csr_matrix(self.todense())

    #def torep(self):
    #    """
    #    Return a "representation" object for this error generator.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        if self.sparse:
    #            A = self.err_gen_mx
    #            return replib.DMOpRepSparse(
    #                _np.ascontiguousarray(A.data),
    #                _np.ascontiguousarray(A.indices, _np.int64),
    #                _np.ascontiguousarray(A.indptr, _np.int64))
    #        else:
    #            return replib.DMOpRepDense(_np.ascontiguousarray(self.err_gen_mx, 'd'))
    #    else:
    #        raise NotImplementedError("torep(%s) not implemented for %s objects!" %
    #                                  (self._evotype, self.__class__.__name__))

    def get_taylor_order_terms(self, order, max_poly_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the gate's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the gate's parent (usually a :class:`Model`), not the
        gate's local parameter array (i.e. that returned from `to_vector`).


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
        assert(order == 0), \
            "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        assert(return_coeff_polys is False)
        if self.Lterms is None:
            Ltermdict, basis = self.LtermdictAndBasis
            self.Lterms, self.Lterm_coeffs = self._init_terms(Ltermdict, basis, self._evotype, self.dim, max_poly_vars)
        return self.Lterms  # terms with local-index polynomial coefficients

    #def get_direct_order_terms(self, order): # , order_base=None - unused currently b/c order is always 0...
    #    v = self.to_vector()
    #    poly_terms = self.get_taylor_order_terms(order)
    #    return [ term.evaluate_coeff(v) for term in poly_terms ]

    def get_total_term_magnitude(self):
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
        # return (sum of absvals of term coeffs)
        assert(self.Lterms is not None), "Must call `get_taylor_order_terms` before calling get_total_term_magnitude!"
        vtape, ctape = self.Lterm_coeffs
        return _abs_sum_bulk_eval_compact_polys_complex(vtape, ctape, self.to_vector(), len(self.Lterms))

    def get_total_term_magnitude_deriv(self):
        """
        Get the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params()
        """
        # In general: d(|x|)/dp = d( sqrt(x.r^2 + x.im^2) )/dp = (x.r*dx.r/dp + x.im*dx.im/dp) / |x| = Re(x * conj(dx/dp))/|x|  # noqa: E501
        # The total term magnitude in this case is sum_i( |coeff_i| ) so we need to compute:
        # d( sum_i( |coeff_i| )/dp = sum_i( d(|coeff_i|)/dp ) = sum_i( Re(coeff_i * conj(d(coeff_i)/dp)) / |coeff_i| )

        wrtInds = _np.ascontiguousarray(_np.arange(self.num_params()), _np.int64)  # for Cython arg mapping
        vtape, ctape = self.Lterm_coeffs
        coeff_values = _bulk_eval_compact_polys_complex(vtape, ctape, self.to_vector(), (len(self.Lterms),))
        coeff_deriv_polys = _compact_deriv(vtape, ctape, wrtInds)
        coeff_deriv_vals = _bulk_eval_compact_polys_complex(coeff_deriv_polys[0], coeff_deriv_polys[1],
                                                            self.to_vector(), (len(self.Lterms), len(wrtInds)))
        abs_coeff_values = _np.abs(coeff_values)
        abs_coeff_values[abs_coeff_values < 1e-10] = 1.0  # so ratio is 0 in cases where coeff_value == 0
        ret = _np.sum(_np.real(coeff_values[:, None] * _np.conj(coeff_deriv_vals))
                      / abs_coeff_values[:, None], axis=0)  # row-sum
        assert(_np.linalg.norm(_np.imag(ret)) < 1e-8)
        return ret.real

        #DEBUG
        #ret2 = _np.empty(self.num_params(),'d')
        #eps = 1e-8
        #orig_vec = self.to_vector().copy()
        #f0 = sum([abs(coeff) for coeff in coeff_values])
        #for i in range(self.num_params()):
        #    v = orig_vec.copy()
        #    v[i] += eps
        #    new_coeff_values = _bulk_eval_compact_polys_complex(vtape, ctape, v, (len(self.Lterms),))
        #    ret2[i] = ( sum([abs(coeff) for coeff in new_coeff_values]) - f0 ) / eps

        #test3 = _np.linalg.norm(ret-ret2)
        #print("TEST3 = ",test3)
        #if test3 > 10.0:
        #    import bpdb; bpdb.set_trace()
        #return ret

    def num_params(self):
        """
        Get the number of independent parameters which specify this gate.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.paramvals)

    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this gate.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.paramvals

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
        assert(len(v) == self.num_params())
        self.paramvals = v
        if self._evotype == "densitymx":
            self._update_rep()
        if not nodirty: self.dirty = True

    def get_coeffs(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients
        of this error generator.  Note that these are not necessarily the
        parameter values, as these coefficients are generally functions of
        the parameters (so as to keep the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool
            Whether to also return a :class:`Basis` containing the elements
            with which the error generator terms were constructed.

        logscale_nonham : bool, optional
            Whether or not the non-hamiltonian error generator coefficients
            should be scaled so that the returned dict contains:
            `(1 - exp(-d^2 * coeff)) / d^2` instead of `coeff`.  This
            essentially converts the coefficient into a rate that is
            the contribution this term would have within a depolarizing
            channel where all stochastic generators had this same coefficient.
            This is the value returned by :method:`get_error_rates`.

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients.

        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `Ltermdict` to basis matrices.
        """
        hamC, otherC = _gt.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.param_mode, self.nonham_mode, self.Lmx)

        Ltermdict_and_maybe_basis = _gt.projections_to_lindblad_terms(
            hamC, otherC, self.ham_basis, self.other_basis, self.nonham_mode, return_basis)

        if logscale_nonham:
            Ltermdict = Ltermdict_and_maybe_basis if return_basis else Ltermdict_and_maybe_basis
            d2 = self.dim
            for k in Ltermdict.keys():
                if k[0] == "S":  # reverse mapping: err_coeff -> err_rate
                    Ltermdict[k] = (1 - _np.exp(-d2 * Ltermdict[k])) / d2  # err_rate = (1-exp(-d^2*errgen_coeff))/d^2

        return Ltermdict_and_maybe_basis

    def get_error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this
        error generator (pertaining to the *channel* formed by
        exponentiating this object).

        The "error rate" for an individual Hamiltonian error is the angle
        about the "axis" (generalized in the multi-qubit case)
        corresponding to a particular basis element, i.e. `theta` in
        the unitary channel `U = exp(i * theta/2 * BasisElement)`.

        The "error rate" for an individual Stochastic error is the
        contribution that basis element's term would have to the
        error rate of a depolarization channel.  For example, if
        the rate corresponding to the term ('S','X') is 0.01 this
        means that the coefficient of the rho -> X*rho*X-rho error
        generator is set such that if this coefficient were used
        for all 3 (X,Y, and Z) terms the resulting depolarizing
        channel would have error rate 3*0.01 = 0.03.

        Note that because error generator terms do not necessarily
        commute with one another, the sum of the returned error
        rates is not necessarily the error rate of the overall
        channel.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case.
        """
        return self.get_coeffs(return_basis=False, logscale_nonham=True)

    def set_coeffs(self, lindblad_term_dict, action="update", logscale_nonham=False):
        """
        Sets the coefficients of terms in this error generator.  The dictionary
        `lindblad_term_dict` has tuple-keys describing the type of term and the basis
        elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are the coefficients of these error generators,
            and should be real except for the 2-basis-label case.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :method:`get_errgen_coeffs`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :method:`set_error_rates`.

        Returns
        -------
        None
        """
        existing_Ltermdict, basis = self.get_coeffs(return_basis=True, logscale_nonham=False)

        if action == "reset":
            for k in existing_Ltermdict:
                existing_Ltermdict[k] = 0.0

        for k, v in lindblad_term_dict.items():
            if logscale_nonham and k[0] == "S":
                # treat the value being set in lindblad_term_dict as the *channel* stochastic error rate, and
                # set the errgen coefficient to the value that would, in a depolarizing channel, give
                # that per-Pauli (or basis-el general?) stochastic error rate. See lindbladtools.py also.
                # errgen_coeff = -log(1-d^2*err_rate) / d^2
                d2 = self.dim
                v = -_np.log(1 - d2 * v) / d2

            if k not in existing_Ltermdict:
                raise KeyError("Invalid L-term descriptor (key) `%s`" % str(k))
            elif action == "update" or action == "reset":
                existing_Ltermdict[k] = v
            elif action == "add":
                existing_Ltermdict[k] += v
            else:
                raise ValueError('Invalid `action` argument: must be one of "update", "add", or "reset"')

        hamC, otherC, _, _ = \
            _gt.lindblad_terms_to_projections(existing_Ltermdict, basis, self.nonham_mode)
        pvec = _gt.lindblad_projections_to_paramvals(
            hamC, otherC, self.param_mode, self.nonham_mode, truncate=True)  # shouldn't need to truncate
        self.from_vector(pvec)

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in this error generator so that the
        contributions of the resulting channel's error rate are given by
        the values in `lindblad_term_dict`.  See :method:`get_error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case, when they may be complex.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_coeffs(lindblad_term_dict, action, logscale_nonham=True)

    def transform(self, s):
        """
        Update error generator E with inv(s) * E * s,

        Generally, the transform function updates the *parameters* of
        the gate such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the gate parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.
        """
        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.get_transform_matrix()
            Uinv = s.get_transform_matrix_inverse()

            #conjugate Lindbladian exponent by U:
            err_gen_mx = self.tosparse() if self.sparse else self.todense()
            err_gen_mx = _mt.safedot(Uinv, _mt.safedot(err_gen_mx, U))
            trunc = bool(isinstance(s, _gaugegroup.UnitaryGaugeGroupElement))
            self._set_params_from_matrix(err_gen_mx, truncate=trunc)
            self.dirty = True
            #Note: truncate=True above for unitary transformations because
            # while this trunctation should never be necessary (unitaries map CPTP -> CPTP)
            # sometimes a unitary transform can modify eigenvalues to be negative beyond
            # the tight tolerances checked when truncate == False. Maybe we should be able
            # to give a tolerance as `truncate` in the future?

        else:
            raise ValueError("Invalid transform for this LindbladErrorgen: type %s"
                             % str(type(s)))

    def spam_transform(self, s, typ):
        """
        Update operation matrix G with inv(s) * G OR G * s,
        depending on the value of `typ`.

        This functions as `transform(...)` but is used when this
        Lindblad-parameterized gate is used as a part of a SPAM
        vector.  When `typ == "prep"`, the spam vector is assumed
        to be `rho = dot(self, <spamvec>)`, which transforms as
        `rho -> inv(s) * rho`, so `self -> inv(s) * self`. When
        `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
        `self` is NOT `self.dag` here), and `e.dag -> e.dag * s`
        so that `self -> self * s`.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        typ : { 'prep', 'effect' }
            Which type of SPAM vector is being transformed (see above).
        """
        assert(typ in ('prep', 'effect')), "Invalid `typ` argument: %s" % typ

        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.get_transform_matrix()
            Uinv = s.get_transform_matrix_inverse()
            err_gen_mx = self.tosparse() if self.sparse else self.todense()

            #just act on postfactor and Lindbladian exponent:
            if typ == "prep":
                err_gen_mx = _mt.safedot(Uinv, err_gen_mx)
            else:
                err_gen_mx = _mt.safedot(err_gen_mx, U)

            self._set_params_from_matrix(err_gen_mx, truncate=True)
            self.dirty = True
            #Note: truncate=True above because some unitary transforms seem to
            ## modify eigenvalues to be negative beyond the tolerances
            ## checked when truncate == False.
        else:
            raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
                             % str(type(s)))

    def _d_hdp(self):
        return self.hamGens.transpose((1, 2, 0))  # PRETRANS
        #return _np.einsum("ik,akl,lj->ija", self.leftTrans, self.hamGens, self.rightTrans)

    def _d_odp(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH - 1 if (bsH > 0) else 0
        d2 = self.dim

        assert(bsO > 0), "Cannot construct dOdp when other_basis_size == 0!"
        if self.nonham_mode == "diagonal":
            otherParams = self.paramvals[nHam:]

            # Derivative of exponent wrt other param; shape == [d2,d2,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [d2,d2,1]
            if self.param_mode == "depol":  # all coeffs same & == param^2
                assert(len(otherParams) == 1), "Should only have 1 non-ham parameter in 'depol' case!"
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None] * 2*otherParams[0]
                dOdp = _np.sum(_np.transpose(self.otherGens, (1, 2, 0)), axis=2)[:, :, None] * 2 * otherParams[0]
            elif self.param_mode == "reldepol":  # all coeffs same & == param
                assert(len(otherParams) == 1), "Should only have 1 non-ham parameter in 'reldepol' case!"
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None]
                dOdp = _np.sum(_np.transpose(self.otherGens, (1, 2, 0)), axis=2)[:, :, None] * 2 * otherParams[0]
            elif self.param_mode == "cptp":  # (coeffs = params^2)
                #dOdp  = _np.einsum('alj,a->lja', self.otherGens, 2*otherParams)
                dOdp = _np.transpose(self.otherGens, (1, 2, 0)) * 2 * otherParams  # just a broadcast
            else:  # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('alj->lja', self.otherGens)
                dOdp = _np.transpose(self.otherGens, (1, 2, 0))

        elif self.nonham_mode == "diag_affine":
            otherParams = self.paramvals[nHam:]
            # Note: otherGens has shape (2,bsO-1,d2,d2) with diag-term generators
            # in first "row" and affine generators in second row.

            # Derivative of exponent wrt other param; shape == [d2,d2,2,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [d2,d2,bs]
            if self.param_mode == "depol":  # all coeffs same & == param^2
                diag_params = otherParams[0:1]
                dOdp = _np.empty((d2, d2, bsO), 'complex')
                #dOdp[:,:,0]  = _np.einsum('alj->lj', self.otherGens[0]) * 2*diag_params[0] # single diagonal term
                #dOdp[:,:,1:] = _np.einsum('alj->lja', self.otherGens[1]) # no need for affine_params
                dOdp[:, :, 0] = _np.sum(self.otherGens[0], axis=0) * 2 * diag_params[0]  # single diagonal term
                dOdp[:, :, 1:] = _np.transpose(self.otherGens[1], (1, 2, 0))  # no need for affine_params
            elif self.param_mode == "reldepol":  # all coeffs same & == param^2
                dOdp = _np.empty((d2, d2, bsO), 'complex')
                #dOdp[:,:,0]  = _np.einsum('alj->lj', self.otherGens[0]) # single diagonal term
                #dOdp[:,:,1:] = _np.einsum('alj->lja', self.otherGens[1]) # affine part: each gen has own param
                dOdp[:, :, 0] = _np.sum(self.otherGens[0], axis=0)  # single diagonal term
                dOdp[:, :, 1:] = _np.transpose(self.otherGens[1], (1, 2, 0))  # affine part: each gen has own param
            elif self.param_mode == "cptp":  # (coeffs = params^2)
                diag_params = otherParams[0:bsO - 1]
                dOdp = _np.empty((d2, d2, 2, bsO - 1), 'complex')
                #dOdp[:,:,0,:] = _np.einsum('alj,a->lja', self.otherGens[0], 2*diag_params)
                #dOdp[:,:,1,:] = _np.einsum('alj->lja', self.otherGens[1]) # no need for affine_params
                dOdp[:, :, 0, :] = _np.transpose(self.otherGens[0], (1, 2, 0)) * 2 * diag_params  # broadcast works
                dOdp[:, :, 1, :] = _np.transpose(self.otherGens[1], (1, 2, 0))  # no need for affine_params
            else:  # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('ablj->ljab', self.otherGens) # -> shape (d2,d2,2,bsO-1)
                dOdp = _np.transpose(self.otherGens, (2, 3, 0, 1))  # -> shape (d2,d2,2,bsO-1)

        else:  # nonham_mode == "all" ; all lindblad terms included
            assert(self.param_mode in ("cptp", "unconstrained"))

            if self.param_mode == "cptp":
                L, Lbar = self.Lmx, self.Lmx.conjugate()
                F1 = _np.tril(_np.ones((bsO - 1, bsO - 1), 'd'))
                F2 = _np.triu(_np.ones((bsO - 1, bsO - 1), 'd'), 1) * 1j

                # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
                # Note: replacing einsums here results in at least 3 numpy calls (probably slower?)
                dOdp = _np.einsum('amlj,mb,ab->ljab', self.otherGens, Lbar, F1)  # only a >= b nonzero (F1)
                dOdp += _np.einsum('malj,mb,ab->ljab', self.otherGens, L, F1)    # ditto
                dOdp += _np.einsum('bmlj,ma,ab->ljab', self.otherGens, Lbar, F2)  # only b > a nonzero (F2)
                dOdp += _np.einsum('mblj,ma,ab->ljab', self.otherGens, L, F2.conjugate())  # ditto
            else:  # "unconstrained"
                F0 = _np.identity(bsO - 1, 'd')
                F1 = _np.tril(_np.ones((bsO - 1, bsO - 1), 'd'), -1)
                F2 = _np.triu(_np.ones((bsO - 1, bsO - 1), 'd'), 1) * 1j

                # Derivative of exponent wrt other param; shape == [d2,d2,bs-1,bs-1]
                #dOdp  = _np.einsum('ablj,ab->ljab', self.otherGens, F0)  # a == b case
                #dOdp += _np.einsum('ablj,ab->ljab', self.otherGens, F1) + \
                #           _np.einsum('balj,ab->ljab', self.otherGens, F1) # a > b (F1)
                #dOdp += _np.einsum('balj,ab->ljab', self.otherGens, F2) - \
                #           _np.einsum('ablj,ab->ljab', self.otherGens, F2) # a < b (F2)
                tmp_ablj = _np.transpose(self.otherGens, (2, 3, 0, 1))  # ablj -> ljab
                tmp_balj = _np.transpose(self.otherGens, (2, 3, 1, 0))  # balj -> ljab
                dOdp = tmp_ablj * F0  # a == b case
                dOdp += tmp_ablj * F1 + tmp_balj * F1  # a > b (F1)
                dOdp += tmp_balj * F2 - tmp_ablj * F2  # a < b (F2)

        # apply basis transform
        tr = len(dOdp.shape)  # tensor rank
        assert((tr - 2) in (1, 2)), "Currently, dodp can only have 1 or 2 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
        return _np.real(dOdp)

    def _d2_odp2(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH - 1 if (bsH > 0) else 0
        d2 = self.dim

        assert(bsO > 0), "Cannot construct dOdp when other_basis_size == 0!"
        if self.nonham_mode == "diagonal":
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams)

            # Derivative of exponent wrt other param; shape == [d2,d2,nP,nP]
            if self.param_mode == "depol":
                assert(nP == 1)
                #d2Odp2  = _np.einsum('alj->lj', self.otherGens)[:,:,None,None] * 2
                d2Odp2 = _np.sum(self.otherGens, axis=0)[:, :, None, None] * 2
            elif self.param_mode == "cptp":
                assert(nP == bsO - 1)
                #d2Odp2  = _np.einsum('alj,aq->ljaq', self.otherGens, 2*_np.identity(nP,'d'))
                d2Odp2 = _np.transpose(self.otherGens, (1, 2, 0))[:, :, :, None] * 2 * _np.identity(nP, 'd')
            else:  # param_mode == "unconstrained" or "reldepol"
                assert(nP == bsO - 1)
                d2Odp2 = _np.zeros([d2, d2, nP, nP], 'd')

        elif self.nonham_mode == "diag_affine":
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams)

            # Derivative of exponent wrt other param; shape == [d2,d2,nP,nP]
            if self.param_mode == "depol":
                assert(nP == bsO)  # 1 diag param + (bsO-1) affine params
                d2Odp2 = _np.empty((d2, d2, nP, nP), 'complex')
                #d2Odp2[:,:,0,0]  = _np.einsum('alj->lj', self.otherGens[0]) * 2 # single diagonal term
                d2Odp2[:, :, 0, 0] = _np.sum(self.otherGens[0], axis=0) * 2  # single diagonal term
                d2Odp2[:, :, 1:, 1:] = 0  # 2nd deriv wrt. all affine params == 0
            elif self.param_mode == "cptp":
                assert(nP == 2 * (bsO - 1)); hnP = bsO - 1  # half nP
                d2Odp2 = _np.empty((d2, d2, nP, nP), 'complex')
                #d2Odp2[:,:,0:hnP,0:hnP] = _np.einsum('alj,aq->ljaq', self.otherGens[0], 2*_np.identity(nP,'d'))
                d2Odp2[:, :, 0:hnP, 0:hnP] = _np.transpose(self.otherGens[0], (1, 2, 0))[
                    :, :, :, None] * 2 * _np.identity(nP, 'd')
                d2Odp2[:, :, hnP:, hnP:] = 0  # 2nd deriv wrt. all affine params == 0
            else:  # param_mode == "unconstrained" or "reldepol"
                assert(nP == 2 * (bsO - 1))
                d2Odp2 = _np.zeros([d2, d2, nP, nP], 'd')

        else:  # nonham_mode == "all" : all lindblad terms included
            nP = bsO - 1
            if self.param_mode == "cptp":
                d2Odp2 = _np.zeros([d2, d2, nP, nP, nP, nP], 'complex')  # yikes! maybe make this SPARSE in future?

                #Note: correspondence w/Erik's notes: a=alpha, b=beta, q=gamma, r=delta
                # indices of d2Odp2 are [i,j,a,b,q,r]

                def iter_base_ab_qr(ab_inc_eq, qr_inc_eq):
                    """ Generates (base,ab,qr) tuples such that `base` runs over
                        all possible 'other' params and 'ab' and 'qr' run over
                        parameter indices s.t. ab > base and qr > base.  If
                        ab_inc_eq == True then the > becomes a >=, and likewise
                        for qr_inc_eq.  Used for looping over nonzero hessian els. """
                    for _base in range(nP):
                        start_ab = _base if ab_inc_eq else _base + 1
                        start_qr = _base if qr_inc_eq else _base + 1
                        for _ab in range(start_ab, nP):
                            for _qr in range(start_qr, nP):
                                yield (_base, _ab, _qr)

                for base, a, q in iter_base_ab_qr(True, True):  # Case1: base=b=r, ab=a, qr=q
                    d2Odp2[:, :, a, base, q, base] = self.otherGens[a, q] + self.otherGens[q, a]
                for base, a, r in iter_base_ab_qr(True, False):  # Case2: base=b=q, ab=a, qr=r
                    d2Odp2[:, :, a, base, base, r] = -1j * self.otherGens[a, r] + 1j * self.otherGens[r, a]
                for base, b, q in iter_base_ab_qr(False, True):  # Case3: base=a=r, ab=b, qr=q
                    d2Odp2[:, :, base, b, q, base] = 1j * self.otherGens[b, q] - 1j * self.otherGens[q, b]
                for base, b, r in iter_base_ab_qr(False, False):  # Case4: base=a=q, ab=b, qr=r
                    d2Odp2[:, :, base, b, base, r] = self.otherGens[b, r] + self.otherGens[r, b]

            else:  # param_mode == "unconstrained"
                d2Odp2 = _np.zeros([d2, d2, nP, nP, nP, nP], 'd')  # all params linear

        # apply basis transform
        tr = len(d2Odp2.shape)  # tensor rank
        assert((tr - 2) in (2, 4)), "Currently, d2Odp2 can only have 2 or 4 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
        return _np.real(d2Odp2)

    def deriv_wrt_params(self, wrt_filter=None):
        """
        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per gate parameter.

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        assert(not self.sparse), \
            "LindbladErrorgen.deriv_wrt_params(...) can only be called when using *dense* basis elements!"

        d2 = self.dim
        bsH = self.ham_basis_size
        bsO = self.other_basis_size

        #Deriv wrt hamiltonian params
        if bsH > 0:
            dH = self._d_hdp()
            dH = dH.reshape((d2**2, bsH - 1))  # [iFlattenedOp,iHamParam]
        else:
            dH = _np.empty((d2**2, 0), 'd')  # so concat works below

        #Deriv wrt other params
        if bsO > 0:
            dO = self._d_odp()
            dO = dO.reshape((d2**2, -1))  # [iFlattenedOp,iOtherParam]
        else:
            dO = _np.empty((d2**2, 0), 'd')  # so concat works below

        derivMx = _np.concatenate((dH, dO), axis=1)
        assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)  # allowed to be complex?
        derivMx = _np.real(derivMx)

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this error generator with respect to
        its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1, wrt_filter2 : list
            Lists of indices of the paramters to take first and second
            derivatives with respect to.  If None, then derivatives are
            taken with respect to all of the gate's parameters.

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        assert(not self.sparse), \
            "LindbladErrorgen.hessian_wrt_params(...) can only be called when using *dense* basis elements!"

        d2 = self.dim
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH - 1 if (bsH > 0) else 0

        #Split hessian in 4 pieces:   d2H  |  dHdO
        #                             dHdO |  d2O
        # But only d2O is non-zero - and only when cptp == True

        nTotParams = self.num_params()
        hessianMx = _np.zeros((d2**2, nTotParams, nTotParams), 'd')

        #Deriv wrt other params
        if bsO > 0:  # if there are any "other" params
            nP = nTotParams - nHam  # num "other" params, e.g. (bsO-1) or (bsO-1)**2
            d2Odp2 = self._d2_odp2()
            d2Odp2 = d2Odp2.reshape((d2**2, nP, nP))

            #d2Odp2 has been reshape so index as [iFlattenedOp,iDeriv1,iDeriv2]
            assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
            hessianMx[:, nHam:, nHam:] = _np.real(d2Odp2)  # d2O block of hessian

        if wrt_filter1 is None:
            if wrt_filter2 is None:
                return hessianMx
            else:
                return _np.take(hessianMx, wrt_filter2, axis=2)
        else:
            if wrt_filter2 is None:
                return _np.take(hessianMx, wrt_filter1, axis=1)
            else:
                return _np.take(_np.take(hessianMx, wrt_filter1, axis=1),
                                wrt_filter2, axis=2)

    def onenorm_upperbound(self):
        """
        Returns an upper bound on the 1-norm for this error generator (
        viewed as a matrix).

        Returns
        -------
        float
        """
        # computes sum of 1-norms of error generator terms multiplied by abs(coeff) values
        # because ||A + B|| <= ||A|| + ||B|| and ||cA|| == abs(c)||A||
        return self._onenorm_upbound

    def __str__(self):
        s = "Lindblad error generator with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params())
        return s
