import sys as _sys
import numpy as _np
import scipy as _sp
import scipy.optimize as _spo
from scipy.stats import chi2 as _chi2
import warnings as _warnings
import time as _time

import itertools as _itertools
from functools import reduce as _reduce
from functools import lru_cache as _lru_cache

try:
    from ...tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None

try:
    import cvxpy as _cp
except ImportError:
    _cp = None


REBUILD = True
REVERT_MSG_THRESHOLD = 10.0  # larger values = fewer messages
MAX_RESIDUAL_TVD_REDUCTION_PER_ITER = 0.3
OBJ_CHK_TOL = 1e-6  # tolerance used to check that objective fn decreases when it should
ZERO_RTVD_THRESHOLD = 1e-5  # threshold for considering a residualTVD == 0 (and not needing to compute higher weights)

# Make numpy raise exceptions on wrong input, rather than just warnings
# Useful for correctly handling logs of negative numbers
#_np.seterr(invalid='raise', divide='raise')  # don't do this globally in pyGSTi - use only for debugging!

# The "zero" used in positive-probability constraints.  Cannot be exactly 0
# because this causes problems when evaluating the log inside convex solver
# objective functions.
CONSTRAINT_ZERO = 0.0  # 5e-10

default_cvxpy_solver_args = {
    "all": dict(warm_start=True),
    "SCS": dict(eps=2e-6, max_iters=1000),
    "kicked_SCS": dict(eps=1e-7, max_iters=10000)
}


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def default_cvxpy_args(solver):
    addl_args = default_cvxpy_solver_args['all'].copy()
    addl_args.update(default_cvxpy_solver_args.get(solver, {}))
    return addl_args


def remove_kicked(s):
    if s.startswith("kicked_"):
        return s[len("kicked_"):]
    return s


def print_revert_msg(formatted_str, tup, verbosity):
    greater, lesser = tup
    if verbosity > 0 and (greater - lesser) / (greater + lesser + 1e-6) > REVERT_MSG_THRESHOLD:
        print("REVERTING: " + (formatted_str % tup))


# ------------------------------------------------------------------------------
# Parameterizing weight-k stochastic matrices: utility functions
# ------------------------------------------------------------------------------
def unit_vector(a, b):
    """Returns the unit vector of length 'b' with the 'a'th element = 1"""
    tmp = _np.zeros(b)
    tmp[a] = 1
    return tmp


def matrix_units(dim):
    """ Returns a list of all matrix units of dimension `dim` """
    return [_np.reshape(unit_vector(a, dim**2), (dim, dim)) for a in range(dim**2)]


def multikron(a):
    """ Kronecker product of all the elements of `a` """
    return _reduce(_np.kron, a)


# These are useful for making arbitrary matrices and then sticking in the right number of identities:
def interior_tensor_product(mx, dim_a, dim_b, e=None):
    """
    `mx` is an operator on two subsystems of dimension dim_a and dim_b
    `mx = sum_i A_i \otimes B_i` where A_i is an operator on subsystem a and B_i is an operator on subsystem b
    Return: sum_i A_i \otimes e \otimes B_i
    """
    assert _np.shape(mx) == (dim_a * dim_b, dim_a * dim_b), "Dimensions do not agree with matrix size"
    assert _np.shape(e)[0] == _np.shape(e)[1], "e should be a square matrix"
    basis_a = matrix_units(dim_a)
    basis_b = matrix_units(dim_b)
    return sum((_np.trace(_np.dot(mx, _np.kron(unit_a, unit_b).T)) * multikron([unit_a, e, unit_b])
                for unit_a in basis_a for unit_b in basis_b))


def swell_slow(mx, which_bits, n_bits=4):
    # M a transition matrix on bits b1..bn
    # Return a transition matrix on all bits
    assert all([bit < n_bits for bit in which_bits]), "You've specified bits not in the register"

    which_bits = _np.array(which_bits)

    if set(which_bits) == set(_np.arange(n_bits)):
        return mx

    for ind in range(n_bits):
        if ind in which_bits:
            continue
        else:
            dim_before = 2**(sum(which_bits < ind))
            dim_after = 2**(sum(which_bits > ind))
            mx = interior_tensor_product(mx, dim_before, dim_after, _np.eye(2))
            which_bits = _np.sort(_np.append(which_bits, ind))
            return swell_slow(mx, which_bits, n_bits)


def swell(mx, which_bits, n_bits=4):
    # M a transition matrix on bits b1..bn
    # Return a transition matrix on all bits
    assert all([bit < n_bits for bit in which_bits]), "You've specified bits not in the register"

    which_bits = _np.array(which_bits)

    if set(which_bits) == set(_np.arange(n_bits)):
        return mx

    # *** Below is a special case of construction found in DMOpRep_Embedded.__cinit__ ***
    #  (where each sector/component has dimension 2 - a classical bit)
    action_inds = which_bits  # the indices that correspond to mx indices
    numBasisEls = _np.array([2] * n_bits, _np.int64)

    # numBasisEls_noop_blankaction is just numBasisEls with actionInds == 1
    numBasisEls_noop_blankaction = numBasisEls.copy()
    numBasisEls_noop_blankaction[action_inds] = 1

    # multipliers to go from per-label indices to tensor-product-block index
    # e.g. if map(len,basisInds) == [1,4,4] then multipliers == [ 16 4 1 ]
    multipliers = _np.array(_np.flipud(_np.cumprod([1] + [2] * (n_bits - 1))), _np.int64)

    # noop_incrementers[i] specifies how much the overall vector index
    #  is incremented when the i-th "component" digit is advanced
    dec = 0
    noop_incrementers = _np.empty(n_bits, _np.int64)
    for i in range(n_bits - 1, -1, -1):
        noop_incrementers[i] = multipliers[i] - dec
        dec += (numBasisEls_noop_blankaction[i] - 1) * multipliers[i]

    # self.baseinds specifies the contribution from the "active
    #  component" digits to the overall vector index.
    baseinds = _np.empty(2**len(action_inds), _np.int64)
    basisInds_action = [[0, 1]] * len(action_inds)

    for ii, op_b in enumerate(_itertools.product(*basisInds_action)):
        vec_index = 0
        for j, bInd in zip(action_inds, op_b):
            vec_index += multipliers[j] * bInd
        baseinds[ii] = vec_index

    ret = _np.zeros((2**n_bits, 2**n_bits), 'd')  # final "swelled" matrix
    mx = _np.ascontiguousarray(mx)
    ret = _np.ascontiguousarray(ret)
    _fastcalc.fast_add_embeded(mx, ret, noop_incrementers, numBasisEls_noop_blankaction, baseinds)

    #CHECK DEBUG
    #check = swell_slow(mx, which_bits, n_bits)
    #assert(_np.allclose(check, ret))

    return ret


# ------------------------------------------------------------------------------
# Functions to handle parameter counting for stochastic matrices
# ------------------------------------------------------------------------------
def n_matrices_per_weight(weight, n_bits):
    """ The number of submatrices there are for `weight` """
    return int(_sp.special.binom(n_bits, weight))


def n_parameters_per_matrix(weight, n_bits):
    """ The number of parameters needed to define a weight-w transition submatrix on `n_bits`"""
    return 2**weight * (2**weight - 1)


def n_parameters(weight, n_bits):
    """ The number of parameters needed to define a complete weight-w transition matrix"""
    n_w = n_parameters_per_matrix(weight, n_bits)

    # Number of ways to pick weight bits out of n_bits
    n_a = n_matrices_per_weight(weight, n_bits)

    return n_w * n_a


def transition_matrix(v, dimension):
    """
    Produce a transition matrix of a given dimension given a parameter vector v.
    The only enforced constraint here is that the columns sum to 1
    """
    assert len(v) == dimension * (dimension - 1), f"Parameter vector must have length {dimension*(dimension-1)}."
    for ind in range(dimension):
        v = _np.insert(v, dimension * ind + ind, 1 - sum(v[dimension * ind:dimension * (ind + 1) - 1]))
    return _np.reshape(v, (dimension, dimension)).T


def comprehensive_transition_matrix(v, weight, n_bits):
    """ Build a generic weight-n transition_matrix """
    assert len(v) == n_parameters(weight, n_bits), "v is the wrong dimension"

    n_w = n_parameters_per_matrix(weight, n_bits)
    n_a = n_matrices_per_weight(weight, n_bits)

    vs = _np.reshape(v, (n_a, n_w))

    pairs = list(_itertools.combinations(_np.arange(n_bits), weight))
    ctm = sum((swell(transition_matrix(v, 2**weight), pair, n_bits)
               for v, pair in zip(vs, pairs))) / n_a

    return ctm


def nlogp(n, p):
    """n*log(p) such that if n == 0 the product is 0 too"""
    return 0 if n == 0 else n * _np.log(max(p, 1e-8))


def log_likelihood(data, probs):
    """ Compute log likelihood of a probability distribution over bitstrings given data """
    # Assume data is given as counts
    return _np.sum([nlogp(n, p) for n, p in zip(data, probs) if n > 0])


def max_log_likelihood(data):
    """ Compute log likelihood of a probability distribution over bitstrings given data """
    # Assume data is given as counts
    tot = sum(data)
    return _np.sum([nlogp(n, n / tot) for n in data if n > 0])


@_lru_cache(maxsize=100)
def _build_basis_slow(weight, n_bits):
    """
    Build a basis of matrices for constructing the transition matrix
    T = I + sum_i a_i G_i
    also builds the constraint matrix, C:
    C . a <= 1
    """
    _warnings.warn(("You're using a slow version of the basis-building code used by the disturbance calculations"
                    " - compile pyGSTi's C extensions to make this go faster."))
    n_w = n_parameters_per_matrix(weight, n_bits)
    n_a = n_matrices_per_weight(weight, n_bits)
    dim = 2**n_bits

    my_basis = []
    my_constraints = []
    # All sets of qubits of given weight on n_bits
    pairs = list(_itertools.combinations(_np.arange(n_bits), weight))

    for ind in range(n_w * n_a):
        v = unit_vector(ind, n_w * n_a)
        vs = _np.reshape(v, (n_a, n_w))
        ctm = sum((swell_slow(transition_matrix(v, 2**weight), pair, n_bits)
                   for v, pair in zip(vs, pairs))) - n_a * _np.eye(dim)
        my_basis += [ctm]
        my_constraints += [-_np.diag(ctm)]

    return my_basis, _np.array(my_constraints, dtype='int').T


@_lru_cache(maxsize=100)
def _build_basis_fast(weight, n_bits):
    """
    Build a basis of matrices for constructing the transition matrix
    T = I + sum_i a_i G_i
    also builds the constraint matrix, C:
    C . a <= 1
    """
    n_w = n_parameters_per_matrix(weight, n_bits)
    n_a = n_matrices_per_weight(weight, n_bits)
    dim = 2**n_bits

    my_basis = []
    my_constraints = []
    # All sets of qubits of given weight on n_bits
    pairs = list(_itertools.combinations(_np.arange(n_bits), weight))

    for ind in range(n_w * n_a):
        v = unit_vector(ind, n_w * n_a)
        vs = _np.reshape(v, (n_a, n_w))
        ctm = sum((swell(transition_matrix(v, 2**weight), pair, n_bits)
                   for v, pair in zip(vs, pairs)))
        ctm -= n_a * _np.eye(dim)
        my_basis += [ctm]
        my_constraints += [-_np.diag(ctm)]

    return my_basis, _np.array(my_constraints, dtype='int').T


#Select fast version if it's available
build_basis = _build_basis_fast if (_fastcalc is not None) else _build_basis_slow


class ResidualTVD:
    """
    Computes the "weight-X residual TVD": the TVD between two probability
    distributions up to weight-X transformations.

    This corresponds to optimizing abs(Q - T*P) where P and Q are the two
    probability distributions and T is a transition matrix.
    """

    def __init__(self, weight, n_bits, initial_treg_factor=1e-3, solver="SCS"):
        """
        Create a ResidualTVD function object.

        Parameters
        ----------
        weight : int
            The weight: all stochastic errors of this weight or below are
            considered "free", i.e. contribute nothing, to this residual TVD.

        n_bits : int
            The number of bits (qubits).

        initial_treg_factor : float, optional
            The magnitude of an internal penalty factor on the off-diagonals of
            the transition matrix (T), intended to eliminate unnecessarily-large
            T matrices which move a large proportion of probability between
            near-zero elements of both P and Q.  You should only adjust this
            if you know what you're doing.

        solver : str, optional
            The name of the solver to used (see `cvxpy.installed_solvers()`)
        """

        self.exactly_zero = bool(weight == n_bits)
        self.n_bits = n_bits
        self.n = int(2**n_bits)
        self.weight = weight
        self.dim = n_parameters(weight, n_bits)
        self.solver = solver
        self.initial_treg_factor = initial_treg_factor
        self.warning_msg = None

        # Hold values *separate* from cvxpy variables as we sometimes need to revert
        # cvxpy optimizations which actually move values in a way that gives a *worse*
        # objective function.
        self.t_params = _np.zeros(self.dim)

        # cvxpy parameters
        self.P = _cp.Parameter(shape=(self.n,), nonneg=True, value=_np.zeros(self.n))
        self.Q = _cp.Parameter(shape=(self.n,), nonneg=True, value=_np.zeros(self.n))

        if weight == 0: return  # special case; nothing more needed

        # Initialze a regularization factor to keep the optimizer from putting large elements
        # in T that move weight between near-zero elements of both p and q.  We might need
        # to adjust this later, so make it a parameter.
        self.Treg_factor = _cp.Parameter(nonneg=True, value=self.initial_treg_factor)

        # Build the basis and the constrain matrix - the basis used to construct the T vector
        self.t_basis, self.cons = build_basis(self.weight, self.n_bits)

        self._build_problem()

    def build_transfer_mx(self, t_params=None, apply_abs=True):
        """ Builds transition matrix from a vector of parameters """
        if t_params is None: t_params = self.t_params
        tmx = _np.sum([t_params[ind] * self.t_basis[ind] for ind in range(self.dim)], axis=0) + _np.eye(self.n)
        return _np.abs(tmx) if apply_abs else tmx

    def _build_problem(self):
        # Initialize the variables - the parameters used to define the T matrix
        self.T_params = _cp.Variable(self.dim, value=self.t_params.copy(), nonneg=True)

        # Constraints
        # T must be stochastic, so
        #    column sums must be 1 <-- enforced by construction of T
        #    T must have no negative elements so:
        #        1. Keep all the diagonal elements positive
        #        2. Keep all the off-diagonal elements positive
        bounds = _np.ones(self.n)
        self.constraints = [self.cons @ self.T_params <= bounds,
                            self.T_params >= 0]

        # Form objective.
        self.T = _cp.sum([self.T_params[ind] * self.t_basis[ind] for ind in range(self.dim)]) \
            + _np.eye(2**self.n_bits)
        self.resid_tvd = _cp.sum(_cp.abs(self.Q - self.T @ self.P)) / 2
        self.obj = _cp.Minimize(self.resid_tvd + self.Treg_factor * _cp.norm(self.T_params, 1))

        # Form the problem.
        self.prob = _cp.Problem(self.obj, self.constraints)

    def _rebuild_problem(self):
        # Set variable values
        self.T_params.value[:] = self.t_params.copy()

    def _obj(self, t_params):  # objective function for sanity checking cvxpy
        p = self.P.value
        q = self.Q.value
        tmx = self.build_transfer_mx(t_params)
        return _np.sum(_np.abs(q - _np.dot(tmx, p))) / 2

    def __call__(self, p, q, verbosity=1, warn=True):
        """
        Compute the residual TVD.

        Parameters
        ----------
        p, q : numpy array
            The reference and test probability distributions, respectively,
            given as an array of probabilities, one for each 2**n_bits bit string.

        verbosity : int, optional
            Sets the level of detail for messages printed to the console (higher = more detail).

        warn : bool, optional
            Whether warning messages should be issued if problems are encountered.

        Returns
        -------
        float
        """
        if self.exactly_zero: return 0.0  # shortcut for trivial case

        if self.weight == 0:
            return _np.sum(_np.abs(q - p)) / 2

        #Set parameter values
        self.P.value[:] = p[:]
        self.Q.value[:] = q[:]

        treg_factor_ok = False
        self.Treg_factor.value = self.initial_treg_factor
        while not treg_factor_ok:

            obj1 = self._obj(self.t_params)
            if REBUILD:
                self._rebuild_problem()
            else:
                self._build_problem()

            self.prob.solve(solver=remove_kicked(self.solver), verbose=(verbosity > 1),
                            **default_cvxpy_args(self.solver))

            failed = self.T.value is None  # or self.resid_tvd.value is None

            if not failed:  # sanity check
                t_chk = self.build_transfer_mx(self.T_params.value)
                assert(_np.linalg.norm(_np.abs(self.T.value) - t_chk) < 1e-6)

            self.warning_msg = None
            if failed:
                if self.solver == "SCS":
                    #raise ValueError("ResidualTVD: Convex optimizer failure")
                    for eps in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                        if REBUILD:
                            self._rebuild_problem()
                        else:
                            self._build_problem()
                        self.prob.solve(solver=remove_kicked(self.solver), verbose=(verbosity > 1), eps=eps)
                        failed = self.T.value is None  # or self.resid_tvd.value is None

                        if not failed:
                            t_chk = self.build_transfer_mx(self.T_params.value)
                            assert(_np.linalg.norm(self.T.value - t_chk) < 1e-6)

                            if eps > 1e-4:
                                self.warning_msg = ("ResidualTVD: Needed to increase eps to %g."
                                                    "  The resulting ResidualTVD values are less precise.") % eps
                                if warn: print(self.warning_msg)
                            break
                    else:
                        raise ValueError("ResidualTVD: Convex optimizer failure")
                else:
                    raise ValueError("ResidualTVD: Convex optimizer failure")

            #check that Treg_factor term doesn't dominate

            # Update: just leave this alone, since norm-penalty doesn't get reported - TODO later
            treg_factor_ok = True

            # ------------------------------------------------------------------
            #EXPERIMENTAL algorithms for updating Treg_factor ------------------
            # ------------------------------------------------------------------

            #resid_tvd = self._obj(self.T_params.value)
            #if resid_tvd > 10 * self.Treg_factor.value * _np.linalg.norm(self.T_params.value, 1):
            #    Treg_factor_ok = True
            #else:
            #    self.Treg_factor.value = resid_tvd / 10  # self.Treg_factor.value / 10

            #obj2 = self._obj(self.T_params.value)
            #if obj2 < obj1:
            #    Treg_factor_ok = True
            #else:
            #    #maybe penalty term dominated - reduce norm(tparams) penalty term
            #    self.T_params.value[:] = self.t_params[:]  #REVERT
            #    self.T.value[:, :] = _np.sum([self.t_params[ind] * self.t_basis[ind]
            #             for ind in range(self.dim)], axis=0) + _np.eye(self.n)  # REVERT
            #    self.Treg_factor.value = self.Treg_factor.value / 10
            #    if self.Treg_factor.value > 1e-7:
            #        print("REDUCING treg factor to: ", self.Treg_factor.value)
            #    else:
            #        Treg_factor_ok = True  # give up!

        if self.Treg_factor.value != self.initial_treg_factor:
            if verbosity > 0: print("NOTE: Treg_factor was reduced to %g." % self.Treg_factor.value)
            #_warnings.warn(("Initial Treg_factor (%g) was too large, and was reduced to %g."
            #                " Consider reducing the initial value to avoid repeating calculations.")
            #               % (self.initial_treg_factor, self.Treg_factor.value))

        obj2 = self._obj(self.T_params.value)
        if obj2 <= obj1:
            self.t_params[:] = self.T_params.value[:]
        else:
            print_revert_msg("ResidualTVD failed to reduce objective function (%g > %g)", (obj2, obj1), verbosity)
            self.T_params.value[:] = self.t_params[:]
            self.T.value[:, :] = self.build_transfer_mx(self.t_params)

        return self._obj(self.t_params)  # not self.obj.value b/c that has additional norm regularization


class RegularizedDeltaLikelihood:
    """
    The max - log-likelihood regularized by a "fixed-transition-matrix residual TVD".
    The 'alpha' parameter determines the strength of the regularizaton.  The objective
    function is:
        (max_logL - logL) + alpha * fixed_T_residual_tvd
    """

    def __init__(self, data_p, data_q, solver="SCS"):
        """
        Initialize a RegularizedLikelihood function object.

        Parameters
        ----------
        data_p, data_q : numpy array
            Arrays of outcome counts from the reference and test experiments,
            respectively.  Each array has one element per 2^n_bits bit string.

        solver : str, optional
            The name of the solver to used (see `cvxpy.installed_solvers()`)
        """
        self.data_P = data_p
        self.data_Q = data_q
        self.solver = solver
        self.warning_msg = None

        self.n = len(data_p)

        # Hold values *separate* from cvxpy variables as we sometimes need to revert
        # cvxpy optimizations which actually move values in a way that gives a *worse*
        # objective function.
        self.p = _np.array(self.data_P) / _np.sum(self.data_P)
        self.q = _np.array(self.data_Q) / _np.sum(self.data_Q)

        # cvxpy parameters
        self.T = _cp.Parameter(shape=(self.n, self.n), nonneg=True, value=_np.eye(self.n))
        self.alpha = _cp.Parameter(nonneg=True, value=1.0)

        self.max_logl = max_log_likelihood(data_p) + max_log_likelihood(data_q)
        self._build_problem()

    def _build_problem(self):
        #HACK: cvxpy seems non-deterministic usin SCS, and doesn't reliably return
        # the same result given the same problem unless we re-init the problem like this:
        self.P = _cp.Variable(self.n, nonneg=True, value=self.p.copy())
        self.Q = _cp.Variable(self.n, nonneg=True, value=self.q.copy())

        self.constraints = [self.P >= CONSTRAINT_ZERO, _cp.sum(self.P) == 1,
                            self.Q >= CONSTRAINT_ZERO, _cp.sum(self.Q) == 1]

        # Form objective.
        llp = _cp.sum([num * _cp.log(prob) for num, prob in zip(self.data_P, self.P) if num > 0])
        llq = _cp.sum([num * _cp.log(prob) for num, prob in zip(self.data_Q, self.Q) if num > 0])
        self.log_likelihood = llp + llq
        self.residual_tvd = _cp.sum(_cp.abs(self.Q - self.T @ self.P)) / 2
        self.objective = _cp.Minimize((self.max_logl - self.log_likelihood) + self.alpha * self.residual_tvd)
        self.prob = _cp.Problem(self.objective, self.constraints)

    def _rebuild_problem(self):
        # Set variable values
        self.P.value[:] = self.p.copy()
        self.Q.value[:] = self.q.copy()

    def _obj(self, p, q):  # objective function for sanity checking cvxpy
        alpha = self.alpha.value  # a parameter
        tmx = self.T.value  # a parameter
        delta_logl = self.max_logl - (log_likelihood(self.data_P, p)
                                      + log_likelihood(self.data_Q, q))
        residual_tvd = _np.sum(_np.abs(q - _np.dot(tmx, p))) / 2
        return delta_logl + alpha * residual_tvd

    def _delta_logl_value(self):
        dlogl = self.max_logl - (log_likelihood(self.data_P, self.p)
                                 + log_likelihood(self.data_Q, self.q))
        assert(dlogl >= 0)
        return dlogl

    def __call__(self, log10_alpha, tmx, verbosity=1, warn=True):
        """
        Computes the regularized log-likelihood:
        (max_logL - logL) + alpha * fixed_T_residual_tvd

        Parameters
        ----------
        log10_alpha : float
            log10(alpha), where alpha sets the strength of the regularization.

        T : numpy array
            The (fixed) transition matrix used in fixed_T_residual_tvd.

        verbosity : int, optional
            Sets the level of detail for messages printed to the console (higher = more detail).

        warn : bool, optional
            Whether warning messages should be issued if problems are encountered.

        Returns
        -------
        float
        """

        #Set parameter values
        self.T.value = tmx
        self.alpha.value = 10.0**log10_alpha

        obj1 = self._obj(self.p, self.q)
        if REBUILD:
            self._rebuild_problem()
        else:
            self._build_problem()
        self.prob.solve(solver=remove_kicked(self.solver), verbose=(verbosity > 1), **default_cvxpy_args(self.solver))
        failed = self.P.value is None or self.Q.value is None

        self.warning_msg = None
        if failed:
            if self.solver == "SCS":
                if verbosity > 0: print("RegularizedLikelihood: Convex optimizer failure")
                for eps in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    #if verbosity > 0: print("EPS = ", eps)
                    if REBUILD:
                        self._rebuild_problem()
                    else:
                        self._build_problem()
                    self.prob.solve(solver=remove_kicked(self.solver), verbose=(verbosity > 1), eps=eps)
                    failed = self.P.value is None or self.Q.value is None

                    if not failed:
                        if eps > 1e-4:
                            self.warning_msg = ("RegularizedLikelihood: Needed to increase eps to %g."
                                                "  The resulting ResidualTVD values are less precise.") % eps
                            if verbosity > 0 and warn: print(self.warning_msg)
                        break
                else:
                    raise ValueError("RegularizedLikelihood: Convex optimizer failure")
            else:
                raise ValueError("RegularizedLikelihood: Convex optimizer failure")

        obj2 = self._obj(self.P.value / sum(self.P.value), self.Q.value / sum(self.Q.value))
        if obj2 <= obj1:
            self.p[:] = self.P.value[:]
            self.q[:] = self.Q.value[:]
            self.p /= sum(self.p)  # ensure sum(p) == 1 (cvxpy doesn't always obey constraints exactly)
            self.q /= sum(self.q)  # ensure sum(q) == 1 (cvxpy doesn't always obey constraints exactly)
        else:
            print_revert_msg("RegularizedLikelihood failed to reduce objective function (%g > %g)",
                             (obj2, obj1), verbosity)
            self.P.value[:] = self.p[:]
            self.Q.value[:] = self.q[:]

        # Note: we just return the logl value, not the regularized
        # objective value (self.objective.value)
        return self._delta_logl_value()


class ProfileLikelihood:
    """
    The profile likelihood obtained by maximizing the likelihood on level-sets of
    constant weight-X residual-TVD.

    ProfileLikelihood(residual_TVD) values are evaluated by optimizing the function:

    alpha*ResidualTVD(p,q;weight) - log(Likelihood(p,q;data_ref,data_test))

    for a fixed value of alpha, yielding a single (residual_TVD, ProfileLikelihood) point.
    The optimization is implemented as an alternating minimization between
    optimize-T (ResidualTVD) and optimize-(P,Q) (RegularizedLikelihood) steps.
    """

    def __init__(self, weight, n_bits, data_ref, data_test, solver="SCS"):
        """
        Create a ProfileLikelihood function object.

        Parameters
        ----------
        weight : int
            The weight: all stochastic errors of this weight or below are
            considered "free", i.e. contribute nothing, to the residual TVD.

        n_bits : int
            The number of bits (qubits).

        data_ref, data_test : numpy array
            Arrays of outcome counts from the reference and test experiments,
            respectively.  Each array has one element per 2^n_bits bit string.

        solver : str, optional
            The name of the solver to used (see `cvxpy.installed_solvers()`)
        """
        self.weight = weight
        self.n_bits = n_bits
        self.data_ref = data_ref
        self.data_test = data_test
        self.solver = solver

        # Initialize the two solvers
        self.residual_tvd = ResidualTVD(weight, n_bits, solver=solver)
        self.reg_likelihood = RegularizedDeltaLikelihood(data_ref, data_test, solver=solver)

        # Initialize self.p, self.q, and self.T
        self._init_starting_values()

        # Store the log-likelihood with *no* regularization (alpha=0)
        # in case this is useful (this only depends on the data)
        self.max_logl = max_log_likelihood(data_ref) + max_log_likelihood(data_test)

    def _init_starting_values(self):
        # Initialize p, q, and T to a standard set of initial
        #  values before beginning an alternating-minimization.

        # Initialize p and q to their ML estimates
        self.p = _np.array(self.data_ref) / _np.sum(self.data_ref)
        self.q = _np.array(self.data_test) / _np.sum(self.data_test)
        # Initialize T for the ML estimates of P and Q
        self.t_params = _np.zeros(self.residual_tvd.dim)

        #Sync values in contained objectives
        self.residual_tvd.P.value[:] = self.p[:]
        self.residual_tvd.Q.value[:] = self.q[:]
        self.residual_tvd.t_params[:] = self.t_params[:]
        self.reg_likelihood.p[:] = self.p[:]
        self.reg_likelihood.q[:] = self.q[:]
        self.reg_likelihood.T.value[:, :] = self.residual_tvd.build_transfer_mx(self.t_params)

    def _obj(self, log10_alpha, p=None, q=None, tmx=None):  # for debugging
        if p is None: p = self.p
        if q is None: q = self.q
        if tmx is None: tmx = self.residual_tvd.build_transfer_mx(self.t_params)
        logl = (log_likelihood(self.data_ref, p)
                + log_likelihood(self.data_test, q)) - self.max_logl
        residual_tvd = _np.sum(_np.abs(q - _np.dot(tmx, p))) / 2
        return 10**log10_alpha * residual_tvd - logl

    def _iterate(self, log10_alpha, verbosity, warn):
        # Minimize over p and q
        tmx_raw = self.residual_tvd.build_transfer_mx(self.t_params, apply_abs=False)
        tmx = self.residual_tvd.build_transfer_mx(self.t_params)

        obj1 = self._obj(log10_alpha)  # ; print("obj1 = ",obj1)
        delta_logl = self.reg_likelihood(log10_alpha, tmx, verbosity=verbosity, warn=warn)
        self.p[:] = self.reg_likelihood.p[:]
        self.q[:] = self.reg_likelihood.q[:]

        obj2 = self._obj(log10_alpha)  # ; print("obj2 = ",obj2)
        assert(obj2 <= obj1 + OBJ_CHK_TOL)

        # Minimize over T
        #res_tvd_curT = _np.sum(_np.abs(self.q - _np.dot(T, self.p))) / 2  # uses "current" T
        res_tvd = self.residual_tvd(self.p, self.q, verbosity=verbosity, warn=warn)

        if self.weight != 0:  # weight = 0 case has no T matrix
            # we limit the change in p_prime = T*p:
            # |T'*p - T*p| = |((1-d)*Ts + (d-1)*T)*p| <= |(1-d)(Ts-T)|*|p| = (1-d)|Ts-T|*|p|
            # so, if we want delta_p_prime < eps then set (1-d) = eps / (|Ts-T|*|p|)
            # where norms are the vector 1-norm and inherited matrix 1-norm
            pre_res_tvd = _np.sum(_np.abs(self.q - _np.dot(tmx, self.p))) / 2  # uses "old" T
            eps = max(MAX_RESIDUAL_TVD_REDUCTION_PER_ITER, 0.1 * pre_res_tvd)  # only allow step of 0.1*existing tvd
            tmxs = self.residual_tvd.T.value

            damping = max(0, 1 - eps / max((_np.linalg.norm(_np.abs(tmxs) - tmx, ord=1)
                                            * _np.linalg.norm(self.p, ord=1)), 1e-6))
            self.t_params[:] = damping * self.t_params + (1 - damping) * self.residual_tvd.T_params.value
            self.residual_tvd.t_params[:] = self.t_params[:]  # back-propagate damped t_params to ResidualTVD
            self.residual_tvd.T_params.value[:] = self.t_params[:]  # needed?
            new_tmx = self.residual_tvd.build_transfer_mx(self.t_params)
            assert(_np.allclose(new_tmx, _np.abs(damping * tmx_raw + (1 - damping) * tmxs)))

            new_res_tvd = _np.sum(_np.abs(self.q - _np.dot(new_tmx, self.p))) / 2
            best_res_tvd = _np.sum(_np.abs(self.q - _np.dot(_np.abs(tmxs), self.p))) / 2

            assert(-OBJ_CHK_TOL < pre_res_tvd - new_res_tvd < eps)
            #print("DEBUG TVD: ", pre_res_tvd, new_res_tvd, best_res_tvd, res_tvd)
            assert(abs(best_res_tvd - res_tvd) <= OBJ_CHK_TOL)

        else:
            new_res_tvd = res_tvd  # no damping in weight=0 case

        obj3 = self._obj(log10_alpha)  # ; print("obj3 = ",obj3)
        assert(obj3 <= obj2 + OBJ_CHK_TOL)

        return new_res_tvd, delta_logl

    def __call__(self, log10_alpha=0, maxiters=20, reltol=1e-5, abstol=1e-5, verbosity=1, warn=True):
        """
        Compute an (x,y) = (residualTVD, ProfileLikelihood(residualTVD)) point
        given a fixed value of alpha, by minimizing (w.r.t p and q):

        alpha*ResidualTVD(p,q;weight) - log(Likelihood(p,q;data_ref,data_test))

        Parameters
        ----------
        log10_alpha : float
            log10(alpha), where alpha sets the strength of the regularization.

        maxiters : int, optional
            The maximum number of alternating-minimization iterations to allow
            before giving up and deeming the final result "ok".

        reltol : float, optional
            The relative tolerance used to within the alternating minimization.

        abstol : float, optional
            The absolute tolerance used to within the alternating minimization.

        verbosity : int, optional
            Sets the level of detail for messages printed to the console (higher = more detail).

        warn : bool, optional
            Whether warning messages should be issued if problems are encountered.

        Returns
        -------
        residualTVD : float
        ProfileLikelihood(residualTVD) : float
        """
        self._init_starting_values()
        last_residual_tvd = last_dlog_likelihood = -1.0e100  # a sentinel
        last_obj = None
        for ind in range(maxiters):
            residual_tvd, delta_log_likelihood = self._iterate(log10_alpha, verbosity - 1, warn)
            rel_rtvd = abs(last_residual_tvd - residual_tvd) / (abs(residual_tvd) + abstol)
            rel_logl = abs(last_dlog_likelihood - delta_log_likelihood) / (abs(delta_log_likelihood) + abstol)
            last_residual_tvd, last_dlog_likelihood = residual_tvd, delta_log_likelihood
            obj = delta_log_likelihood + 10**(log10_alpha) * residual_tvd

            if verbosity > 0:
                print("Iteration %d: dlogL=%g, residualTVD=%g (rel change=%g, %g): %g" %
                      (ind, delta_log_likelihood, residual_tvd, rel_logl, rel_rtvd, obj))
            assert(last_obj is None or obj <= last_obj), \
                "Alternating minimization failed to decrease objective function!"

            if (rel_logl < reltol or abs(delta_log_likelihood) < abstol) \
               and (rel_rtvd < reltol or abs(residual_tvd) < abstol):
                if verbosity > 0: print("Converged!")
                break
        else:
            if verbosity > 0:
                print("Maxium iterations (%d) reached before converging." % maxiters)

        return residual_tvd, delta_log_likelihood

    def at_logl_value(self, logl_value, maxiters=20, search_tol=0.1, reltol=1e-5, abstol=1e-5,
                      init_log10_alpha=3, verbosity=1):
        max_logl = self.max_logl
        res_tvd, delta_logl = self.at_delta_logl_value(max_logl - logl_value, maxiters,
                                                       search_tol, reltol, abstol, init_log10_alpha, verbosity)
        return res_tvd, max_logl - delta_logl

    def at_delta_logl_value(self, delta_logl_value, maxiters=20, search_tol=0.1, reltol=1e-5, abstol=1e-5,
                            init_log10_alpha=3, verbosity=1):
        """
        Compute an (x,y) = (residualTVD, ProfileLikelihood(residualTVD)) point
        such that ProfileLikelihood(residualTVD) is within `search_tol` of `logl_value`.

        Parameters
        ----------
        delta_logl_value : float
            the target profile (max - log-likelihood) value.

        maxiters : int, optional
            The maximum number of alternating-minimization iterations to allow
            before giving up and deeming the final result "ok".

        search_tol : float, optional
            The tolerance used when testing whether an obtained profile delta-log-likelihood
            value is close enough to `delta_logl_value`.

        reltol : float, optional
            The relative tolerance used to within the alternating minimization.

        abstol : float, optional
            The absolute tolerance used to within the alternating minimization.

        init_log10_alpha : float, optional
            The initial log10(alpha) value to use.  This shouldn't matter except
            that better initial values will cause the routine to run faster.

        verbosity : int, optional
            Sets the level of detail for messages printed to the console (higher = more detail).

        Returns
        -------
        residualTVD : float
        ProfileLikelihood(residualTVD) : float
        """
        log10_alpha = init_log10_alpha
        left = None; left_val = None
        right = None; right_val = None
        bracket_is_substantial = True
        res_tvd = None  # in case first evaluation fails
        it = 0

        if verbosity > 0: print("Searching for delta logl value = %.3f +/- %.3f" % (delta_logl_value, search_tol))
        while bracket_is_substantial:
            res_tvd, delta_logl = self(log10_alpha, maxiters, reltol, abstol, verbosity - 1, warn=False)

            if verbosity > 0:
                print("Binary search (iter %d): log10(a)=%.3f in [%.3f,%.3f]"
                      % (it, log10_alpha, left or _np.nan, right or _np.nan),
                      "dlogl=%.6f resTVD=%.6f" % (delta_logl, res_tvd))

                if (left_val and left_val > delta_logl) or (right_val and right_val < delta_logl):
                    print("WARNING: value looks suspicious!  Dlogl=%s should have been in (%s, %s)!"
                          % (delta_logl, str(right_val), str(left_val)))

            if abs(delta_logl - delta_logl_value) < search_tol:
                return res_tvd, delta_logl

            if res_tvd < abstol / 10.0:  # small residualTVD value => increasing alpha doesn't help, we're already at 0
                right = log10_alpha; right_val = delta_logl

            if delta_logl > delta_logl_value:
                # delta_logl too high, need less residualTVD penalty => decrease alpha
                right = log10_alpha; right_val = delta_logl
            else:
                # delta_logl too low, need more residualTVD penalty => increase alpha
                left = log10_alpha; left_val = delta_logl

            if left is not None and right is not None:
                if right_val - left_val > 1e-6:
                    gamma = (delta_logl_value - left_val) / (right_val - left_val)
                    #log10_alpha = _np.clip((1 - gamma) * left + gamma * right, left, right)
                    log10_alpha = _np.clip(_np.log10((1 - gamma) * 10**left + gamma * 10**right), left, right)
                else:
                    log10_alpha = (left + right) / 2.0
                bracket_is_substantial = (right - left) / (left + right) > 1e-6  # convergence criterion
            elif left is None:  # right was just updated -> decrease alpha
                log10_alpha -= 1  # decrease alpha by 1 order of magnitude
            else:
                log10_alpha += 1
            it += 1

        if verbosity > 0:
            if self.reg_likelihood.warning_msg: print(self.reg_likelihood.warning_msg)
            if self.residual_tvd.warning_msg: print(self.residual_tvd.warning_msg)

        if res_tvd > abstol and abs(delta_logl - delta_logl_value) < 4 * search_tol:
            # Only make a fuss if it's 4x the given tolerance
            _warnings.warn(("A binary search could not pinpoint the desired dlogL value within tolerance %g."
                            " (It achieved %g instead of the desired %g).  This could invalidate the computed"
                            " error bars.") % (4 * search_tol, delta_logl, delta_logl_value))
        # Otherwise we're against the "wall" where the ResidTVD==0, and
        # it's likely that logl_value can't be attained (so don't warn about it).

        return res_tvd, delta_logl

    def at_2llr_value(self, two_llr_value, maxiters=20, search_tol=0.1,
                      reltol=1e-5, abstol=1e-5, init_log10_alpha=3, verbosity=1):
        """
        Similar to :method:`at_delta_logl_value` except target is a 2*log-likelihood-ratio
        value, i.e. 2*(max_logL - logL).
        """
        # llr = max_logl - logl => delta_logl = two_llr_value/2.0
        return self.at_delta_logl_value(two_llr_value / 2.0, maxiters,
                                        search_tol, reltol, abstol, init_log10_alpha, verbosity)

    def at_confidence(self, confidence_percent, maxiters=20, search_tol=0.1,
                      reltol=1e-5, abstol=1e-5, init_log10_alpha=3, verbosity=1):
        """
        Similar to :method:`at_logl_value` except target is a given percent confidence
        value, yielding a (residualTVD, ProfileLikelihood(residualTVD)) point that lies
        on one end of a `confidence_percent`% confidence interval of the residualTVD.

        Note that `confidence_percent` should be a number between 0 and 100, *not* 0 and 1.
        """
        if confidence_percent <= 1.0:
            _warnings.warn(("`confidence_percent` <= 1.0 may be a mistake - "
                            "this should be value between 0 and 100, not 0 and 1."))
        return self.at_2llr_value(_chi2.ppf(confidence_percent / 100.0, df=1), maxiters,
                                  search_tol, reltol, abstol, init_log10_alpha, verbosity)


class ResidualTVDWithConfidence:
    """
    Residual TVD with error bars given by an assumed-symmetric confidence-region.

    The residual TVD is computed using :class:`ResidualTVD`.  A confidence region
    is constructed by finding where the :class:`ProfileLikelihood` is reduced from
    its maximum by an amount given by the desired confidence level.  This locates one
    side of the confidence region, and it is assumed to be symmetric.
    """

    def __init__(self, weight, n_bits, data_ref, data_test, solver="SCS", initial_treg_factor=1e-3):
        """
        Create a ResidualTVDWithConfidence function object.

        Parameters
        ----------
        weight : int
            The weight: all stochastic errors of this weight or below are
            considered "free", i.e. contribute nothing, to this residual TVD.

        n_bits : int
            The number of bits (qubits).

        data_ref, data_test : numpy array
            Arrays of outcome counts from the reference and test experiments,
            respectively.  Each array has one element per 2^n_bits bit string.

        solver : str, optional
            The name of the solver to used (see `cvxpy.installed_solvers()`)

        initial_treg_factor : float, optional
            The magnitude of an internal penalty factor on the off-diagonals of
            the T matrix (see :class:`ResidualTVD`).
        """
        self.exactly_zero = bool(weight == n_bits)
        self.residual_tvd = ResidualTVD(weight, n_bits, initial_treg_factor, solver=solver)
        self.profile_likelihood = ProfileLikelihood(
            weight, n_bits, data_ref, data_test, solver)
        self.pML = _np.array(data_ref) / _np.sum(data_ref)
        self.qML = _np.array(data_test) / _np.sum(data_test)

    def __call__(self, confidence_percent=68.0, maxiters=20, search_tol=0.1,
                 reltol=1e-5, abstol=1e-5, init_log10_alpha=3, verbosity=1):
        """
        Compute the ResidualTVD and its `confidence_percent`% confidence interval.

        Parameters
        ----------
        confidence_percent : float
            The confidence level desired for the computed error bars.  Note that this
            number can range between 0 and 100, not 0 and 1.

        maxiters : int, optional
            The maximum number of alternating-minimization iterations to allow within
            the profile-loglikelihood computation before giving up and deeming
            the final result "ok".

        search_tol : float, optional
            The tolerance on the log-likelihood used when trying to locate the
            (residualTVD, logL) pair with logL at the edge of the confidence interval.

        reltol : float, optional
            The relative tolerance used to within profile likelihood.

        abstol : float, optional
            The absolute tolerance used to within profile likelihood.

        init_log10_alpha : float, optional
            The initial log10(alpha) value to use within profile likelihood
            evaluations.  Only change this if you know what you're doing.

        verbosity : int, optional
            Sets the level of detail for messages printed to the console (higher = more detail).
        """
        if self.exactly_zero: return 0.0, 0.0  # shortcut for trivial case
        resid_tvd = self.residual_tvd(self.pML, self.qML)
        # print("ResidTVD = ",resid_tvd)
        resid_tvd_at_edge_of_cr, _ = self.profile_likelihood.at_confidence(
            confidence_percent, maxiters, search_tol, reltol, abstol, init_log10_alpha, verbosity)
        # print("ResidTVD @ CR-edge = ",resid_tvd_at_edge_of_cr)
        return resid_tvd, resid_tvd - resid_tvd_at_edge_of_cr


class ProfileLikelihoodPlot:
    def __init__(self, profile_likelihood, mode="auto-cr", maxiters=20,
                 search_tol=0.1, reltol=1e-5, abstol=1e-5, log10_alpha_values=None, num_auto_pts=10, verbosity=1):
        """
        Creates a plot of the profile log-likelihood.

        Parameters
        ----------
        profile_likelihood : ProfileLikelihood
            The profile likelihood to plot

        mode : {"auto-cr", "auto-fullrange", "manual"}
            How to decide what domain/range to plot.  "auto-cr" plots the region
            of the profile likelihood relevant to finding a confidence region.
            "auto-fullrange" plots the entire range of log-likelihood values, from
            the maximum to the amount it is reduced when the residual-TVD reaches 0.
            "manual" lets the user specify the log10(alpha) values to use (given
            in the `log10_alpha_values` argument).

        maxiters : int, optional
            The maximum number of alternating-minimization iterations to allow before
            giving up and deeming the final result "ok".

        search_tol : float, optional
            The tolerance on the log-likelihood used when trying to locate a (residualTVD, logL)
            pair with a particular logL.

        reltol : float, optional
            The relative tolerance used to within profile likelihood.

        abstol : float, optional
            The absolute tolerance used to within profile likelihood.

        log10_alpha_values : list, optional
            A list of log10(alpha) values to use to determing the (x,y)=(residualTVD, logL)
            points to plot when `mode == "manual"`.

        num_auto_pts : int, optional
            The number of points to include in the plot when `mode` is "auto-cr" or "auto-fullrange".

        verbosity : int, optional
            Sets the level of detail for messages printed to the console (higher = more detail).
        """
        # Place to dump the results
        self.profile_likelihood = profile_likelihood
        self.mode = mode
        self.residual_tvds = []
        self.log_likelihoods = []
        self.ps = []
        self.ts = []
        self.qs = []

        if mode.startswith("auto"):
            assert(log10_alpha_values is None)
            self._compute_pts_auto(mode, maxiters, search_tol, reltol, abstol, num_auto_pts, verbosity)
        elif mode == "manual":
            assert(log10_alpha_values is not None), "Must specify `log10_alpha_values` for manual mode!"
            self.log10_alphas = log10_alpha_values
            self._compute_pts_manual(log10_alpha_values, maxiters, reltol, abstol, verbosity)
        else:
            raise ValueError("Invalid mode: %s" % mode)

    def _compute_pts_manual(self, log10_alpha_values, maxiters,
                            reltol, abstol, verbosity):
        for log10_alpha in log10_alpha_values:
            residual_tvd, log_likelihood = self.profile_likelihood(
                log10_alpha, maxiters, reltol, abstol, verbosity)

            self.residual_tvds += [residual_tvd]
            self.log_likelihoods += [log_likelihood]
            self.ps += [self.profile_likelihood.p]
            self.ts += [_np.dot(self.profile_likelihood.T, self.profile_likelihood.p)]
            self.qs += [self.profile_likelihood.q]

        return self.residual_tvds, self.log_likelihoods

    def _get_minlogl(self, search_tol, maxiters, reltol, abstol, verbosity):
        large_log10_alpha = 3
        min_residual_tvd = 1.0
        min_logl = None  # in case failure on first eval
        while min_residual_tvd > search_tol:
            min_residual_tvd, min_logl = self.profile_likelihood(
                large_log10_alpha, maxiters, reltol, abstol, verbosity)
            large_log10_alpha += 1  # increase by 3 orders of magnitude
        return min_logl

    def _compute_pts_auto(self, mode, maxiters, search_tol, reltol, abstol, num_pts, verbosity):
        max_logl = self.profile_likelihood.max_logl

        if mode == "auto-cr":
            offset_to_cr_edge = _chi2.ppf(0.95, df=1) / 2.0  # delta logL to get to 95% CR edge
            min_logl = max_logl - 2 * offset_to_cr_edge  # range is 2x to put CR edge in middle of range.
        elif mode == "auto-fullrange":
            min_logl = self._get_minlogl(search_tol, maxiters, reltol, abstol, verbosity)
        else:
            raise ValueError("Invalid 'auto' mode: %s" % mode)

        desired_logl_values = _np.linspace(min_logl, max_logl, num_pts)
        for logl in desired_logl_values:
            residual_tvd, log_likelihood = self.profile_likelihood.at_logl_value(
                logl, maxiters, search_tol, reltol, abstol, verbosity=1)

            self.residual_tvds += [residual_tvd]
            self.log_likelihoods += [log_likelihood]
            self.ps += [self.profile_likelihood.p]
            self.ts += [_np.dot(self.profile_likelihood.residual_tvd.build_transfer_mx(), self.profile_likelihood.p)]
            self.qs += [self.profile_likelihood.q]

        return self.residual_tvds, self.log_likelihoods

    def make_plot(self, xlim=None, ylim=None, figsize=(10, 7), title=None):
        """
        Creates the plot figure using matplotlib.  Arguments are familiar plot variables.
        """
        from matplotlib import pyplot as plt

        xs, ys = self.residual_tvds, self.log_likelihoods
        plt.figure(figsize=figsize)
        plt.scatter(xs, ys)
        plt.title("Profile Likelihood" if (title is None) else title, fontsize=22)
        plt.xlabel('Residual TVD', fontsize=16)
        plt.ylabel('Log Likelihood', fontsize=16)
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        else:
            plt.xlim(_np.min(xs), _np.max(xs))

        if ylim:
            plt.ylim(ylim[0], ylim[1])
        else:
            plt.ylim(_np.min(ys), _np.max(ys))

        ax = plt.gca()
        ax.ticklabel_format(useOffset=False)


def compute_disturbances_with_confidence(n_bits, data_ref, data_test, confidence_percent=68.0,
                                         max_weight=4, maxiters=20, search_tol=0.1, reltol=1e-5,
                                         abstol=1e-5, solver="SCS", initial_treg_factor=1e-3, verbosity=1):
    """
    Compute the weight-X distrubances between two data sets (including error bars).

    This function is computes the weight-X disturbance, defined as the difference between
    the weight-(X-1) and weight-X residual TVDs, (evaluated at the ML probability
    distributions implied by the data) for all weights up to `max_weight`.  It also
    uses the data to compute `confidence_percent`% confidence intervals for each residualTVD
    and adds these in quadrature to arrive at error bars on each weight-X disturbance.

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    confidence_percent : float or None, optional
        The confidence level desired for the computed error bars.  Note that this
        number can range between 0 and 100, not 0 and 1.  If None, then no error
        bars are computed.

    max_weight : int, optional
        The maximum weight disturbance to compute.  Typically this is the same
        as `n_bits`.

    maxiters : int, optional
        The maximum number of alternating-minimization iterations to allow within
        the profile-loglikelihood computation before giving up and deeming
        the final result "ok".

    search_tol : float, optional
        The tolerance on the log-likelihood used when trying to locate the
        (residualTVD, logL) pair with logL at the edge of the confidence interval.

    reltol : float, optional
        The relative tolerance used to within profile likelihood.

    abstol : float, optional
        The absolute tolerance used to within profile likelihood.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    initial_treg_factor : float, optional
        The magnitude of an internal penalty factor on the off-diagonals of
        the T matrix (see :class:`ResidualTVD`).

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    Returns
    -------
    list
        A list of the disturbances by weight.  The lists i-th element is a
        `(disturbance, errorbar_length)` tuple for the weight (i+1) disturbance.
        That is, the weight (i+1) disturbance = `disturbance +/- errorbar_length`.
    """
    rtvds_by_weight = compute_residual_tvds(n_bits, data_ref, data_test, confidence_percent,
                                            max_weight, maxiters, search_tol, reltol,
                                            abstol, solver, initial_treg_factor, verbosity)
    rtvds = [value_and_errorbar[0] for value_and_errorbar in rtvds_by_weight]
    errorbars = [value_and_errorbar[1] for value_and_errorbar in rtvds_by_weight]

    disturbance_by_weight = []
    for i in range(1, max_weight + 1):
        eb = _np.sqrt(errorbars[i - 1]**2 + errorbars[i]**2) \
            if (confidence_percent is not None) else None
        disturbance_by_weight.append((rtvds[i - 1] - rtvds[i], eb))
    return disturbance_by_weight


def compute_ovd_over_tvd_ratio(n_bits, data_ref, data_test, p_ideal, return_all=False):
    """
    TODO: docstring
    """
    p_ml = _np.array(data_ref) / _np.sum(data_ref)
    q_ml = _np.array(data_test) / _np.sum(data_test)
    ratio = _np.zeros(p_ml.shape, 'd')
    nonzero_inds = _np.where(p_ideal > 0)[0]
    ratio[nonzero_inds] = p_ideal[nonzero_inds] / p_ml[nonzero_inds]
    tvd = _np.sum(_np.abs(q_ml - p_ml)) / 2
    ovd = _np.sum(ratio * _np.maximum(p_ml - q_ml, 0))
    r = ovd / tvd
    return r if (not return_all) else (r, ovd, tvd)


def compute_ovd_corrected_disturbances_noconfidence(n_bits, data_ref, data_test, p_ideal,
                                                    max_weight=4, maxiters=20, search_tol=0.1, reltol=1e-5,
                                                    abstol=1e-5, solver="SCS", initial_treg_factor=1e-3, verbosity=1):
    """
    Compute the weight-X distrubances between two data sets (including error bars).

    This function is computes the weight-X OVD-corrected disturbances, defined as the
    scaled difference between the weight-(X-1) and weight-X residual TVDs, for all weights up to
    `max_weight`.  Each difference is scaled by the ratio of the original variation distance (OVD)
    and the TVD, that is, multipled by r = OVD/TVD.

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    p_ideal : numpy array
        The ideal probability distribution (of both reference and test experiments).

    max_weight : int, optional
        The maximum weight disturbance to compute.  Typically this is the same
        as `n_bits`.

    maxiters : int, optional
        The maximum number of alternating-minimization iterations to allow within
        the profile-loglikelihood computation before giving up and deeming
        the final result "ok".

    search_tol : float, optional
        The tolerance on the log-likelihood used when trying to locate the
        (residualTVD, logL) pair with logL at the edge of the confidence interval.

    reltol : float, optional
        The relative tolerance used to within profile likelihood.

    abstol : float, optional
        The absolute tolerance used to within profile likelihood.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    initial_treg_factor : float, optional
        The magnitude of an internal penalty factor on the off-diagonals of
        the T matrix (see :class:`ResidualTVD`).

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    Returns
    -------
    list
        A list of the OVD-corrected disturbances by weight.  The lists i-th element is
        the weight (i+1) disturbance.  The `max_weight`-th element is the OVD/TVD ratio.
    """
    rtvds_by_weight = compute_residual_tvds(n_bits, data_ref, data_test, None,
                                            max_weight, maxiters, search_tol, reltol,
                                            abstol, solver, initial_treg_factor, verbosity)

    rtvds = [value_and_errorbar[0] for value_and_errorbar in rtvds_by_weight]
    scale_fctr = compute_ovd_over_tvd_ratio(n_bits, data_ref, data_test, p_ideal)

    disturbance_by_weight = []
    for i in range(1, max_weight + 1):
        disturbance_by_weight.append(scale_fctr * (rtvds[i - 1] - rtvds[i]))
    disturbance_by_weight.append(scale_fctr)

    return disturbance_by_weight


def compute_residual_tvds(n_bits, data_ref, data_test, confidence_percent=68.0,
                          max_weight=4, maxiters=20, search_tol=0.1, reltol=1e-5,
                          abstol=1e-5, solver="SCS", initial_treg_factor=1e-3, verbosity=1):
    """
    Compute the weight-X residual TVDs between two data sets (including error bars).

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    confidence_percent : float or None, optional
        The confidence level desired for the computed error bars.  Note that this
        number can range between 0 and 100, not 0 and 1.  If None, then no error
        bars are computed.

    max_weight : int, optional
        The maximum weight residual TVD to compute.  Typically this is the same
        as `n_bits`.

    maxiters : int, optional
        The maximum number of alternating-minimization iterations to allow within
        the profile-loglikelihood computation before giving up and deeming
        the final result "ok".

    search_tol : float, optional
        The tolerance on the log-likelihood used when trying to locate the
        (residualTVD, logL) pair with logL at the edge of the confidence interval.

    reltol : float, optional
        The relative tolerance used to within profile likelihood.

    abstol : float, optional
        The absolute tolerance used to within profile likelihood.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    initial_treg_factor : float, optional
        The magnitude of an internal penalty factor on the off-diagonals of
        the T matrix (see :class:`ResidualTVD`).

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    Returns
    -------
    list
        A list of the residual TVDs by weight.  The lists i-th element is a
        `(residual_tvd, errorbar_length)` tuple for the weight (i+1) residual TVD.
        That is, the weight (i+1) residual TVD = `residual_tvd +/- errorbar_length`.
    """
    residualtvd_by_weight = []
    last_rtvd = None; last_errorbar = None
    for weight in range(0, max_weight + 1):
        t0 = _time.time()

        if last_rtvd is not None and last_rtvd < ZERO_RTVD_THRESHOLD:
            if verbosity > 1:
                print("Approximating weight-%d residual TVD as zero (b/c weight-%d r-TVD < %g)"
                      % (weight, weight - 1, ZERO_RTVD_THRESHOLD))
            residualtvd_by_weight.append((0.0, 0.0))  # or use previous value and error bar?
            continue

        if verbosity > 0:
            print("Computing weight-%d residual TVD..." % weight, end='')
        if confidence_percent is not None:
            residual_tvd_fn = ResidualTVDWithConfidence(weight, n_bits, data_ref, data_test,
                                                        solver, initial_treg_factor)
            resid_tvd, errorbar = residual_tvd_fn(confidence_percent, maxiters,
                                                  search_tol, reltol, abstol, verbosity=verbosity - 2)
        else:
            p_ml = _np.array(data_ref) / _np.sum(data_ref)
            q_ml = _np.array(data_test) / _np.sum(data_test)
            residual_tvd_fn = ResidualTVD(weight, n_bits, solver=solver)
            resid_tvd = residual_tvd_fn(p_ml, q_ml, verbosity=verbosity - 2)
            errorbar = None

        # added a tolerance to the line below so this doesn't trigger with resid_tvd is barely above the last rtvd
        if last_rtvd is not None and resid_tvd > last_rtvd + 1e-6:
            #Try recomputing with kicked parameters
            solver = "kicked_" + solver
            kicked_args = default_cvxpy_args(solver)
            if len(kicked_args) > 0:
                if verbosity > 0:
                    print("Adjusting solver to use %s b/c residual TVD didn't decrease like it should have (%g > %g)"
                          % (str(kicked_args), resid_tvd, last_rtvd))
                if confidence_percent is not None:
                    residual_tvd_fn = ResidualTVDWithConfidence(weight, n_bits, data_ref, data_test,
                                                                solver, initial_treg_factor)
                    resid_tvd, errorbar = residual_tvd_fn(confidence_percent, maxiters, search_tol,
                                                          reltol, abstol, verbosity=verbosity - 2)
                else:
                    p_ml = _np.array(data_ref) / _np.sum(data_ref)
                    q_ml = _np.array(data_test) / _np.sum(data_test)
                    residual_tvd_fn = ResidualTVD(weight, n_bits, solver=solver)
                    resid_tvd = residual_tvd_fn(p_ml, q_ml, verbosity=verbosity - 2)
                    errorbar = None
            else:
                if verbosity > 0:
                    print("Warning! Residual TVD didn't decrease like it should (but no adjustments for %s)." % solver)
            solver = remove_kicked(solver)

        if last_rtvd is not None and resid_tvd > last_rtvd + 1e-6:
            if verbosity > 0:
                print(("Warning! Residual TVD *still* didn't decrease like it should have - "
                       "just using lower weight solution."))
            resid_tvd, errorbar = last_rtvd, last_errorbar

        residualtvd_by_weight.append((resid_tvd, errorbar))
        last_rtvd = resid_tvd
        last_errorbar = errorbar
        eb_str = (" +/- %.3g" % errorbar) if (errorbar is not None) else ""
        if verbosity > 0:
            print(" %5.1fs\t\t%.3g%s" % (_time.time() - t0, resid_tvd, eb_str))

    return residualtvd_by_weight


def resample_data(data, n_data_points=None, seed=None):
    """ Sample from the ML probability distrubution of `data`."""
    if seed is not None: _np.random.seed(seed)
    if n_data_points is None: n_data_points = _np.sum(data)
    p_ml = _np.array(data) / _np.sum(data)
    resampled = _np.random.multinomial(n_data_points, p_ml)
    return resampled


def compute_disturbances_bootstrap_rawdata(n_bits, data_ref, data_test, num_bootstrap_samples=20,
                                           max_weight=4, solver="SCS", verbosity=1, seed=0,
                                           return_resampled_data=False, add_one_to_data=True):
    """
    Compute the weight-X distrubances between two data sets (including error bars).

    This function is computes the weight-X disturbance, defined as the difference between
    the weight-(X-1) and weight-X residual TVDs, (evaluated at the ML probability
    distributions implied by the data) for all weights up to `max_weight`.  It also
    uses the data to compute 1-sigma error bar for each value using the boostrap method.

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    num_bootstrap_samples : int
        The number of boostrap (re-)samples to use.

    max_weight : int, optional
        The maximum weight disturbance to compute.  Typically this is the same
        as `n_bits`.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    add_one_to_data : bool, optional
        Sets whether the bootstrap should be calculated after adding a single fake count to every
        possible outcome.

    Returns
    -------
    disturbance_by_weight_ML : numpy.ndarray
        The ML disturbances by weight (length `max_weight`)

    bootstrap_disturbances_by_weight : numpy.ndarray
        A (max_weight, num_bootstrap_samples) sized array of each disturbance
        computed for each of the `num_bootstrap_samples` re-sampled data sets.
    """
    #p_ml = _np.array(data_ref) / _np.sum(data_ref)
    #q_ml = _np.array(data_test) / _np.sum(data_test)

    if verbosity > 0:
        print("Computing base disturbances")
    dist_by_weight_ml = compute_disturbances_with_confidence(
        n_bits, data_ref, data_test, None, max_weight, solver=solver, verbosity=verbosity - 1)

    dist_by_weight = _np.zeros((max_weight, num_bootstrap_samples), 'd')
    resampled_data = []

    bootstrap_data_ref = data_ref + _np.ones(len(data_ref), dtype='int')
    bootstrap_data_test = data_test + _np.ones(len(data_test), dtype='int')

    for i in range(num_bootstrap_samples):
        if verbosity > 0:
            print("Analyzing bootstrap sample %d of %d..." % (i + 1, num_bootstrap_samples), end='')
            _sys.stdout.flush(); tStart = _time.time()
        redata_ref = resample_data(bootstrap_data_ref, seed=seed + i)
        redata_test = resample_data(bootstrap_data_test, seed=seed + num_bootstrap_samples + i)
        if return_resampled_data:
            resampled_data.append((redata_ref, redata_test))

        try:
            disturbances = compute_disturbances_with_confidence(
                n_bits, redata_ref, redata_test, None, max_weight, solver=solver, verbosity=verbosity - 2)
        except Exception:
            try:
                if verbosity > 0: print("\nFalling back on ECOS")
                disturbances = compute_disturbances_with_confidence(
                    n_bits, redata_ref, redata_test, None, max_weight, solver="ECOS",
                    verbosity=verbosity - 2)
            except Exception:
                if verbosity > 0: print("\nFailed using %s and ECOS - reporting nans" % solver)
                for w in range(max_weight):
                    dist_by_weight[w, i] = _np.nan

        for w in range(max_weight):
            dist_by_weight[w, i] = disturbances[w][0]

        if verbosity > 0:
            print(" (%.1fs)" % (_time.time() - tStart))

    dist_ml = _np.array([dist_by_weight_ml[w][0] for w in range(max_weight)], 'd')

    if return_resampled_data:
        return dist_ml, dist_by_weight, resampled_data
    else:
        return dist_ml, dist_by_weight


def compute_ovd_corrected_disturbances_bootstrap_rawdata(n_bits, data_ref, data_test, p_ideal, num_bootstrap_samples=20,
                                                         max_weight=4, solver="SCS", verbosity=1, seed=0,
                                                         return_resampled_data=False, add_one_to_data=True):
    """
    Compute the weight-X distrubances between two data sets (including error bars).

    This function is computes the weight-X disturbance, defined as the difference between
    the weight-(X-1) and weight-X residual TVDs, (evaluated at the ML probability
    distributions implied by the data) for all weights up to `max_weight`.  It also
    uses the data to compute 1-sigma error bar for each value using the boostrap method.

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    p_ideal : numpy array
        The ideal probability distribution (of both reference and test experiments).

    num_bootstrap_samples : int
        The number of boostrap (re-)samples to use.

    max_weight : int, optional
        The maximum weight disturbance to compute.  Typically this is the same
        as `n_bits`.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    add_one_to_data : bool, optional
        Sets whether the bootstrap should be calculated after adding a single fake count to every
        possible outcome.

    Returns
    -------
    disturbance_by_weight_ML : numpy.ndarray
        The ML OVD-corrected disturbances by weight, with the OVD/TVD ratio tagged on at
        the end (so the length is `max_weight + 1`)

    bootstrap_disturbances_by_weight : numpy.ndarray
        A (max_weight + 1, num_bootstrap_samples) sized array of each disturbance and the
        OVD/TVD ratio (included as the final row in this matrix) for each of the
        `num_bootstrap_samples` re-sampled data sets.
    """
    #p_ml = _np.array(data_ref) / _np.sum(data_ref)
    #q_ml = _np.array(data_test) / _np.sum(data_test)

    if verbosity > 0:
        print("Computing base disturbances")
    dist_by_weight_ml = compute_ovd_corrected_disturbances_noconfidence(
        n_bits, data_ref, data_test, p_ideal, max_weight, solver=solver, verbosity=verbosity - 1)

    dist_by_weight = _np.zeros((max_weight, num_bootstrap_samples), 'd')
    resampled_data = []

    bootstrap_data_ref = data_ref + _np.ones(len(data_ref), dtype='int')
    bootstrap_data_test = data_test + _np.ones(len(data_test), dtype='int')

    for i in range(num_bootstrap_samples):
        if verbosity > 0:
            print("Analyzing bootstrap sample %d of %d..." % (i + 1, num_bootstrap_samples), end='')
            _sys.stdout.flush(); tStart = _time.time()
        redata_ref = resample_data(bootstrap_data_ref, seed=seed + i)
        redata_test = resample_data(bootstrap_data_test, seed=seed + num_bootstrap_samples + i)
        if return_resampled_data:
            resampled_data.append((redata_ref, redata_test))

        try:
            disturbances = compute_ovd_corrected_disturbances_noconfidence(
                n_bits, redata_ref, redata_test, p_ideal, max_weight, solver=solver, verbosity=verbosity - 2)
        except Exception:
            try:
                if verbosity > 0: print("\nFalling back on ECOS")
                disturbances = compute_ovd_corrected_disturbances_noconfidence(
                    n_bits, redata_ref, redata_test, p_ideal, max_weight, solver="ECOS",
                    verbosity=verbosity - 2)
            except Exception:
                if verbosity > 0: print("\nFailed using %s and ECOS - reporting nans" % solver)
                for w in range(max_weight):
                    dist_by_weight[w, i] = _np.nan

        for w in range(max_weight + 1):  # +1 includes r=OVD/TVD ratio (scale factor)
            dist_by_weight[w, i] = disturbances[w][0]

        if verbosity > 0:
            print(" (%.1fs)" % (_time.time() - tStart))

    dist_ml = _np.array([dist_by_weight_ml[w][0] for w in range(max_weight + 1)], 'd')

    if return_resampled_data:
        return dist_ml, dist_by_weight, resampled_data
    else:
        return dist_ml, dist_by_weight


def compute_disturbances_from_bootstrap_rawdata(ml_disturbances, bootstrap_disturbances,
                                                num_bootstrap_samples='all'):
    """
    Compute 1-sigma error bars for a set of disturbances (given by `ml_disturbances`)
    using boostrap data.

    Parameters
    ----------
    ml_disturbances : numpy.ndarray
        The disturbances by weight (length `max_weight`) for the maximum-likelhood
        (ML) distribution of some set of data.

    bootstrap_disturbances : numpy.ndarray
        A (max_weight, num_bootstrap_samples) sized array where each column is
        the set of by-weight disturbances for a distribution corresponding to a
        re-sampled bootstrap data set.

    num_bootstrap_samples : int or tuple or 'all'
        How many bootstrap samples to use when computing the boostrap error bars.
        This number can be less than the total number of bootstrap samples to test
        how using fewer boostrap samples would have performed.  `'all'` means to
        use all available bootstrap samples.  If a tuple, then each entry should be
        an integer and a series of error bars is returned (instead of a single one)
        corresponding to using each number of samples.

    Returns
    -------
    list
        A list of the disturbances by weight.  The lists i-th element is a
        `(disturbance, errorbar_length)` tuple for the weight (i+1) disturbance.
        That is, the weight (i+1) disturbance = `disturbance +/- errorbar_length`.
        If `num_bootstrap_samples` is a tuple, then elements are instead
        `(disturbance, errorbar_length1, errorbar_length2, ...)` where error bar
        lengths correspond to entries in `num_bootstrap_samples`.
    """
    if not isinstance(num_bootstrap_samples, (list, tuple)):
        num_bootstrap_samples = (num_bootstrap_samples,)

    max_weight = len(ml_disturbances)
    rms_disturbance_error = {w: () for w in range(max_weight)}
    for w in range(max_weight):
        for nsamples in num_bootstrap_samples:
            if nsamples == 'all': nsamples = len(bootstrap_disturbances[w])
            if nsamples == 0: continue  # zero boot strap samples => no error bars
            # error_vec = error in weight-(w+1) disturbance for each bootstrap sample
            error_vec = bootstrap_disturbances[w][0:nsamples] - ml_disturbances[w]
            rms_disturbance_error[w] += (_np.sqrt(_np.mean(error_vec**2)),)

    return [(ml_disturbances[w],) + rms_disturbance_error[w] for w in range(max_weight)]


def compute_disturbances(n_bits, data_ref, data_test, num_bootstrap_samples=20,
                         max_weight=4, solver="SCS", verbosity=1, add_one_to_data=True):
    """
    Compute the weight-X disturbances between two data sets (including error bars).

    This function is computes the weight-X disturbance, defined as the difference between
    the weight-(X-1) and weight-X residual TVDs, (evaluated at the ML probability
    distributions implied by the data) for all weights up to `max_weight`.  It also
    uses the data to compute 1-sigma error bar for each value using the boostrap method.

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    num_bootstrap_samples : int
        The number of boostrap (re-)samples to use.  If 0, then error bars are not computed.

    max_weight : int, optional
        The maximum weight disturbance to compute.  Typically this is the same
        as `n_bits`.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    add_one_to_data : bool, optional
        Sets whether the bootstrap should be calculated after adding a single fake count to every
        possible outcome.

    Returns
    -------
    list
        A list of the disturbances by weight.  The lists i-th element is a
        `(disturbance, errorbar_length)` tuple for the weight (i+1) disturbance.
        That is, the weight (i+1) disturbance = `disturbance +/- errorbar_length`.
    """
    dist_ml, dist = compute_disturbances_bootstrap_rawdata(
        n_bits, data_ref, data_test, num_bootstrap_samples,
        max_weight, solver, verbosity, add_one_to_data=add_one_to_data)
    return compute_disturbances_from_bootstrap_rawdata(dist_ml, dist)


def compute_ovd_corrected_disturbances(n_bits, data_ref, data_test, p_ideal, num_bootstrap_samples=20,
                                       max_weight=4, solver="SCS", verbosity=1, add_one_to_data=True):
    """
    Compute the weight-X disturbances between two data sets (including error bars).

    This function is computes the weight-X disturbance, defined as the difference between
    the weight-(X-1) and weight-X residual TVDs, (evaluated at the ML probability
    distributions implied by the data) for all weights up to `max_weight`.  It also
    uses the data to compute 1-sigma error bar for each value using the boostrap method.

    Parameters
    ----------
    n_bits : int
        The number of bits (qubits).

    data_ref, data_test : numpy array
        Arrays of outcome counts from the reference and test experiments,
        respectively.  Each array has one element per 2^n_bits bit string.

    num_bootstrap_samples : int
        The number of boostrap (re-)samples to use.  If 0, then error bars are not computed.

    max_weight : int, optional
        The maximum weight disturbance to compute.  Typically this is the same
        as `n_bits`.

    solver : str, optional
        The name of the solver to used (see `cvxpy.installed_solvers()`)

    verbosity : int, optional
        Sets the level of detail for messages printed to the console (higher = more detail).

    add_one_to_data : bool, optional
        Sets whether the bootstrap should be calculated after adding a single fake count to every
        possible outcome.

    Returns
    -------
    list
        A list of the disturbances by weight.  The lists i-th element is a
        `(disturbance, errorbar_length)` tuple for the weight (i+1) disturbance.
        That is, the weight (i+1) disturbance = `disturbance +/- errorbar_length`.
        the `max_weight`-th element gives the OVD/TVD ratio used to correct the
        TVD-based disturbance values, along with its error bar.
    """
    dist_ml, dist = compute_ovd_corrected_disturbances_bootstrap_rawdata(
        n_bits, data_ref, data_test, p_ideal, num_bootstrap_samples,
        max_weight, solver, verbosity, add_one_to_data=add_one_to_data)
    return compute_disturbances_from_bootstrap_rawdata(dist_ml, dist)


#TODO: move to unit tests
#sig_i = _np.array([[1., 0], [0, 1]], dtype='complex')
#sig_x = _np.array([[0, 1], [1, 0]], dtype='complex')
#sig_y = _np.array([[0, -1], [1, 0]], dtype='complex') * 1.j
#sig_z = _np.array([[1, 0], [0, -1]], dtype='complex')
#sig_m = (sig_x - 1.j * sig_y) / 2.
#sig_p = (sig_x + 1.j * sig_y) / 2.
#
#def test():
#    """ Unit tests for this module - a work in progress """
#    # This is a test of the above functions ... should all be 0
#    assert(_np.count_nonzero(
#        interior_tensor_product(multikron([sigX,sigZ,sigI]), 2,4, sigI) -
#        multikron([sigX,sigI,sigZ,sigI]))==0)
#    assert(_np.count_nonzero(swell(sigX,[1],3) - multikron([sigI,sigX,sigI]))==0)
#    assert(_np.count_nonzero(swell(sigX,[0],3) - multikron([sigX,sigI,sigI]))==0)
#    assert(_np.count_nonzero(swell(_np.kron(sigX,sigX),[1,3],4) - multikron([sigI,sigX,sigI,sigX]))==0)
#
#    # Test the above functions - How many parameters for a weight-k stochastic matrix on 4 bits?
#    assert([n_parameters(weight,4) for weight in [1,2,3,4]] == [8, 72, 224, 240])
#
#    # TODO - more unittests
