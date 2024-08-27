"""
Tools for working with representations of the special unitary group, SU(2).
"""
import numpy as np
import pygsti.modelmembers
import scipy.linalg as la
from pygsti.tools.su2tools import SU2, Spin72
from pygsti.tools.optools import unitary_to_superop
from pygsti.baseobjs.basis import Basis, BuiltinBasis
from typing import List
from tqdm import tqdm
import functools


def get_M():
    # define a bunch of constants to reduce the risk of typos.
    from numpy import sqrt
    s1b2   = sqrt(1/2)
    s7b6   = sqrt(7/6)
    s1b42  = sqrt(1/42)
    s3b14  = sqrt(3/14)
    s3b22  = sqrt(3/22)
    s1b858 = sqrt(1/858)
    s1b546 = sqrt(1/546)
    s3b286 = sqrt(3/286)
    s3b182 = sqrt(3/182)
    s1b66  = sqrt(1/66)
    s7b22  = sqrt(7/22)
    s1b154 = sqrt(1/154)
    s7b78  = sqrt(7/78)

    row1 = [     s1b2,        s1b2,         s1b2,          s1b2   ]
    row2 = [     s7b6,    5 * s1b42,        s3b14,         s1b42  ]
    row3 = [     s7b6,        s1b42,        s3b14,     5 * s1b42  ]
    row4 = [ 7 * s1b66,   5 * s1b66,    7 * s1b66,         s3b22  ]
    row5 = [     s7b22,  13 * s1b154,   3 * s1b154,    9 * s1b154 ]
    row6 = [     s7b78,  23 * s1b546,  17 * s1b546,    5 * s3b182 ]
    row7 = [     s1b66,   5 * s1b66,    3 * s3b22,     5 * s1b66  ]
    row8 = [     s1b858,  7 * s1b858,   7 * s3b286,   35 * s1b858 ]
    Mhalf = np.vstack([row1, row2, row3, row4, row5, row6, row7, row8])
    M = np.hstack([Mhalf, Mhalf[:, ::-1]])
    signs = np.ones((8,8))
    signs[1, 4:8] = -1
    signs[2, 2:6] = -1
    signs[3,   :] = [1,  -1,  -1,  -1,   1,   1,   1,  -1]
    signs[4,   :] = [1,  -1,  -1,   1,   1,  -1,  -1,   1]
    signs[5,   :] = [1,  -1,   1,   1,  -1,  -1,   1,  -1]
    signs[6,   :] = [1,  -1,   1,  -1,  -1,   1,  -1,   1]
    signs[7,   :] = [1,  -1,   1,  -1,   1,  -1,   1,  -1]
    M = signs * M
    M = 0.5*M
    # A construction that's more error prone but appearently equivalent.
    #
    # row1 = [    s1b2,         s1b2,         s1b2,          s1b2,         s1b2,          s1b2,          s1b2,          s1b2]
    # row2 = [    s7b6,     5 * s1b42,        s3b14,         s1b42,   -1 * s1b42,    -1 * s3b14,    -5 * s1b42,    -1 * s7b6]
    # row3 = [    s7b6,         s1b42,   -1 * s3b14,    -5 * s1b42,   -5 * s1b42,    -1 * s3b14,         s1b42,         s7b6]
    # row4 = [7 * s1b66,   -5 * s1b66,   -7 * s1b66,    -1 * s3b22,        s3b22,     7 * s1b66,     5 * s1b66,    -7 * s1b66]
    # row5 = [    s7b22,  -13 * s1b154,  -3 * s1b154,    9 * s1b154,   9 * s1b154,   -3 * s1b154,  -13 * s1b154,        s7b22]
    # row6 = [    s7b78,  -23 * s1b546,  17 * s1b546,    5 * s3b182,  -5 * s3b182,  -17 * s1b546,   23 * s1b546,   -1 * s7b78]
    # row7 = [    s1b66,   -5 * s1b66,    3 * s3b22,    -5 * s1b66,   -5 * s1b66,     3 * s3b22,    -5 * s1b66,         s1b66]
    # row8 = [    s1b858,  -7 * s1b858,   7 * s3b286,  -35 * s1b858,  35 * s1b858,   -7 * s3b286,    7 * s1b858,   -1 * s1b858]
    # M0 = np.vstack([row1, row2, row3, row4, row5, row6, row7, row8])
    return M


def get_F():

    F = np.zeros((8,8))
    F[0,  :8] = 1.0
    F[1, 1:8] = [59 / 63, 17 / 21,   13 / 21,    23 / 63,     1 / 21,    -1 / 3,    -7 / 9    ]
    F[2, 2:8] =         [  7 / 15,    1 / 21,    -1 / 3,    -11 / 21,    -1 / 3,     7 / 15   ]
    F[3, 3:8] =                   [ -31 / 77,  -101 / 231,    1 / 77,    17 / 33,   -7 / 33   ]
    F[4, 4:8] =                              [    1 / 9,    103 / 231,   -1 / 3,     7 / 99   ]
    F[5, 5:8] =                                         [   -33 / 91,    53 / 429,  -7 / 429  ]
    F[6, 6:8] =                                                       [  -1 / 39,    1 / 429  ]
    F[7, 7:8] =                                                                   [ -1 / 6435 ]

    tril_ind = np.tril_indices(8, -1)
    F[tril_ind] = F.T[tril_ind]
    # F = F / 8 <--- that scaling is likely a mistake, per Robin's email.
    return F


@functools.cache
def get_operator_basischangers(dim, from_basis, to_basis):
    """
    Return matrices to_mx, from_mx so that a square matrix A of order "dim"
    in basis from_basis can be converted into its represention in to_basis by
    B = to_mx @ A @ from_mx
    """
    if from_basis == to_basis:
        return np.eye(dim), np.eye(dim)
    if not isinstance(from_basis, Basis):
        from_basis = BuiltinBasis(from_basis, dim, sparse=False)
    if not isinstance(to_basis, Basis):
        to_basis = BuiltinBasis(to_basis, dim, sparse=False)
    to_mx = from_basis.create_transform_matrix(to_basis)
    from_mx = to_basis.create_transform_matrix(from_basis)
    return to_mx, from_mx


@functools.cache
def get_vector_basischanger(dim, from_basis, to_basis):
    if from_basis == to_basis:
        return np.eye(dim)
    if not isinstance(from_basis, Basis):
        from_basis = BuiltinBasis(from_basis, dim, sparse=False)
    if not isinstance(to_basis, Basis):
        to_basis = BuiltinBasis(to_basis, dim, sparse=False)
    to_mx = from_basis.create_transform_matrix(to_basis)
    return to_mx


def unitary_to_superoperator(U, basis):
    superop_std = pygsti.tools.unitary_to_std_process_mx(U)
    if basis == 'std':
        return superop_std
    else:
        to_mx, from_mx = get_operator_basischangers(superop_std.shape[0], 'std', basis)
        return to_mx @ superop_std @ from_mx


def default_povm(d, basis):
    effects = [np.zeros((d,d)) for _ in range(d)]
    for i, e in enumerate(effects):
        e[i,i] = 1.0
    povm = np.vstack([pygsti.tools.stdmx_to_vec(e, basis).ravel() for e in effects])
    return povm


class SU2RBSim:

    def __init__(self, su2class,  N: int, lengths: List[int]):
        self._su2rep = su2class
        self._unitary_dim = su2class.eigJx.size
        self._superop_dim = self._unitary_dim ** 2
        self._noise_channel = np.eye(self._superop_dim)
        self._povm = default_povm(self._unitary_dim, 'std')
        self.N = N
        self.lengths = np.array(lengths)
        pass

    def _set_error_channel_Jz_dephasing(self, gamma: float, power: float):
        assert gamma >= 0
        if gamma == 0:
            self._noise_channel = np.eye(self._superop_dim)
            return
        E_matrixunit = np.zeros(2*(self._superop_dim,))
        for ell in range(self._superop_dim):
            i = ell  % self._unitary_dim
            j = ell // self._unitary_dim
            E_matrixunit[ell,ell] = np.exp(-gamma * abs(i - j)**power)
        self._noise_channel = E_matrixunit
        return

    def set_error_channel_exponential(self, gamma: float):
        self._set_error_channel_Jz_dephasing(gamma, 1.0)
        return

    def set_error_channel_gaussian(self, gamma: float):
        self._set_error_channel_Jz_dephasing(gamma, 2.0)
        pass

    def set_error_channel_rotate_Jz2(self, theta: float):
        U = la.expm(1j * theta * self._su2rep.Jz @ self._su2rep.Jz)
        self._noise_channel = unitary_to_superop(U, 'std')
        pass

    def set_error_channel_exponential_compose_rotate_Jz2(self, gamma:float, theta:float):
        self.set_error_channel_exponential(gamma)
        E0 = self._noise_channel
        self.set_error_channel_rotate_Jz2(theta)
        E1 = self._noise_channel
        self._noise_channel = E1 @ E0

    def probabilities(self, seed=0):
        np.random.seed(seed)
        dim = self._unitary_dim
        statepreps = self._povm
        num_statepreps = statepreps.shape[0]
        probs = np.zeros((num_statepreps, self.lengths.size, self.N, dim))
        # probs[i,j,k,ell] = probability of measuring outcome ell 
        #    ... when running the k-th circuit of length lengths[j] 
        #    ... given preparation in state i.

        all_circuits = SU2.rb_circuits_by_angles(self.N, self.lengths, seed)
        # ^ For each k in range(N), all_circuits[j][k] is a (lengths[j]+1)-by-3 array.
        for j in tqdm(range(self.lengths.size)):
            fixed_length_circuits = all_circuits[j]
            probs[:,j,:,:] = self.process_circuits(fixed_length_circuits, statepreps)
        return probs

    def process_circuits(self, fixedlen_circuits, statepreps):
        N = len(fixedlen_circuits)
        num_statepreps = statepreps.size // (self._unitary_dim**2)
        statepreps = statepreps.reshape((num_statepreps, -1)) # make sure it's a bunch of row vectors.

        block = np.zeros(shape=(num_statepreps, N, self._unitary_dim))
        # block[i,k,ell] = the probability of measuring outcome ell after running the k-th circuit, given i-th starting state.
        for k, angles in enumerate(fixedlen_circuits):
            # angles has three columns. Each row of angles specifies an element of SU(2).
            # The sequence of SU(2) elements induced by the rows of angles defines a circuit. 
            unitaries = self._su2rep.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
            for i,superket in enumerate(statepreps):
                for U in unitaries:
                    densitymx_in = superket.reshape(U.shape)
                    densitymx_out = U @ densitymx_in @ U.T.conj()
                    superket = densitymx_out.ravel()
                    superket = self._noise_channel @ superket
                block[i,k,:] = self._povm @ superket
        return block

    @staticmethod
    def synspam_transform(_probs, M=None, shots_per_circuit=np.inf, seed=0):
        #  probs[i,j,k,ell] = probability of measuring outcome ell when running the k-th lengths[j] circuit given preparation in state i.
        if M is None:
            M = get_M()

        num_statepreps, num_lengths, circuits_per_length, num_povm_effects = _probs.shape
        assert circuits_per_length > 1
        assert M.shape == (num_statepreps, num_povm_effects)
        
        g = np.random.default_rng(seed)
        
        def mean_empirical_distribution(distributions):
            if shots_per_circuit == np.inf:
                empirical_distn = distributions
            else:
                empirical_distn = g.multinomial(shots_per_circuit, distributions) / shots_per_circuit
            return np.mean(empirical_distn, axis=0)

        # check each _probs[i,j,k,:] for negative values and normalize to a probability distribution
        tol = 1e-14
        temp = _probs.copy()
        temp[temp >= 0] = 0
        assert np.all(la.norm(temp, axis=3, ord=2) <= tol)
        _probs[_probs < 0] = 0.0
        _probs /= np.sum(_probs, axis=3)[:,:,:,np.newaxis]

        diag_statepreps  = np.diag_indices(num_statepreps)
        diag_povmeffects = np.diag_indices(num_povm_effects)
        synthetic_probs  = np.zeros((num_povm_effects, num_lengths))
        survival_probs   = np.zeros((num_statepreps,   num_lengths))
        for j in range(num_lengths):
            P = np.zeros((num_statepreps, num_statepreps))
            # ^ This will be row-stochastic, like a state transition matrix for a Markov chain.
            for i in range(num_statepreps):
                P[i,:] = mean_empirical_distribution(_probs[i,j,:,:])
            Q = M @ P @ M.T
            survival_probs[ :, j] = P[diag_statepreps]
            synthetic_probs[:, j] = Q[diag_povmeffects]

        return synthetic_probs, survival_probs


class Spin72CharacterRBSim(SU2RBSim):

    def __init__(self, N: int, lengths: List[int]):
        SU2RBSim.__init__(self, Spin72, N, lengths)
        pass

    def data_generators(self, seed=0, characters_only=False):
        """
        The loop over i in range(8) could be vectorized away if we could use the same
        collection of circuits for each state prep.
        """

        np.random.seed(seed)
        dim = self._unitary_dim
        statepreps = self._povm
        num_statepreps = statepreps.shape[0]
        probs = np.zeros((num_statepreps, self.lengths.size, self.N, dim))
        chars = np.zeros((num_statepreps, self.lengths.size, self.N, dim))
        # probs[i,j,k,ell] = probability of measuring outcome ell 
        #    ... when running the k-th circuit of length lengths[j] 
        #    ... given preparation in state i.

        all_circuits = SU2.rb_circuits_by_angles(self.N, self.lengths, seed)
        # ^ For each k in range(N), all_circuits[j][k] is a (lengths[j]+1)-by-3 array.
        for j in tqdm(range(self.lengths.size)):
            fixedlen_circuits = all_circuits[j]
            pb, cb = self.process_circuits(fixedlen_circuits, statepreps, characters_only)
            probs[:,j,:,:] = pb
            chars[:,j,:,:] = cb

        return probs, chars
    
    def process_circuits(self, fixedlen_circuits, statepreps, characters_only):

        # Set dimensions
        num_effects = self._unitary_dim
        num_irreps  = len(Spin72.irrep_labels)
        if num_effects != num_irreps:
            raise NotImplementedError()
        assert statepreps.ndim == 2
        num_statepreps = statepreps.shape[0]
        N = len(fixedlen_circuits)
        
        probs = np.zeros(shape=(num_statepreps, N, num_effects))
        # probs[i,k,ell] = the probability of measuring outcome ell after running the k-th circuit, given i-th starting state.
        chars = np.zeros(shape=(num_statepreps, N, num_irreps))
        # chars[i,k,ell] = the ell-th irrep's character for the unitary induced by the (noiseless version of the) k-th circuit, given the i-th starting state.
        #       --> Fun fact, this doesn't depend on i!
        error_free = la.norm(self._noise_channel - np.eye(64)) <= 1e-16
        for k, angles in enumerate(fixedlen_circuits):
            # angles has three columns. Each row of angles specifies an element of SU(2).
            # The sequence of SU(2) elements induced by the rows of angles defines a circuit. 
            a,b,g = angles.T # vectors of 
            char_U = SU2.unitaries_from_angles(a[0], b[0], g[0])[0]
            unitaries = self._su2rep.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
            chars[0,k,:] = [Spin72.character_from_unitary(char_U, j) for j in Spin72.irrep_labels]
            if characters_only:
                continue
            for i,superket in enumerate(statepreps):
                if error_free:
                    superket = superket.copy()
                    superop = unitary_to_superoperator(char_U, 'std')
                    superket = superop @ superket
                for U in unitaries:
                    densitymx_in = superket.reshape(U.shape)
                    densitymx_out = U @ densitymx_in @ U.T.conj()
                    superket = densitymx_out.ravel()
                    superket = self._noise_channel @ superket
                probs[i,k,:] = self._povm @ superket
        chars[1:,:,:] =  np.broadcast_to(chars[0,:,:], (num_statepreps-1, N, num_irreps))
        return probs

