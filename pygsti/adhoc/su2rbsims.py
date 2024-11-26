"""
Tools for working with representations of the special unitary group, SU(2).
"""
import numpy as np
import pygsti.modelmembers
import scipy.linalg as la
from pygsti.tools.su2tools import SU2, Spin72
from pygsti.tools.optools import unitary_to_superop
from pygsti.baseobjs.basis import Basis, BuiltinBasis
from typing import List, Tuple, Callable, Union
from tqdm import tqdm
import functools
from matplotlib import pyplot as plt
from matplotlib import axes as plt_axes
import scipy.sparse as spar
import pathlib
import pickle


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


def raw_su2_rb_design(N: int, lengths: np.ndarray, seed: int, character=False):
    np.random.seed(seed)
    all_circuits = []
    invert_from = 1 if character else 0
    assert 0 not in lengths
    for ell in lengths:
        fixedlen_circuits = []
        for _ in range(N):
            onecircuit_angles = np.zeros((3, ell+1))
            onecircuit_angles[:, 0:ell] = np.row_stack(SU2.random_euler_angles(ell))
            onecircuit_angles[:, ell:]  = np.row_stack(SU2.composition_inverse(
                *onecircuit_angles[:, invert_from:ell]
            ))
            fixedlen_circuits.append(onecircuit_angles.T)
        all_circuits.append(np.array(fixedlen_circuits))
    return all_circuits


def empirical_distribution(outcome_distributions, shots_per_circuit, g):
    if shots_per_circuit == np.inf:
        empirical_distn = outcome_distributions.copy()
    else:
        empirical_distn = g.multinomial(shots_per_circuit, outcome_distributions) / shots_per_circuit
    return empirical_distn


def sanitize_probs(_probs):
    tol = 1e-14
    temp = _probs.copy()
    temp[temp >= 0] = 0
    assert np.all(la.norm(temp, axis=3, ord=2) <= tol)
    _probs[_probs < 0] = 0.0
    _probs /= np.sum(_probs, axis=3)[:,:,:,np.newaxis]

"""
Classic RB (although allowing for multiple state preps, and allowing full POVMs rather than 2-element POVMs)
"""


class SU2RBDesign:

    def __init__(self, su2rep, N, lengths, statepreps, povm, seed, __character__=False):
        self.su2rep = su2rep
        self.N = N
        self.lengths = lengths
        self.all_circuits = raw_su2_rb_design(N, lengths, seed, __character__)
        self.statepreps = statepreps
        self.povm = povm
        assert self.statepreps.ndim == 2
        assert self.povm.ndim == 2
        assert self.statepreps.shape[1] == self.povm.shape[1]
        return
    
    @property
    def num_effects(self) -> int:
        return self.povm.shape[0]

    @property
    def num_statepreps(self) -> int:
        return self.statepreps.shape[0]
    
    @property
    def num_lens(self) -> int:
        return self.lengths.size


class SU2RBSim:

    def __init__(self, design, __allow_characters__=False):
        self._unitary_dim = design.su2rep.eigJx.size
        self._superop_dim = self._unitary_dim ** 2
        self._noise_channel = np.eye(self._superop_dim)
        self.design = design
        if not __allow_characters__:
            assert type(design) == SU2RBDesign
        self.probs = None
        self.nonspam_compositions = None
        self._noise_channel_repr = 'superoperator'
        pass
    
    @property
    def N(self) -> int:
        return self.design.N

    @property
    def num_statepreps(self) -> int:
        return self.design.num_statepreps
    
    @property
    def num_effects(self) -> int:
        return self.design.num_effects
    
    @property
    def num_lens(self) -> int:
        return self.design.lengths.size

    @property
    def su2rep(self):
        return self.design.su2rep

    @property
    def main_filename(self):
        profile = self._noise_channel_info[0]
        assert profile in {'rotate_Jz', 'Jz_dephasing'}

        rate_or_angle, power = self._noise_channel_info[1]
        ratestr = f'{rate_or_angle:6.4f}'[2:] # just grabs four digits after the decimal place.
        max_circuit_length = self.design.lengths[-1]
        typestr = 'character' if isinstance(self.design, SU2CharacterRBDesign) else 'standard'
        n = self.N
        prefix = '/Users/rjmurr/Documents/pygsti-general/scripts/demos/workspace'
        if profile == 'rotate_Jz':
            fn = f'{prefix}/su2_{typestr}_rb_N_{n}_maxlen_{max_circuit_length}_rotateJz_angle_{ratestr}_power_{power}.pkl'
            return fn
        else:
            fn = f'{prefix}/su2_{typestr}_rb_N_{n}_maxlen_{max_circuit_length}_dephasing_rate_{ratestr}_power_{power}.pkl'
            return fn

    @property
    def nonspam_comp_filename(self):
        fn = self.main_filename.strip('.pkl') + '_nonspam_comp.pkl'
        return fn
    
    def load_nonspam_compositions(self):
        if self.nonspam_compositions is not None:
            return
        fn = self.nonspam_comp_filename
        if pathlib.Path(fn).exists():
            f = open(fn, 'rb')
            self.nonspam_compositions = pickle.load(f)
            f.close()
        return

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
        if self._noise_channel_repr == 'superoperator':
            self._noise_channel = spar.dia_array(E_matrixunit)
            self._noise_channel_info = ('Jz_dephasing', (gamma, power))
        else:
            raise ValueError()
        return

    def set_error_channel_exponential(self, gamma: float):
        self._set_error_channel_Jz_dephasing(gamma, 1.0)
        return

    def set_error_channel_gaussian(self, gamma: float):
        self._set_error_channel_Jz_dephasing(gamma, 2.0)
        return

    def set_error_channel_rotate_Jz_pow(self, theta: float, pow: int):
        dZ = self.su2rep.SPINS
        exp_1jthetaJzpow = np.exp(1j * theta * (dZ ** pow))
        U = np.diag(exp_1jthetaJzpow)
        if self._noise_channel_repr == 'superoperator':
            self._noise_channel = unitary_to_superop(U, 'std')
        elif self._noise_channel_repr == 'unitary':
            self._noise_channel = U
        else:
            raise ValueError()
        self._noise_channel_info = ('rotate_Jz', (theta, pow))
        return

    def set_error_channel_rotate_Jz2(self, theta: float):
        self.set_error_channel_rotate_Jz_pow(theta, 2)
        return

    def set_error_channel_gaussian_compose_rotate_Jz2(self, gamma:float, theta:float):
        self._noise_channel_repr = 'superoperator'
        self.set_error_channel_gaussian(gamma)
        E0 = self._noise_channel
        self.set_error_channel_rotate_Jz2(theta)
        E1 = self._noise_channel
        self._noise_channel = E1 @ E0
        self._noise_channel_info = ('gaussian_compose_rotate_Jz2', (gamma, theta))

    @property
    def noise_channel_parameter_names(self):
        if self._noise_channel_info is None:
            return ('Identity',)
        ncname = self._noise_channel_info[0]
        if ncname == 'Jz_dephasing':
            return (r'$\gamma$', '$p$')
        if ncname == 'rotate_Jz':
            return (r'$\theta$', '$p$')
        raise NotImplementedError()
    
    @property
    def noise_channel_math_expr(self):
        if self._noise_channel_info is None:
            return ('Id',)
        pnames = [s.strip('$') for s in self.noise_channel_parameter_names]
        ncname = self._noise_channel_info[0]
        if ncname == 'Jz_dephasing':
            rhoij = '\\rho_{ij}'
            return f'elementwise scaling ${rhoij} \\leftarrow {rhoij} \\cdot \\exp(-{pnames[0]} \\cdot |i - j|^{pnames[1]})$'
        if ncname == 'rotate_Jz':
            im = 'i'
            jzpow = 'J_z^' + pnames[1]
            return f'evolve $\\rho \\mapsto U \\rho U^\\dagger$ where $U =$ expm($-${im}$ {pnames[0]} \\cdot {jzpow}$)'
        raise NotImplementedError()

    def compute_probabilities(self):
        probs = np.zeros((self.num_statepreps, self.num_lens, self.N,  self.num_effects))
        # probs[i,j,k,ell] = probability of measuring outcome ell 
        #    ... when running the k-th circuit of length lengths[j] 
        #    ... given preparation in state i.
        all_circuits = self.design.all_circuits
        # ^ For each k in range(N), all_circuits[j][k] is a (lengths[j]+1)-by-3 array.
        for j, fixed_length_circuits in tqdm(enumerate(all_circuits)):
            probs[:,j,:,:] = self._process_circuits(fixed_length_circuits)
        self.probs = probs
        return

    def _process_circuits(self, fixedlen_circuits, skip_first_noise=False):
        assert self._noise_channel_repr in ('superoperator', 'unitary')
        block = np.zeros(shape=(self.num_statepreps, self.N, self.num_effects))
        # block[i,k,ell] = the probability of measuring outcome ell after running the k-th circuit, given i-th starting state.
        for k, angles in enumerate(fixedlen_circuits):
            # angles has three columns. Each row of angles specifies an element of SU(2).
            # The sequence of SU(2) elements induced by the rows of angles defines a circuit. 
            unitaries = self.su2rep.unitaries_from_angles(angles[:, 0], angles[:, 1], angles[:, 2])
            for i,superket in enumerate(self.design.statepreps):
                skip_next_noise = skip_first_noise
                if self._noise_channel_repr == 'superoperator':
                    # simulate by alternating between density matrix and superket reps
                    for U in unitaries:
                        densitymx_in = superket.reshape(U.shape)
                        densitymx_out = U @ densitymx_in @ U.T.conj()
                        superket = densitymx_out.ravel()
                        if not skip_next_noise:
                            superket = self._noise_channel @ superket
                        skip_next_noise = False
                else:
                    # simulate the entire circuit in density matrix rep
                    #   could cut the matmuls in half by working with sqrtm(rho)
                    densitymx_out = superket.reshape((self._unitary_dim, self._unitary_dim))
                    for U in unitaries:
                        W = U if skip_next_noise else self._noise_channel @ U
                        densitymx_out = W @ densitymx_out @ W.T.conj()
                        skip_next_noise = False
                    superket = densitymx_out.ravel()
                block[i,k,:] = self.design.povm @ superket
        return block

    def compute_nonspam_compositions(self, skip_first_noise=False):
        """
        out.shape == (self.num_lens, self.N, self._superop_dim, self._superop_dim)
        out[a, b] = superoperator representing the composition of all non-spam operations
                    in the b-th circuit of length self.design.lengths[a].
        """
        assert self._noise_channel_repr in ('superoperator', 'unitary')
        all_circuits = self.design.all_circuits
        out = []
        for fixedlen_circuits in tqdm(all_circuits):
            temp = []
            for angles in fixedlen_circuits:
                unitaries = self.su2rep.unitaries_from_angles(*angles.T)
                skip_next_noise = skip_first_noise
                if self._noise_channel_repr == 'unitary':
                    V = np.eye(self._unitary_dim, dtype=np.complex128)
                    for U in unitaries:
                        V = U @ V
                        V = V if skip_next_noise else self._noise_channel @ V
                        skip_next_noise = False
                    S = unitary_to_superoperator(V, 'std')
                else:
                    S = np.eye(self._superop_dim, dtype=np.complex128)
                    for U in unitaries:
                        T = unitary_to_superoperator(U, 'std')
                        S = T @ S
                        S = S if skip_next_noise else self._noise_channel @ S
                        skip_next_noise = False
                    pass
                temp.append(S)
            out.append(np.array(temp))
        out = np.array(out)
        self.nonspam_compositions = out
        return

    @staticmethod
    def synspam_transform(_probs, M=None, shots_per_circuit=np.inf, seed=0):
        #  probs[i,j,k,ell] = probability of measuring outcome ell when running the k-th lengths[j] circuit given preparation in state i.
        if M is None:
            M = get_M()

        num_statepreps, num_lengths, circuits_per_length, num_povm_effects = _probs.shape
        assert circuits_per_length > 1
        assert M.shape == (num_statepreps, num_povm_effects)
        
        g = np.random.default_rng(seed)

        # check each _probs[i,j,k,:] for negative values and normalize to a probability distribution
        sanitize_probs(_probs)

        diag_statepreps  = np.diag_indices(num_statepreps)
        diag_povmeffects = np.diag_indices(num_povm_effects)
        synthetic_probs  = np.zeros((num_povm_effects, num_lengths))
        survival_probs   = np.zeros((num_statepreps,   num_lengths))
        for j in range(num_lengths):
            P = np.zeros((num_statepreps, num_statepreps))
            # ^ This will be row-stochastic, like a
            #   state transition matrix for a Markov chain.
            for i in range(num_statepreps):
                P[i,:] = empirical_distribution(_probs[i,j,:,:], shots_per_circuit, g).mean(axis=0)
                """
                # ^ Compare to SU2CharacterRBSim.character_transform. This is equivalent to 
                #   that function's tensor contraction (see "tensordot") where "currchars" is
                #   an array of all ones.
                """
            Q = M @ P @ M.T
            survival_probs[ :, j] = P[diag_statepreps]
            synthetic_probs[:, j] = Q[diag_povmeffects]

        return synthetic_probs, survival_probs


"""
Character RB (although allowing for multiple state preps, and allowing full POVMs rather than 2-element POVMs)
"""


class SU2CharacterRBDesign(SU2RBDesign):

    def __init__(self, su2rep, N, lengths, statepreps, povm, seed):
        SU2RBDesign.__init__(self, su2rep, N, lengths, statepreps, povm, seed, __character__=True)
        self.angles2irrepchars = su2rep.angles2irrepchars
        self._angles2irrepcharcores = su2rep.angles2irrepcharcores
        self.unitaries_from_angles = su2rep.unitaries_from_angles
        self.irrep_sizes = su2rep.irrep_block_sizes
        self.num_irreps = self.irrep_sizes.size
        self.chars = np.zeros(shape=(self.lengths.size, self.N, self.num_irreps))
        # chars[j,k,ell] = the ell-th irrep's character for the unitary induced by the (noiseless version of the) k-th circuit of length lengths[j]
        self._charcores = np.zeros_like(self.chars)
        self._compute_chars()
        self._ssr1rb = False

    """Inherited properties: N, num_effects, num_lens, num_statepreps
    """

    @property
    def angles2irrepcharcores(self) -> Callable:
        if not hasattr(self, '_angles2irrepcharcores'):
            su2rep = Spin72 if self.irrep_sizes.size == 8 else SU2
            self._angles2irrepcharcores = su2rep.angles2irrepcharcores
        return self._angles2irrepcharcores

    @property
    def charcores(self) -> np.ndarray:
        if not hasattr(self, '_charcores'):
            self._charcores = np.zeros_like(self.chars)
            for j in range(self.num_lens):
                circuits = np.array(self.all_circuits[j])
                firstgate_angles = circuits[:,0,:]
                self._charcores[j,:,:] = self.angles2irrepcharcores(firstgate_angles)
        return self._charcores

    @property
    def ssr1rb(self) -> bool:
        if not hasattr(self, '_ssr1rb'):
            self._ssr1rb = False
        return self._ssr1rb

    def _compute_chars(self):
        # This is a separate function since it might take a while.
        for j in range(self.num_lens):
            circuits = np.array(self.all_circuits[j])
            firstgate_angles = circuits[:,0,:]
            self.chars[j,:,:] = self.angles2irrepchars(firstgate_angles)
            self.charcores[j,:,:] = self.angles2irrepcharcores(firstgate_angles)
        return


class SU2CharacterRBSim(SU2RBSim):

    def __init__(self, design : SU2CharacterRBDesign):
        assert isinstance(design, SU2CharacterRBDesign)
        SU2RBSim.__init__(self, design, __allow_characters__=True)
        return
    
    """Inherited properties: N, num_effects, num_lens, num_statepreps
    """

    @property
    def chars(self) -> np.ndarray:
        return self.design.chars

    @property
    def charcores(self) -> np.ndarray:
        return self.design.charcores
    
    @property
    def ssr1rb(self) -> bool:
        return self.design.ssr1rb

    def _process_circuits(self, fixedlen_circuits):
        if la.norm(self._noise_channel - np.eye(self._noise_channel.shape[0])) <= 1e-16:
            # We're simulating character RB on a noiseless system;
            # we can evaluate the circuit faster since we know in
            # advance that the composition of all rotations is equal
            # to the first rotation.
            probs = np.zeros(shape=(self.num_statepreps, self.N, self.num_effects))
            # probs[i,k,ell] = the probability of measuring outcome ell after running the k-th circuit, given i-th starting state.
            for k, angles in enumerate(fixedlen_circuits):
                a,b,g = angles.T
                char_U = self.design.unitaries_from_angles(a[0], b[0], g[0])[0]
                superop = unitary_to_superoperator(char_U, 'std')
                for i,superket in enumerate(self.design.statepreps):
                    superket = superket.copy()
                    superket = superop @ superket
                    probs[i,k,:] = self.design.povm @ superket
        else:
            probs = SU2RBSim._process_circuits(self, fixedlen_circuits, True)
        return probs

    @staticmethod
    def character_transform(_probs, _chars, irrep_sizes, shots_per_circuit=np.inf, seed=0):
        # Inputs:
        #   probs[i,j,k,ell] = probability of measuring outcome ell when running the k-th lengths[j] circuit given preparation in state i.
        #
        #   chars[  j,k,x] = the value of the x^th irrep's character function for the "hidden" initial gate in the k-th circuit of length lengths[j];
        #       ^ the same circuits and hidden initial gates were used for all statepreps.
        #
        # Output
        #   arr[i,ell,x,j] = empirical mean of x^th irrep's (scaled) character function times the indicator variable that we end in state ell, having started in state i and run a circuit of depth lengths[j].
        #                    (the mean being over the k=0,...,N-1 random circuits)
        #
        num_statepreps, num_lens, circuits_per_length, num_effects = _probs.shape
        assert circuits_per_length > 1
        num_irreps = irrep_sizes.size
        assert _chars.shape == (num_lens, circuits_per_length, num_irreps)

        sanitize_probs(_probs)
        g = np.random.default_rng(seed)
        probs = empirical_distribution(_probs, shots_per_circuit, g)
        wchars = _chars * irrep_sizes[np.newaxis, np.newaxis, :]

        permprobs = np.moveaxis(probs, (1,2,3), (2,3,1))
        # permprobs[a,b,c,d] = prob of measuring b given starting in a, for a circuit of length lengths[c], and in fact the d-th such circuit.
        permwchars = np.moveaxis(wchars, (0,1), (1,0))
        # permchars[x,y,z] = value of the z-th irreps character function for the hidden initial gate in the x-th circuit of length lengths[y]
        
        arr = np.zeros((num_statepreps, num_effects, num_irreps, num_lens))
        for j in range(num_lens):
            currprobs = permprobs[:,:,j,:]
            currchars = permwchars[:,j,:]
            P = np.tensordot(currprobs, currchars, axes=1) / circuits_per_length
            # P[u,v,w] = [w-th irrep character-weighted] transition probability to state v from state u
            arr[:,:,:,j] = P
        return arr

    @staticmethod
    def synspam_character_transform(_probs, _chars, _irrep_sizes, M=None, shots_per_circuit=np.inf, seed=0):
        # Inputs:
        #   probs[i,j,k,ell] = probability of measuring outcome ell when running the k-th lengths[j] circuit given preparation in state i.
        #
        #   chars[  j,k,x] = the value of the x^th irrep's character function for the "hidden" initial gate in the k-th circuit of length lengths[j];
        #       ^ the same circuits and hidden initial gates were used for all statepreps.
        #
        # Output
        #   synthetic_probs[ell,j] = character-weighted synthetic survival probability for circuits of length lengths[j] that start in state ell,
        #   where empirical distributions are computed with shots_per_circuit shots of each circuit.
        #   
        #   None (retained for compatibility reasons)
        #
        # Notes
        #   It's also valid to use
        #       chars[  j,k,x] = P_k(beta) where beta is the middle Euler angle of the "hidden"
        #                        initial gate in the k-th circuit of length lengths[j].
        #   In that case this is the transform for SSR1RB.
        #
        arr = SU2CharacterRBSim.character_transform(
            _probs, _chars, _irrep_sizes, shots_per_circuit, seed
        )
        num_statepreps, num_effects, num_irreps, num_lens = arr.shape
        if num_irreps != num_effects:
            raise NotImplementedError()
        if M is None:
            M = get_M()
        assert M.shape == (num_effects, num_statepreps)

        synthetic_probs  = np.zeros((num_effects, num_lens))
        for j in range(num_lens):
            P = arr[:,:,:,j]
            # P[u,v,w] = [w-th irrep character-weighted] transition probability
            #             to state v from state u, for circuits of length lengths[j]
            for k in range(num_irreps):
                block = P[:,:,k]
                row_M_k = M[k,:]
                synthetic_probs[k,j] = row_M_k @ block @ row_M_k

        return synthetic_probs, None


class Analysis:

    plot_colors = default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    @staticmethod
    def exp_decay_logvar(x,a,b):
        return a * np.exp(-b * x)
    
    @staticmethod
    def exp_decay(x,a,b):
        return a * b ** x

    @staticmethod
    def fit_exp(x, y, fitlog):
        assert x.size == y.size
        assert x.ndim == y.ndim == 1

        if fitlog:
            model = Analysis.exp_decay_logvar
            p0 = np.array([1, 0.05])
            bounds = ((-np.inf, -np.inf), (np.inf, np.inf))
        else:
            p0 = np.array([1, np.exp(-0.05)])
            model = Analysis.exp_decay
            bounds = ((-np.inf, 0), (np.inf, 1))

        from scipy.optimize import curve_fit
        p, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=6000)
        stddevs = np.sqrt(np.diag(pcov))
        if fitlog:
            stddevs[1] = np.exp(-p[1])*abs(2*np.sinh(stddevs[1]))
            p[1] = np.exp(-p[1])
        return p, stddevs
    
    @staticmethod
    def fit_exp_batch(x, ys, fitlog):
        num_series = ys.shape[0]
        ps = np.zeros((num_series, 2))
        sigmas = np.zeros((num_series, 2))
        for i,y in enumerate(ys):
            p, sigma = Analysis.fit_exp(x, y, fitlog)
            ps[i,:] = p
            sigmas[i,:] = sigma
        return ps, sigmas

    @staticmethod
    def fit_and_get_rates(x, ys, fitlog, F=None):
        assert np.all(x > 0)
        if F is None:
            F = get_F()
        num_series = ys.shape[0]
        assert F.shape == (num_series, num_series)
        ps, sigmas = Analysis.fit_exp_batch(x, ys, fitlog)
        f_mu = ps[:,1]
        rates = la.solve(F, f_mu)
        return rates, ps, sigmas

    @staticmethod
    def fit_and_plot(x, pkm, ax : plt_axes.Axes, fitlog):
        ps, sigmas = Analysis.fit_exp_batch(x, pkm, fitlog)
        f_mu = ps[:,1]
        f_sig = sigmas[:,1]
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        model = Analysis.exp_decay_logvar if fitlog else Analysis.exp_decay
        num_series = f_mu.size
        for i in range(num_series):
            y = pkm[i,:]
            textstr = f'{i}: f = {f_mu[i]:.3f}±({f_sig[i]:.3f})'
            ax.scatter(x, y, label=textstr, color=default_colors[i % len(default_colors)], s=20)
            ax.plot(x, model(x, ps[i,0], ps[i,1]), color=default_colors[i % len(default_colors)], linestyle='-')
        ax.legend()
        return ps, sigmas

    @staticmethod
    def fit_and_plot_with_rates_stderrors(x, ys, ax : plt_axes.Axes, fitlog, F=None):
        if F is None:
            F = get_F()
        rates, ps, ps_sigmas = Analysis.fit_and_get_rates(x, ys, fitlog)
        f_mu = ps[:,1]
        f_sig = ps_sigmas[:,1]
        invF = la.inv(F)  # doing this properly by factoring F isn't worth it.
        rates = invF @ f_mu
        rates_sig = np.sqrt(np.diag( invF @ np.diag(f_sig**2) @ invF.T ))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        model = Analysis.exp_decay_logvar if fitlog else Analysis.exp_decay
        for i,r in enumerate(rates):
            y = ys[i,:]
            if fitlog:
                textstr = f'{i}: f = {f_mu[i]:.3f}±({f_sig[i]:.3f}) | rate = {r:.3f}'
            else:
                textstr = f'{i}: f = {f_mu[i]:.3f} | rate = {r:.3f}±({rates_sig[i]:.4f})'
            ax.scatter(x, y, label=textstr, color=default_colors[i % len(default_colors)], s=20)
            ax.plot(x, model(x, ps[i,0], ps[i,1]), color=default_colors[i % len(default_colors)], linestyle='-')
        ax.legend()
        return rates, rates_sig, ps, ps_sigmas
    
    @staticmethod
    def fit_and_plot_with_rates(x, ys, ax : plt_axes.Axes, fitlog, F=None):
        rates, _, ps, sigmas = Analysis.fit_and_plot_with_rates_stderrors(x, ys, ax, fitlog, F)
        return rates, ps, sigmas

    @staticmethod
    def synspam_rb_fig(_rbm : Union[SU2RBSim, SU2CharacterRBSim], fitlog=False, figax=None, rate_stderrs=False):
        sps = np.inf
        if type(_rbm) == SU2RBSim:
            pkm, _ = SU2RBSim.synspam_transform(_rbm.probs, shots_per_circuit=sps)
            rbtype = 'Standard'
        elif isinstance(_rbm, SU2CharacterRBSim):
            if _rbm.ssr1rb:
                rbtype = 'Rank-1 Character'
                chararg = _rbm.charcores
            else:
                rbtype = 'Character'
                chararg = _rbm.chars
            pkm, _ = SU2CharacterRBSim.synspam_character_transform(_rbm.probs, chararg, Spin72.irrep_block_sizes, shots_per_circuit=sps)
        else:
            raise ValueError()
        if figax is None:
            fig, axs = plt.subplots(nrows=1, ncols=1)
            showfig = True
            fig.suptitle(f'{rbtype} RB with synSPAM, {sps} shot/circuit. Noise model {_rbm._noise_channel_info}')
        else:
            fig, axs = figax
            showfig = False
            names = _rbm.noise_channel_parameter_names
            axs.set_title(f'{names[0]} = {_rbm._noise_channel_info[1][0]}, {names[1]} = {_rbm._noise_channel_info[1][1]} ')
        if not rate_stderrs:
            out = Analysis.fit_and_plot_with_rates(_rbm.design.lengths + 1, pkm, axs, fitlog)
        else:
            assert not fitlog
            out = Analysis.fit_and_plot_with_rates_stderrors(_rbm.design.lengths + 1, pkm, axs, fitlog)
        if showfig:
            fig.show()
        return out
