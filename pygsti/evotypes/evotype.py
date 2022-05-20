import importlib as _importlib

from . import basereps as _basereps


class Evotype(object):
    """
    The base class for all other evotype classes.

    Provides an interface for creating representations.  The `create_*` methods specify an API used by the
    operation classes so they can create the representation they need.

    Parameters
    ----------
    name : str
        The (module) name of the evolution type

    prefer_dense_reps : bool, optional
        Whether the dense representation provided by this evolution type should be preferred
        over more specific types, such as those for composed, embedded, and exponentiated
        operations.  Most often this is set to `True` when using a :class:`MatrixForwardSimulator`
        in order to get a performance gain.
    """
    default_evotype = None

    _reptype_to_attrs = {
        'dense superop': 'OpRepDenseSuperop',
        'dense unitary': 'OpRepDenseUnitary',
        'composed': 'OpRepComposed',
        'embedded': 'OpRepEmbedded',
        'experrorgen': 'OpRepExpErrorgen',
        'stochastic': 'OpRepStochastic',
        'sum': 'OpRepSum',
        'clifford': 'OpRepClifford',
        'repeated': 'OpRepRepeated',
        'standard': 'OpRepStandard',
        'sparse superop': 'OpRepSparse',
        'lindblad errorgen': 'OpRepLindbladErrorgen',
        'dense state': 'StateRepDense',
        'pure state': 'StateRepDensePure',
        'computational state': 'StateRepComputational',
        'composed state': 'StateRepComposed',
        'tensorproduct state': 'StateRepTensorProduct',
        'conjugatedstate effect': 'EffectRepConjugatedState',
        'computational effect': 'EffectRepComputational',
        'tensorproduct effect': 'EffectRepTensorProduct',
        'composed effect': 'EffectRepComposed',
        'term': 'TermRep',
        'direct term': 'TermDirectRep'
    }

    @classmethod
    def cast(cls, obj, default_prefer_dense_reps=False):
        if isinstance(obj, Evotype):
            return obj
        elif obj == "default":
            return Evotype(cls.default_evotype, default_prefer_dense_reps)
        else:  # assume obj is a string naming an evotype
            return Evotype(str(obj), default_prefer_dense_reps)

    def __init__(self, name, prefer_dense_reps=False):
        self.name = name
        self.module = _importlib.import_module("pygsti.evotypes." + self.name)
        self.prefer_dense_reps = prefer_dense_reps

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['module']  # can't pickle module
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.module = _importlib.import_module("pygsti.evotypes." + self.name)

    def __hash__(self):
        return hash((self.name, self.prefer_dense_reps))

    def __eq__(self, other_evotype):
        if isinstance(other_evotype, Evotype):
            return self.name == other_evotype.name
        elif other_evotype == "default":
            return self.name == self.default_evotype
        else:
            return self.name == str(other_evotype)

    def __str__(self):
        return self.name

    @property
    def minimal_space(self):
        return self.module.minimal_space

    def minimal_dim(self, state_space):
        return state_space.udim if self.minimal_space == 'Hilbert' else state_space.dim

    def supported_reptypes(self):
        return [reptype for reptype, attr in self._reptype_to_attrs.items() if hasattr(self.module, attr)]

    def supports(self, reptype):
        return hasattr(self.module, self._reptype_to_attrs[reptype])

    def create_dense_superop_rep(self, mx, super_basis, state_space):  # process_mx=None,
        return self.module.OpRepDenseSuperop(mx, super_basis, state_space)

    def create_dense_unitary_rep(self, mx, super_basis, state_space):  # process_mx=None,
        return self.module.OpRepDenseUnitary(mx, super_basis, state_space)

    def create_composed_rep(self, factor_op_reps, state_space):
        return self.module.OpRepComposed(factor_op_reps, state_space)

    def create_embedded_rep(self, state_space, targetLabels, embedded_rep):
        return self.module.OpRepEmbedded(state_space, targetLabels, embedded_rep)

    def create_experrorgen_rep(self, errorgen_rep):
        return self.module.OpRepExpErrorgen(errorgen_rep)

    def create_identitypluserrorgen_rep(self, errorgen_rep):
        return self.module.OpRepIdentityPlusErrorgen(errorgen_rep)

    def create_stochastic_rep(self, stochastic_basis, basis, initial_rates, seed_or_state, state_space):
        return self.module.OpRepStochastic(stochastic_basis, basis, initial_rates, seed_or_state, state_space)

    def create_kraus_rep(self, basis, kraus_reps, state_space):
        return self.module.OpRepKraus(basis, kraus_reps, state_space)

    def create_sum_rep(self, factor_reps, state_space):
        return self.module.OpRepSum(factor_reps, state_space)

    def create_clifford_rep(self, unitarymx, symplecticrep, super_basis, state_space):
        return self.module.OpRepClifford(unitarymx, symplecticrep, super_basis, state_space)

    def create_repeated_rep(self, rep_to_repeat, num_repetitions, state_space):
        return self.module.OpRepRepeated(rep_to_repeat, num_repetitions, state_space)

    def create_standard_rep(self, standard_name, super_basis, state_space):
        return self.module.OpRepStandard(standard_name, super_basis, state_space)

    def create_sparse_rep(self, data, indices, indptr, state_space):
        return self.module.OpRepSparse(data, indices, indptr, state_space)

    def create_lindblad_errorgen_rep(self, lindblad_coefficient_blocks, state_space):
        return self.module.OpRepLindbladErrorgen(lindblad_coefficient_blocks, state_space)

    # STATE REPS
    def create_dense_state_rep(self, vec, super_basis, state_space):
        return self.module.StateRepDense(vec, state_space, super_basis)

    def create_pure_state_rep(self, purevec, super_basis, state_space):
        return self.module.StateRepDensePure(purevec, state_space, super_basis)

    def create_computational_state_rep(self, zvals, super_basis, state_space):
        return self.module.StateRepComputational(zvals, super_basis, state_space)

    def create_composed_state_rep(self, state_rep, op_rep, state_space):
        return self.module.StateRepComposed(state_rep, op_rep, state_space)

    def create_tensorproduct_state_rep(self, factor_state_reps, state_space):
        return self.module.StateRepTensorProduct(factor_state_reps, state_space)

    # EFFECT REPS
    def create_conjugatedstate_effect_rep(self, state_rep):
        return self.module.EffectRepConjugatedState(state_rep)

    def create_computational_effect_rep(self, zvals, super_basis, state_space):
        return self.module.EffectRepComputational(zvals, super_basis, state_space)

    def create_tensorproduct_effect_rep(self, povm_factors, effect_labels, state_space):
        return self.module.EffectRepTensorProduct(povm_factors, effect_labels, state_space)

    def create_composed_effect_rep(self, errmap_rep, effect_rep, errmap_name, state_space):
        return self.module.EffectRepComposed(errmap_rep, effect_rep, errmap_name, state_space)

    #POVM REPS
    def create_composed_povm_rep(self, errmap_rep, base_povm_rep, state_space):
        return self.module.ComposedPOVMRep(errmap_rep, base_povm_rep, state_space)

    def create_computational_povm_rep(self, nqubits, qubit_filter):
        return self.module.ComputationalPOVMRep(nqubits, qubit_filter)

    # TERM REPS
    def create_term_rep(self, coeff, mag, logmag, pre_state, post_state,
                        pre_effect, post_effect, pre_ops, post_ops):
        try:  # see if module implements its own term rep, otherwise use "stock" version
            return self.module.TermRep(coeff, mag, logmag, pre_state, post_state,
                                       pre_effect, post_effect, pre_ops, post_ops)
        except Exception:
            return _basereps.StockTermRep(coeff, mag, logmag, pre_state, post_state,
                                          pre_effect, post_effect, pre_ops, post_ops)

    def create_direct_term_rep(self, coeff, mag, logmag, pre_state, post_state,
                               pre_effect, post_effect, pre_ops, post_ops):
        try:  # see if module implements its own term rep, otherwise use "stock" version
            return self.module.TermDirectRep(coeff, mag, logmag, pre_state, post_state,
                                             pre_effect, post_effect, pre_ops, post_ops)
        except Exception:
            return _basereps.StockTermDirectRep(coeff, mag, logmag, pre_state, post_state,
                                                pre_effect, post_effect, pre_ops, post_ops)

    def conjugate_state_term_rep(self, term_rep):
        """ Turns a state term => effect term via conjugation of the state """
        coeff = term_rep.coeff
        mag = term_rep.magnitude
        logmag = term_rep.logmagnitude
        pre_effect = self.create_conjugatedstate_effect_rep(term_rep.pre_state)
        post_effect = self.create_conjugatedstate_effect_rep(term_rep.post_state)

        try:  # see if module implements its own term rep, otherwise use "stock" version
            return self.module.TermRep(coeff, mag, logmag, None, None,
                                       pre_effect, post_effect, term_rep.pre_ops, term_rep.post_ops)
        except Exception:
            return _basereps.StockTermRep(coeff, mag, logmag, None, None,
                                          pre_effect, post_effect, term_rep.pre_ops, term_rep.post_ops)


try:
    from . import densitymx as _dummy
    Evotype.default_evotype = "densitymx"
except ImportError:
    Evotype.default_evotype = "densitymx_slow"
