import importlib as _importlib


class Evotype(object):
    """
    The base class for all other evotype classes.

    Provides an interface for creating representations.  The `create_*` methods specify an API used by the
    operation classes so they can create the representation they need.
    """
    defaut_evotype = None

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, Evotype):
            return obj
        elif obj == "default":
            return Evotype(cls.default_evotype)
        else:  # assume obj is a string naming an evotype
            return Evotype(str(obj))

    def __init__(self, name):
        if ':' in name:
            self.name, self.term_evotype = name.split(':')  # e.g. 'pathintegral:statevec'
        else:
            self.name, self.term_evotype = name, None
        self.module = _importlib.import_module("pygsti.evotypes." + name)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['module']  # can't pickle module
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.module = _importlib.import_module("pygsti.evotypes." + self.name)

    def __eq__(self, other_evotype):
        if isinstance(other_evotype, Evotype):
            return self.name == other_evotype.name
        else:
            return self.name == str(other_evotype)

    def __str__(self):
        return self.name

    def create_dense_rep(self, mx, state_space):  # process_mx=None,
        return self.module.OpRepDense(mx, state_space)

    def create_denseunitary_rep(self, mx, state_space):  # process_mx=None,
        return self.module.OpRepDenseUnitary(mx, state_space)

    def create_composed_rep(self, factor_op_reps, state_space):
        return self.module.OpRepComposed(factor_op_reps, state_space)

    def create_embedded_rep(self, state_space, targetLabels, embedded_rep):
        return self.module.OpRepEmbedded(state_space, targetLabels, embedded_rep)

    def create_experrorgen_rep(self, errorgen_rep):
        return self.module.OpRepExpErrorgen(errorgen_rep)

    def create_stochastic_rep(self, basis, rate_poly_dicts, initial_rates, seed_or_state, state_space):
        return self.module.OpRepStochastic(basis, rate_poly_dicts, initial_rates, seed_or_state, state_space)

    def create_sum_rep(self, factor_reps, state_space):
        return self.module.OpRepSum(factor_reps, state_space)

    def create_clifford_rep(self, unitarymx, symplecticrep, state_space):
        return self.module.OpRepClifford(unitarymx, symplecticrep, state_space)

    def create_repeated_rep(self, rep_to_repeat, num_repetitions, state_space):
        return self.module.OpRepRepeated(rep_to_repeat, num_repetitions, state_space)

    def create_standard_rep(self, standard_name, state_space):
        return self.module.OpRepStandard(standard_name, state_space)

    def create_sparse_rep(self, data, indices, indptr, state_space):
        return self.module.OpRepSparse(data, indices, indptr, state_space)

    def create_lindblad_errorgen_rep(self, lindblad_term_dict, basis, state_space):
        return self.module.OpRepLindbladErrorgen(lindblad_term_dict, basis, state_space)



    def create_dense_state_rep(self, vec, state_space):
        return self.module.StateRepDense(vec, state_space)

    def create_pure_state_rep(self, purevec, basis, state_space):
        return self.module.StateRepPure(purevec, basis, state_space)

    def create_computational_state_rep(self, zvals, state_space):
        return self.module.StateRepComputational(zvals, state_space)

    def create_composed_state_rep(self, state_rep, op_rep, state_space):
        return self.module.StateRepComposed(state_rep, op_rep, state_space)

    def create_tensorproduct_state_rep(self, factor_state_reps, state_space):
        return self.module.StateRepTensorProduct(factor_state_reps, state_space)



    def create_conjugatedstate_effect_rep(self, state_rep):
        return self.module.EffectRepConjugatedState(state_rep)

    def create_computational_effect_rep(self, zvals, state_space):
        return self.module.EffectRepComputational(zvals, state_space)

    def create_tensorproduct_effect_rep(self, povm_factors, effect_labels, state_space):
        return self.module.EffectRepTensorProduct(povm_factors, effect_labels, state_space)

    def create_composed_effect_rep(self, errmap_rep, effect_rep, errmap_name, state_space):
        return self.module.EffectRepComposed(errmap_rep, effect_rep, errmap_name, state_space)
    

try:
    from . import densitymx as _dummy
    Evotype.default_evotype = "densitymx"
except ImportError:
    Evotype.default_evotype = "densitymx_slow"
