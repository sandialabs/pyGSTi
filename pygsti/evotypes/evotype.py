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
    defaut_evotype = None

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
        #REMOVE - and get rid of module_name variable
        #if ':' in name:
        #    i = name.index(':')
        #    module_name, sub_evotype_name = name[0:i], name[i + 1:]  # e.g. 'pathintegral:statevec'
        #else:
        #    module_name, sub_evotype_name = name, None
        module_name = name

        self.module = _importlib.import_module("pygsti.evotypes." + module_name)
        self.prefer_dense_reps = prefer_dense_reps
        #REMOVE self.sub_evotype = Evotype(sub_evotype_name) if (sub_evotype_name is not None) else None

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
        elif other_evotype == "default":
            return self.name == self.default_evotype
        else:
            return self.name == str(other_evotype)

    def __str__(self):
        return self.name

    def create_dense_superop_rep(self, mx, state_space):  # process_mx=None,
        return self.module.OpRepDenseSuperop(mx, state_space)

    def create_dense_unitary_rep(self, mx, super_basis, state_space):  # process_mx=None,
        return self.module.OpRepDenseUnitary(mx, super_basis, state_space)

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

    def create_clifford_rep(self, unitarymx, symplecticrep, super_basis, state_space):
        return self.module.OpRepClifford(unitarymx, symplecticrep, super_basis, state_space)

    def create_repeated_rep(self, rep_to_repeat, num_repetitions, state_space):
        return self.module.OpRepRepeated(rep_to_repeat, num_repetitions, state_space)

    def create_standard_rep(self, standard_name, super_basis, state_space):
        return self.module.OpRepStandard(standard_name, super_basis, state_space)

    def create_sparse_rep(self, data, indices, indptr, state_space):
        return self.module.OpRepSparse(data, indices, indptr, state_space)

    def create_lindblad_errorgen_rep(self, lindblad_term_dict, basis, state_space):
        return self.module.OpRepLindbladErrorgen(lindblad_term_dict, basis, state_space)

    # STATE REPS
    def create_dense_state_rep(self, vec, state_space):
        return self.module.StateRepDense(vec, state_space)

    def create_pure_state_rep(self, purevec, super_basis, state_space):
        return self.module.StateRepPure(purevec, super_basis, state_space)

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


try:
    from . import densitymx as _dummy
    Evotype.default_evotype = "densitymx"
except ImportError:
    Evotype.default_evotype = "densitymx_slow"
