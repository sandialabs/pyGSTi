import importlib as _importlib


class Evotype(object):
    """
    The base class for all other evotype classes.

    Provides an interface for creating representations.  The `create_*` methods specify an API used by the
    operation classes so they can create the representation they need.
    """

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, Evotype):
            return obj
        else:  # assume obj is a string naming an evotype
            return Evotype(str(obj))

    def __init__(self, name):
        self.name = name
        self.module = _importlib.import_module("pygsti.evotypes." + name)
        self.term_evotype = None  # maybe parse this out of name? TODO , e.g. 'terms:statevec'? OR 'pathintegral:statevec'

    def __eq__(self, other_evotype):
        if isinstance(other_evotype, Evotype):
            return self.name == other_evotype.name
        else:
            return self.name == str(other_evotype)

    def create_dense_rep(self, dim=None):  # process_mx=None, 
        return self.module.OpRepDense(dim)
        #ret.base[:,:] = process_mx  # HACK -----------------------------------------------------------------------------------------
        #return ret

    def create_composed_rep(self, factor_op_reps, dim):
        return self.module.OpRepComposed(factor_op_reps, dim)

    def create_embedded_rep(self, state_space_labels, targetLabels, embedded_rep):
        return self.module.OpRepEmbedded(state_space_labels, targetLabels, embedded_rep)

    def create_experrorgen_rep(self, errorgen_rep):
        return self.module.OpRepExpErrorgen(errorgen_rep)

    def create_stochastic_rep(self, basis, dim):
        return self.module.OpRepStochastic(basis, dim)

    def create_sum_rep(self, factor_reps, dim):
        return self.module.OpRepSum(factor_reps, dim)

    def create_clifford_rep(self, unitarymx, symplecticrep):
        return self.module.OpRepClifford(unitarymx, symplecticrep)

    def create_repeated_rep(self, rep_to_repeat, num_repetitions, dim):
        return self.module.OpRepRepeated(rep_to_repeat, num_repetitions, dim)

    def create_standard_rep(self, standard_name):
        return self.module.OpRepStandard(standard_name)

    def create_sparse_rep(self, data, indices, indptr):
        return self.module.OpRepSparse(data, indices, indptr)

    def create_lindblad_errorgen_rep(self, lindblad_term_dict, basis):
        return self.module.OpRepLindbladErrorgen(lindblad_term_dict, basis)



    def create_dense_state_rep(self, vec):
        return self.module.StateRepDense(vec)

    def create_pure_state_rep(self, purevec, basis):
        return self.module.StateRepPure(purevec, basis)

    def create_computational_state_rep(self, zvals):
        return self.module.StateRepComputational(zvals)

    def create_composed_state_rep(self, state_rep, op_rep):
        return self.module.StateRepComposed(state_rep, op_rep)

    def create_tensorproduct_state_rep(self, factor_state_reps):
        return self.module.StateRepTensorProduct(factor_state_reps)



    def create_conjugatedstate_effect_rep(self, state_rep):
        return self.module.EffectRepConjugatedState(state_rep)

    def create_computational_effect_rep(self, zvals):
        return self.module.EffectRepComputational(zvals)

    def create_tensorproduct_effect_rep(self, povm_factors, effect_labels):
        return self.module.EffectRepTensorProduct(povm_factors, effect_labels)

    def create_composed_effect_rep(self, errmap_rep, effect_rep, errmap_name):
        return self.module.EffectRepComposed(errmap_rep, effect_rep, errmap_name)
    
