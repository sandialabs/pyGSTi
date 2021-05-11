


class EvoType(object):
    """
    The base class for all other evotype classes.

    Provides an interface for creating representations.  The `create_*` methods specify an API used by the
    operation classes so they can create the representation they need.
    """

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, EvoType):
            return obj
        else:  # assume obj is a string naming an evotype
            return Evotype(str(obj))

    def __init__(self, name):
        self.name = name
        self.module = XXX

    def create_dense_rep(self, process_mx):
        return self.module.OpRepDense(process_mx) #?? or just have evotype.DenseRep ??

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

    def create_XXX(self, XXX):
        pass


    
