

class OpRepBase(XXX):
    # These were taken from LinearOperator



class OpRepComposed(XXX):

    def __init__(self, factor_op_reps, dim):
        pass

    def reinit_factor_op_reps(factor_op_reps):
        pass  # TODO



class OpRepEmbedded(XXX):

    def __init__(self, state_space_labels, target_labels, embedded_rep):
        pass
    


class OpRepExpErrorgen(XXX):

    def __init__(self, errorgen_rep):

        #TODO: make terms init-able from sparse elements, and below code work with a *sparse* unitary_postfactor
        #termtype = "dense" if evotype == "svterm" else "clifford"  # TODO - split to cterm case
        
        #TODO REMOVE
        # Store *unitary* as self.unitary_postfactor - NOT a superop
        #if unitary_postfactor is not None:  # can be None
        #    op_std = _bt.change_basis(unitary_postfactor, self.errorgen.matrix_basis, 'std')
        #    self.unitary_postfactor = _gt.process_mx_to_unitary(op_std)
        #
        #    # automatically "up-convert" operation to CliffordOp if needed
        #    if termtype == "clifford":
        #        self.unitary_postfactor = CliffordOp(self.unitary_postfactor)
        #else:
        #    self.unitary_postfactor = None
        pass


class OpRepStochastic(XXX):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state):
        pass

    def update_rates(self, rates):
        pass


class OpRepSum(XXX):
    def __init__(self, factor_reps, dim):
        pass

    def reinit_factor_reps(self, factor_reps):
        pass


class OpRepStandard(XXX):
    def __init__(self, name):
        std_unitaries = _itgs.standard_gatename_unitaries()
        if self.name not in std_unitaries:
            raise ValueError("Name '%s' not in standard unitaries" % self.name)

        U = std_unitaries[self.name]
        # do we need this?


class OpRepLindbladErrorgen(XXX):

    def __init__(self, lindblad_term_dict, basis):
        assert(not self.sparse), "Sparse bases are not supported for term-based evolution"
        self.LtermdictAndBasis = (lindblad_term_dict, basis)  # HACK
        self.Lterms, self.Lterm_coeffs = None, None
        # # OLD: do this lazily now that we need max_polynomial_vars...
        # self._init_terms(lindblad_term_dict, basis, evotype, dim, max_polynomial_vars)
