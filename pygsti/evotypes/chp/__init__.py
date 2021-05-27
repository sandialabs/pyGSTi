


class OpRepComposed(XXX):

    def __init__(self, factor_op_reps, dim):
        pass

    def reinit_factor_op_reps(factor_op_reps):
        pass  # TODO

    def get_chp_str(self, targets=None):
        """Return a string suitable for printing to a CHP input file from all underlying operations.

        Parameters
        ----------
        targets: list of int
            Qubits to be applied to (if None, uses stored CHP strings directly)

        Returns
        -------
        s : str
            String of CHP code
        """
        s = ""
        for op in self.factorops:
            s += op.get_chp_str(targets)
        return s


class OpRepEmbedded(XXX):

    def __init__(self, state_space_labels, target_labels, embedded_rep):
        # assert that all state space labels == qubits, since we only know
        # how to embed cliffords on qubits...
        assert(len(state_space_labels.labels) == 1
               and all([ld == 2 for ld in state_space_labels.labeldims.values()])), \
            "All state space labels must correspond to *qubits*"

        #TODO: enfore this another way? 
        #assert(self.embedded_op._evotype == 'chp'), \
        #    "Embedded op must also have CHP evotype instead of %s" % self.embedded_op._evotype
        assert(isinstance(embedded_rep, OpRepBase))  # needs to point to chp.OpRep class??
        
        op_nqubits = (embedded_rep.dim - 1).bit_length()
        assert(len(target_labels) == op_nqubits), \
            "Inconsistent number of qubits in `target_labels` ({0}) and CHP `embedded_op` ({1})".format(
                len(target_labels), op_nqubits)

        qubitLabels = state_space_labels.labels[0]
        qubit_indices = _np.array([qubitLabels.index(targetLbl)
                                   for targetLbl in target_labels], _np.int64)

        nQubits = state_space_labels.nqubits
        assert(nQubits is not None), "State space does not contain a definite number of qubits!"

        # Store qubit indices as targets for later use
        self.target_indices = qubit_indices

        #TODO - figure out what this means - I think there wasn't a CHP embedded rep class before?
        rep = opDim  # Don't set representation again, just use embedded_op calls later


    def get_chp_str(self, targets=None):  # => chpstr property? TODO
        """Return a string suitable for printing to a CHP input file from the embedded operations.

        Just calls underlying get_chp_str but with an extra layer of target redirection.

        Parameters
        ----------
        targets: list of int
            Qubits to be applied to (if None, uses stored CHP strings directly).

        Returns
        -------
        s : str
            String of CHP code
        """
        target_indices = list(self.target_indices)

        # Targets are for the full embedded operation so we need to map these to the actual targets of the CHP operation
        if targets is not None:
            assert len(targets) == self.state_space_labels.nqubits, \
                "Got {0} targets instead of required {1}".format(len(targets), self.state_space_labels.nqubits)
            target_indices = [targets[ti] for ti in self.target_indices]

        return self.embedded_op.get_chp_str(target_indices)



class OpRepStochastic(XXX):

    def __init__(self, basis, rate_poly_dicts, initial_rates, seed_or_state):

        self.basis = basis
        assert (basis.name == 'pp'), "Only Pauli basis is allowed for 'chp' evotype"

        if isinstance(seed_or_state, _RandomState):
            self.rand_state = seed_or_state
        else:
            self.rand_state = _RandomState(seed_or_state)

        #TODO: need to fix this: `basis` above functions as basis to make superoperators out of, but here we have
        # a CHP stochastic op which is given a basis for the space - e.g. a dim=2 vector space for 1 qubit, so
        # we need to distinguish/specify the basis better for this... and what about rate_poly_dicts (see svterm)
        nqubits = (self.basis.dim - 1).bit_length()
        assert(self.basis.dim == 4**nqubits), "Must have an integral number of qubits"

        std_chp_ops = _itgs.standard_gatenames_chp_conversions()

        # For CHP, need to make a Composed + EmbeddedOp for the super operators
        # For lower overhead, make this directly using the rep instead of with objects
        self.stochastic_superops = []
        for label in self.basis.labels[1:]:
            combined_chp_ops = []

            for i, pauli in enumerate(label):
                name = 'Gi' if pauli == "I" else 'G%spi' % pauli.lower()
                chp_op = std_chp_ops[name]
                chp_op_targeted = [op.replace('0', str(i)) for op in chp_op]
                combined_chp_ops.extend(chp_op_targeted)

            rep = replib.CHPOpRep(combined_chp_ops, nqubits)
            self.stochastic_superops.append(LinearOperator(rep, 'chp'))
        self.rates = initial_rates

    def update_rates(self, rates):
        self.rates[:] = rates

    def get_chp_str(self, targets=None):
        """Return a string suitable for printing to a CHP input file after stochastically selecting operation.

        Parameters
        ----------
        targets: list of int
            Qubits to be applied to (if None, uses stored CHP strings directly)

        Returns
        -------
        s : str
            String of CHP code
        """
        assert (self._evotype == 'chp'), "Must have 'chp' evotype to use get_chp_str"

        rates = self.rates
        all_rates = [*rates, 1.0 - sum(rates)]  # Include identity so that probabilities are 1
        index = self.rand_state.choice(self.basis.size, p=all_rates)

        # If first entry, no operation selected
        if index == self.basis.size - 1:
            return ''

        op = self.stochastic_superops[index]
        chp_ops = op._rep.chp_ops
        nqubits = op._rep.nqubits

        if targets is not None:
            assert len(targets) == nqubits, "Got {0} targets instead of required {1}".format(len(targets), nqubits)
            target_map = {str(i): str(t) for i, t in enumerate(targets)}

        s = ""
        for op in chp_ops:
            # Substitute if alternate targets provided
            if targets is not None:
                op_str = ''.join([target_map[c] if c in target_map else c for c in op])
            else:
                op_str = op

            s += op_str + '\n'

        return s


class OpRepStandard(XXX):
    def __init__(self, name):
        self.name = name
        std_chp_ops = _itgs.standard_gatenames_chp_conversions()
        if self.name not in std_chp_ops:
            raise ValueError("Name '%s' not in standard CHP operations" % self.name)

        native_ops = std_chp_ops[self.name]
        nqubits = 2 if any(['c' in n for n in native_ops]) else 1

        rep = replib.CHPOpRep(native_ops, nqubits)
