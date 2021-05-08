class StaticStandardOp(DenseOperator):
    """
    An operation that is completely fixed, or "static" (i.e. that posesses no parameters)
    that can be constructed from "standard" gate names (as defined in pygsti.tools.internalgates).

    Parameters
    ----------
    name : str
        Standard gate name

    evotype : {"statevec", "densitymx", "svterm", "cterm"}
        The evolution type.
        - "statevec": Unitary from standard_gatename_unitaries is used directly
        - "densitymx", "svterm", "cterm": Pauli transfer matrix is built from standard_gatename_unitaries
          (i.e. basis = 'pp')
        - "chp": CHP compilation taken from standard_gatenames_chp_conversions
    """
    def __init__(self, name, evotype):
        self.name = name

        if evotype in ('statevec', 'densitymx', 'svterm', 'cterm'):
            std_unitaries = _itgs.standard_gatename_unitaries()
            if self.name not in std_unitaries:
                raise ValueError("Name '%s' not in standard unitaries" % self.name)

            U = std_unitaries[self.name]

            if evotype == 'statevec':
                rep = replib.SVOpRepDense(LinearOperator.convert_to_matrix(U))
            else:  # evotype in ('densitymx', 'svterm', 'cterm')
                ptm = _gt.unitary_to_pauligate(U)
                rep = replib.DMOpRepDense(LinearOperator.convert_to_matrix(ptm))
        elif evotype == 'chp':
            std_chp_ops = _itgs.standard_gatenames_chp_conversions()
            if self.name not in std_chp_ops:
                raise ValueError("Name '%s' not in standard CHP operations" % self.name)

            native_ops = std_chp_ops[self.name]
            nqubits = 2 if any(['c' in n for n in native_ops]) else 1

            rep = replib.CHPOpRep(native_ops, nqubits)
        else:
            raise ValueError("Invalid evotype for a StaticStandardOp: %s" % evotype)

        LinearOperator.__init__(self, rep, evotype)

    # TODO: This should not be necessary to define, but is here as a temporary measure
    # This will likely be removed as "dense" is reworked in the evotype refactor
    @property
    def base(self):
        """
        The underlying dense process matrix.
        """
        if self._evotype in ['statevec', 'densitymx', 'svterm', 'cterm']:
            return self._rep.base
        else:
            raise NotImplementedError('No base available for evotype "%s"' % self._evotype)

    def __str__(self):
        s = "%s with name %s and evotype %s\n" % (self.__class__.__name__, self.name, self._evotype)
        if self._evotype in ['statevec', 'densitymx', 'svterm', 'cterm']:
            s += _mt.mx_to_string(self.base, width=4, prec=2)
        elif self._evotype == 'chp':
            s += 'CHP operations: ' + ','.join(self._rep.chp_ops) + '\n'
        return s
