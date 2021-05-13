

#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT

cdef extern from "fastreps.h" namespace "CReps":
    cdef cppclass EffectCRep:
        EffectCRep() except +
        EffectCRep(INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim

    cdef cppclass EffectCRep_Dense(EffectCRep):
        EffectCRep_Dense() except +
        EffectCRep_Dense(double*,INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        double* _dataptr

    cdef cppclass EffectCRep_TensorProd(EffectCRep):
        EffectCRep_TensorProd() except +
        EffectCRep_TensorProd(double*, INT*, INT, INT, INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        INT _nfactors
        INT _max_factor_dim

    cdef cppclass EffectCRep_Computational(EffectCRep):
        EffectCRep_Computational() except +
        EffectCRep_Computational(INT, INT, double, INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim

    cdef cppclass EffectCRep_Errgen(EffectCRep):
        EffectCRep_Errgen() except +
        EffectCRep_Errgen(OpCRep*, EffectCRep*, INT, INT) except +
        double probability(StateCRep* state)
        double probability_using_cache(StateCRep* state, StateCRep* precomp_state, INT& precomp_id)
        INT _dim
        INT _errgen_id



class EffectRep(_basereps_cython.EffectRep):
    cdef EffectCRep* c_effect

    def __cinit__(self):
        self.c_effect = NULL

    def __dealloc__(self):
        if self.c_effect != NULL:
            del self.c_effect

    def __reduce__(self):
        return (EffectRep, ())

    @property
    def dim(self):
        return self.c_effect._dim

    def probability(self, StateRep state not None):
        #unnecessary (just put in signature): cdef StateRep st = <StateRep?>state
        return self.c_effect.probability(state.c_state)


cdef class EffectRepConjugatedState(EffectRep):
    cdef public StateRep state_rep

    def __cinit__(self, StateRep state_rep):
        self.state_rep = state_rep
        self.c_effect = new EffectCRep_Dense(<double*>self.state_rep.base.data,
                                               <INT>self.state_rep.base.shape[0])

    def __str__(self):
        return str([ (<EffectCRep_Dense*>self.c_effect)._dataptr[i] for i in range(self.c_effect._dim)])

    def __reduce__(self):
        return (EffectRepDense, (self.state_rep,))


cdef class EffectRepComputational(EffectRep):
    cdef public np.ndarray zvals

    def __cinit__(self, np.ndarray[np.int64_t, ndim=1, mode='c'] zvals, INT dim):
        # cdef INT dim = 4**zvals.shape[0] -- just send as argument
        cdef INT nfactors = zvals.shape[0]
        cdef double abs_elval = 1/(np.sqrt(2)**nfactors)
        cdef INT base = 1
        cdef INT zvals_int = 0
        for i in range(nfactors):
            zvals_int += base * zvals[i]
            base = base << 1 # *= 2
        self.zvals = zvals
        self.c_effect = new EffectCRep_Computational(nfactors, zvals_int, abs_elval, dim)

    def __reduce__(self):
        return (EffectRepComputational, (self.zvals, self.c_effect._dim))

    #Add party & to_dense from slow version?


cdef class EffectRepTensorProduct(EffectRep):
    cdef public object povm_factors
    cdef public object effect_labels
    cdef public np.ndarray kron_array
    cdef public np.ndarray factor_dims

    def __init__(self, povm_factors, effect_labels):
        #Arrays for speeding up kron product in effect reps
        cdef INT max_factor_dim = max(fct.dim for fct in povm_factors)
        cdef np.ndarray[double, ndim=2, mode='c'] kron_array = \
            _np.ascontiguousarray(_np.empty((len(povm_factors), max_factor_dim), 'd'))
        cdef np.ndarray[np.int64_t, ndim=1, mode='c'] factordims = \
            _np.ascontiguousarray(_np.array([fct.dim for fct in povm_factors], _np.int64))

        cdef INT dim = _np.product(factordims)
        cdef INT nfactors = len(self.povm_factors)
        self.povm_factors = povm_factors
        self.effect_labels = effect_labels
        self.kron_array = kron_array
        self.factor_dims = factordims
        self.c_effect = new EffectCRep_TensorProd(<double*>kron_array.data,
                                                  <INT*>factor_dims.data,
                                                  nfactors, max_factor_dim, dim)
        self.factor_effects_have_changed()  # computes self.kron_array

    def __reduce__(self):
        return (EffectRepTensorProduct, (self.povm_factors, self.effect_labels))

    def _fill_fast_kron(self):
        """ Fills in self._fast_kron_array based on current self.factors """
        for i, (factor_dim, Elbl) in enumerate(zip(self.factor_dims, self.effect_labels)):
                self.kron_array[i][0:factor_dim] = self.povm_factors[i][Elbl].to_dense()

    def factor_effects_have_changed(self):
        self._fill_fast_kron()  # updates effect reps

    #TODO: Take to_dense from slow version?


cdef class EffectRepCompsed(EffectRep):
    cdef public OpRep op_rep
    cdef public EffectRep effect_rep
    cdef public object op_id

    def __cinit__(self, OpRep op_rep not None, EffectRep effect_rep not None, op_id):
        cdef INT dim = effect_rep.c_effect._dim
        self.op_id = op_id
        self.op_rep = op_rep
        self.effect_rep = effect_rep
        self.c_effect = new EffectCRep_Composed(op_rep.c_op,
                                                effect_rep.c_effect,
                                                <INT>op_id, dim)

    def __reduce__(self):
        return (EffectRepComposed, (self.op_rep, self.effect_rep, self.op_id))
