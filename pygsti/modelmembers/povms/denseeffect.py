

class DensePOVMEffect(_POVMEffect):
    """
    A POVM effect vector that behaves like a numpy array.

    This class is the common base class for parameterizations of an effect vector
    that have a dense representation and can be accessed like a numpy array.

    Parameters
    ----------
    vec : numpy.ndarray
        The SPAM vector as a dense numpy array.

    evotype : {"statevec", "densitymx"}
        The evolution type.

    Attributes
    ----------
    _base_1d : numpy.ndarray
        Direct access to the underlying 1D array.

    base : numpy.ndarray
        Direct access the the underlying data as column vector,
        i.e, a (dim,1)-shaped array.
    """

    def __init__(self, vec, evotype):
        #dtype = complex if evotype == "statevec" else 'd'
        vec = _np.asarray(vec, dtype='d')
        vec.shape = (vec.size,)  # just store 1D array flatten
        vec = _np.require(vec, requirements=['OWNDATA', 'C_CONTIGUOUS'])
        evotype = _Evotype.cast(evotype)
        rep = evotype.create_dense_effect_rep(vec)
        super(DensePOVMEffect, self).__init__(rep, evotype)
        assert(self._base_1d.flags['C_CONTIGUOUS'] and self._base_1d.flags['OWNDATA'])

    def to_dense(self, scratch=None):
        """
        Return this SPAM vector as a (dense) numpy array.

        The memory in `scratch` maybe used when it is not-None.

        Parameters
        ----------
        scratch : numpy.ndarray, optional
            scratch space available for use.

        Returns
        -------
        numpy.ndarray
        """
        #don't use scratch since we already have memory allocated
        return self._base_1d  # *must* be a numpy array for Cython arg conversion

    @property
    def _base_1d(self):
        """
        Direct access to the underlying 1D array.
        """
        return self._rep.base

    @property
    def base(self):
        """
        Direct access the the underlying data as column vector, i.e, a (dim,1)-shaped array.
        """
        bv = self._base_1d.view()
        bv.shape = (bv.size, 1)  # 'base' is by convention a (N,1)-shaped array
        return bv

    def __copy__(self):
        # We need to implement __copy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __copy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def __deepcopy__(self, memo):
        # We need to implement __deepcopy__ because we defer all non-existing
        # attributes to self.base (a numpy array) which *has* a __deepcopy__
        # implementation that we don't want to use, as it results in just a
        # copy of the numpy array.
        cls = self.__class__
        cpy = cls.__new__(cls)
        memo[id(self)] = cpy
        for k, v in self.__dict__.items():
            setattr(cpy, k, _copy.deepcopy(v, memo))
        return cpy

    #Access to underlying array
    def __getitem__(self, key):
        self.dirty = True
        return self.base.__getitem__(key)

    def __getslice__(self, i, j):
        self.dirty = True
        return self.__getitem__(slice(i, j))  # Called for A[:]

    def __setitem__(self, key, val):
        self.dirty = True
        return self.base.__setitem__(key, val)

    def __getattr__(self, attr):
        #use __dict__ so no chance for recursive __getattr__
        if '_rep' in self.__dict__:  # sometimes in loading __getattr__ gets called before the instance is loaded
            ret = getattr(self.base, attr)
        else:
            raise AttributeError("No attribute:", attr)
        self.dirty = True
        return ret

    #Mimic array
    def __pos__(self): return self.base
    def __neg__(self): return -self.base
    def __abs__(self): return abs(self.base)
    def __add__(self, x): return self.base + x
    def __radd__(self, x): return x + self.base
    def __sub__(self, x): return self.base - x
    def __rsub__(self, x): return x - self.base
    def __mul__(self, x): return self.base * x
    def __rmul__(self, x): return x * self.base
    def __truediv__(self, x): return self.base / x
    def __rtruediv__(self, x): return x / self.base
    def __floordiv__(self, x): return self.base // x
    def __rfloordiv__(self, x): return x // self.base
    def __pow__(self, x): return self.base ** x
    def __eq__(self, x): return self.base == x
    def __len__(self): return len(self.base)
    def __int__(self): return int(self.base)
    def __long__(self): return int(self.base)
    def __float__(self): return float(self.base)
    def __complex__(self): return complex(self.base)

    def __str__(self):
        s = "%s with dimension %d\n" % (self.__class__.__name__, self.dim)
        s += _mt.mx_to_string(self.to_dense(), width=4, prec=2)
        return s
