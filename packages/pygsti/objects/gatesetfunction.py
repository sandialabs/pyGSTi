from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" The GateSetFunction class """

class GateSetFunction(object):
    def __init__(self, gateset, dependencies):
        self.base_gateset = gateset
        self.dependencies = dependencies

    def evaluate(self, gateset):
        return None
    
    def evaluate_nearby(self, nearby_gateset):
        # do stuff assuming nearby_gateset is eps away from gateset
        return self.evaluate(nearby_gateset)

    def get_dependencies(self):
        return self.dependencies
          #determines which variations in gateset are used when computing confidence regions

def spamfn_factory(fn):
    """
    Creates a class that evaluates `fn(preps,effects,...)`, where `preps` and
    `effects` are lists of the preparation and POVM effect  SPAM vectors of a
    GateSet, respectively, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset, ...)` where `gateset` is a GateSet and `...` are optional
        additional arguments that are passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        def __init__(self, gateset, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset, ["spam"])
            
        def evaluate(self, gateset):
            return fn(gateset.get_preps(), gateset.get_effects(),
                      *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp

#Note: the 'basis' argument is unnecesary here, as it could be passed as an additional arg
def gatefn_factory(fn):
    """
    Creates a class that evaluates `fn(gate,basis,...)`, where `gate` is a
    single gate matrix, `basis` describes what basis it's in, and `...` are
    additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset, gl, ...)` where `gateset` is a GateSet, `gl` is a gate
        label, and `...` are optional additional arguments passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        def __init__(self, gateset, gl, *args, **kwargs):
            self.gl = gl
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset, ["gate:"+gl])
            
        def evaluate(self, gateset):
            return fn(gateset.gates[self.gl], gateset.basis,
                      *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


#Note: the 'gate2' and 'basis' arguments are unnecesary here, as they could be
# passed as additional args
def gatesfn_factory(fn):
    """
    Creates a class that evaluates `fn(gate1,gate2,basis,...)`, where `gate1`
    and `gate2` are a single gate matrices, `basis` describes what basis they're
    in, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset1, gateset2, gl, ...)` where `gateset1` and `gateset2` are
        GateSets (only `gateset1` and `gate1` are varied when computing a
        confidence region), `gl` is a gate label, and `...` are optional
        additional arguments passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        def __init__(self, gateset1, gateset2, gl, *args, **kwargs):
            self.other_gateset = gateset2
            self.gl = gl
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset1, ["gate:"+gl])
            
        def evaluate(self, gateset):
            return fn(gateset.gates[self.gl], self.other_gateset.gates[self.gl],
                      gateset.basis, *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def vecfn_factory(fn):
    """
    Creates a class that evaluates `fn(vec,basis,...)`, where `vec` is a
    single SPAM vector, `basis` describes what basis it's in, and `...` are
    additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset, lbl, typ, ...)` where `gateset` is a GateSet, `lbl` is
        SPAM vector label, `typ` is either `"prep"` or `"effect"` (the type of
        the SPAM vector), and `...` are optional additional arguments
        passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        def __init__(self, gateset, lbl, typ, *args, **kwargs):
            self.lbl = lbl
            self.typ = typ
            self.args = args
            self.kwargs = kwargs
            assert(typ in ['prep','effect']), "`typ` argument must be either 'prep' or 'effect'"
            GateSetFunction.__init__(self, gateset, [typ + ":" + lbl])
            
        def evaluate(self, gateset):
            vecsrc = gateset.preps if self.typ == "prep" else gateset.effects
            return fn(vecsrc[self.lbl], gateset.basis,
                      *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def vecsfn_factory(fn):
    """
    Creates a class that evaluates `fn(vec1, vec2, basis,...)`, where `vec1`
    and `vec2` are SPAM vectors, `basis` describes what basis they're in, and
    `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset1, gateset2, lbl, typ, ...)` where `gateset1` and `gateset2`
        are GateSets (only `gateset1` and `vec1` are varied when computing a
        confidence region), `lbl` is a SPAM vector label, `typ` is either
        `"prep"` or `"effect"` (the type of the SPAM vector), and `...` are
        optional additional arguments passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        def __init__(self, gateset1, gateset2, lbl, typ, *args, **kwargs):
            self.other_gateset = gateset2
            self.lbl = lbl
            self.typ = typ
            self.args = args
            self.kwargs = kwargs
            assert(typ in ['prep','effect']), "`typ` argument must be either 'prep' or 'effect'"
            self.other_vecsrc = self.other_gateset.preps if self.typ == "prep" \
                                else self.other_gateset.effects
            GateSetFunction.__init__(self, gateset1, [typ + ":" + lbl])
            
        def evaluate(self, gateset):
            vecsrc = gateset.preps if self.typ == "prep" else gateset.effects
            return fn(vecsrc[self.lbl], self.other_vecsrc[self.lbl],
                      gateset.basis,  *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def gatesetfn_factory(fn):
    """
    Creates a class that evaluates `fn(gateset,...)`, where `gateset` is a
    `GateSet` object and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the single `gateset` parameter discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset, ...)` where `gateset` is a GateSet, and `...` are
        optional additional arguments passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        def __init__(self, gateset, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset, ["all"])
            
        def evaluate(self, gateset):
            return fn(gateset, *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp

