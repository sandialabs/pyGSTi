""" Defines the GateSetFunction class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

class GateSetFunction(object):
    """ 
    Encapsulates a "function of a GateSet" that one may want to compute
    confidence-interval error bars for based on a confidence region of
    the functions gate set argument.  The "function" may have other parameters,
    and the reason for defining it as a class is so that it can hold 

    1. relevant "meta" arguments in addition to the central GateSet, and 
    2. information to speed up function evaluations at nearby GateSet "points",
       for computing finite-difference derivatives.
    """ 
    
    def __init__(self, gateset, dependencies):
        """
        Creates a new GateSetFunction object.

        Parameters
        ----------
        gateset : GateSet
            A sample gate set giving the constructor a template for what 
            type/parameterization of gate set to expect in calls to 
            :func:`evaluate`.

        dependencies : list
            A list of *type:label* strings, or the special strings `"all"` and
            `"spam"`, indicating which GateSet parameters the function depends
            upon. Here *type* can be `"gate"`, `"prep"`, `"povm"`, or 
            `"instrument"`, and  *label* can be any of the corresponding labels
            found in the gate sets being evaluated.  The reason for specifying
            this at all is to speed up computation of the finite difference
            derivative needed to find the error bars.
        """
        self.base_gateset = gateset
        self.dependencies = dependencies

    def evaluate(self, gateset):
        """ Evaluate this gate-set-function at `gateset`."""
        return None
    
    def evaluate_nearby(self, nearby_gateset):
        """ 
        Evaluate this gate-set-function at `nearby_gateset`, which can
        be assumed is very close to the `gateset` provided to the last
        call to :func:`evaluate`.
        """
        # do stuff assuming nearby_gateset is eps away from gateset
        return self.evaluate(nearby_gateset)

    def get_dependencies(self):
        """
        Return the dependencies of this gate-set-function.

        Returns
        -------
        list
            A list of *type:label* strings, or the special strings `"all"` and
            `"spam"`, indicating which GateSet parameters the function depends
            upon. Here *type* can be `"gate"`, `"prep"`, `"povm"`, or 
            `"instrument"` and *label* can be any of the corresponding labels
            found in the gate sets being evaluated.
        """
        return self.dependencies
          #determines which variations in gateset are used when computing confidence regions

def spamfn_factory(fn):
    """
    Ceates a class that evaluates 
    `fn(preps,povms,...)`, where `preps` and `povms` are lists of the
    preparation SPAM vectors and POVMs of a GateSet, respectively,
    and `...` are additional arguments (see below).

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
        """ GateSetFunction class created by spamfn_factory """
        def __init__(self, gateset, *args, **kwargs):
            """ 
            Creates a new GateSetFunction dependent only on its GateSet
            argument's SPAM vectors.
            """
            self.args = args
            self.kwargs = kwargs
            GateSetFunction.__init__(self, gateset, ["spam"])
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
            return fn(list(gateset.preps.values()),
                      list(gateset.povms.values()),
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
        """ GateSetFunction class created by gatefn_factory """
        def __init__(self, gateset, gl, *args, **kwargs):
            """ Creates a new GateSetFunction dependent on a single gate"""
            self.gl = gl
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset, ["gate:"+str(gl)])
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
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
        """ GateSetFunction class created by gatesfn_factory """
        def __init__(self, gateset1, gateset2, gl, *args, **kwargs):
            """ Creates a new GateSetFunction dependent on a single gate"""
            self.other_gateset = gateset2
            self.gl = gl
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset1, ["gate:"+str(gl)])
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
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
        """ GateSetFunction class created by vecfn_factory """
        def __init__(self, gateset, lbl, typ, *args, **kwargs):
            """ Creates a new GateSetFunction dependent on a single SPAM vector"""
            self.lbl = lbl
            self.typ = typ
            self.args = args
            self.kwargs = kwargs
            assert(typ in ['prep','effect']), "`typ` argument must be either 'prep' or 'effect'"
            if typ == 'effect':
                typ = "povm"
                lbl,_ = lbl.split(":") #for "effect"-mode, lbl must == "povmLbl:ELbl"
                                       # and GateSetFunction depends on entire POVM
            GateSetFunction.__init__(self, gateset, [typ + ":" + str(lbl)]) 
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
            if self.typ == "prep":
                return fn(gateset.preps[self.lbl], gateset.basis,
                      *self.args, **self.kwargs)
            else:
                povmlbl,Elbl = self.lbl.split(":") #for effect, lbl must == "povmLbl:ELbl"
                return fn(gateset.povms[povmlbl][Elbl], gateset.basis,
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
        """ GateSetFunction class created by vecsfn_factory """
        def __init__(self, gateset1, gateset2, lbl, typ, *args, **kwargs):
            """ Creates a new GateSetFunction dependent on a single SPAM vector"""
            self.other_gateset = gateset2
            self.lbl = lbl
            self.typ = typ
            self.args = args
            self.kwargs = kwargs
            assert(typ in ['prep','effect']), "`typ` argument must be either 'prep' or 'effect'"
            if typ == 'effect':
                typ = "povm"
                lbl,_ = lbl.split(":") #for "effect"-mode, lbl must == "povmLbl:ELbl"
                                       # and GateSetFunction depends on entire POVM
            self.other_vecsrc = self.other_gateset.preps if self.typ == "prep" \
                                else self.other_gateset.povms
            GateSetFunction.__init__(self, gateset1, [typ + ":" + str(lbl)])
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
            if self.typ == "prep":
                return fn(gateset.preps[self.lbl], self.other_vecsrc[self.lbl],
                      gateset.basis,  *self.args, **self.kwargs)
            else:
                povmlbl,Elbl = self.lbl.split(":") #for effect, lbl must == "povmLbl:ELbl"
                return fn(gateset.povms[povmlbl][Elbl], self.other_vecsrc[povmlbl][Elbl],
                          gateset.basis,  *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def povmfn_factory(fn):
    """
    Ceates a class that evaluates 
    `fn(gateset,...)` where `gateset` is the entire GateSet (and it is assumed
    that `fn` is only a function of the POVM effect elements of the gate set),
    and `...` are additional arguments (see below).
    
    Parameters
    ----------
    fn : function
        A function of at least the one parameter as discussed above.

    Returns
    -------
    cls : class
        A :class:`GateSetFunction`-derived class initialized by
        `cls(gateset, ...)` where `gateset` is a GateSet and `...` are optional
        additional arguments that are passed to `fn`.
    """
    class GSFTemp(GateSetFunction):
        """ GateSetFunction class created by povmfn_factory """
        def __init__(self, gateset, *args, **kwargs):
            """ 
            Creates a new GateSetFunction dependent on all of its
            GateSet argument's effects
            """
            self.args = args
            self.kwargs = kwargs
            dps = ["povm:%s"%l for l in gateset.povms]
            GateSetFunction.__init__(self, gateset, dps)
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
            return fn(gateset, *self.args, **self.kwargs)
        
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
        """ GateSetFunction class created by gatesetfn_factory """
        def __init__(self, gateset, *args, **kwargs):
            """ 
            Creates a new GateSetFunction dependent on all of its GateSet
            argument's paramters
            """
            self.args = args
            self.kwargs = kwargs        
            GateSetFunction.__init__(self, gateset, ["all"])
            
        def evaluate(self, gateset):
            """ Evaluate this gate-set-function at `gateset`."""
            return fn(gateset, *self.args, **self.kwargs)
        
    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp

