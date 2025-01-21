"""
Defines the ModelFunction class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.models.explicitmodel import ExplicitOpModel as _ExplicitOpModel
from pygsti.models.localnoisemodel import LocalNoiseModel as _LocalNoiseModel

class ModelFunction(object):
    """
    A "function of a model" for which one may want to compute error bars.

    Specifically, one may want to compute confidence-interval error bars based
    on a confidence region of the function's model argument.

    The "function" may have other parameters, and the reason for defining it as
    a class is so that it can hold

    1. relevant "meta" arguments in addition to the central Model, and
    2. information to speed up function evaluations at nearby Model "points",
       for computing finite-difference derivatives.

    Parameters
    ----------
    model : Model
        A sample model giving the constructor a template for what
        type/parameterization of model to expect in calls to
        :func:`evaluate`.

    dependencies : list
        A list of (*type*,*label*) tuples, or the special strings `"all"` and
        `"spam"`, indicating which Model parameters the function depends
        upon. Here *type* can be `"gate"`, `"prep"`, `"povm"`, or
        `"instrument"`, and  *label* can be any of the corresponding labels
        found in the models being evaluated.  The reason for specifying
        this at all is to speed up computation of the finite difference
        derivative needed to find the error bars.
    """

    def __init__(self, model, dependencies):
        """
        Creates a new ModelFunction object.

        Parameters
        ----------
        model : Model
            A sample model giving the constructor a template for what
            type/parameterization of model to expect in calls to
            :func:`evaluate`.

        dependencies : list
            A list of (*type*,*label*) tuples, or the special strings `"all"` and
            `"spam"`, indicating which Model parameters the function depends
            upon. Here *type* can be `"gate"`, `"prep"`, `"povm"`, or
            `"instrument"`, and  *label* can be any of the corresponding labels
            found in the models being evaluated.  The reason for specifying
            this at all is to speed up computation of the finite difference
            derivative needed to find the error bars.
        """
        self.base_model = model
        self.dependencies = dependencies

    def evaluate(self, model):
        """
        Evaluate this gate-set-function at `model`.

        Parameters
        ----------
        model : Model
            The "point" at which to evaluate this function.

        Returns
        -------
        object
        """
        return None

    def evaluate_nearby(self, nearby_model):
        """
        Evaluate this model-function at `nearby_model`.

        `nearby_model` can be assumed to be very close to the `model` provided
        to the last call to :meth:`evaluate`.

        Parameters
        ----------
        nearby_model : Model
            A nearby "point" to evaluate this function at.

        Returns
        -------
        object
        """
        # do stuff assuming nearby_model is eps away from model
        return self.evaluate(nearby_model)

    def list_dependencies(self):
        """
        Return the dependencies of this model-function.

        Returns
        -------
        list
            A list of (*type*,*label*) tuples, or the special strings `"all"` and
            `"spam"`, indicating which Model parameters the function depends
            upon. Here *type* can be `"gate"`, `"prep"`, `"povm"`, or
            `"instrument"`, and  *label* can be any of the corresponding labels
            found in the models being evaluated.
        """
        return self.dependencies
        #determines which variations in model are used when computing confidence regions


def spamfn_factory(fn):
    """
    Creates a class that evaluates `fn(preps,povms,...)`.

    Here `preps` and `povms` are lists of the preparation SPAM vectors and POVMs
    of a Model, respectively, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model, ...)` where `model` is a Model and `...` are optional
        additional arguments that are passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by spamfn_factory """

        def __init__(self, model, *args, **kwargs):
            """
            Creates a new ModelFunction dependent only on its Model
            argument's SPAM vectors.
            """
            self.args = args
            self.kwargs = kwargs
            ModelFunction.__init__(self, model, ["spam"])

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            return fn(list(model.preps.values()),
                      list(model.povms.values()),
                      *self.args, **self.kwargs)

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp

#Note: the 'basis' argument is unnecesary here, as it could be passed as an additional arg


def opfn_factory(fn):
    """
    Creates a class that evaluates `fn(gate,basis,...)`.

    Hhere `gate` is a single operation matrix, `basis` describes what basis it's
    in, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model, gl, ...)` where `model` is a Model, `gl` is a gate
        label, and `...` are optional additional arguments passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by opfn_factory """

        def __init__(self, model, gl, *args, **kwargs):
            """ Creates a new ModelFunction dependent on a single gate"""
            self.gl = gl
            self.args = args
            self.kwargs = kwargs
            ModelFunction.__init__(self, model, [("gate", gl)])

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            return fn(model.operations[self.gl].to_dense(on_space='HilbertSchmidt'), model.basis,
                      *self.args, **self.kwargs)

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


#Note: the 'op2' and 'basis' arguments are unnecesary here, as they could be
# passed as additional args
def opsfn_factory(fn):
    """
    Creates a class that evaluates `fn(op1,op2,basis,...)`.

    Here `op1` and `op2` are a single operation matrices, `basis` describes what
    basis they're in, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model1, model2, gl, ...)` where `model1` and `model2` are
        Models (only `model1` and `op1` are varied when computing a
        confidence region), `gl` is a operation label, and `...` are optional
        additional arguments passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by opsfn_factory """

        def __init__(self, model1, model2, gl, *args, **kwargs):
            """ Creates a new ModelFunction dependent on a single gate"""
            self.other_model = model2
            self.gl = gl
            self.args = args
            self.kwargs = kwargs
            ModelFunction.__init__(self, model1, [("gate", gl)])

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            if isinstance(model, _ExplicitOpModel):
                return fn(model.operations[self.gl].to_dense(on_space='HilbertSchmidt'),
                          self.other_model.operations[self.gl].to_dense(on_space='HilbertSchmidt'),
                          model.basis, *self.args, **self.kwargs)  # assume functions want *dense* gates
            elif isinstance(model, _LocalNoiseModel):
                return fn(model.operation_blks['gates'][self.gl].to_dense(on_space='HilbertSchmidt'),
                          self.other_model.operation_blks['gates'][self.gl].to_dense(on_space='HilbertSchmidt'),
                          model.basis, *self.args, **self.kwargs)  # assume functions want *dense* gates
            else:
                raise ValueError(f"Unsupported model type: {type(model)}!")

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def instrumentfn_factory(fn):
    """
    Creates a class that evaluates `fn(instrument1,instrument2,basis,...)`.

    Here `instrument1` and `instrument2` are a :class:`Instrument`s, `basis`
    describes what basis they're in, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model1, model2, gl, ...)` where `model1` and `model2` are
        Models (only `model1` and `op1` are varied when computing a
        confidence region), `gl` is a operation label, and `...` are optional
        additional arguments passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by opsfn_factory """

        def __init__(self, model1, model2, instrument_lbl, *args, **kwargs):
            """ Creates a new ModelFunction dependent on a single gate"""
            self.other_model = model2
            self.il = instrument_lbl
            self.args = args
            self.kwargs = kwargs
            ModelFunction.__init__(self, model1, [("instrument", instrument_lbl)])

        def evaluate(self, model):
            """ Evaluate this model-function at `model`."""
            return fn(model.instruments[self.il], self.other_model.instruments[self.il],
                      model.basis, *self.args, **self.kwargs)  # assume functions want *dense* gates

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def vecfn_factory(fn):
    """
    Creates a class that evaluates `fn(vec,basis,...)`.

    Here `vec` is a single SPAM vector, `basis` describes what basis it's in,
    and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model, lbl, typ, ...)` where `model` is a Model, `lbl` is
        SPAM vector label, `typ` is either `"prep"` or `"effect"` (the type of
        the SPAM vector), and `...` are optional additional arguments
        passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by vecfn_factory """

        def __init__(self, model, lbl, typ, *args, **kwargs):
            """ Creates a new ModelFunction dependent on a single SPAM vector"""
            self.lbl = lbl
            self.typ = typ
            self.args = args
            self.kwargs = kwargs
            assert(typ in ['prep', 'effect']), "`typ` argument must be either 'prep' or 'effect'"
            if typ == 'effect':
                typ = "povm"
                lbl, _ = lbl.split(":")  # for "effect"-mode, lbl must == "povmLbl:ELbl"
                # and ModelFunction depends on entire POVM
            ModelFunction.__init__(self, model, [(typ, lbl)])

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            if self.typ == "prep":
                return fn(model.preps[self.lbl].to_dense(on_space='HilbertSchmidt'), model.basis,
                          *self.args, **self.kwargs)
            else:
                povmlbl, Elbl = self.lbl.split(":")  # for effect, lbl must == "povmLbl:ELbl"
                return fn(model.povms[povmlbl][Elbl].to_dense(on_space='HilbertSchmidt'), model.basis,
                          *self.args, **self.kwargs)

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def vecsfn_factory(fn):
    """
    Creates a class that evaluates `fn(vec1, vec2, basis,...)`.

    Hhere `vec1` and `vec2` are SPAM vectors, `basis` describes what basis
    they're in, and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the two parameters as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model1, model2, lbl, typ, ...)` where `model1` and `model2`
        are Models (only `model1` and `vec1` are varied when computing a
        confidence region), `lbl` is a SPAM vector label, `typ` is either
        `"prep"` or `"effect"` (the type of the SPAM vector), and `...` are
        optional additional arguments passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by vecsfn_factory """

        def __init__(self, model1, model2, lbl, typ, *args, **kwargs):
            """ Creates a new ModelFunction dependent on a single SPAM vector"""
            self.other_model = model2
            self.lbl = lbl
            self.typ = typ
            self.args = args
            self.kwargs = kwargs
            assert(typ in ['prep', 'effect']), "`typ` argument must be either 'prep' or 'effect'"
            if typ == 'effect':
                typ = "povm"
                lbl, _ = lbl.split(":")  # for "effect"-mode, lbl must == "povmLbl:ELbl"
                # and ModelFunction depends on entire POVM
            self.other_vecsrc = self.other_model.preps if self.typ == "prep" \
                else self.other_model.povms
            ModelFunction.__init__(self, model1, [(typ, lbl)])

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            if self.typ == "prep":
                return fn(model.preps[self.lbl].to_dense(on_space='HilbertSchmidt'),
                          self.other_vecsrc[self.lbl].to_dense(on_space='HilbertSchmidt'),
                          model.basis, *self.args, **self.kwargs)
            else:
                povmlbl, Elbl = self.lbl.split(":")  # for effect, lbl must == "povmLbl:ELbl"
                return fn(model.povms[povmlbl][Elbl].to_dense(on_space='HilbertSchmidt'),
                          self.other_vecsrc[povmlbl][Elbl].to_dense(on_space='HilbertSchmidt'),
                          model.basis, *self.args, **self.kwargs)

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def povmfn_factory(fn):
    """
    Creates a class that evaluates `fn(model,...)` where `fn` *only* depends on the POVM effect elements `model`.

    Here `model` is the entire Model and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the one parameter as discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model, ...)` where `model` is a Model and `...` are optional
        additional arguments that are passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by povmfn_factory """

        def __init__(self, model, *args, **kwargs):
            """
            Creates a new ModelFunction dependent on all of its
            Model argument's effects
            """
            self.args = args
            self.kwargs = kwargs
            dps = [("povm", l) for l in model.povms]
            ModelFunction.__init__(self, model, dps)

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            return fn(model, *self.args, **self.kwargs)

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp


def modelfn_factory(fn):
    """
    Creates a class that evaluates `fn(model,...)`.

    Here `model` is a `Model` object and `...` are additional arguments (see below).

    Parameters
    ----------
    fn : function
        A function of at least the single `model` parameter discussed above.

    Returns
    -------
    cls : class
        A :class:`ModelFunction`-derived class initialized by
        `cls(model, ...)` where `model` is a Model, and `...` are
        optional additional arguments passed to `fn`.
    """
    class GSFTemp(ModelFunction):
        """ ModelFunction class created by modelfn_factory """

        def __init__(self, model, *args, **kwargs):
            """
            Creates a new ModelFunction dependent on all of its Model
            argument's paramters
            """
            self.args = args
            self.kwargs = kwargs
            ModelFunction.__init__(self, model, ["all"])

        def evaluate(self, model):
            """ Evaluate this gate-set-function at `model`."""
            return fn(model, *self.args, **self.kwargs)

    GSFTemp.__name__ = fn.__name__ + str("_class")
    return GSFTemp
