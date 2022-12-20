"""
Defines objective-function objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import sys as _sys
import time as _time
import pathlib as _pathlib

import numpy as _np

from pygsti import tools as _tools
from pygsti.layouts.distlayout import DistributableCOPALayout as _DistributableCOPALayout
from pygsti.tools import slicetools as _slct, mpitools as _mpit, sharedmemtools as _smt
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter


def _objfn(objfn_cls, model, dataset, circuits=None,
           regularization=None, penalties=None, op_label_aliases=None,
           comm=None, mem_limit=None, method_names=None, array_types=None,
           mdc_store=None, verbosity=0, **addl_args):
    """
    A convenience function for creating an objective function.

    Takes a number of common parameters and automates the creation of
    intermediate objects like a :class:`ResourceAllocation` and
    :class:`CircuitList`.

    Parameters
    ----------
    objfn_cls : class
        The :class:`MDCObjectiveFunction`-derived class to create.

    model : Model
        The model.

    dataset : DataSet
        The data.

    circuits : list, optional
        The circuits.

    regularization : dict, optional
        A dictionary of regularization values.

    penalties : dict, optional
        A dictionary of penalty values.

    op_label_aliases : dict, optional
        An alias dictionary.

    comm : mpi4py.MPI.Comm, optional
        For splitting load among processors.

    mem_limit : int, optional
        Rough memory limit in bytes.

    method_names : tuple
        A tuple of the method names of the returned objective function
        that will be called (used to estimate memory and setup resource division)

    array_types : tuple
        A tuple of array types that will be allocated, in addition to those contained in
        the returned objective functon itself and within the methods given by `method_names`.

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    verbosity : int or VerbosityPrinter, optional
        Amount of information printed to stdout.

    Returns
    -------
    ObjectiveFunction
    """

    if mdc_store is None:

        if circuits is None:
            circuits = list(dataset.keys())

        if op_label_aliases:
            circuits = _CircuitList(circuits, op_label_aliases)

        resource_alloc = _ResourceAllocation(comm, mem_limit)
        ofn = objfn_cls.create_from(model, dataset, circuits, regularization, penalties,
                                    resource_alloc, verbosity=verbosity,
                                    method_names=method_names if (method_names is not None) else ('fn',),
                                    array_types=array_types if (array_types is not None) else (),
                                    **addl_args)

    else:
        #Create directly from store object, which contains everything else
        assert(model is None and dataset is None and circuits is None and comm is None and mem_limit is None)
        # Note: allow method_names and array_types to be non-None and still work with mdc_store, since
        # the way this function is used in chi2fns.py and likelihoodfns.py hard-codes these values.
        ofn = objfn_cls(mdc_store, regularization, penalties, verbosity=0, **addl_args)

    return ofn

    #def __len__(self):
    #    return len(self.circuits)


class ObjectiveFunctionBuilder(_NicelySerializable):
    """
    A factory class for building objective functions.

    This is useful because often times the user will want to
    specify some but not all of the information needed to create
    an actual objective function object.  Namely, regularization
    and penalty values are known ahead of time, while the model,
    dataset, and circuits are supplied later, internally, when
    running a protocol.

    Parameters
    ----------
    cls_to_build : class
        The :class:`MDCObjectiveFunction`-derived objective function class to build.

    name : str, optional
        A name for the built objective function (can be anything).

    description : str, optional
        A description for the built objective function (can be anything)

    regularization : dict, optional
        Regularization values (allowed keys depend on `cls_to_build`).

    penalties : dict, optional
        Penalty values (allowed keys depend on `cls_to_build`).
    """

    @classmethod
    def cast(cls, obj):
        """
        Cast `obj` to an `ObjectiveFunctionBuilder` instance.

        If `obj` is already an `ObjectiveFunctionBuilder` instance, it is simply returned.
        Otherwise a new `ObjectiveFunctionBuilder` instance is created from `obj` if possible.

        Parameters
        ----------
        obj : None or str or dict or list or tuple or ObjectiveFunctionBuilder
            Object to cast.

        Returns
        -------
        ObjectiveFunctionBuilder
        """
        if isinstance(obj, cls): return obj
        elif obj is None: return cls.create_from()
        elif isinstance(obj, str): return cls.create_from(objective=obj)
        elif isinstance(obj, dict): return cls.create_from(**obj)
        elif isinstance(obj, (list, tuple)): return cls(*obj)
        else: raise ValueError("Cannot create an %s object from '%s'" % (cls.__name__, str(type(obj))))

    @classmethod
    def create_from(cls, objective='logl', freq_weighted_chi2=False):
        """
        Creates common :class:`ObjectiveFunctionBuilder`s from a few arguments.

        Parameters
        ----------
        objective : {'logl', 'chi2'}, optional
            The objective function type: log-likelihood or chi-squared.

        freq_weighted_chi2 : bool, optional
            Whether to use 1/frequency values as the weights in the `"chi2"` case.

        Returns
        -------
        ObjectiveFunctionBuilder
        """
        if objective == "chi2":
            if freq_weighted_chi2:
                builder = FreqWeightedChi2Function.builder(
                    name='fwchi2',
                    description="Freq-weighted sum of Chi^2",
                    regularization={'min_freq_clip_for_weighting': 1e-4})
            else:
                builder = Chi2Function.builder(
                    name='chi2',
                    description="Sum of Chi^2",
                    regularization={'min_prob_clip_for_weighting': 1e-4})

        elif objective == "logl":
            builder = PoissonPicDeltaLogLFunction.builder(
                name='dlogl',
                description="2*Delta(log(L))",
                regularization={'min_prob_clip': 1e-4,
                                'radius': 1e-4},
                penalties={'cptp_penalty_factor': 0,
                           'spam_penalty_factor': 0})

        elif objective == "tvd":
            builder = TVDFunction.builder(
                name='tvd',
                description="Total Variational Distance (TVD)")

        else:
            raise ValueError("Invalid objective: %s" % objective)
        assert(isinstance(builder, cls)), "This function should always return an ObjectiveFunctionBuilder!"
        return builder

    def __init__(self, cls_to_build, name=None, description=None, regularization=None, penalties=None, **kwargs):
        super().__init__()
        self.name = name if (name is not None) else cls_to_build.__name__
        self.description = description if (description is not None) else "_objfn"  # "Sum of Chi^2" OR "2*Delta(log(L))"
        self.cls_to_build = cls_to_build
        self.regularization = regularization
        self.penalties = penalties
        self.additional_args = kwargs

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'description': self.description,
                      'class_to_build': self.cls_to_build.__module__ + '.' + self.cls_to_build.__name__,
                      'regularization': self.regularization,
                      'penalties': self.penalties,
                      'additional_arguments': self.additional_args,
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.io.metadir import _class_for_name
        return cls(_class_for_name(state['class_to_build']), state['name'], state['description'],
                   state['regularization'], state['penalties'], *state['additional_arguments'])

    def compute_array_types(self, method_names, forwardsim):
        return self.cls_to_build.compute_array_types(method_names, forwardsim)

    def build(self, model, dataset, circuits, resource_alloc=None, verbosity=0):
        """
        Build an objective function.  This is the workhorse method of an :class:`ObjectiveFunctionBuilder`.

        Arguments are the additional information needed to construct a
        :class:`MDCObjectiveFunction` object, beyond what is stored in
        this builder object.

        Parameters
        ----------
        model : Model
            The model.

        dataset : DataSet.
            The data set.

        circuits : list
            The circuits.

        resource_alloc : ResourceAllocation, optional
            Available resources and how they should be allocated for objective
            function computations.

        verbosity : int, optional
            Level of detail to print to stdout.

        Returns
        -------
        MDCObjectiveFunction
        """
        return self.cls_to_build.create_from(model=model, dataset=dataset, circuits=circuits,
                                             resource_alloc=resource_alloc, verbosity=verbosity,
                                             regularization=self.regularization, penalties=self.penalties,
                                             name=self.name, description=self.description, **self.additional_args)

    def build_from_store(self, mdc_store, verbosity=0):
        """
        Build an objective function.  This is a workhorse method of an :class:`ObjectiveFunctionBuilder`.

        Takes a single "store" argument (apart from `verbosity`) that encapsulates all the remaining
        ingredients needed to build a :class:`MDCObjectiveFunction` object (beyond what is stored in
        this builder object).

        Parameters
        ----------
        mdc_store : ModelDatasetCircuitsStore
            The store object, which doubles as a cache for reused information.

        verbosity : int, optional
            Level of detail to print to stdout.

        Returns
        -------
        MDCObjectiveFunction
        """
        return self.cls_to_build(mdc_store, verbosity=verbosity,
                                 regularization=self.regularization, penalties=self.penalties,
                                 name=self.name, description=self.description, **self.additional_args)


class ObjectiveFunction(object):
    """
    So far, this is just a base class for organizational purposes
    """

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        For instance, if the objective function is DeltaLogL then this function would
        multiply `objective_function_value` by 2, whereas in the case of a chi-squared
        objective function this function just return `objective_function_value`.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        raise ValueError("This objective function does not have chi2_k distributed values!")


class RawObjectiveFunction(ObjectiveFunction):
    """
    An objective function that acts on probabilities and counts directly.

    Every :class:`RawObjectiveFunction` is assumed to perform a "local" function
    element-wise on the vectors of probabilities, counts (usually for a single outcome),
    and total-counts (usually for all the outcomes in a group), and sum the results
    to arrive at the final objective function's value.

    That is, the function must be of the form:
    `objective_function = sum_i local_function(probability_i, counts_i, total_counts_i)`.

    Each element of this sum (`local_function(probability_i, counts_i, total_counts_i)`)
    is called a *term* of the objective function.  A vector contains the square-roots
    of the terms is referred to as the *least-squares vector* (since least-squares
    optimizers use this vector as their objective function) and is abbreviated "lsvec".

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """

    def __init__(self, regularization=None, resource_alloc=None, name=None, description=None, verbosity=0):
        """
        Create a raw objective function.

        A raw objective function acts on "raw" probabilities and counts,
        and is usually a statistic comparing the probabilities to count data.

        Parameters
        ----------
        regularization : dict, optional
            Regularization values.

        resource_alloc : ResourceAllocation, optional
            Available resources and how they should be allocated for computations.

        name : str, optional
            A name for this objective function (can be anything).

        description : str, optional
            A description for this objective function (can be anything)

        verbosity : int, optional
            Level of detail to print to stdout.
        """
        self.resource_alloc = _ResourceAllocation.cast(resource_alloc)
        self.printer = _VerbosityPrinter.create_printer(verbosity, self.resource_alloc.comm)
        self.name = name if (name is not None) else self.__class__.__name__
        self.description = description if (description is not None) else "_objfn"

        if regularization is None: regularization = {}
        self.set_regularization(**regularization)

    def set_regularization(self):
        """
        Set regularization values.
        """
        pass  # no regularization parameters

    def _intermediates(self, probs, counts, total_counts, freqs):
        """ Intermediate values used by multiple functions (similar to a temporary cache) """
        return ()  # no intermdiate values

    def fn(self, probs, counts, total_counts, freqs):
        """
        Evaluate the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        Returns
        -------
        float
        """
        return _np.sum(self.terms(probs, counts, total_counts, freqs))

    def jacobian(self, probs, counts, total_counts, freqs):
        """
        Evaluate the derivative of the objective function with respect to the probabilities.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each argument, corresponding to
            the derivative with respect to each element of `probs`.
        """
        return self.dterms(probs, counts, total_counts, freqs)  # same as dterms b/c only i-th term depends on p_i

    def hessian(self, probs, counts, total_counts, freqs):
        """
        Evaluate the Hessian of the objective function with respect to the probabilities.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each argument, corresponding to
            the 2nd derivative with respect to each element of `probs`.  Note that this
            is not a 2D matrix because all off-diagonal elements of the Hessian are
            zero (because only the i-th term depends on the i-th probability).
        """
        return self.hterms(probs, counts, total_counts, freqs)  # same as dterms b/c only i-th term depends on p_i

    def terms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-(probability, count, total-count) values
        that get summed together to result in the objective function value.
        These are the "local" or "per-element" values of the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        return self.lsvec(probs, counts, total_counts, freqs, intermediates)**2

    def lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        return _np.sqrt(self.terms(probs, counts, total_counts, freqs, intermediates))

    def dterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        return 2 * self.lsvec(probs, counts, total_counts, freqs, intermediates) \
            * self.dlsvec(probs, counts, total_counts, freqs, intermediates)

    def dlsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # lsvec = sqrt(terms)
        # dlsvec = 0.5/lsvec * dterms
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        lsvec = self.lsvec(probs, counts, total_counts, freqs, intermediates)
        pt5_over_lsvec = _np.where(lsvec < 1e-100, 0.0, 0.5 / _np.maximum(lsvec, 1e-100))  # lsvec=0 is *min* w/0 deriv
        dterms = self.dterms(probs, counts, total_counts, freqs, intermediates)
        return pt5_over_lsvec * dterms

    def dlsvec_and_lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the least-squares vector together with the vector itself.

        This is sometimes more computationally efficient than calling :method:`dlsvec` and
        :method:`lsvec` separately, as the former call may require computing the latter.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        dlsvec: numpy.ndarray
            A 1D array of length equal to that of each array argument.

        lsvec: numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        #Similar to above, just return lsvec too
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        lsvec = self.lsvec(probs, counts, total_counts, freqs, intermediates)
        dlsvec = self.dlsvec(probs, counts, total_counts, freqs, intermediates)
        return dlsvec, lsvec

    def hterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of the `local_function` at each
        (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # terms = lsvec**2
        # dterms/dp = 2*lsvec*dlsvec/dp
        # d2terms/dp2 = 2*[ (dlsvec/dp)^2 + lsvec*d2lsvec/dp2 ]
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        return 2 * (self.dlsvec(probs, counts, total_counts, freqs, intermediates)**2
                    + self.lsvec(probs, counts, total_counts, freqs, intermediates)
                    * self.hlsvec(probs, counts, total_counts, freqs, intermediates))

    def hlsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of `sqrt(local_function)` at each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # lsvec = sqrt(terms)
        # dlsvec/dp = 0.5 * terms^(-0.5) * dterms/dp
        # d2lsvec/dp2 = -0.25 * terms^(-1.5) * (dterms/dp)^2 + 0.5 * terms^(-0.5) * d2terms_dp2
        #             = 0.5 / sqrt(terms) * (d2terms_dp2 - 0.5 * (dterms/dp)^2 / terms)
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        terms = self.terms(probs, counts, total_counts, freqs, intermediates)
        dterms = self.dterms(probs, counts, total_counts, freqs, intermediates)
        hterms = self.hterms(probs, counts, total_counts, freqs, intermediates)
        return 0.5 / _np.sqrt(terms) * (hterms - 0.5 * dterms**2 / terms)

    #Required zero-term methods for omitted probs support in model-based objective functions
    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        raise NotImplementedError("Derived classes must implement this!")

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        raise NotImplementedError("Derived classes must implement this!")

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        raise NotImplementedError("Derived classes must implement this!")


class ModelDatasetCircuitsStore(object):
    """
    Contains all the information that we'd like to persist when performing
    (multiple) evaluations of the same circuits using the same model and
    data set.  For instance, the evaluation of mubltiple (different) objective
    functions.

    This class holds only quantities that do *not* depend on the contained
    model's parameters.  See :class:`EvaluatedObjectiveFunction` for a class (TODO??)
    that holds the values of an objective function at a certain parameter-space
    point.
    """
    def __init__(self, model, dataset, circuits=None, resource_alloc=None, array_types=(),
                 precomp_layout=None, verbosity=0):
        self.dataset = dataset
        self.model = model
        #self.opBasis = mdl.basis
        self.resource_alloc = _ResourceAllocation.cast(resource_alloc)
        # expand = ??? get from model based on fwdsim type?

        circuit_list = circuits if (circuits is not None) else list(dataset.keys())
        bulk_circuit_list = circuit_list if isinstance(
            circuit_list, _CircuitList) else _CircuitList(circuit_list)
        self.circuits = bulk_circuit_list

        #The model's forward simulator gets to determine how the circuit outcome
        # probabilities (and other results) are stored in arrays - this makes sense
        # because it understands how to make this layout amenable to fast computation.
        if precomp_layout is None:
            self.layout = model.sim.create_layout(bulk_circuit_list, dataset, self.resource_alloc,
                                                  array_types, verbosity=verbosity)  # a CircuitProbabilityArrayLayout
        else:
            self.layout = precomp_layout
        self.array_types = array_types

        if isinstance(self.layout, _DistributableCOPALayout):  # then store global circuit liste separately
            self.global_circuits = self.circuits
            self.circuits = _CircuitList(self.layout.circuits, self.global_circuits.op_label_aliases,
                                         self.global_circuits.circuit_weights, name=None)
        else:
            self.global_circuits = self.circuits

        #self.circuits = bulk_circuit_list[:]
        #self.circuit_weights = bulk_circuit_list.circuit_weights
        self.ds_circuits = self.circuits.apply_aliases()

        # computed by add_count_vectors
        self.counts = None
        self.total_counts = None
        self.freqs = None

        # computed by add_omitted_freqs
        self.firsts = None
        self.indicesOfCircuitsWithOmittedData = None
        self.dprobs_omitted_rowsum = None

        self.time_dependent = False  # indicates whether the data should be treated as time-resolved

        #if not self.cache.has_evaltree():
        #    subcalls = self.get_evaltree_subcalls()
        #    evt_resource_alloc = _ResourceAllocation(self.raw_objfn.comm, evt_mlim,
        #                                             self.raw_objfn.profiler, self.raw_objfn.distribute_method)
        #    self.cache.add_evaltree(self.mdl, self.dataset, bulk_circuit_list, evt_resource_alloc,
        #                            subcalls, self.raw_objfn.printer - 1)
        #self.eval_tree = self.cache.eval_tree
        #self.lookup = self.cache.lookup
        #self.outcomes_lookup = self.cache.outcomes_lookup
        #self.wrt_block_size = self.cache.wrt_block_size
        #self.wrt_block_size2 = self.cache.wrt_block_size2

        #convenience attributes (could make properties?)
        if isinstance(self.layout, _DistributableCOPALayout):
            self.global_nelements = self.layout.global_num_elements
            self.global_nparams = self.layout.global_num_params
            self.global_nparams2 = self.layout.global_num_params2
            self.host_nelements = self.layout.host_num_elements
            self.host_nparams = self.layout.host_num_params
            self.host_nparams2 = self.layout.host_num_params2
            self.nelements = _slct.length(self.layout.host_element_slice)  # just for *this* proc
            self.nparams = _slct.length(self.layout.host_param_slice) \
                if self.layout.host_param_slice else self.model.num_params
            self.nparams2 = _slct.length(self.layout.host_param2_slice) \
                if self.layout.host_param2_slice else self.model.num_params
            assert(self.global_nparams is None or self.global_nparams == self.model.num_params)
        else:
            self.global_nelements = self.host_nelements = self.nelements = len(self.layout)
            self.global_nparams = self.host_nparams = self.nparams = self.model.num_params
            self.global_nparams2 = self.host_nparams2 = self.nparams2 = self.model.num_params

    @property
    def opBasis(self):
        return self.model.basis

    def num_data_params(self):
        """
        The number of degrees of freedom in the data used by this objective function.

        Returns
        -------
        int
        """
        return self.dataset.degrees_of_freedom(self.ds_circuits,
                                               aggregate_times=not self.time_dependent)

    def add_omitted_freqs(self, printer=None, force=False):
        """
        Detect omitted frequences (assumed to be 0) so we can compute objective fn correctly
        """
        if self.firsts is None or force:
            # FUTURE: add any tracked memory? self.resource_alloc.add_tracked_memory(...)
            self.firsts = []; self.indicesOfCircuitsWithOmittedData = []
            for i, c in enumerate(self.circuits):
                indices = _slct.to_array(self.layout.indices_for_index(i))
                lklen = _slct.length(self.layout.indices_for_index(i))
                if 0 < lklen < self.model.compute_num_outcomes(c):
                    self.firsts.append(indices[0])
                    self.indicesOfCircuitsWithOmittedData.append(i)
            if len(self.firsts) > 0:
                self.firsts = _np.array(self.firsts, 'i')
                self.indicesOfCircuitsWithOmittedData = _np.array(self.indicesOfCircuitsWithOmittedData, 'i')
                self.dprobs_omitted_rowsum = _np.empty((len(self.firsts), self.nparams), 'd')
                #if printer: printer.log("SPARSE DATA: %d of %d rows have sparse data" %
                #                        (len(self.firsts), len(self.circuits)))
            else:
                self.firsts = None  # no omitted probs

    def add_count_vectors(self, force=False):
        """
        Ensure this store contains count and total-count vectors.
        """
        if self.counts is None or self.total_counts is None or force:
            #Assume if an item is not None the appropriate amt of memory has already been tracked
            if self.counts is None: self.resource_alloc.add_tracked_memory(self.nelements)  # 'e'
            if self.total_counts is None: self.resource_alloc.add_tracked_memory(self.nelements)  # 'e'
            if self.freqs is None: self.resource_alloc.add_tracked_memory(self.nelements)  # 'e'

            # Note: in distributed case self.layout only holds *local* quantities (e.g.
            # the .ds_circuits are a subset of all the circuits and .nelements is the local
            # number of elements).
            counts = _np.empty(self.nelements, 'd')
            totals = _np.empty(self.nelements, 'd')

            for (i, circuit) in enumerate(self.ds_circuits):
                cnts = self.dataset[circuit].counts
                totals[self.layout.indices_for_index(i)] = sum(cnts.values())  # dataset[opStr].total
                counts[self.layout.indices_for_index(i)] = [cnts.get(x, 0) for x in self.layout.outcomes_for_index(i)]

            if self.circuits.circuit_weights is not None:
                for i in range(len(self.ds_circuits)):  # multiply N's by weights
                    counts[self.layout.indices_for_index(i)] *= self.circuits.circuit_weights[i]
                    totals[self.layout.indices_for_index(i)] *= self.circuits.circuit_weights[i]

            self.counts = counts
            self.total_counts = totals
            self.freqs = counts / totals


class EvaluatedModelDatasetCircuitsStore(ModelDatasetCircuitsStore):
    """
    Additionally holds quantities at a specific model-parameter-space point.
    """

    def __init__(self, mdc_store, verbosity):
        super().__init__(mdc_store.model, mdc_store.dataset, mdc_store.global_circuits, mdc_store.resource_alloc,
                         mdc_store.array_types, mdc_store.layout, verbosity)

        # Memory check - see if there's enough memory to hold all the evaluated quantities
        #persistent_mem = self.layout.memory_estimate()
        #in_gb = 1.0 / 1024.0**3  # in gigabytes
        #if self.raw_objfn.mem_limit is not None:
        #    in_gb = 1.0 / 1024.0**3  # in gigabytes
        #    cur_mem = _profiler._get_max_mem_usage(self.raw_objfn.comm)  # is this what we want??
        #    if self.raw_objfn.mem_limit - cur_mem < persistent_mem:
        #        raise MemoryError("Memory limit ({}-{} GB) is < memory required to hold final results "
        #                          "({} GB)".format(self.raw_objfn.mem_limit * in_gb, cur_mem * in_gb,
        #                                           persistent_mem * in_gb))
        #
        #    self.gthrMem = int(0.1 * (self.raw_objfn.mem_limit - persistent_mem - cur_mem))
        #    evt_mlim = self.raw_objfn.mem_limit - persistent_mem - self.gthrMem - cur_mem
        #    self.raw_objfn.printer.log("Memory limit = %.2fGB" % (self.raw_objfn.mem_limit * in_gb))
        #    self.raw_objfn.printer.log("Cur, Persist, Gather = %.2f, %.2f, %.2f GB" %
        #                               (cur_mem * in_gb, persistent_mem * in_gb, self.gthrMem * in_gb))
        #    assert(evt_mlim > 0), 'Not enough memory, exiting..'
        #else:
        #    evt_mlim = None

        #Note: don't add any tracked memory to self.resource_alloc, as none is used yet.
        self.probs = None
        self.dprobs = None
        self.jac = None
        self.v = None  # for time dependence - rename to objfn_terms or objfn_lsvec?


class MDCObjectiveFunction(ObjectiveFunction, EvaluatedModelDatasetCircuitsStore):
    """
    An objective function whose probabilities and counts are given by a Model and DataSet, respectively.

    Instances of this class glue a model, dataset, and circuit list to a
    "raw" objective function, resulting in an objective function that is a
    function of model-parameters and contains counts based on a data set.

    The model is treated as a function that computes probabilities (as a function of
    the model's parameters) for each circuit outcome, and the data set as a function
    that similarly computes counts (and total-counts).

    Parameters
    ----------
    raw_objfn : RawObjectiveFunction
        The raw objective function - specifies how probability and count values
        are turned into objective function values.

    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.

    Attributes
    ----------
    name : str
        The name of this objective function.

    description : str
        A description of this objective function.
    """

    @classmethod
    def create_from(cls, raw_objfn, model, dataset, circuits, resource_alloc=None, verbosity=0, array_types=()):
        mdc_store = ModelDatasetCircuitsStore(model, dataset, circuits, resource_alloc, array_types)
        return cls(raw_objfn, mdc_store, verbosity)

    @classmethod
    def _array_types_for_method(cls, method_name, fsim):
        if method_name == 'fn': return cls._array_types_for_method('terms', fsim)
        if method_name == 'jacobian': return cls._array_types_for_method('dterms', fsim)
        if method_name == 'terms': return cls._array_types_for_method('lsvec', fsim) + ('e',)  # extra 'E' for **2
        if method_name == 'dterms': return cls._array_types_for_method('dlsvec', fsim) + ('ep',)
        if method_name == 'percircuit': return cls._array_types_for_method('terms', fsim) + ('c',)
        if method_name == 'dpercircuit': return cls._array_types_for_method('dterms', fsim) + ('cp',)
        return ()

    def __init__(self, raw_objfn, mdc_store, verbosity=0):
        """
        Create a new MDCObjectiveFunction.

        mdc_store is thought to be a normal MDC store, but could also be an evaluated one,
        in which case should we take enable_hessian from it?
        """
        EvaluatedModelDatasetCircuitsStore.__init__(self, mdc_store, verbosity)
        self.raw_objfn = raw_objfn

    @property
    def name(self):
        """
        Name of this objective function.
        """
        return self.raw_objfn.name

    @property
    def description(self):
        """
        A description of this objective function.
        """
        return self.raw_objfn.description

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        For instance, if the objective function is DeltaLogL then this function would
        multiply `objective_function_value` by 2, whereas in the case of a chi-squared
        objective function this function just return `objective_function_value`.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return self.raw_objfn.chi2k_distributed_qty(objective_function_value)

    def lsvec(self, paramvec=None, oob_check=False):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        oob_check : bool, optional
            Whether the objective function should raise an error if it is being
            evaluated in an "out of bounds" region.

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def dlsvec(self, paramvec=None):
        """
        The derivative (jacobian) of the least-squares vector.

        Derivatives are taken with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def terms(self, paramvec=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-circuit-outcome values that get summed together
        to result in the objective function value.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        with self.resource_alloc.temporarily_track_memory(self.nelements):  # 'e'
            return self.lsvec(paramvec)**2

    def dterms(self, paramvec=None):
        """
        Compute the jacobian of the terms of the objective function.

        The "terms" are the per-circuit-outcome values that get summed together
        to result in the objective function value.  Differentiation is with
        respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        with self.resource_alloc.temporarily_track_memory(self.nelements * self.nparams):  # 'ep'
            lsvec = self.lsvec(paramvec)  # least-squares objective fn: v is a vector s.t. obj_fn = ||v||^2 (L2 norm)
            dlsvec = self.dlsvec(paramvec)  # jacobian of dim N x M where N = len(v) and M = len(pv)
            assert(dlsvec.shape == (len(lsvec), self.nparams)), "dlsvec returned a Jacobian with the wrong shape!"
            return 2.0 * lsvec[:, None] * dlsvec  # terms = lsvec**2, so dterms = 2*lsvec*dlsvec

    def percircuit(self, paramvec=None):
        """
        Compute the per-circuit contributions to this objective function.

        These values collect (sum) together the contributions of
        the outcomes of a single circuit.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nCircuits,)` where `nCircuits` is the number
            of circuits (specified when this objective function was constructed).
        """
        num_circuits = len(self.circuits)

        with self.resource_alloc.temporarily_track_memory(num_circuits):  # 'c'
            terms = self.terms(paramvec)

            #Aggregate over outcomes:
            # obj_per_el[iElement] contains contributions per element - now aggregate over outcomes
            # percircuit[iCircuit] will contain contributions for each original circuit (aggregated over outcomes)
            percircuit = _np.empty(num_circuits, 'd')
            for i in range(num_circuits):
                percircuit[i] = _np.sum(terms[self.layout.indices_for_index(i)], axis=0)
            return percircuit

    def dpercircuit(self, paramvec=None):
        """
        Compute the jacobian of the per-circuit contributions of this objective function.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nCircuits, nParams)` where `nCircuits` is the number
            of circuits and `nParams` is the number of model parameters (the circuits
            and model were specified when this objective function was constructed).
        """
        num_circuits = len(self.circuits)

        with self.resource_alloc.temporarily_track_memory(num_circuits * self.nparams):  # 'cp'
            dterms = self.dterms(paramvec)

            #Aggregate over outcomes:
            # obj_per_el[iElement] contains contributions per element - now aggregate over outcomes
            # percircuit[iCircuit] will contain contributions for each original circuit (aggregated over outcomes)
            dpercircuit = _np.empty((num_circuits, self.nparams), 'd')
            for i in range(num_circuits):
                dpercircuit[i] = _np.sum(dterms[self.layout.indices_for_index(i)], axis=0)
            return dpercircuit

    def lsvec_percircuit(self, paramvec=None):
        """
        Compute the square root of per-circuit contributions to this objective function.

        These values are primarily useful for interfacing with a least-squares
        optimizer.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nCircuits,)` where `nCircuits` is the number
            of circuits (specified when this objective function was constructed).
        """
        return _np.sqrt(self.percircuit(paramvec))

    def dlsvec_percircuit(self, paramvec=None):
        """
        Compute the jacobian of the sqrt(per-circuit) values given by :method:`lsvec_percircuit`.

        This jacobian is primarily useful for interfacing with a least-squares optimizer.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nCircuits, nParams)` where `nCircuits` is the number
            of circuits and `nParams` is the number of model parameters (the circuits
            and model were specified when this objective function was constructed).
        """
        denom = self.lsvec_percircuit(paramvec)
        denom = _np.clip(denom, 1e-10, None)

        # Note: don't need paramvec here since above call sets it
        return (0.5 / denom)[:, None] * self.dpercircuit()

    def fn_local(self, paramvec=None):
        """
        Evaluate the *local* value of this objective function.

        When the objective function's layout is distributed, each processor only holds a
        portion of the objective function terms, and this function returns only the
        sum of these local terms.  See :method:`fn` for the global objective function value.


        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        float
        """
        return _np.sum(self.terms(paramvec))

    def fn(self, paramvec=None):
        """
        Evaluate the value of this objective function.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        float
        """
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        local = _np.array([self.fn_local(paramvec)], 'd')
        unit_ralloc = self.layout.resource_alloc('atom-processing')  # proc group that computes same els
        self.resource_alloc.allreduce_sum(result, local, unit_ralloc)
        global_fnval = result[0]
        _smt.cleanup_shared_ndarray(result_shm)
        return global_fnval

    def jacobian(self, paramvec=None):
        """
        Compute the Jacobian of this objective function.

        Derivatives are takes with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nParams,)` where `nParams` is the number
            of model parameters.
        """
        return _np.sum(self.dterms(paramvec), axis=0)

    def hessian(self, paramvec=None):
        """
        Compute the Hessian of this objective function.

        Derivatives are takes with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nParams, nParams)` where `nParams` is the number
            of model parameters.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def approximate_hessian(self, paramvec=None):
        """
        Compute an approximate Hessian of this objective function.

        This is typically much less expensive than :method:`hessian` and
        does not require that `enable_hessian=True` was set upon initialization.
        It computes an approximation to the Hessian that only utilizes the
        information in the Jacobian. Derivatives are takes with respect to model
        parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nParams, nParams)` where `nParams` is the number
            of model parameters.
        """
        raise NotImplementedError("Derived classes should implement this!")

    #MOVED - but these versions have updated names
    #def _persistent_memory_estimate(self, num_elements=None):
    #    #  Estimate & check persistent memory (from allocs within objective function)
    #    """
    #    Compute the amount of memory needed to perform evaluations of this objective function.
    #
    #    This number includes both intermediate and final results, and assumes
    #    that the types of evauations given by :method:`_evaltree_subcalls`
    #    are required.
    #
    #    Parameters
    #    ----------
    #    num_elements : int, optional
    #        The number of elements (circuit outcomes) that will be computed.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    if num_elements is None:
    #        nout = int(round(_np.sqrt(self.mdl.dim)))  # estimate of avg number of outcomes per string
    #        nc = len(self.circuits)
    #        ne = nc * nout  # estimate of the number of elements (e.g. probabilities, # LS terms, etc) to compute
    #    else:
    #        ne = num_elements
    #    np = self.mdl.num_params
    #
    #    # "persistent" memory is that used to store the final results.
    #    obj_fn_mem = FLOATSIZE * ne
    #    jac_mem = FLOATSIZE * ne * np
    #    hess_mem = FLOATSIZE * ne * np**2
    #    persistent_mem = 4 * obj_fn_mem + jac_mem  # 4 different objective-function sized arrays, 1 jacobian array?
    #    if any([nm == "bulk_fill_hprobs" for nm in self._evaltree_subcalls()]):
    #        persistent_mem += hess_mem  # we need room for the hessian too!
    #    # TODO: what about "bulk_hprobs_by_block"?
    #
    #    return persistent_mem
    #
    #def _evaltree_subcalls(self):
    #    """
    #    The types of calls that will be made to an evaluation tree.
    #
    #    This information is used for memory estimation purposes.
    #
    #    Returns
    #    -------
    #    list
    #    """
    #    calls = ["bulk_fill_probs", "bulk_fill_dprobs"]
    #    if self.enable_hessian: calls.append("bulk_fill_hprobs")
    #    return calls
    #
    #def num_data_params(self):
    #    """
    #    The number of degrees of freedom in the data used by this objective function.
    #
    #    Returns
    #    -------
    #    int
    #    """
    #    return self.dataset.degrees_of_freedom(self.ds_circuits,
    #                                               aggregate_times=not self.time_dependent)

    #def _precompute_omitted_freqs(self):
    #    """
    #    Detect omitted frequences (assumed to be 0) so we can compute objective fn correctly
    #    """
    #    self.firsts = []; self.indicesOfCircuitsWithOmittedData = []
    #    for i, c in enumerate(self.circuits):
    #        lklen = _slct.length(self.lookup[i])
    #        if 0 < lklen < self.mdl.compute_num_outcomes(c):
    #            self.firsts.append(_slct.to_array(self.lookup[i])[0])
    #            self.indicesOfCircuitsWithOmittedData.append(i)
    #    if len(self.firsts) > 0:
    #        self.firsts = _np.array(self.firsts, 'i')
    #        self.indicesOfCircuitsWithOmittedData = _np.array(self.indicesOfCircuitsWithOmittedData, 'i')
    #        self.dprobs_omitted_rowsum = _np.empty((len(self.firsts), self.nparams), 'd')
    #        self.raw_objfn.printer.log("SPARSE DATA: %d of %d rows have sparse data" %
    #                                   (len(self.firsts), len(self.circuits)))
    #    else:
    #        self.firsts = None  # no omitted probs
    #
    #def _compute_count_vectors(self):
    #    """
    #    Ensure self.cache contains count and total-count vectors.
    #    """
    #    if not self.cache.has_count_vectors():
    #        self.cache.add_count_vectors(self.dataset, self.ds_circuits, self.circuit_weights)
    #    return self.cache.counts, self.cache.total_counts

    def _construct_hessian(self, counts, total_counts, prob_clip_interval):
        """
        Framework for constructing a hessian matrix row by row using a derived
        class's `_hessian_from_hprobs` method.  This function expects that this
        objective function has been setup for hessian computation, and it's evaltree
        may be split in order to facilitate this.
        """
        #Note - we could in the future use comm to distribute over
        # subtrees here.  We currently don't because we parallelize
        # over columns and it seems that in almost all cases of
        # interest there will be more hessian columns than processors,
        # so adding the additional ability to parallelize over
        # subtrees would just add unnecessary complication.

        #Note2: this function follows a similar pattern to DistributableForwardSimulator's
        # _bulk_fill_hprobs method, which will also uses the layout's param_dimension_blk_sizes
        # to divide up the computation of the Hessian of each circuit outcome probability
        # individually.

        blk_size1, blk_size2 = self.layout.param_dimension_blk_sizes
        atom_resource_alloc = self.layout.resource_alloc('atom-processing')
        #param_resource_alloc = self.layout.resource_alloc('param-processing')
        param2_resource_alloc = self.layout.resource_alloc('param2-processing')

        layout = self.layout
        global_param_slice = layout.global_param_slice
        global_param2_slice = layout.global_param2_slice

        my_nparams1 = _slct.length(layout.host_param_slice)  # the number of params this processor is supposed to
        my_nparams2 = _slct.length(layout.host_param2_slice)  # compute (a subset of those its host computes)

        row_parts = int(round(my_nparams1 / blk_size1)) if (blk_size1 is not None) else 1
        col_parts = int(round(my_nparams2 / blk_size2)) if (blk_size2 is not None) else 1
        row_parts = max(row_parts, 1)  # can't have 0 parts!
        col_parts = max(col_parts, 1)

        blocks1 = _mpit.slice_up_range(my_nparams1, row_parts)
        blocks2 = _mpit.slice_up_range(my_nparams2, col_parts)

        blocks1 = [_slct.shift(s, global_param_slice.start) for s in blocks1]
        blocks2 = [_slct.shift(s, global_param2_slice.start) for s in blocks2]
        slicetup_list = list(_itertools.product(blocks1, blocks2))  # *global* parameter indices

        #TODO: see if we can reimplement this 2x speedup - with layout-assigned portions of H the below code won't work
        ##cull out lower triangle blocks, which have no overlap with
        ## the upper triangle of the hessian
        #slicetup_list = [(slc1, slc2) for slc1, slc2 in slicetup_list_all
        #                 if slc1.start <= slc2.stop]  # these are the local "blocks" for this proc

        #UPDATE: use shared memory, so allocate within loop b/c need different shared memory chunks
        # when different processors on same node are give different atoms.
        #  Allocate memory (alloc max required & take views)
        # max_nelements = max([self.layout.atoms[i].num_elements for i in my_atom_indices])
        # probs_mem = _np.empty(max_nelements, 'd')

        rank = self.resource_alloc.comm_rank
        sub_rank = atom_resource_alloc.comm_rank

        with self.resource_alloc.temporarily_track_memory(my_nparams1 * my_nparams2):  # (atom_hessian)
            # Each atom-processor (atom_resource_alloc) contains processors assigned to *disjoint*
            # sections of the Hessian, so these processors can all act simultaneously to fill out a
            # full-size `atom_hessian`.  Then we'll need to add together the contributions from
            # different atom processors at the end.
            atom_hessian = _np.zeros((my_nparams1, my_nparams2), 'd')

            tm = _time.time()

            #Loop over atoms
            for iAtom, atom in enumerate(layout.atoms):  # iterates over *local* atoms
                atom_nelements = atom.num_elements

                if self.raw_objfn.printer.verbosity > 3 or (self.raw_objfn.printer.verbosity == 3 and sub_rank == 0):
                    print("rank %d: %gs: beginning atom %d/%d, atom-size (#circuits) = %d"
                          % (rank, _time.time() - tm, iAtom + 1, len(layout.atoms), atom_nelements))
                    _sys.stdout.flush()

                # Create views into pre-allocated memory
                probs = _np.empty(atom_nelements, 'd')  # Note: this doesn't need to be shared as we never gather it

                # Take portions of count arrays for this subtree
                atom_counts = counts[atom.element_slice]
                atom_total_counts = total_counts[atom.element_slice]
                freqs = atom_counts / atom_total_counts
                assert(len(atom_counts) == len(probs))

                #compute probs separately
                self.model.sim._bulk_fill_probs_atom(probs, atom, atom_resource_alloc)  # need to reach into internals!
                if prob_clip_interval is not None:
                    _np.clip(probs, prob_clip_interval[0], prob_clip_interval[1], out=probs)

                k, kmax = 0, len(slicetup_list)
                blk_rank = param2_resource_alloc.comm_rank
                for (slice1, slice2, hprobs, dprobs12) in self.model.sim._iter_atom_hprobs_by_rectangle(
                        atom, slicetup_list, True, param2_resource_alloc):
                    local_slice1 = _slct.shift(slice1, -global_param_slice.start)  # indices into atom_hessian
                    local_slice2 = _slct.shift(slice2, -global_param2_slice.start)  # indices into atom_hessian

                    if self.raw_objfn.printer.verbosity > 3 or \
                       (self.raw_objfn.printer.verbosity == 3 and blk_rank == 0):
                        print("rank %d: %gs: block %d/%d, atom %d/%d, atom-size (#circuits) = %d"
                              % (self.resource_alloc.comm_rank, _time.time() - tm, k + 1, kmax, iAtom + 1,
                                 len(layout.atoms), atom.num_elements))
                        _sys.stdout.flush(); k += 1

                    hessian_blk = self._hessian_from_block(hprobs, dprobs12, probs, atom_counts,
                                                           atom_total_counts, freqs, param2_resource_alloc)
                    #NOTE: _hessian_from_hprobs MAY modify hprobs and dprobs12
                    #NOTE2: we don't account for memory within _hessian_from_block - maybe we should?

                    atom_hessian[local_slice1, local_slice2] += hessian_blk

        #This shouldn't be necessary:
        ##copy upper triangle to lower triangle (we only compute upper)
        #for i in range(final_hessian.shape[0]):
        #    for j in range(i + 1, final_hessian.shape[1]):
        #        final_hessian[j, i] = final_hessian[i, j]

        return atom_hessian  # (my_nparams1, my_nparams2)

    def _hessian_from_block(self, hprobs, dprobs12, probs, counts, total_counts, freqs, resource_alloc):
        raise NotImplementedError("Derived classes should implement this!")

    def _gather_hessian(self, local_hessian):
        nparams = self.model.num_params
        interatom_ralloc = self.layout.resource_alloc('param2-interatom')
        param2_ralloc = self.layout.resource_alloc('param2-processing')
        my_nparams1, my_nparams2 = local_hessian.shape
        global_param_slice = self.layout.global_param_slice
        global_param2_slice = self.layout.global_param2_slice

        # also could use self.resource_alloc.temporarily_track_memory(self.global_nelements**2): # 'PP'
        with self.resource_alloc.temporarily_track_memory(nparams * nparams):  # 'PP' (final_hessian)
            final_hessian_blk, final_hessian_blk_shm = _smt.create_shared_ndarray(
                interatom_ralloc, (my_nparams1, my_nparams2), 'd')
            final_hessian, final_hessian_shm = _smt.create_shared_ndarray(
                self.resource_alloc, (nparams, nparams), 'd') \
                if self.resource_alloc.host_index == 0 else None

            #We've computed 'local_hessian': the portion of the total hessian assigned to
            # this processor's atom-proc and param-procs (param slices).  Now, we sum all such atom_hessian pieces
            # corresponding to the same param slices (but different atoms).  This is the "param2-interatom" comm.

            #Note: really we just need a reduce_sum here - getting the sum on the root procs is sufficient
            interatom_ralloc.allreduce_sum(final_hessian_blk, local_hessian, unit_ralloc=param2_ralloc)

            if self.resource_alloc.comm is not None:
                self.resource_alloc.comm.barrier()  # make sure reduce call above finishes (needed?)

            #Finally, we need to gather the different pieces on each root param2-interatom proc into the final hessian
            self.resource_alloc.gather(final_hessian, final_hessian_blk, (global_param_slice, global_param2_slice),
                                       unit_ralloc=interatom_ralloc)

            if self.resource_alloc.comm_rank == 0:
                final_hessian_cpy = final_hessian.copy()  # so we don't return shared mem...
            else:
                final_hessian_cpy = None

            if self.resource_alloc.comm is not None:
                self.resource_alloc.host_comm_barrier()  # make sure we don't deallocate too early
            _smt.cleanup_shared_ndarray(final_hessian_blk_shm)
            _smt.cleanup_shared_ndarray(final_hessian_shm)
        return final_hessian_cpy  # (N,N)


#NOTE on chi^2 expressions:
#in general case:   chi^2 = sum (p_i-f_i)^2/p_i  (for i summed over outcomes)
#in 2-outcome case: chi^2 = (p+ - f+)^2/p+ + (p- - f-)^2/p-
#                         = (p - f)^2/p + (1-p - (1-f))^2/(1-p)
#                         = (p - f)^2 * (1/p + 1/(1-p))
#                         = (p - f)^2 * ( ((1-p) + p)/(p*(1-p)) )
#                         = 1/(p*(1-p)) * (p - f)^2

class RawChi2Function(RawObjectiveFunction):
    """
    The function `N(p-f)^2 / p`

    Note that this equals `Nf (1-x)^2 / x` where `x := p/f`.

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """
    def __init__(self, regularization=None, resource_alloc=None, name="chi2", description="Sum of Chi^2", verbosity=0):
        super().__init__(regularization, resource_alloc, name, description, verbosity)

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return objective_function_value

    def set_regularization(self, min_prob_clip_for_weighting=1e-4):
        """
        Set regularization values.

        Parameters
        ----------
        min_prob_clip_for_weighting : float, optional
            Cutoff for probability `prob` in `1 / prob` weighting factor (the maximum
            of `prob` and `min_prob_clip_for_weighting` is used in the denominator).

        Returns
        -------
        None
        """
        self.min_prob_clip_for_weighting = min_prob_clip_for_weighting

    def lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        return (probs - freqs) * self._weights(probs, freqs, total_counts)  # Note: ok if this is negative

    def dlsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        weights = self._weights(probs, freqs, total_counts)
        return weights + (probs - freqs) * self._dweights(probs, freqs, weights)

    def hlsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of `sqrt(local_function)` at each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # lsvec = (p-f)*sqrt(N/cp) = (p-f)*w
        # dlsvec/dp = w + (p-f)*dw/dp
        # d2lsvec/dp2 = dw/dp + (p-f)*d2w/dp2 + dw/dp = 2*dw/dp + (p-f)*d2w/dp2
        weights = self._weights(probs, freqs, total_counts)
        return 2 * self._dweights(probs, freqs, weights) + (probs - freqs) * self._hweights(probs, freqs, weights)

    def hterms_alt(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Alternate computation of the 2nd derivatives of the terms of this objective function.

        This should give exactly the same results as :method:`hterms`, but may be a little faster.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # v = N * (p-f)**2 / p  => dv/dp = 2N * (p-f)/p - N * (p-f)**2 / p**2 = 2N * t - N * t**2
        # => d2v/dp2 = 2N*dt - 2N*t*dt = 2N(1-t)*dt
        cprobs = _np.clip(probs, self.min_prob_clip_for_weighting, None)
        iclip = (cprobs == self.min_prob_clip_for_weighting)
        t = ((probs - freqs) / cprobs)  # should think of as (p-f)/p
        dtdp = (1.0 - t) / cprobs  # 1/p - (p-f)/p**2 => 1/cp - (p-f)/cp**2 = (1-t)/cp
        d2v_dp2 = 2 * total_counts * (1.0 - t) * dtdp
        d2v_dp2[iclip] = 2 * total_counts[iclip] / self.min_prob_clip_for_weighting
        # with cp constant v = N*(p-f)**2/cp => dv/dp = 2N*(p-f)/cp => d2v/dp2 = 2N/cp
        return d2v_dp2

    #Required zero-term methods for omitted probs support in model-based objective functions
    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        clipped_probs = _np.clip(probs, self.min_prob_clip_for_weighting, None)
        return total_counts * probs**2 / clipped_probs

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        clipped_probs = _np.clip(probs, self.min_prob_clip_for_weighting, None)
        return _np.where(probs == clipped_probs, total_counts, 2 * total_counts * probs / clipped_probs)

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        clipped_probs = _np.clip(probs, self.min_prob_clip_for_weighting, None)
        return _np.where(probs == clipped_probs, 0.0, 2 * total_counts / clipped_probs)

    #Support functions
    def _weights(self, p, f, total_counts):
        """
        Get the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        total_counts : numpy.ndarray
            The total counts.

        Returns
        -------
        numpy.ndarray
        """
        cp = _np.clip(p, self.min_prob_clip_for_weighting, None)
        return _np.sqrt(total_counts / cp)  # nSpamLabels x nCircuits array (K x M)

    def _dweights(self, p, f, wts):  # derivative of weights w.r.t. p
        """
        Get the derivative of the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        wts : numpy.ndarray
            The weights, as computed by :method:`_weights`.

        Returns
        -------
        numpy.ndarray
        """
        cp = _np.clip(p, self.min_prob_clip_for_weighting, None)
        dw = -0.5 * wts / cp   # nSpamLabels x nCircuits array (K x M)
        dw[p < self.min_prob_clip_for_weighting] = 0.0
        return dw

    def _hweights(self, p, f, wts):  # 2nd derivative of weights w.r.t. p
        # wts = sqrt(N/cp), dwts = (-1/2) sqrt(N) *cp^(-3/2), hwts = (3/4) sqrt(N) cp^(-5/2)
        """
        Get the 2nd derivative of the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        wts : numpy.ndarray
            The weights, as computed by :method:`_weights`.

        Returns
        -------
        numpy.ndarray
        """
        cp = _np.clip(p, self.min_prob_clip_for_weighting, None)
        hw = 0.75 * wts / cp**2   # nSpamLabels x nCircuits array (K x M)
        hw[p < self.min_prob_clip_for_weighting] = 0.0
        return hw


class RawChiAlphaFunction(RawObjectiveFunction):
    """
    The function `N[x + 1/(alpha * x^alpha) - (1 + 1/alpha)]` where `x := p/f`.

    This function interpolates between the log-likelihood function (alpha=>0)
    and the chi2 function (alpha=1).

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    alpha : float, optional
        The alpha parameter, which lies in the interval (0,1].
    """
    def __init__(self, regularization=None, resource_alloc=None, name="chialpha", description="Sum of ChiAlpha",
                 verbosity=0, alpha=1):
        super().__init__(regularization, resource_alloc, name, description, verbosity)
        self.alpha = alpha

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return objective_function_value

    def set_regularization(self, pfratio_stitchpt=0.01, pfratio_derivpt=0.01, radius=None, fmin=None):
        """
        Set regularization values.

        Parameters
        ----------
        pfratio_stitchpt : float, optional
            The x-value (x = probility/frequency ratio) below which the function is
            replaced with it second-order Taylor expansion.

        pfratio_derivpt : float, optional
            The x-value at which the Taylor expansion derivatives are evaluated at.  If
            this is the same as `pfratio_stitchpt` then the function is smooth to 2nd
            order at this point.  However, choosing a larger value of `pfratio_derivpt`
            will make the stitched part of the function less steep, which is sometimes
            more helpful to an optimizer than having the stitch-point be smooth.

        radius : float, optional
            If `radius` is not None then a "harsh" method of regularizing the zero-frequency
            terms (where the local function = `N*p`) is used.  Specifically, for `p < radius`
            we splice in the cubic polynomial, `-(1/3)*p^3/r^2 + p^2/r + (1/3)*r` (where `r == radius`).
            This has the nice properties that 1) it matches the value, first-derivative,
            and second derivative of `N*p` at `p=r` and 2) it, like `N*p` has a minimum at `p=0`
            with value `0`.  The `radius` dictates the amount of curvature or sharpness of this
            stitching function, with smaller values making the function more pointed.  We recommend
            making this value smaller than the smallest expected frequencies, so as not to alter
            the objective function in regions we near the ML point.  If `radius` is None, then
            `fmin` is used to handle the zero-frequency terms.

        fmin : float, optional
            The minimum expected frequency.  When `radius` is None a "relaxed" regularization of
            the zero-frequency terms is used that stitches the quadratic `N * C * p^2` to `N*p` when
            `p < 1/C`, with `C = 1/(2 fmin) * (1 + alpha) / pfratio_derivpt^(2 + alpha)`.  This
            matches the value of the stitch and `N*p` at `p=1/C` but *not* the derivative, but
            makes up for this by being less steep - the value of `C` is chosen so that the derivative
            (steepness) of the zero-frequency terms at the stitch point is similar to the regular
            nonzero-frequency terms at their stitch points.

        Returns
        -------
        None
        """
        self.x0 = pfratio_stitchpt
        self.x1 = pfratio_derivpt

        if radius is None:
            #Infer the curvature of the regularized zero-f-term functions from
            # the largest curvature we use at the stitch-points of nonzero-f terms.
            assert(fmin is not None), "Must specify 'fmin' when radius is None (should be smalled allowed frequency)."
            self.radius = None
            self.zero_freq_terms = self._zero_freq_terms_relaxed
            self.zero_freq_dterms = self._zero_freq_dterms_relaxed
            self.zero_freq_hterms = None  # no hessian support
            self.fmin = fmin  # = max(1e-7, _np.min(freqs_nozeros))  # lowest non-zero frequency
        else:
            #Use radius to specify the curvature/"roundness" of f == 0 terms,
            # though this uses a more aggressive p^3 function to penalize negative probs.
            self.radius = radius
            self.zero_freq_terms = self._zero_freq_terms_harsh
            self.zero_freq_dterms = self._zero_freq_dterms_harsh
            self.zero_freq_hterms = None  # no hessian support
            self.fmin = None

    def _intermediates(self, probs, counts, total_counts, freqs):
        """ Intermediate values used by both terms(...) and dterms(...) """
        freqs_nozeros = _np.where(counts == 0, 1.0, freqs)
        x = probs / freqs_nozeros
        itaylor = x < self.x0  # indices where we patch objective function with taylor series
        c0 = 1. - 1. / (self.x1**(1 + self.alpha))
        c1 = 0.5 * (1. + self.alpha) / self.x1**(2 + self.alpha)
        return x, itaylor, c0, c1

    def terms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-(probability, count, total-count) values
        that get summed together to result in the objective function value.
        These are the "local" or "per-element" values of the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)

        x0 = self.x0
        x, itaylor, c0, c1 = intermediates
        xt = x.copy(); xt[itaylor] = x0  # so we evaluate expression below at x0 (first taylor term) at itaylor indices
        terms = counts * (xt + 1.0 / (self.alpha * xt**self.alpha) - (1.0 + 1.0 / self.alpha))
        terms = _np.where(itaylor, terms + c0 * counts * (x - x0) + c1 * counts * (x - x0)**2, terms)
        terms = _np.where(counts == 0, self.zero_freq_terms(total_counts, probs), terms)
        return terms

    def dterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)

        x0 = self.x0
        x, itaylor, c0, c1 = intermediates
        dterms = total_counts * (1 - 1. / x**(1. + self.alpha))
        dterms_taylor = total_counts * (c0 + 2 * c1 * (x - x0))
        dterms[itaylor] = dterms_taylor[itaylor]
        dterms = _np.where(counts == 0, self.zero_freq_dterms(total_counts, probs), dterms)
        return dterms

    def hterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of the `local_function` at each
        (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise NotImplementedError("Hessian not implemented for ChiAlpha function yet")

    def hlsvec(self, probs, counts, total_counts, freqs):
        """
        Compute the 2nd derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of `sqrt(local_function)` at each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise NotImplementedError("Hessian not implemented for ChiAlpha function yet")

    #Required zero-term methods for omitted probs support in model-based objective functions
    def _zero_freq_terms_harsh(self, total_counts, probs):
        a = self.radius
        return total_counts * _np.where(probs >= a, probs,
                                        (-1.0 / (3 * a**2)) * probs**3 + probs**2 / a + a / 3.0)

    def _zero_freq_dterms_harsh(self, total_counts, probs):
        a = self.radius
        return total_counts * _np.where(probs >= a, 1.0, (-1.0 / a**2) * probs**2 + 2 * probs / a)

    def _zero_freq_terms_relaxed(self, total_counts, probs):
        c1 = (0.5 / self.fmin) * (1. + self.alpha) / (self.x1**(2 + self.alpha))
        p0 = 1.0 / c1
        return total_counts * _np.where(probs > p0, probs, c1 * probs**2)

    def _zero_freq_dterms_relaxed(self, total_counts, probs):
        c1 = (0.5 / self.fmin) * (1. + self.alpha) / (self.x1**(2 + self.alpha))
        p0 = 1.0 / c1
        return total_counts * _np.where(probs > p0, 1.0, 2 * c1 * probs)


class RawFreqWeightedChi2Function(RawChi2Function):

    """
    The function `N(p-f)^2 / f`

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """
    def __init__(self, regularization=None, resource_alloc=None, name="fwchi2",
                 description="Sum of freq-weighted Chi^2", verbosity=0):
        super().__init__(regularization, resource_alloc, name, description, verbosity)

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return objective_function_value  # default is to assume the value *is* chi2_k distributed

    def set_regularization(self, min_freq_clip_for_weighting=1e-4):
        """
        Set regularization values.

        Parameters
        ----------
        min_freq_clip_for_weighting : float, optional
            The minimum frequency that will be used in the `1/f` weighting factor.
            That is, the weighting factor is the `1 / max(f, min_freq_clip_for_weighting)`.

        Returns
        -------
        None
        """
        self.min_freq_clip_for_weighting = min_freq_clip_for_weighting

    def _weights(self, p, f, total_counts):
        #Note: this could be computed once and cached?
        """
        Get the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        total_counts : numpy.ndarray
            The total counts.

        Returns
        -------
        numpy.ndarray
        """
        return _np.sqrt(total_counts / _np.clip(f, self.min_freq_clip_for_weighting, None))

    def _dweights(self, p, f, wts):
        """
        Get the derivative of the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        wts : numpy.ndarray
            The weights, as computed by :method:`_weights`.

        Returns
        -------
        numpy.ndarray
        """
        return _np.zeros(len(p), 'd')

    def _hweights(self, p, f, wts):
        """
        Get the 2nd derivative of the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        wts : numpy.ndarray
            The weights, as computed by :method:`_weights`.

        Returns
        -------
        numpy.ndarray
        """
        return _np.zeros(len(p), 'd')

    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return total_counts * probs**2 / self.min_freq_clip_for_weighting  # N * p^2 / fmin

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return 2 * total_counts * probs / self.min_freq_clip_for_weighting

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return 2 * total_counts / self.min_freq_clip_for_weighting


class RawCustomWeightedChi2Function(RawChi2Function):

    """
    The function `custom_weight^2 (p-f)^2`, with custom weights that default to 1.

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    custom_weights : numpy.ndarray, optional
        One-dimensional array of the custom weights, which linearly multiply the
        *least-squares* terms, i.e. `(p - f)`.  If `None`, then unit weights are
        used and the objective function computes the sum of unweighted squares.
    """
    def __init__(self, regularization=None, resource_alloc=None, name="cwchi2",
                 description="Sum of custom-weighted Chi^2", verbosity=0, custom_weights=None):
        super().__init__(regularization, resource_alloc, name, description, verbosity)
        self.custom_weights = custom_weights

    def set_regularization(self):
        """
        Set regularization values.

        Returns
        -------
        None
        """
        pass

    def _weights(self, p, f, total_counts):
        #Note: this could be computed once and cached?
        """
        Get the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        total_counts : numpy.ndarray
            The total counts.

        Returns
        -------
        numpy.ndarray
        """
        if self.custom_weights is not None:
            return self.custom_weights
        else:
            return _np.ones(len(p), 'd')

    def _dweights(self, p, f, wts):
        """
        Get the derivative of the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        wts : numpy.ndarray
            The weights, as computed by :method:`_weights`.

        Returns
        -------
        numpy.ndarray
        """
        return _np.zeros(len(p), 'd')

    def _hweights(self, p, f, wts):
        """
        Get the 2nd derivative of the chi2 weighting factor.

        Parameters
        ----------
        p : numpy.ndarray
            The probabilities.

        f : numpy.ndarray
            The frequencies

        wts : numpy.ndarray
            The weights, as computed by :method:`_weights`.

        Returns
        -------
        numpy.ndarray
        """
        return _np.zeros(len(p), 'd')

    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        if self.custom_weights is not None:
            return self.custom_weights**2 * probs**2  # elementwise cw^2 * p^2
        else:
            return probs**2  # p^2

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        if self.custom_weights is not None:
            return 2 * self.custom_weights**2 * probs
        else:
            return 2 * probs  # p^2

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        if self.custom_weights is not None:
            return 2 * self.custom_weights**2
        else:
            return 2 * _np.ones(len(probs))


# The log(Likelihood) within the Poisson picture is:                                                                                                    # noqa
#                                                                                                                                                       # noqa
# L = prod_{i,sl} lambda_{i,sl}^N_{i,sl} e^{-lambda_{i,sl}} / N_{i,sl}!                                                                                 # noqa
#                                                                                                                                                       # noqa
# Where lamba_{i,sl} := p_{i,sl}*N[i] is a rate, i indexes the operation sequence,                                                                      # noqa
#  and sl indexes the spam label.  N[i] is the total counts for the i-th circuit, and                                                                   # noqa
#  so sum_{sl} N_{i,sl} == N[i]. We can ignore the p-independent N_j! and take the log:                                                                 # noqa
#                                                                                                                                                       # noqa
# log L = sum_{i,sl} N_{i,sl} log(N[i]*p_{i,sl}) - N[i]*p_{i,sl}                                                                                        # noqa
#       = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}   (where we ignore the p-independent log(N[i]) terms)                                       # noqa
#                                                                                                                                                       # noqa
# The objective function computes the negative log(Likelihood) as a vector of leastsq                                                                   # noqa
#  terms, where each term == sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} )                                                                        # noqa
#                                                                                                                                                       # noqa
# See LikelihoodFunctions.py for details on patching                                                                                                    # noqa
# The log(Likelihood) within the standard picture is:
#
# L = prod_{i,sl} p_{i,sl}^N_{i,sl}
#
# Where i indexes the operation sequence, and sl indexes the spam label.
#  N[i] is the total counts for the i-th circuit, and
#  so sum_{sl} N_{i,sl} == N[i]. We take the log:
#
# log L = sum_{i,sl} N_{i,sl} log(p_{i,sl})
#
# The objective function computes the negative log(Likelihood) as a vector of leastsq
#  terms, where each term == sqrt( N_{i,sl} * -log(p_{i,sl}) )
#
# See LikelihoodFunction.py for details on patching
class RawPoissonPicDeltaLogLFunction(RawObjectiveFunction):
    """
    The function `N*f*log(f/p) - N*(f-p)`.

    Note that this equals `Nf(-log(x) - 1 + x)` where `x := p/f`.

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """
    def __init__(self, regularization=None,
                 resource_alloc=None, name='dlogl', description="2*Delta(log(L))", verbosity=0):
        super().__init__(regularization, resource_alloc, name, description, verbosity)

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return 2 * objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

    def set_regularization(self, min_prob_clip=1e-4, pfratio_stitchpt=None, pfratio_derivpt=None,
                           radius=1e-4, fmin=None):
        """
        Set regularization values.

        Parameters
        ----------
        min_prob_clip : float, optional
            The probability below which the objective function is replaced with its
            second order Taylor expansion.  This must be `None` if `pfratio_stitchpt`
            is not None, this specifies an alternate stitching method where the
            stitch-point is given in `x=p/f` units.

        pfratio_stitchpt : float, optional
            The x-value (x = probility/frequency ratio) below which the function is
            replaced with it second order Taylor expansion.  Conflicts with
            `min_prob_clip`, which specifies an alternate stitching method.

        pfratio_derivpt : float, optional
            Specified if and only if `pfratio_stitchpt` is.  The x-value at which the
            Taylor expansion derivatives are evaluated at.  If this is the same as
            `pfratio_stitchpt` then the function is smooth to 2nd order at this point.
            However, choosing a larger value of `pfratio_derivpt` will make the stitched
            part of the function less steep, which is sometimes more helpful to an
            optimizer than having the stitch-point be smooth.

        radius : float, optional
            If `radius` is not None then a "harsh" method of regularizing the zero-frequency
            terms (where the local function = `N*p`) is used.  Specifically, for `p < radius`
            we splice in the cubic polynomial, `-(1/3)*p^3/r^2 + p^2/r + (1/3)*r` (where `r == radius`).
            This has the nice properties that 1) it matches the value, first-derivative,
            and second derivative of `N*p` at `p=r` and 2) it, like `N*p` has a minimum at `p=0`
            with value `0`.  The `radius` dictates the amount of curvature or sharpness of this
            stitching function, with smaller values making the function more pointed.  We recommend
            making this value smaller than the smallest expected frequencies, so as not to alter
            the objective function in regions we near the ML point.  If `radius` is None, then
            `fmin` is used to handle the zero-frequency terms.

        fmin : float, optional
            The minimum expected frequency.  When `radius` is None a "relaxed" regularization of
            the zero-frequency terms is used that stitches the quadratic `N * C * p^2` to `N*p` when
            `p < 1/C`, with `C = 1/(2 fmin) * (1 + alpha) / pfratio_derivpt^(2 + alpha)`.  This
            matches the value of the stitch and `N*p` at `p=1/C` but *not* the derivative, but
            makes up for this by being less steep - the value of `C` is chosen so that the derivative
            (steepness) of the zero-frequency terms at the stitch point is similar to the regular
            nonzero-frequency terms at their stitch points.

        Returns
        -------
        None
        """
        if min_prob_clip is not None:
            assert(pfratio_stitchpt is None and pfratio_derivpt is None), \
                "Cannot specify pfratio and min_prob_clip arguments as non-None!"
            self.min_p = min_prob_clip
            self.regtype = "minp"
        else:
            assert(min_prob_clip is None), "Cannot specify pfratio and min_prob_clip arguments as non-None!"
            self.x0 = pfratio_stitchpt
            self.x1 = pfratio_derivpt
            self.regtype = "pfratio"

        if radius is None:
            #Infer the curvature of the regularized zero-f-term functions from
            # the largest curvature we use at the stitch-points of nonzero-f terms.
            assert(self.regtype == 'pfratio'), "Must specify `radius` when %s regularization type" % self.regtype
            assert(fmin is not None), "Must specify 'fmin' when radius is None (should be smalled allowed frequency)."
            self.radius = None
            self.zero_freq_terms = self._zero_freq_terms_relaxed
            self.zero_freq_dterms = self._zero_freq_dterms_relaxed
            self.zero_freq_hterms = self._zero_freq_hterms_relaxed
            self.fmin = fmin  # = max(1e-7, _np.min(freqs_nozeros))  # lowest non-zero frequency
        else:
            #Use radius to specify the curvature/"roundness" of f == 0 terms,
            # though this uses a more aggressive p^3 function to penalize negative probs.
            assert(fmin is None), "Cannot specify 'fmin' when radius is specified."
            self.radius = radius
            self.zero_freq_terms = self._zero_freq_terms_harsh
            self.zero_freq_dterms = self._zero_freq_dterms_harsh
            self.zero_freq_hterms = self._zero_freq_hterms_harsh
            self.fmin = None

    def _intermediates(self, probs, counts, total_counts, freqs):
        """ Intermediate values used by both terms(...) and dterms(...) """
        # Quantities depending on data only (not probs): could be computed once and
        # passed in as arguments to this (and other) functions?
        freqs_nozeros = _np.where(counts == 0, 1.0, freqs)

        if self.regtype == 'pfratio':
            x0 = self.x0
            x1 = self.x1
            x = probs / freqs_nozeros  # objective is -Nf*(log(x) + 1 - x)
            pos_x = _np.where(x < x0, x0, x)
            c0 = counts * (1 - 1 / x1)  # deriv wrt x at x == x1 (=min_p)
            c1 = 0.5 * counts / (x1**2)  # 0.5 * 2nd deriv at x1
            return x, pos_x, c0, c1, freqs_nozeros

        elif self.regtype == 'minp':
            freq_term = counts * (_np.log(freqs_nozeros) - 1.0)
            pos_probs = _np.where(probs < self.min_p, self.min_p, probs)
            c0 = total_counts - counts / self.min_p
            c1 = 0.5 * counts / (self.min_p**2)
            return freq_term, pos_probs, c0, c1, freqs_nozeros

        else:
            raise ValueError("Invalid regularization type: %s" % self.regtype)

    def terms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-(probability, count, total-count) values
        that get summed together to result in the objective function value.
        These are the "local" or "per-element" values of the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)

        if self.regtype == 'pfratio':
            x0 = self.x0
            x, pos_x, c0, c1, _ = intermediates
            terms = -counts * (1.0 - pos_x + _np.log(pos_x))
            #Note: order of +/- terms above is important to avoid roundoff errors when x is near 1.0
            # (see patching line below).  For example, using log(x) + 1 - x causes significant loss
            # of precision because log(x) is tiny and so is |1-x| but log(x) + 1 == 1.0.

            # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
            terms = _np.maximum(terms, 0)
            # quadratic extrapolation of logl at x0 for probabilities/frequencies < x0
            terms = _np.where(x < x0, terms + c0 * (x - x0) + c1 * (x - x0)**2, terms)
            #terms = _np.where(x > 1 / x0, terms + T * (x - x0) + T2 * (x - x0)**2, terms)

        elif self.regtype == 'minp':
            freq_term, pos_probs, c0, c1, _ = intermediates
            terms = freq_term - counts * _np.log(pos_probs) + total_counts * pos_probs

            # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
            terms = _np.maximum(terms, 0)
            # quadratic extrapolation of logl at min_p for probabilities < min_p
            terms = _np.where(probs < self.min_p,
                              terms + c0 * (probs - self.min_p) + c1 * (probs - self.min_p)**2, terms)
        else:
            raise ValueError("Invalid regularization type: %s" % self.regtype)

        terms = _np.where(counts == 0, self.zero_freq_terms(total_counts, probs), terms)
        # special handling for f == 0 terms
        # using cubit rounding of function that smooths N*p for p>0:
        #  has minimum at p=0; matches value, 1st, & 2nd derivs at p=a.

        if terms.size > 0 and _np.min(terms) < 0.0:
            #Since we set terms = _np.maximum(terms, 0) above we know it was the regularization that caused this
            if self.regtype == 'minp':
                raise ValueError(("Regularization => negative terms!  Is min_prob_clip (%g) too large? "
                                  "(it should be smaller than the smallest frequency)") % self.min_p)
            else:
                raise ValueError("Regularization => negative terms!")

        return terms

    def lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        # lsvec = sqrt(terms), but don't use base class fn b/c of special taylor patch...
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        lsvec = _np.sqrt(self.terms(probs, counts, total_counts, freqs, intermediates))

        if self.regtype == "pfratio":  # post-sqrt(v) 1st order taylor patch for x near 1.0 - maybe unnecessary
            freqs_nozeros = _np.where(counts == 0, 1.0, freqs)
            x = probs / freqs_nozeros  # objective is -Nf*(log(x) + 1 - x)
            lsvec = _np.where(_np.abs(x - 1) < 1e-6, _np.sqrt(counts) * _np.abs(x - 1) / _np.sqrt(2), lsvec)

        return lsvec

    def dterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)

        if self.regtype == 'pfratio':
            x0 = self.x0
            x, pos_x, c0, c1, freqs_nozeros = intermediates
            dterms = (total_counts * (-1 / pos_x + 1))
            dterms_taylor = (c0 + 2 * c1 * (x - x0)) / freqs_nozeros
            #dterms_taylor2 = (T + 2 * T2 * (x - x0)) / self.freqs_nozeros
            dterms = _np.where(x < x0, dterms_taylor, dterms)
            #terms = _np.where(x > 1 / x0, dprobs_taylor2, dterms)

        elif self.regtype == 'minp':
            _, pos_probs, c0, c1, freqs_nozeros = intermediates
            dterms = total_counts - counts / pos_probs
            dterms_taylor = c0 + 2 * c1 * (probs - self.min_p)
            dterms = _np.where(probs < self.min_p, dterms_taylor, dterms)

        dterms_zerofreq = self.zero_freq_dterms(total_counts, probs)
        dterms = _np.where(counts == 0, dterms_zerofreq, dterms)
        return dterms

    def hterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of the `local_function` at each
        (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # terms = Nf*(log(f)-log(p)) + N*(p-f)  OR const + S*(p - minp) + S2*(p - minp)^2
        # dterms/dp = -Nf/p + N  OR  c0 + 2*S2*(p - minp)
        # d2terms/dp2 = Nf/p^2   OR  2*S2
        if(self.regtype != "minp"):
            raise NotImplementedError("Hessian only implemented for 'minp' regularization type so far.")

        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        _, pos_probs, c0, c1, freqs_nozeros = intermediates
        d2terms_dp2 = _np.where(probs < self.min_p, 2 * c1, counts / pos_probs**2)
        zfc = _np.where(probs >= self.radius, 0.0,
                        total_counts * ((-2.0 / self.radius**2) * probs + 2.0 / self.radius))
        d2terms_dp2 = _np.where(counts == 0, zfc, d2terms_dp2)
        return d2terms_dp2  # a 1D array of d2(logl)/dprobs2 values; shape = (nEls,)

    #Required zero-term methods for omitted probs support in model-based objective functions
    def _zero_freq_terms_harsh(self, total_counts, probs):
        a = self.radius
        return total_counts * _np.where(probs >= a, probs,
                                        (-1.0 / (3 * a**2)) * probs**3 + probs**2 / a + a / 3.0)

    def _zero_freq_dterms_harsh(self, total_counts, probs):
        a = self.radius
        return total_counts * _np.where(probs >= a, 1.0, (-1.0 / a**2) * probs**2 + 2 * probs / a)

    def _zero_freq_hterms_harsh(self, total_counts, probs):
        a = self.radius
        return total_counts * _np.where(probs >= a, 0.0, (-2.0 / a**2) * probs + 2 / a)

    def _zero_freq_terms_relaxed(self, total_counts, probs):
        # quadratic N*C0*p^2 that == N*p at p=1/C0.
        # Pick C0 so it is ~ magnitude of curvature at patch-pt p/f = x1
        # Note that at d2f/dx2 at x1 is 0.5 N*f / x1^2 so d2f/dp2 = 0.5 (N/f) / x1^2  (dx/dp = 1/f)
        # Thus, we want C0 ~ 0.5(N/f)/x1^2; the largest this value can be is when f=fmin
        c1 = (0.5 / self.fmin) * 1.0 / (self.x1**2)
        p0 = 1.0 / c1
        return total_counts * _np.where(probs > p0, probs, c1 * probs**2)

    def _zero_freq_dterms_relaxed(self, total_counts, probs):
        c1 = (0.5 / self.fmin) * 1.0 / (self.x1**2)
        p0 = 1.0 / c1
        return total_counts * _np.where(probs > p0, 1.0, 2 * c1 * probs)

    def _zero_freq_hterms_relaxed(self, total_counts, probs):
        raise NotImplementedError()  # This is straightforward, but do it later.


class RawDeltaLogLFunction(RawObjectiveFunction):
    """
    The function `N*f*log(f/p)`.

    Note that this equals `-Nf log(x)` where `x := p/f`.

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """
    def __init__(self, regularization=None,
                 resource_alloc=None, name='dlogl', description="2*Delta(log(L))", verbosity=0):
        super().__init__(regularization, resource_alloc, name, description, verbosity)

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return 2 * objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

    def set_regularization(self, min_prob_clip=1e-4, pfratio_stitchpt=None, pfratio_derivpt=None):
        """
        Set regularization values.

        Parameters
        ----------
        min_prob_clip : float, optional
            The probability below which the objective function is replaced with its
            second order Taylor expansion.  This must be `None` if `pfratio_stitchpt`
            is not None, this specifies an alternate stitching method where the
            stitch-point is given in `x=p/f` units.

        pfratio_stitchpt : float, optional
            The x-value (x = probility/frequency ratio) below which the function is
            replaced with it second order Taylor expansion.  Conflicts with
            `min_prob_clip`, which specifies an alternate stitching method.

        pfratio_derivpt : float, optional
            Specified if and only if `pfratio_stitchpt` is.  The x-value at which the
            Taylor expansion derivatives are evaluated at.  If this is the same as
            `pfratio_stitchpt` then the function is smooth to 2nd order at this point.
            However, choosing a larger value of `pfratio_derivpt` will make the stitched
            part of the function less steep, which is sometimes more helpful to an
            optimizer than having the stitch-point be smooth.

        Returns
        -------
        None
        """
        if min_prob_clip is not None:
            assert(pfratio_stitchpt is None and pfratio_derivpt is None), \
                "Cannot specify pfratio and min_prob_clip arguments as non-None!"
            self.min_p = min_prob_clip
            self.regtype = "minp"
        else:
            assert(min_prob_clip is None), "Cannot specify pfratio and min_prob_clip arguments as non-None!"
            self.x0 = pfratio_stitchpt
            self.x1 = pfratio_derivpt
            self.regtype = "pfratio"

    def _intermediates(self, probs, counts, total_counts, freqs):
        """ Intermediate values used by both terms(...) and dterms(...) """
        # Quantities depending on data only (not probs): could be computed once and
        # passed in as arguments to this (and other) functions?
        freqs_nozeros = _np.where(counts == 0, 1.0, freqs)

        if self.regtype == 'pfratio':
            x0 = self.x0
            x1 = self.x1
            x = probs / freqs_nozeros  # objective is -Nf*log(x)
            pos_x = _np.where(x < x0, x0, x)
            c0 = -counts * (1 / x1)  # deriv wrt x at x == x1 (=min_p)
            c1 = 0.5 * counts / (x1**2)  # 0.5 * 2nd deriv at x1
            return x, pos_x, c0, c1, freqs_nozeros

        elif self.regtype == 'minp':
            freq_term = counts * _np.log(freqs_nozeros)  # objective is Nf*(log(f) - log(p))
            pos_probs = _np.where(probs < self.min_p, self.min_p, probs)
            c0 = -counts / self.min_p
            c1 = 0.5 * counts / (self.min_p**2)
            return freq_term, pos_probs, c0, c1, freqs_nozeros

        else:
            raise ValueError("Invalid regularization type: %s" % self.regtype)

    def terms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-(probability, count, total-count) values
        that get summed together to result in the objective function value.
        These are the "local" or "per-element" values of the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)

        if self.regtype == 'pfratio':
            x0 = self.x0
            x, pos_x, c0, c1, freqs_nozeros = intermediates
            terms = -counts * _np.log(pos_x)
            terms = _np.where(x < x0, terms + c0 * (x - x0) + c1 * (x - x0)**2, terms)

        elif self.regtype == 'minp':
            freq_term, pos_probs, c0, c1, _ = intermediates
            terms = freq_term - counts * _np.log(pos_probs)
            terms = _np.where(probs < self.min_p,
                              terms + c0 * (probs - self.min_p) + c1 * (probs - self.min_p)**2, terms)
        else:
            raise ValueError("Invalid regularization type: %s" % self.regtype)

        terms = _np.where(counts == 0, 0.0, terms)
        #Note: no penalty for omitted probabilities (objective fn == 0 whenever counts == 0)
        return terms

    def dterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)

        if self.regtype == 'pfratio':
            x0 = self.x0
            x, pos_x, c0, c1, freqs_nozeros = intermediates
            dterms = total_counts * (-1 / pos_x)  # note Nf/p = N/x
            dterms_taylor = (c0 + 2 * c1 * (x - x0)) / freqs_nozeros  # (...) is df/dx and want df/dp = df/dx * (1/f)
            dterms = _np.where(x < x0, dterms_taylor, dterms)

        elif self.regtype == 'minp':
            _, pos_probs, c0, c1, freqs_nozeros = intermediates
            dterms = -counts / pos_probs
            dterms_taylor = c0 + 2 * c1 * (probs - self.min_p)
            dterms = _np.where(probs < self.min_p, dterms_taylor, dterms)

        dterms = _np.where(counts == 0, 0.0, dterms)
        return dterms

    def hterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of the `local_function` at each
        (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        # terms = Nf*log(p) OR const + S*(p - minp) + S2*(p - minp)^2
        # dterms/dp = Nf/p  OR  c0 + 2*S2*(p - minp)
        # d2terms/dp2 = -Nf/p^2   OR  2*S2
        if(self.regtype != "minp"):
            raise NotImplementedError("Hessian only implemented for 'minp' regularization type so far.")

        if intermediates is None:
            intermediates = self._intermediates(probs, counts, total_counts, freqs)
        _, pos_probs, c0, c1, freqs_nozeros = intermediates
        d2terms_dp2 = _np.where(probs < self.min_p, 2 * c1, counts / pos_probs**2)
        d2terms_dp2 = _np.where(counts == 0, 0.0, d2terms_dp2)
        return d2terms_dp2  # a 1D array of d2(logl)/dprobs2 values; shape = (nEls,)

    def lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        # lsvec = sqrt(terms), but terms are not guaranteed to be positive!
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("LogL objective function cannot produce a LS-vector b/c terms are not necessarily positive!")

    def dlsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("LogL objective function cannot produce a LS-vector b/c terms are not necessarily positive!")

    def dlsvec_and_lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the least-squares vector together with the vector itself.

        This is sometimes more computationally efficient than calling :method:`dlsvec` and
        :method:`lsvec` separately, as the former call may require computing the latter.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        dlsvec: numpy.ndarray
            A 1D array of length equal to that of each array argument.

        lsvec: numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("LogL objective function cannot produce a LS-vector b/c terms are not necessarily positive!")

    def hlsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of `sqrt(local_function)` at each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("LogL objective function cannot produce a LS-vector b/c terms are not necessarily positive!")

    #Required zero-term methods for omitted probs support in model-based objective functions
    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return _np.zeros(len(probs), 'd')

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return _np.zeros(len(probs), 'd')

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return _np.zeros(len(probs), 'd')


class RawMaxLogLFunction(RawObjectiveFunction):
    """
    The function `N*f*log(f)` (note this doesn't depend on the probability!).

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """
    def __init__(self, regularization=None,
                 resource_alloc=None, name='maxlogl', description="Max LogL", verbosity=0, poisson_picture=True):
        super().__init__(regularization, resource_alloc, name, description, verbosity)
        self.poisson_picture = poisson_picture

    def terms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-(probability, count, total-count) values
        that get summed together to result in the objective function value.
        These are the "local" or "per-element" values of the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        freqs_nozeros = _np.where(counts == 0, 1.0, freqs)
        if self.poisson_picture:
            terms = counts * (_np.log(freqs_nozeros) - 1.0)
        else:
            terms = counts * _np.log(freqs_nozeros)
        terms[counts == 0] = 0.0
        return terms

    def dterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        return _np.zeros(len(probs), 'd')

    def hterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of the `local_function` at each
        (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        return _np.zeros(len(probs), 'd')

    def lsvec(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("MaxLogL objective function cannot produce a LS-vector: terms are not necessarily positive!")

    def dlsvec(self, probs, counts, total_counts, freqs):
        """
        Compute the derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("MaxLogL objective function cannot produce a LS-vector: terms are not necessarily positive!")

    def dlsvec_and_lsvec(self, probs, counts, total_counts, freqs):
        """
        Compute the derivatives of the least-squares vector together with the vector itself.

        This is sometimes more computationally efficient than calling :method:`dlsvec` and
        :method:`lsvec` separately, as the former call may require computing the latter.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        dlsvec: numpy.ndarray
            A 1D array of length equal to that of each array argument.

        lsvec: numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("MaxLogL objective function cannot produce a LS-vector: terms are not necessarily positive!")

    def hlsvec(self, probs, counts, total_counts, freqs):
        """
        Compute the 2nd derivatives of the least-squares vector of this objective function.

        Note that because each `lsvec` element only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of `sqrt(local_function)` at each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise ValueError("LogL objective function cannot produce a LS-vector b/c terms are not necessarily positive!")

    #Required zero-term methods for omitted probs support in model-based objective functions
    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return _np.zeros(len(probs), 'd')

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return _np.zeros(len(probs), 'd')

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return _np.zeros(len(probs), 'd')


class RawTVDFunction(RawObjectiveFunction):
    """
    The function `0.5 * |p-f|`.

    Parameters
    ----------
    regularization : dict, optional
        Regularization values.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.
    """
    def __init__(self, regularization=None,
                 resource_alloc=None, name='tvd', description="Total Variational Distance (TVD)", verbosity=0):
        super().__init__(regularization, resource_alloc, name, description, verbosity)

    def terms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-(probability, count, total-count) values
        that get summed together to result in the objective function value.
        These are the "local" or "per-element" values of the objective function.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        return 0.5 * _np.abs(probs - freqs)

    def dterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise derivative (or, the diagonal of a jacobian matrix),
        i.e. the resulting values are the derivatives of the `local_function` at
        each (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise NotImplementedError("Derivatives not implemented for TVD yet!")

    def hterms(self, probs, counts, total_counts, freqs, intermediates=None):
        """
        Compute the 2nd derivatives of the terms of this objective function.

        Note that because each term only depends on the corresponding probability,
        this is just an element-wise 2nd derivative, i.e. the resulting values are
        the 2nd-derivatives of the `local_function` at each
        (probability, count, total-count) value.

        Parameters
        ----------
        probs : numpy.ndarray
            Array of probability values.

        counts : numpy.ndarray
            Array of count values.

        total_counts : numpy.ndarray
            Array of total count values.

        freqs : numpy.ndarray
            Array of frequency values.  This should always equal `counts / total_counts`
            but is supplied separately to increase performance.

        intermediates : tuple, optional
            Used internally to speed up computations.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to that of each array argument.
        """
        raise NotImplementedError("Derivatives not implemented for TVD yet!")

    #Required zero-term methods for omitted probs support in model-based objective functions
    def zero_freq_terms(self, total_counts, probs):
        """
        Evaluate objective function terms with zero frequency (where count and frequency are zero).

        Such terms are treated specially because, for some objective functions,
        having zero frequency is a special case and must be handled differently.

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        return 0.5 * _np.abs(probs)

    def zero_freq_dterms(self, total_counts, probs):
        """
        Evaluate the derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        raise NotImplementedError("Derivatives not implemented for TVD yet!")

    def zero_freq_hterms(self, total_counts, probs):
        """
        Evaluate the 2nd derivative of zero-frequency objective function terms.

        Zero frequency terms are treated specially because, for some objective functions,
        these are a special case and must be handled differently.  Derivatives are
        evaluated element-wise, i.e. the i-th element of the returned array is the
        2nd derivative of the i-th term with respect to the i-th probability (derivatives
        with respect to all other probabilities are zero because of the function structure).

        Parameters
        ----------
        total_counts : numpy.ndarray
            The total counts.

        probs : numpy.ndarray
            The probabilities.

        Returns
        -------
        numpy.ndarray
            A 1D array of the same length as `total_counts` and `probs`.
        """
        raise NotImplementedError("Derivatives not implemented for TVD yet!")


class TimeIndependentMDCObjectiveFunction(MDCObjectiveFunction):
    """
    A time-independent model-based (:class:`MDCObjectiveFunction`-derived) objective function.

    Parameters
    ----------
    raw_objfn : RawObjectiveFunction
        The raw objective function - specifies how probability and count values
        are turned into objective function values.

    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """

    @classmethod
    def builder(cls, name=None, description=None, regularization=None, penalties=None, **kwargs):
        """
        Create an :class:`ObjectiveFunctionBuilder` that builds an objective function of this type.

        Parameters
        ----------
        name : str, optional
            A name for the built objective function (can be anything).

        description : str, optional
            A description for the built objective function (can be anything)

        regularization : dict, optional
            Regularization values.

        penalties : dict, optional
            Penalty values.

        Returns
        -------
        ObjectiveFunctionBuilder
        """
        return ObjectiveFunctionBuilder(cls, name, description, regularization, penalties, **kwargs)

    @classmethod
    def _create_mdc_store(cls, model, dataset, circuits, resource_alloc,
                          method_names=('fn',), array_types=(), verbosity=0):
        # Note: array_types should not include the types used by the created objective function (store) or by the
        # intermediate variables in `method_names` methods.  It is for *additional* arrays, usually the intermediates
        # used within an optimization or other computation that uses this objective function.

        #Array types are used to construct memory estimates (as a function of element number, etc) for layout creation.
        # They account for memory used in:
        #  1) an optimization method (if present),
        #  2a) memory taken by (this) store itself - mirrors allocations in __init__ below.
        #  2b) intermediate memory allocated by methods of the created object (possibly an objective function)
        array_types += cls.compute_array_types(method_names, model.sim)
        return ModelDatasetCircuitsStore(model, dataset, circuits, resource_alloc, array_types, None, verbosity)

    @classmethod
    def create_from(cls, raw_objfn, model, dataset, circuits, resource_alloc=None, penalties=None,
                    verbosity=0, method_names=('fn',), array_types=()):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(raw_objfn, mdc_store, penalties, verbosity)

    @classmethod
    def _array_types_for_method(cls, method_name, fsim):
        #FUTURE: add more from with the raw_objfn calls within each of these fns, e.g. 'lsvec'?
        if method_name == 'lsvec': return fsim._array_types_for_method('bulk_fill_probs') + ('e',)
        if method_name == 'terms': return fsim._array_types_for_method('bulk_fill_probs') + ('e',)
        if method_name == 'dlsvec': return fsim._array_types_for_method('bulk_fill_dprobs') + ('e', 'e')
        if method_name == 'dterms': return fsim._array_types_for_method('bulk_fill_dprobs')
        if method_name == 'hessian_brute': return fsim._array_types_for_method('bulk_fill_hprobs') \
           + ('e', 'e', 'epp', 'epp', 'PP')
        if method_name == 'hessian': return fsim._array_types_for_method('_iter_atom_hprobs_by_rectangle') + ('PP',)
        if method_name == 'approximate_hessian': return fsim._array_types_for_method('bulk_fill_dprobs') + ('e', 'PP')
        return super()._array_types_for_method(method_name, fsim)

    @classmethod
    def compute_array_types(cls, method_names, fsim):
        # array types for "persistent" arrays - those allocated as part of this object
        # (not-intermediate). These are filled in other routines and *not* included in
        # the output of _array_types_for_method since these are *not* allocated in methods.
        array_types = ('e',) * 4  # self.probs + 3x add_count_vectors
        if any([x in ('dlsvec', 'dterms', 'dpercircuit', 'jacobian', 'approximate_hessian', 'hessian')
                for x in method_names]):
            array_types += ('ep',)

        # array types for methods
        for method_name in method_names:
            array_types += cls._array_types_for_method(method_name, fsim)

        return array_types

    def __init__(self, raw_objfn, mdc_store, penalties=None, verbosity=0):

        super().__init__(raw_objfn, mdc_store, verbosity)

        if penalties is None: penalties = {}
        self.ex = self.set_penalties(**penalties)
        self.local_ex = self.ex if self._process_penalties else 0
        # "extra" (i.e. beyond the (circuit,spamlabel)) rows of jacobian

        #Setup underlying EvaluatedModelDatasetCircuitsStore object
        #  Allocate peristent memory - (these are members of EvaluatedModelDatasetCircuitsStore)
        self.initial_allocated_memory = self.resource_alloc.allocated_memory

        #Note: allocate probs as a local array in case we want to gather it (though objfn routines don't need this)
        self.probs = self.layout.allocate_local_array('e', 'd', memory_tracker=self.resource_alloc)
        self.obj = self.layout.allocate_local_array('e', 'd', memory_tracker=self.resource_alloc,
                                                    extra_elements=self.ex)

        self.jac = None
        if ('ep' in self.array_types or 'EP' in self.array_types
           or 'epp' in self.array_types or 'EPP' in self.array_types):
            self.jac = self.layout.allocate_local_array('ep', 'd', memory_tracker=self.resource_alloc,
                                                        extra_elements=self.ex)

        #self.maxCircuitLength = max([len(x) for x in self.circuits])
        # If desired, we may need to make it local to this processor, which may not have data for all of self.circuits

        self.add_count_vectors()  # allocates 3x 'E' arrays
        self.add_omitted_freqs()  # sets self.first and more

    def __del__(self):
        # Reset the allocated memory to the value it had in __init__, effectively releasing the allocations made there.
        self.resource_alloc.reset(allocated_memory=self.initial_allocated_memory)
        self.layout.free_local_array(self.probs)
        self.layout.free_local_array(self.obj)
        self.layout.free_local_array(self.jac)

    #Model-based regularization and penalty support functions
    def set_penalties(self, regularize_factor=0, cptp_penalty_factor=0, spam_penalty_factor=0,
                      errorgen_penalty_factor=0, forcefn_grad=None, shift_fctr=100,
                      prob_clip_interval=(-10000, 1000)):

        """
        Set penalty terms.

        Parameters
        ----------
        regularize_factor : float, optional
            The prefactor of a L1 regularization term that penalizes parameter vector
            elements that exceed an absolute value of 1.0.  Adds a penalty term:
            `regularize_factor * max(0, |parameter_value| - 1.0)` for each model parameter.

        cptp_penalty_factor : float, optional
            The prefactor of a term that penalizes non-CPTP operations.  Specifically, adds a
            `cptp_penalty_factor * sqrt(tracenorm(choi_matrix))` penalty utilizing each operation's
            (gate's) Choi matrix.

        spam_penalty_factor : float, optional
            The prefactor of a term that penalizes invalid SPAM operations.  Specifically, adds a
            `spam_penalty_factor * sqrt(tracenorm(spam_op))` penalty where `spam_op` runs over
            each state preparation's density matrix and each effect vector's matrix.

        errorgen_penalty_factor : float, optional
            The prefactor of a term that penalizes nonzero error generators.  Specifically, adds a
            `errorgen_penalty_factor * sqrt(sum_i(|errorgen_coeff_i|))` penalty where the sum ranges
            over all the error generator coefficients of each model operation.

        forcefn_grad : numpy.ndarray, optional
            The gradient of a "forcing function" that is added to the objective function.  This is
            used in the calculation of linear-response error bars.

        shift_fctr : float, optional
            An adjustment prefactor for computing the "shift" that serves as a constant offset of
            the forcing function, i.e. the forcing function (added to the objective function) is
            essentially `ForceFn = force_shift + dot(forcefn_grad, parameter_vector)`, and
            `force_shift = shift_fctr * ||forcefn_grad|| * (||forcefn_grad|| + ||parameter_vector||)`.
            Here `||` indicates a frobenius norm.  The idea behind all this is that `ForceFn` as
            given above *must* remain positive (for least-squares optimization), and so `shift_fctr`
            must be large enough to ensure this is the case.  Usually you don't need to alter the
            default value.

        prob_clip_interval : tuple, optional
            A `(min, max)` tuple that specifies the minium (possibly negative) and maximum values
            allowed for probabilities generated by the model.  If the model gives probabilities
            outside this range they are clipped to `min` or `max`.  These values can be quite
            generous, as the optimizers are quite tolerant of badly behaved probabilities.

        Returns
        -------
        int
            The number of penalty terms.
        """
        self.regularize_factor = regularize_factor
        self.cptp_penalty_factor = cptp_penalty_factor
        self.spam_penalty_factor = spam_penalty_factor
        self.errorgen_penalty_factor = errorgen_penalty_factor
        self.forcefn_grad = forcefn_grad

        self.prob_clip_interval = prob_clip_interval  # not really a "penalty" per se, but including it as one
        # gives the user the ability to easily set it if they ever need to (unlikely)

        self._process_penalties = self.layout.part_of_final_atom_processor \
            if isinstance(self.layout, _DistributableCOPALayout) else True

        ex = 0  # Compute "extra" number of terms/lsvec-element/rows-of-jacobian beyond evaltree elements

        if forcefn_grad is not None:
            ex += forcefn_grad.shape[0]
            ffg_norm = _np.linalg.norm(forcefn_grad)
            start_norm = _np.linalg.norm(self.model.to_vector())
            self.forceShift = ffg_norm * (ffg_norm + start_norm) * shift_fctr
            #used to keep forceShift - _np.dot(forcefn_grad,paramvec) positive (Note: not analytic, just a heuristic!)

        if self.regularize_factor != 0: ex += self.model.num_params  # = self.global_nparams if layout divides params
        if self.cptp_penalty_factor != 0: ex += _cptp_penalty_size(self.model)
        if self.spam_penalty_factor != 0: ex += _spam_penalty_size(self.model)
        if self.errorgen_penalty_factor != 0: ex += _errorgen_penalty_size(self.model)

        return ex

    def _lspenaltyvec(self, paramvec):
        """
        The least-squares penalty vector, an array of the square roots of the penalty terms.

        Parameters
        ----------
        paramvec : numpy.ndarray
            The vector of (model) parameters to evaluate the objective function at.

        Returns
        -------
        numpy.ndarray
        """
        if self.forcefn_grad is not None:
            force_vec = self.forceShift - _np.dot(self.forcefn_grad, self.model.to_vector())
            assert(_np.all(force_vec >= 0)), "Inadequate forcing shift!"
            forcefn_penalty = _np.sqrt(force_vec)
        else: forcefn_penalty = []

        if self.regularize_factor != 0:
            paramvec_norm = self.regularize_factor * _np.array([max(0, absx - 1.0) for absx in map(abs, paramvec)], 'd')
        else: paramvec_norm = []  # so concatenate ignores

        if self.cptp_penalty_factor > 0:
            cp_penalty_vec = _cptp_penalty(self.model, self.cptp_penalty_factor, self.opBasis)
        else: cp_penalty_vec = []  # so concatenate ignores

        if self.spam_penalty_factor > 0:
            spam_penalty_vec = _spam_penalty(self.model, self.spam_penalty_factor, self.opBasis)
        else: spam_penalty_vec = []  # so concatenate ignores

        if self.errorgen_penalty_factor > 0:
            errorgen_penalty_vec = _errorgen_penalty(self.model, self.errorgen_penalty_factor)
        else:
            errorgen_penalty_vec = []

        return _np.concatenate((forcefn_penalty, paramvec_norm, cp_penalty_vec, spam_penalty_vec, errorgen_penalty_vec))

    def _penaltyvec(self, paramvec):
        """
        The penalty vector, an array of all the penalty terms.

        Parameters
        ----------
        paramvec : numpy.ndarray
            The vector of (model) parameters to evaluate the objective function at.

        Returns
        -------
        numpy.ndarray
        """
        return self._lspenaltyvec(paramvec)**2

    def _fill_lspenaltyvec_jac(self, paramvec, lspenaltyvec_jac):
        """
        Fill `lspenaltyvec_jac` with the jacobian of the least-squares (sqrt of the) penalty vector.

        Parameters
        ----------
        paramvec : numpy.ndarray
            The vector of (model) parameters to evaluate the objective function at.

        lspenaltyvec_jac : numpy.ndarray
            The array to fill.

        Returns
        -------
        None
        """
        off = 0

        # wrtslice gives the subset of all the model parameters that this processor is responsible for
        # computing derivatives with respect to.
        wrtslice = self.layout.global_param_slice if isinstance(self.layout, _DistributableCOPALayout) \
            else slice(0, len(paramvec))  # all params

        if self.forcefn_grad is not None:
            n = self.forcefn_grad.shape[0]
            lspenaltyvec_jac[off:off + n, :] = -self.forcefn_grad
            off += n

        if self.regularize_factor > 0:
            n = len(paramvec)
            lspenaltyvec_jac[off:off + n, :] = _np.diag([(self.regularize_factor * _np.sign(x) if abs(x) > 1.0 else 0.0)
                                                         for x in paramvec[wrtslice]])  # (N,N)
            off += n

        if self.cptp_penalty_factor > 0:
            off += _cptp_penalty_jac_fill(
                lspenaltyvec_jac[off:, :], self.model, self.cptp_penalty_factor, self.opBasis, wrtslice)

        if self.spam_penalty_factor > 0:
            off += _spam_penalty_jac_fill(
                lspenaltyvec_jac[off:, :], self.model, self.spam_penalty_factor, self.opBasis, wrtslice)

        if self.errorgen_penalty_factor > 0:
            off += _errorgen_penalty_jac_fill(
                lspenaltyvec_jac[off:, :], self.model, self.errorgen_penalty_factor, wrtslice)

        assert(off == self.local_ex)

    def _fill_dterms_penalty(self, paramvec, terms_jac):
        """
        Fill `terms_jac` with the jacobian of the penalty vector.

        Parameters
        ----------
        paramvec : numpy.ndarray
            The vector of (model) parameters to evaluate the objective function at.

        terms_jac : numpy.ndarray
            The array to fill.

        Returns
        -------
        None
        """
        # terms_penalty = ls_penalty**2
        # terms_penalty_jac = 2 * ls_penalty * ls_penalty_jac
        self._fill_lspenaltyvec_jac(paramvec, terms_jac)
        terms_jac[:, :] *= 2 * self._lspenaltyvec(paramvec)[:, None]

    #Omitted-probability support functions

    def _omitted_prob_first_terms(self, probs):
        """
        Extracts the value of the first term for each circuit that has omitted probabilities.

        Nonzero probabilities may be predicted for circuit outcomes that
        never occur in the data, and therefore do not produce "terms" for
        the objective function sum.  Yet, in many objective functions, zero-
        frequency terms that have non-zero probabilities still produce a
        non-zero contribution and must be included.  This is performed by
        adding these "omitted-probability" contributions to the first
        (nonzero-frequncy, thus present) term corresponding to the given
        circuit.  This function computes these omitted (zero-frequency)
        terms and returns them in an array of length equal to the number
        of circuits with omitted-probability contributions.

        Parameters
        ----------
        probs : numpy.ndarray
            The (full) vector of probabilities. Length is equal to the
            total number of circuit outcomes (not the length of the
            returned array).

        Returns
        -------
        numpy.ndarray
        """
        omitted_probs = 1.0 - _np.array([_np.sum(probs[self.layout.indices_for_index(i)])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        return self.raw_objfn.zero_freq_terms(self.total_counts[self.firsts], omitted_probs)

    def _update_lsvec_for_omitted_probs(self, lsvec, probs):
        """
        Updates the least-squares vector `lsvec`, adding the omitted-probability contributions.

        Parameters
        ----------
        lsvec : numpy.ndarray
            Vector of least-squares (sqrt of terms) objective function values *before* adding
            omitted-probability contributions.  This function updates this array.

        probs : numpy.ndarray
            The (full) vector of probabilities. Length is equal to the
            total number of circuit outcomes.

        Returns
        -------
        None
        """
        # lsvec = sqrt(terms) => sqrt(terms + zerofreqfn(omitted))
        lsvec[self.firsts] = _np.sqrt(lsvec[self.firsts]**2 + self._omitted_prob_first_terms(probs))

    def _update_terms_for_omitted_probs(self, terms, probs):
        """
        Updates the terms vector `terms`, adding the omitted-probability contributions.

        Parameters
        ----------
        terms : numpy.ndarray
            Vector of objective function term values *before* adding
            omitted-probability contributions.  This function updates this array.

        probs : numpy.ndarray
            The (full) vector of probabilities. Length is equal to the
            total number of circuit outcomes.

        Returns
        -------
        None
        """
        # terms => terms + zerofreqfn(omitted)
        terms[self.firsts] += self._omitted_prob_first_terms(probs)

    def _omitted_prob_first_dterms(self, probs):
        """
        Compute the derivative of the first-terms vector returned by :method:`_omitted_prob_first_terms`.

        This derivative is just with respect to the *probabilities*, not the
        model parameters, as it anticipates a final dot product with the jacobian
        of the computed probabilities with respect to the model parameters (see
        :method:`_update_dterms_for_omitted_probs`).

        Parameters
        ----------
        probs : numpy.ndarray
            The (full) vector of probabilities. Length is equal to the
            total number of circuit outcomes.

        Returns
        -------
        numpy.ndarray
            Vector of the derivatives of the term values with respect
            to the corresponding probability.  As such, this is a 1D
            array of length equal to the number of circuits with omitted
            contributions.
        """
        omitted_probs = 1.0 - _np.array([_np.sum(probs[self.layout.indices_for_index(i)])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        return self.raw_objfn.zero_freq_dterms(self.total_counts[self.firsts], omitted_probs)

    def _update_dterms_for_omitted_probs(self, dterms, probs, dprobs_omitted_rowsum):
        # terms => terms + zerofreqfn(omitted)
        # dterms => dterms + dzerofreqfn(omitted) * domitted  (and domitted = (-omitted_rowsum))
        """
        Updates term jacobian to account for omitted probabilities.

        Parameters
        ----------
        dterms : numpy.ndarray
            Jacobian of terms before and omitted-probability contributions are added.
            This array is updated by this function.

        probs : numpy.ndarray
            The (full) vector of probabilities. Length is equal to the
            total number of circuit outcomes.

        dprobs_omitted_rowsum : numpy.ndarray
            An array of shape `(M,N)` where `M` is the number of circuits with
            omitted contributions and `N` is the number of model parameters.  This
            matrix results from summing up the jacobian rows of all the *present*
            probabilities for the circuit corresponding to the row.  That is, the
            i-th row of this matrix contains the summed-up derivatives of all the
            computed probabilities (i.e. present outcomes) for the i-th circuit with
            omitted probabilities. These omitted probabilities are never computed, but
            are inferred as 1.0 minus the present probabilities, so this matrix gives
            the negative of the derivative of the omitted probabilities.

        Returns
        -------
        None
        """
        dterms[self.firsts] -= self._omitted_prob_first_dterms(probs)[:, None] * dprobs_omitted_rowsum

    def _update_dlsvec_for_omitted_probs(self, dlsvec, lsvec, probs, dprobs_omitted_rowsum):
        """
        Updates least-squares vector's jacobian to account for omitted probabilities.

        Parameters
        ----------
        dlsvec : numpy.ndarray
            Jacobian of least-squares vector before and omitted-probability contributions
            are added.  This array is updated by this function.

        lsvec : numpy.ndarray
            The least-squares vector itself, as this is often helpful in this computation.
            Length is equal to the total number of circuit outcomes.

        probs : numpy.ndarray
            The (full) vector of probabilities. Length is equal to the
            total number of circuit outcomes.

        dprobs_omitted_rowsum : numpy.ndarray
            An array of shape `(M,N)` where `M` is the number of circuits with
            omitted contributions and `N` is the number of model parameters.  This
            matrix results from summing up the jacobian rows of all the *present*
            probabilities for the circuit corresponding to the row.  That is, the
            i-th row of this matrix contains the summed-up derivatives of all the
            computed probabilities (i.e. present outcomes) for the i-th circuit with
            omitted probabilities. These omitted probabilities are never computed, but
            are inferred as 1.0 minus the present probabilities, so this matrix gives
            the negative of the derivative of the omitted probabilities.

        Returns
        -------
        None
        """
        # lsvec = sqrt(terms) => sqrt(terms + zerofreqfn(omitted))
        # dlsvec = 0.5 / sqrt(terms) * dterms = 0.5 / lsvec * dterms
        #          0.5 / sqrt(terms + zerofreqfn(omitted)) * (dterms + dzerofreqfn(omitted) * domitted)
        # so dterms = 2 * lsvec * dlsvec, and
        #    new_dlsvec = 0.5 / sqrt(...) * (2 * lsvec * dlsvec + dzerofreqfn(omitted) * domitted)

        lsvec_firsts = lsvec[self.firsts]
        updated_lsvec = _np.sqrt(lsvec_firsts**2 + self._omitted_prob_first_terms(probs))
        updated_lsvec = _np.where(updated_lsvec == 0, 1.0, updated_lsvec)  # avoid 0/0 where lsvec & deriv == 0

        # dlsvec => 0.5 / updated_lsvec * (2 * lsvec * dlsvec + dzerofreqfn(omitted) * domitted) memory efficient:
        dlsvec[self.firsts] *= (lsvec_firsts / updated_lsvec)[:, None]
        dlsvec[self.firsts] -= ((0.5 / updated_lsvec) * self._omitted_prob_first_dterms(probs))[:, None] \
            * dprobs_omitted_rowsum

    def _clip_probs(self):
        """ Clips the potentially shared-mem self.probs according to self.prob_clip_interval """
        if self.prob_clip_interval is not None:
            if isinstance(self.layout, _DistributableCOPALayout):
                if self.layout.resource_alloc('atom-processing').is_host_leader:
                    _np.clip(self.probs, self.prob_clip_interval[0], self.prob_clip_interval[1], out=self.probs)
                self.layout.resource_alloc('atom-processing').host_comm_barrier()
            else:
                _np.clip(self.probs, self.prob_clip_interval[0], self.prob_clip_interval[1], out=self.probs)

    #Objective Function

    def lsvec(self, paramvec=None, oob_check=False):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        oob_check : bool, optional
            Whether the objective function should raise an error if it is being
            evaluated in an "out of bounds" region.

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """

        #DEBUG REMOVE - used for memory profiling
        #import os, psutil
        #process = psutil.Process(os.getpid())
        #def print_mem_usage(prefix):
        #    print("%s: mem usage = %.3f GB" % (prefix, process.memory_info().rss / (1024.0**3)))

        tm = _time.time()
        if paramvec is not None:
            self.model.from_vector(paramvec)
        else:
            paramvec = self.model.to_vector()
        lsvec = self.obj.view()

        # Whether this rank is the "leader" of all the processors accessing the same shared self.jac and self.probs mem.
        #  Only leader processors should modify the contents of the shared memory, so we only apply operations *once*
        #  `unit_ralloc` is the group of all the procs targeting same destination into self.obj
        unit_ralloc = self.layout.resource_alloc('atom-processing')
        shared_mem_leader = unit_ralloc.is_host_leader

        with self.resource_alloc.temporarily_track_memory(self.nelements):  # 'e' (lsvec)
            self.model.sim.bulk_fill_probs(self.probs, self.layout)  # syncs shared mem
            self._clip_probs()  # clips self.probs in place w/shared mem sync

            if oob_check:  # Only used for termgap cases
                if not self.model.sim.bulk_test_if_paths_are_sufficient(self.layout, self.probs, verbosity=1):
                    raise ValueError("Out of bounds!")  # signals LM optimizer

            if shared_mem_leader:
                lsvec_no_penalty = self.raw_objfn.lsvec(self.probs, self.counts, self.total_counts, self.freqs)
                lsvec[:] = _np.concatenate((lsvec_no_penalty, self._lspenaltyvec(paramvec))) \
                    if self._process_penalties else lsvec_no_penalty

        if self.firsts is not None and shared_mem_leader:
            self._update_lsvec_for_omitted_probs(lsvec, self.probs)
        unit_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

        self.raw_objfn.resource_alloc.profiler.add_time("LS OBJECTIVE", tm)
        assert(lsvec.shape == (self.nelements + self.local_ex,))
        return lsvec

    def terms(self, paramvec=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-circuit-outcome values that get summed together
        to result in the objective function value.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        tm = _time.time()
        if paramvec is not None: self.model.from_vector(paramvec)
        else: paramvec = self.model.to_vector()
        terms = self.obj.view()

        # Whether this rank is the "leader" of all the processors accessing the same shared self.jac and self.probs mem.
        #  Only leader processors should modify the contents of the shared memory, so we only apply operations *once*
        #  `unit_ralloc` is the group of all the procs targeting same destination into self.obj
        unit_ralloc = self.layout.resource_alloc('atom-processing')
        shared_mem_leader = unit_ralloc.is_host_leader

        with self.resource_alloc.temporarily_track_memory(self.nelements):  # 'e' (terms)
            self.model.sim.bulk_fill_probs(self.probs, self.layout)
            self._clip_probs()  # clips self.probs in place w/shared mem sync

            if shared_mem_leader:
                terms_no_penalty = self.raw_objfn.terms(self.probs, self.counts, self.total_counts, self.freqs)
                terms[:] = _np.concatenate((terms_no_penalty, self._penaltyvec(paramvec))) \
                    if self._process_penalties else terms_no_penalty

        if self.firsts is not None and shared_mem_leader:
            self._update_terms_for_omitted_probs(terms, self.probs)
        unit_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

        self.raw_objfn.resource_alloc.profiler.add_time("TERMS OBJECTIVE", tm)
        assert(terms.shape == (self.nelements + self.local_ex,))
        return terms

    # Jacobian function
    def dlsvec(self, paramvec=None):
        """
        The derivative (jacobian) of the least-squares vector.

        Derivatives are taken with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        tm = _time.time()
        dprobs = self.jac[0:self.nelements, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.nelements, self.nparams)
        if paramvec is not None:
            self.model.from_vector(paramvec)
        else:
            paramvec = self.model.to_vector()

        # Whether this rank is the "leader" of all the processors accessing the same shared self.jac and self.probs mem.
        #  Only leader processors should modify the contents of the shared memory, so we only apply operations *once*
        #  `unit_ralloc` is the group of all the procs targeting same destination into self.jac
        unit_ralloc = self.layout.resource_alloc('param-processing')
        shared_mem_leader = unit_ralloc.is_host_leader

        with self.resource_alloc.temporarily_track_memory(2 * self.nelements):  # 'e' (dg_dprobs, lsvec)
            #OLD jac distribution method
            # Usual: rootcomm -this_fn-> atom_comm (sub_comm) -> block_comm
            # New: rootcomm -> jac_slice_comm -this_fn-> atom_comm (sub_comm) -> block_comm  (??)
            # it could be that the comm in resource_alloc is already split for jac distribution, but this
            # would mean code that doesn't compute jacobians would be less efficient.
            # maybe hold a jac_slice_comm that is different?
            # break up:  N procs = Natom_eaters * Nprocs_per_atom_eater
            #            Nprocs_per_atom_eater = Nparamblk_eaters * Nprocs_per_paramblk_eater
            #wrtSlice = resource_alloc.jac_slice if (resource_alloc.jac_distribution_method == "columns") \
            #           else slice(0, self.model.num_params)

            self.model.sim.bulk_fill_dprobs(dprobs, self.layout, self.probs)  # wrtSlice)
            self._clip_probs()  # clips self.probs in place w/shared mem sync

            if shared_mem_leader:
                if self.firsts is not None:
                    for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                        self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.layout.indices_for_index(i), :], axis=0)

                #if shared_mem_leader:  # Note: no need for barrier directly below as barrier further down suffices
                dg_dprobs, lsvec = self.raw_objfn.dlsvec_and_lsvec(self.probs, self.counts, self.total_counts,
                                                                   self.freqs)
                dprobs *= dg_dprobs[:, None]
                # (nelements,N) * (nelements,1)   (N = dim of vectorized model)
                # this multiply also computes jac, which is just dprobs
                # with a different shape (jac.shape == [nelements,nparams])

        if shared_mem_leader:  # only "leader" modifies shared mem (dprobs & self.jac)
            if self.firsts is not None:
                #Note: lsvec is assumed to be *not* updated w/omitted probs contribution
                self._update_dlsvec_for_omitted_probs(dprobs, lsvec, self.probs, self.dprobs_omitted_rowsum)

            if self._process_penalties and shared_mem_leader:
                self._fill_lspenaltyvec_jac(paramvec, self.jac[self.nelements:, :])  # jac.shape == (nelements+N,N)
        unit_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

        # REMOVE => unit tests?
        #if self.check_jacobian: _opt.check_jac(lambda v: self.lsvec(
        #    v), paramvec, self.jac, tol=1e-3, eps=1e-6, errType='abs')  # TO FIX

        # dpr has shape == (nCircuits, nDerivCols), weights has shape == (nCircuits,)
        # return shape == (nCircuits, nDerivCols) where ret[i,j] = dP[i,j]*(weights+dweights*(p-f))[i]
        self.raw_objfn.resource_alloc.profiler.add_time("JACOBIAN", tm)
        return self.jac

    def dterms(self, paramvec=None):
        """
        Compute the jacobian of the terms of the objective function.

        The "terms" are the per-circuit-outcome values that get summed together
        to result in the objective function value.  Differentiation is with
        respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        tm = _time.time()
        dprobs = self.jac[0:self.nelements, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.nelements, self.nparams)
        if paramvec is not None:
            self.model.from_vector(paramvec)
        else:
            paramvec = self.model.to_vector()

        # Whether this rank is the "leader" of all the processors accessing the same shared self.jac and self.probs mem.
        #  Only leader processors should modify the contents of the shared memory, so we only apply operations *once*
        #  `unit_ralloc` is the group of all the procs targeting same destination into self.jac
        unit_ralloc = self.layout.resource_alloc('param-processing')
        shared_mem_leader = unit_ralloc.is_host_leader

        with self.resource_alloc.temporarily_track_memory(2 * self.nelements):  # 'e' (dg_dprobs, lsvec)
            self.model.sim.bulk_fill_dprobs(dprobs, self.layout, self.probs)
            self._clip_probs()  # clips self.probs in place w/shared mem sync

            if shared_mem_leader:
                if self.firsts is not None:
                    for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                        self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.layout.indices_for_index(i), :], axis=0)

                #if shared_mem_leader:  # Note: barrier below work suffices for this condition too
                dprobs *= self.raw_objfn.dterms(self.probs, self.counts, self.total_counts, self.freqs)[:, None]
                # (nelements,N) * (nelements,1)   (N = dim of vectorized model)
                # this multiply also computes jac, which is just dprobs
                # with a different shape (jac.shape == [nelements,nparams])

        if shared_mem_leader:
            if self.firsts is not None:
                self._update_dterms_for_omitted_probs(dprobs, self.probs, self.dprobs_omitted_rowsum)

            if self._process_penalties:
                self._fill_dterms_penalty(paramvec, self.jac[self.nelements:, :])  # jac.shape == (nelements+N,N)
        unit_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

        # REMOVE => unit tests
        #if self.check_jacobian: _opt.check_jac(lambda v: self.lsvec(
        #    v), paramvec, self.jac, tol=1e-3, eps=1e-6, errType='abs')  # TO FIX

        # dpr has shape == (nCircuits, nDerivCols), weights has shape == (nCircuits,)
        # return shape == (nCircuits, nDerivCols) where ret[i,j] = dP[i,j]*(weights+dweights*(p-f))[i]
        self.raw_objfn.resource_alloc.profiler.add_time("JACOBIAN", tm)
        return self.jac

    def hessian_brute(self, paramvec=None):
        """
        Computes the Hessian using a brute force approach.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            The hessian of this objective function, of shape `(N,N)` where `N` is
            the number of model parameters.
        """
        if self.firsts is not None:
            raise NotImplementedError("Brute-force Hessian not implemented for sparse data (yet)")

        # Whether this rank is the "leader" of all the processors accessing the same shared self.jac and self.probs mem.
        #  Only leader processors should modify the contents of the shared memory, so we only apply operations *once*
        unit_ralloc = self.layout.resource_alloc('param2-processing')
        shared_mem_leader = unit_ralloc.is_host_leader

        #General idea of what we're doing:
        # Let f(pv) = g(probs(pv)), and let there be nelements elements (i.e. probabilities) & len(pv) == N
        #  => df/dpv = dg/dprobs * dprobs/dpv = (nelements,) * (nelements,N)
        #  => d2f/dpv1dpv2 = d/dpv2( dg/dprobs * dprobs/dpv1 )
        #                  = (d2g/dprobs2 * dprobs/dpv2) * dprobs/dpv1 + dg/dprobs * d2probs/dpv1dpv2
        #                  =  (KM,)       * (KM,N2)       * (KM,N1)    + (KM,)     * (KM,N1,N2)
        # Note that we need to end up with an array with shape (KM,N1,N2), and so we need to swap axes of first term

        if paramvec is not None: self.model.from_vector(paramvec)
        dprobs = self.jac[0:self.nelements, :]  # avoid mem copying: use jac mem for dprobs
        dprobs2 = self.layout.allocate_local_array('ep2', 'd')
        hprobs = self.layout.allocate_local_array('epp', 'd')
        # Note: dprobs2 is needed because param2 slice may be different

        # 'e', 'epp' (dg_dprobs, d2g_dprobs2, temporary variable dprobs_dp2 * dprobs_dp1 )
        with self.resource_alloc.temporarily_track_memory(2 * self.nelements + self.nelements * self.nparams**2):
            self.model.sim.bulk_fill_hprobs(hprobs, self.layout, self.probs, dprobs, dprobs2)
            self._clip_probs()  # clips self.probs in place w/shared mem sync

            dg_dprobs = self.raw_objfn.dterms(self.probs, self.counts, self.total_counts, self.freqs)[:, None, None]
            d2g_dprobs2 = self.raw_objfn.hterms(self.probs, self.counts, self.total_counts, self.freqs)[:, None, None]
            dprobs_dp1 = dprobs[:, :, None]  # (nelements,N,1)
            dprobs_dp2 = dprobs2[:, None, :]  # (nelements,1,N)

            #hessian = d2g_dprobs2 * dprobs_dp2 * dprobs_dp1 + dg_dprobs * hprobs
            # do the above line in a more memory efficient way:
            hessian = hprobs
            if shared_mem_leader:
                hessian *= dg_dprobs
                hessian += d2g_dprobs2 * dprobs_dp2 * dprobs_dp1
            unit_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

        ret = _np.sum(hessian, axis=0)  # sum over operation sequences and spam labels => (N)
        unit_ralloc.host_comm_barrier()  # ensure sum is performed before we free anything
        self.layout.free_local_array(hprobs)
        self.layout.free_local_array(dprobs2)
        #NOTE: it doesn't seem like we need shared memory at all here - only if we were returning
        # a portion of it... would need a fill_ function to make this useful/needed TODO - remove shared mem above?

        return self._gather_hessian(ret)  # ret is just the part of the Hessian that this processor "owns"

    def approximate_hessian(self, paramvec=None):
        #Almost the same as function above but drops hprobs term
        """
        Compute an approximate Hessian of this objective function.

        This is typically much less expensive than :method:`hessian` and
        does not require that `enable_hessian=True` was set upon initialization.
        It computes an approximation to the Hessian that only utilizes the
        information in the Jacobian. Derivatives are takes with respect to model
        parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nParams, nParams)` where `nParams` is the number
            of model parameters.
        """
        if self.firsts is not None:
            raise NotImplementedError("Chi2 hessian not implemented for sparse data (yet)")

        # Whether this rank is the "leader" of all the processors accessing the same shared self.jac and self.probs mem.
        #  Only leader processors should modify the contents of the shared memory, so we only apply operations *once*
        #  `unit_ralloc` is the group of all the procs targeting same destination into self.jac
        #unit_ralloc = self.layout.resource_alloc('param-processing')
        #shared_mem_leader = unit_ralloc.is_host_leader

        if paramvec is not None: self.model.from_vector(paramvec)
        dprobs = self.jac[0:self.nelements, :]  # avoid mem copying: use jac mem for dprobs

        # 'e', 'pp' (d2g_dprobs2, einsum result )
        with self.resource_alloc.temporarily_track_memory(self.nelements + self.nparams**2):
            self.model.sim.bulk_fill_dprobs(dprobs, self.layout, self.probs)
            self._clip_probs()  # clips self.probs in place w/shared mem sync

            d2g_dprobs2 = self.raw_objfn.hterms(self.probs, self.counts, self.total_counts, self.freqs)  # [:,None,None]
            #dprobs_dp1 = dprobs[:, :, None]  # (nelements,N,1)
            #dprobs_dp2 = dprobs[:, None, :]  # (nelements,1,N)

            #hessian = d2g_dprobs2 * dprobs_dp2 * dprobs_dp1  # this creates a huge array - do this instead:
            hessian = _np.einsum('a,ab,ac->bc', d2g_dprobs2, dprobs, dprobs)

        return self._gather_hessian(hessian)  # `hessian` is just the part of the (approximate) Hessian this proc "owns"

    def hessian(self, paramvec=None):
        """
        Compute the Hessian of this objective function.

        Derivatives are takes with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nParams, nParams)` where `nParams` is the number
            of model parameters.
        """
        if self.ex != 0: raise NotImplementedError("Hessian is not implemented for penalty terms yet!")
        if paramvec is not None: self.model.from_vector(paramvec)
        return self._gather_hessian(self._construct_hessian(self.counts, self.total_counts, self.prob_clip_interval))

    def _hessian_from_block(self, hprobs, dprobs12, probs, counts, total_counts, freqs, resource_alloc):
        """ Factored-out computation of hessian from raw components """

        # Note: hprobs, dprobs12, probs are sometimes shared memory, but the caller (e.g. _construct_hessian)
        # ensures that only the "shared memory leader" processor executes this call.  Thus, we may update
        # hprobs, dprobs12, and probs as we see fit without the need to import additional checks withing this fn.

        #General idea of what we're doing:
        # Let f(pv) = g(probs(pv)), and let there be KM elements (i.e. probabilities) & len(pv) == N
        #  => df/dpv = dg/dprobs * dprobs/dpv = (KM,) * (KM,N)
        #  => d2f/dpv1dpv2 = d/dpv2( dg/dprobs * dprobs/dpv1 )
        #                  = (d2g/dprobs2 * dprobs/dpv2) * dprobs/dpv1 + dg/dprobs * d2probs/dpv1dpv2
        #                  =  (KM,)       * (KM,N2)       * (KM,N1)    + (KM,)     * (KM,N1,N2)
        # so: hessian = d2(raw_objfn)/dprobs2 * dprobs12 + d(raw_objfn)/dprobs * hprobs

        dprobs12_coeffs = self.raw_objfn.hterms(probs, counts, total_counts, freqs)
        hprobs_coeffs = self.raw_objfn.dterms(probs, counts, total_counts, freqs)

        if self.firsts is not None:
            #Allocate these above?  Need to know block sizes of dprobs12 & hprobs...
            dprobs12_omitted_rowsum = _np.empty((len(self.firsts),) + dprobs12.shape[1:], 'd')
            hprobs_omitted_rowsum = _np.empty((len(self.firsts),) + hprobs.shape[1:], 'd')

            omitted_probs = 1.0 - _np.array([_np.sum(probs[self.layout.indices_for_index(i)])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                dprobs12_omitted_rowsum[ii, :, :] = _np.sum(dprobs12[self.layout.indices_for_index(i), :, :], axis=0)
                hprobs_omitted_rowsum[ii, :, :] = _np.sum(hprobs[self.layout.indices_for_index(i), :, :], axis=0)

            dprobs12_omitted_coeffs = -self.raw_objfn.zero_freq_hterms(total_counts[self.firsts], omitted_probs)
            hprobs_omitted_coeffs = -self.raw_objfn.zero_freq_dterms(total_counts[self.firsts], omitted_probs)

        # hessian = hprobs_coeffs * hprobs + dprobs12_coeff * dprobs12
        #  but re-using dprobs12 and hprobs memory (which is overwritten!)
        if resource_alloc.is_host_leader:  # hprobs, dprobs12, and probs are shared among resource_alloc procs
            hprobs *= hprobs_coeffs[:, None, None]
            dprobs12 *= dprobs12_coeffs[:, None, None]
            if self.firsts is not None:
                hprobs[self.firsts, :, :] += hprobs_omitted_coeffs[:, None, None] * hprobs_omitted_rowsum
                dprobs12[self.firsts, :, :] += dprobs12_omitted_coeffs[:, None, None] * dprobs12_omitted_rowsum
            hessian = dprobs12; hessian += hprobs
        else:
            hessian = dprobs12
        resource_alloc.host_comm_barrier()  # let root procs finish their work before moving fwd

        # hessian[iElement,iModelParam1,iModelParams2] contains all d2(logl)/d(modelParam1)d(modelParam2) contributions
        # suming over element dimension gets current subtree contribution for (N,N')-sized block of Hessian
        return _np.sum(hessian, axis=0)


class Chi2Function(TimeIndependentMDCObjectiveFunction):
    """
    Model-based chi-squared function: `N(p-f)^2 / p`

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """
    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None, resource_alloc=None,
                    name=None, description=None, verbosity=0, method_names=('fn',), array_types=()):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0):
        raw_objfn = RawChi2Function(regularization, mdc_store.resource_alloc, name, description, verbosity)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class ChiAlphaFunction(TimeIndependentMDCObjectiveFunction):
    """
    Model-based chi-alpha function: `N[x + 1/(alpha * x^alpha) - (1 + 1/alpha)]` where `x := p/f`.

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.

    alpha : float, optional
        The alpha parameter, which lies in the interval (0,1].
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=(), alpha=1):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity, alpha)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0,
                 alpha=1):
        raw_objfn = RawChiAlphaFunction(regularization, mdc_store.resource_alloc, name, description, verbosity, alpha)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class FreqWeightedChi2Function(TimeIndependentMDCObjectiveFunction):
    """
    Model-based frequency-weighted chi-squared function: `N(p-f)^2 / f`

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=()):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0):
        raw_objfn = RawFreqWeightedChi2Function(regularization, mdc_store.resource_alloc, name, description, verbosity)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class CustomWeightedChi2Function(TimeIndependentMDCObjectiveFunction):
    """
    Model-based custom-weighted chi-squared function: `cw^2 (p-f)^2`

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.

    custom_weights : numpy.ndarray, optional
        One-dimensional array of the custom weights, which linearly multiply the
        *least-squares* terms, i.e. `(p - f)`.  If `None`, then unit weights are
        used and the objective function computes the sum of unweighted squares.
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=(), custom_weights=None):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity, custom_weights)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0,
                 custom_weights=None):
        raw_objfn = RawCustomWeightedChi2Function(regularization, mdc_store.resource_alloc, name, description,
                                                  verbosity, custom_weights)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class PoissonPicDeltaLogLFunction(TimeIndependentMDCObjectiveFunction):
    """
    Model-based poisson-picture delta log-likelihood function: `N*f*log(f/p) - N*(f-p)`.

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=()):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0):
        raw_objfn = RawPoissonPicDeltaLogLFunction(regularization, mdc_store.resource_alloc, name, description,
                                                   verbosity)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class DeltaLogLFunction(TimeIndependentMDCObjectiveFunction):
    """
    Model-based delta log-likelihood function: `N*f*log(f/p)`.

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=()):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0):
        raw_objfn = RawDeltaLogLFunction(regularization, mdc_store.resource_alloc, name, description, verbosity)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class MaxLogLFunction(TimeIndependentMDCObjectiveFunction):
    """
    Model-based maximum-model log-likelihood function: `N*f*log(f)`

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=(), poisson_picture=True):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity, poisson_picture)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0,
                 poisson_picture=True):
        raw_objfn = RawMaxLogLFunction(regularization, mdc_store.resource_alloc, name, description, verbosity,
                                       poisson_picture)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class TVDFunction(TimeIndependentMDCObjectiveFunction):
    """
    Model-based TVD function: `0.5 * |p-f|`.

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    description : str, optional
        A description for this objective function (can be anything)

    verbosity : int, optional
        Level of detail to print to stdout.

    enable_hessian : bool, optional
        Whether hessian calculations are allowed.  If `True` then more resources are
        needed.  If `False`, calls to hessian-requiring function will result in an
        error.
    """

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=()):
        mdc_store = cls._create_mdc_store(model, dataset, circuits, resource_alloc, method_names,
                                          array_types, verbosity)
        return cls(mdc_store, regularization, penalties, name, description, verbosity)

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0):
        raw_objfn = RawTVDFunction(regularization, mdc_store.resource_alloc, name, description, verbosity)
        super().__init__(raw_objfn, mdc_store, penalties, verbosity)


class TimeDependentMDCObjectiveFunction(MDCObjectiveFunction):
    """
    A time-dependent model-based objective function

    Parameters
    ----------
    mdl : Model
        The model - specifies how parameter values are turned into probabilities
        for each circuit outcome.

    dataset : DataSet
        The data set - specifies how counts and total_counts are obtained for each
        circuit outcome.

    circuits : list or CircuitList
        The circuit list - specifies what probabilities and counts this objective
        function compares.  If `None`, then the keys of `dataset` are used.

    regularization : dict, optional
        Regularization values.

    penalties : dict, optional
        Penalty values.  Penalties usually add additional (penalty) terms to the sum
        of per-circuit-outcome contributions that evaluate to the objective function.

    resource_alloc : ResourceAllocation, optional
        Available resources and how they should be allocated for computations.

    name : str, optional
        A name for this objective function (can be anything).

    verbosity : int, optional
        Level of detail to print to stdout.
    """

    @classmethod
    def builder(cls, name=None, description=None, regularization=None, penalties=None, **kwargs):
        """
        Create an :class:`ObjectiveFunctionBuilder` that builds an objective function of this type.

        Parameters
        ----------
        name : str, optional
            A name for the built objective function (can be anything).

        description : str, optional
            A description for the built objective function (can be anything)

        regularization : dict, optional
            Regularization values.

        penalties : dict, optional
            Penalty values.

        Returns
        -------
        ObjectiveFunctionBuilder
        """
        return ObjectiveFunctionBuilder(cls, name, description, regularization, penalties, **kwargs)

    #This objective function can handle time-dependent circuits - that is, circuits are treated as
    # potentially time-dependent and mdl as well.  For now, we don't allow any regularization or penalization
    # in this case.

    @classmethod
    def create_from(cls, model, dataset, circuits, regularization=None, penalties=None,
                    resource_alloc=None, name=None, description=None, verbosity=0,
                    method_names=('fn',), array_types=()):
        #Array types are used to construct memory estimates (as a function of element number, etc) for layout creation.
        # They account for memory used in:
        #  1) an optimization method (if present),
        #  2a) memory taken by (this) store itself - mirrors allocations in __init__ below.
        #  2b) intermediate memory allocated by methods of the created object (possibly an objective function)
        array_types += cls.compute_array_types(method_names, model.sim)
        mdc_store = ModelDatasetCircuitsStore(model, dataset, circuits, resource_alloc, array_types)
        return cls(mdc_store, regularization, penalties, name, description, verbosity)

    @classmethod
    def compute_array_types(cls, method_names, fsim):
        # array types for "persistent" arrays
        array_types = ('e',)
        if any([x in ('dlsvec', 'dterms', 'hessian', 'approximate_hessian')
                for x in method_names]): array_types += ('ep',)

        # array types for methods
        for method_name in method_names:
            array_types += cls._array_types_for_method(method_name, fsim)

        return array_types

    def __init__(self, mdc_store, regularization=None, penalties=None, name=None, description=None, verbosity=0):
        dummy = RawObjectiveFunction({}, mdc_store.resource_alloc, name, description, verbosity)
        super().__init__(dummy, mdc_store, verbosity)

        assert(self.resource_alloc.jac_distribution_method is None), \
            "Fancy jacobian distribution not supported for time-dependent objective fns yet!"
        #Note: this means self.nparams == self.global_nparams

        self.time_dependent = True
        self._ds_cache = {}  # cache for dataset-derived quantities that can improve performance.

        if regularization is None: regularization = {}
        self.set_regularization(**regularization)

        if penalties is None: penalties = {}
        self.ex = self.set_penalties(**penalties)  # "extra" (i.e. beyond the (circuit,spamlabel)) rows of jacobian

        #Setup underlying EvaluatedModelDatasetCircuitsStore object
        #  Allocate peristent memory - (these are members of EvaluatedModelDatasetCircuitsStore)
        self.initial_allocated_memory = self.resource_alloc.allocated_memory
        self.v = self.layout.allocate_local_array('e', 'd', memory_tracker=self.resource_alloc)
        self.jac = None

        if 'ep' in self.array_types:
            self.jac = self.layout.allocate_local_array('ep', 'd', memory_tracker=self.resource_alloc,
                                                        extra_elements=self.ex)

        # If desired, we may need to make it local to this processor, which may not have data for all of self.circuits
        self.num_total_outcomes = [self.model.compute_num_outcomes(c) for c in self.circuits]  # to detect sparse-data

    def __del__(self):
        # Reset the allocated memory to the value it had in __init__, effectively releasing the allocations made there.
        self.resource_alloc.reset(allocated_memory=self.initial_allocated_memory)
        self.layout.free_local_array(self.v)
        self.layout.free_local_array(self.jac)

    def set_regularization(self):
        """
        Set regularization values.

        Returns
        -------
        None
        """
        pass  # no regularization

    def set_penalties(self):
        """
        Set penalty terms.
        """
        return 0  # no penalties

    def lsvec(self, paramvec=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        raise NotImplementedError()

    def dlsvec(self, paramvec=None):
        """
        The derivative (jacobian) of the least-squares vector.

        Derivatives are taken with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        raise NotImplementedError()


class TimeDependentChi2Function(TimeDependentMDCObjectiveFunction):
    """
    Chi-squared function that can handle time-dependent circuits and data.

    This objective function can handle time-dependent circuits - that is, circuits are treated as
    potentially time-dependent and mdl as well.  This function currently doesn't support much
    regularization or penalization.
    """

    @classmethod
    def _array_types_for_method(cls, method_name, fsim):
        if method_name == 'lsvec': return fsim._array_types_for_method('bulk_fill_timedep_chi2')
        if method_name == 'dlsvec': return fsim._array_types_for_method('bulk_fill_timedep_dchi2')
        return super()._array_types_for_method(method_name, fsim)

    def set_regularization(self, min_prob_clip_for_weighting=1e-4, radius=1e-4):
        """
        Set regularization values.

        Parameters
        ----------
        min_prob_clip_for_weighting : float, optional
            Cutoff for probability `prob` in `1 / prob` weighting factor (the maximum
            of `prob` and `min_prob_clip_for_weighting` is used in the denominator).

        radius : float, optional
            Affects sharpness of the regularization of zero-frequency terms.

        Returns
        -------
        None
        """
        self.min_prob_clip_for_weighting = min_prob_clip_for_weighting
        self.radius = radius  # parameterizes "roundness" of f == 0 terms

    def set_penalties(self, regularize_factor=0, cptp_penalty_factor=0, spam_penalty_factor=0,
                      errorgen_penalty_factor=0, prob_clip_interval=(-10000, 10000)):
        """
        Set penalty terms.

        Parameters
        ----------
        regularize_factor : float, optional
            The prefactor of a L1 regularization term that penalizes parameter vector
            elements that exceed an absolute value of 1.0.  Adds a penalty term:
            `regularize_factor * max(0, |parameter_value| - 1.0)` for each model parameter.

        cptp_penalty_factor : float, optional
            The prefactor of a term that penalizes non-CPTP operations.  Specifically, adds a
            `cptp_penalty_factor * sqrt(tracenorm(choi_matrix))` penalty utilizing each operation's
            (gate's) Choi matrix.

        spam_penalty_factor : float, optional
            The prefactor of a term that penalizes invalid SPAM operations.  Specifically, adds a
            `spam_penalty_factor * sqrt(tracenorm(spam_op))` penalty where `spam_op` runs over
            each state preparation's density matrix and each effect vector's matrix.

        errorgen_penalty_factor : float, optional
            The prefactor of a term that penalizes nonzero error generators.  Specifically, adds a
            `errorgen_penalty_factor * sqrt(sum_i(|errorgen_coeff_i|))` penalty where the sum ranges
            over all the error generator coefficients of each model operation.

        prob_clip_interval : tuple, optional
            A `(min, max)` tuple that specifies the minium (possibly negative) and maximum values
            allowed for probabilities generated by the model.  If the model gives probabilities
            outside this range they are clipped to `min` or `max`.  These values can be quite
            generous, as the optimizers are quite tolerant of badly behaved probabilities.

        Returns
        -------
        int
            The number of penalty terms.
        """
        assert(regularize_factor == 0 and cptp_penalty_factor == 0 and spam_penalty_factor == 0
               and errorgen_penalty_factor == 0), \
            "Cannot apply regularization or penalization in time-dependent chi2 case (yet)"
        self.prob_clip_interval = prob_clip_interval
        return 0

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

    def lsvec(self, paramvec=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        tm = _time.time()
        if paramvec is not None: self.model.from_vector(paramvec)
        fsim = self.model.sim
        v = self.v
        fsim.bulk_fill_timedep_chi2(v, self.layout, self.ds_circuits, self.num_total_outcomes,
                                    self.dataset, self.min_prob_clip_for_weighting, self.prob_clip_interval,
                                    self._ds_cache)
        self.raw_objfn.resource_alloc.profiler.add_time("Time-dep chi2: OBJECTIVE", tm)
        assert(v.shape == (self.nelements,))  # reshape ensuring no copy is needed
        return v.copy()  # copy() needed for FD deriv, and we don't need to be stingy w/memory at objective fn level

    def dlsvec(self, paramvec=None):
        """
        The derivative (jacobian) of the least-squares vector.

        Derivatives are taken with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        tm = _time.time()
        dprobs = self.jac.view()  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.nelements, self.nparams)
        if paramvec is not None: self.model.from_vector(paramvec)

        fsim = self.model.sim
        fsim.bulk_fill_timedep_dchi2(dprobs, self.layout, self.ds_circuits, self.num_total_outcomes,
                                     self.dataset, self.min_prob_clip_for_weighting, self.prob_clip_interval, None,
                                     self._ds_cache)

        self.raw_objfn.resource_alloc.profiler.add_time("Time-dep chi2: JACOBIAN", tm)
        return self.jac


class TimeDependentPoissonPicLogLFunction(TimeDependentMDCObjectiveFunction):
    """
    Poisson-picture delta log-likelihood function that can handle time-dependent circuits and data.

    This objective function can handle time-dependent circuits - that is, circuits are treated as
    potentially time-dependent and mdl as well.  This function currently doesn't support much
    regularization or penalization.
    """

    @classmethod
    def _array_types_for_method(cls, method_name, fsim):
        if method_name == 'lsvec': return fsim._array_types_for_method('bulk_fill_timedep_loglpp')
        if method_name == 'dlsvec': return fsim._array_types_for_method('bulk_fill_timedep_dloglpp')
        return super()._array_types_for_method(method_name, fsim)

    def set_regularization(self, min_prob_clip=1e-4, radius=1e-4):
        """
        Set regularization values.

        Parameters
        ----------
        min_prob_clip : float, optional
            The probability below which the objective function is replaced with its
            second order Taylor expansion.

        radius : float, optional
            Affects sharpness of the regularization of zero-frequency terms.

        Returns
        -------
        None
        """
        self.min_prob_clip = min_prob_clip
        self.radius = radius  # parameterizes "roundness" of f == 0 terms

    def set_penalties(self, cptp_penalty_factor=0, spam_penalty_factor=0, errorgen_penalty_factor=0,
                      forcefn_grad=None, shift_fctr=100, prob_clip_interval=(-10000, 10000)):
        """
        Set penalties.

        Parameters
        ----------
        cptp_penalty_factor : float, optional
            The prefactor of a term that penalizes non-CPTP operations.  Specifically, adds a
            `cptp_penalty_factor * sqrt(tracenorm(choi_matrix))` penalty utilizing each operation's
            (gate's) Choi matrix.

        spam_penalty_factor : float, optional
            The prefactor of a term that penalizes invalid SPAM operations.  Specifically, adds a
            `spam_penalty_factor * sqrt(tracenorm(spam_op))` penalty where `spam_op` runs over
            each state preparation's density matrix and each effect vector's matrix.

        errorgen_penalty_factor : float, optional
            The prefactor of a term that penalizes nonzero error generators.  Specifically, adds a
            `errorgen_penalty_factor * sqrt(sum_i(|errorgen_coeff_i|))` penalty where the sum ranges
            over all the error generator coefficients of each model operation.

        forcefn_grad : numpy.ndarray, optional
            The gradient of a "forcing function" that is added to the objective function.  This is
            used in the calculation of linear-response error bars.

        shift_fctr : float, optional
            An adjustment prefactor for computing the "shift" that serves as a constant offset of
            the forcing function, i.e. the forcing function (added to the objective function) is
            essentially `ForceFn = force_shift + dot(forcefn_grad, parameter_vector)`, and
            `force_shift = shift_fctr * ||forcefn_grad|| * (||forcefn_grad|| + ||parameter_vector||)`.
            Here `||` indicates a frobenius norm.  The idea behind all this is that `ForceFn` as
            given above *must* remain positive (for least-squares optimization), and so `shift_fctr`
            must be large enough to ensure this is the case.  Usually you don't need to alter the
            default value.

        prob_clip_interval : tuple, optional
            A `(min, max)` tuple that specifies the minium (possibly negative) and maximum values
            allowed for probabilities generated by the model.  If the model gives probabilities
            outside this range they are clipped to `min` or `max`.  These values can be quite
            generous, as the optimizers are quite tolerant of badly behaved probabilities.

        Returns
        -------
        int
            The number of penalty terms.
        """
        assert(cptp_penalty_factor == 0 and spam_penalty_factor == 0 and errorgen_penalty_factor == 0), \
            "Cannot apply CPTP, SPAM, or errorgen penalization in time-dependent logl case (yet)"
        assert(forcefn_grad is None), "forcing functions not supported with time-dependent logl function yet"
        self.prob_clip_interval = prob_clip_interval
        return 0

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return 2 * objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

    def lsvec(self, paramvec=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        tm = _time.time()
        if paramvec is not None: self.model.from_vector(paramvec)
        fsim = self.model.sim
        v = self.v
        fsim.bulk_fill_timedep_loglpp(v, self.layout, self.ds_circuits, self.num_total_outcomes,
                                      self.dataset, self.min_prob_clip, self.radius, self.prob_clip_interval,
                                      self._ds_cache)
        v = _np.sqrt(v)
        v.shape = [self.nelements]  # reshape ensuring no copy is needed

        self.raw_objfn.resource_alloc.profiler.add_time("Time-dep dlogl: OBJECTIVE", tm)
        return v  # Note: no test for whether probs is in [0,1] so no guarantee that
        #      sqrt is well defined unless prob_clip_interval is set within [0,1].

    #  derivative of  sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} ) terms:
    #   == 0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} ) * ( -N_{i,sl} / p_{i,sl} + N[i] ) * dp
    #  with ommitted correction: sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i] * Y(1-other_ps)) terms (Y is a fn of other ps == omitted_probs)  # noqa
    #   == 0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i]*(1-other_ps) ) * ( -N_{i,sl} / p_{i,sl} + N[i] ) * dp_{i,sl} +                   # noqa
    #      0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i]*(1-other_ps) ) * ( N[i]*dY/dp_j(1-other_ps) ) * -dp_j (for p_j in other_ps)      # noqa

    #  if p <  p_min then term == sqrt( N_{i,sl} * -log(p_min) + N[i] * p_min + S*(p-p_min) )
    #   and deriv == 0.5 / sqrt(...) * c0 * dp
    def dlsvec(self, paramvec=None):
        """
        The derivative (jacobian) of the least-squares vector.

        Derivatives are taken with respect to model parameters.

        Parameters
        ----------
        paramvec : numpy.ndarray, optional
            The vector of (model) parameters to evaluate the objective function at.
            If `None`, then the model's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of model parameters.
        """
        tm = _time.time()
        dlogl = self.jac[0:self.nelements, :]  # avoid mem copying: use jac mem for dlogl
        dlogl.shape = (self.nelements, self.nparams)
        if paramvec is not None: self.model.from_vector(paramvec)
        unit_ralloc = self.layout.resource_alloc('param-processing')
        shared_mem_leader = unit_ralloc.is_host_leader

        fsim = self.model.sim
        fsim.bulk_fill_timedep_dloglpp(dlogl, self.layout, self.ds_circuits, self.num_total_outcomes,
                                       self.dataset, self.min_prob_clip, self.radius, self.prob_clip_interval, self.v,
                                       self._ds_cache)

        # want deriv( sqrt(logl) ) = 0.5/sqrt(logl) * deriv(logl)
        v = _np.sqrt(self.v)
        # derivative should not really diverge as v->0 as v=0 is a minimum with zero derivative
        # so we artificially zero out the derivative whenever v < a small positive value to avoid incorrect
        # limiting whereby v == 0 but the derivative is > 0 (but small, e.g. 1e-7).
        pt5_over_v = _np.where(v < 1e-100, 0.0, 0.5 / _np.maximum(v, 1e-100))  # v=0 is *min* w/0 deriv
        dlogl_factor = pt5_over_v
        if shared_mem_leader:
            dlogl *= dlogl_factor[:, None]  # (nelements,N) * (nelements,1)   (N = dim of vectorized model)

        self.raw_objfn.resource_alloc.profiler.add_time("do_mlgst: JACOBIAN", tm)
        return self.jac


def _cptp_penalty_size(mdl):
    return len(mdl.operations)


def _spam_penalty_size(mdl):
    return len(mdl.preps) + sum([len(povm) for povm in mdl.povms.values()])


def _errorgen_penalty_size(mdl):
    return 1


def _cptp_penalty(mdl, prefactor, op_basis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(mdl.operations).
    """
    return prefactor * _np.sqrt(_np.array([_tools.tracenorm(
        _tools.fast_jamiolkowski_iso_std(gate, op_basis)
    ) for gate in mdl.operations.values()], 'd'))


def _spam_penalty(mdl, prefactor, op_basis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(mdl.operations).
    """
    return prefactor * (_np.sqrt(
        _np.array([
            _tools.tracenorm(
                _tools.vec_to_stdmx(prepvec.to_dense(on_space='minimal'), op_basis)
            ) for prepvec in mdl.preps.values()
        ] + [
            _tools.tracenorm(
                _tools.vec_to_stdmx(mdl.povms[plbl][elbl].to_dense(on_space='minimal'), op_basis)
            ) for plbl in mdl.povms for elbl in mdl.povms[plbl]], 'd')
    ))


def _errorgen_penalty(mdl, prefactor):
    """
    Helper function - errorgen penalty: sum_i |errorgen_coeff_i|
    which in least squares optimization means returning an array
    of the sqrt(sum_i |errorgen_coeff_i|) of each gate.

    Note: error generator coefficients *can* be complex.
    """
    val = 0.0
    #for lbl in mdl.primitive_prep_labels:
    #    op = mdl.circuit_layer_operator(lbl, 'prep')
    #    val += _np.sum(_np.abs(op.errorgen_coefficients_array()))

    for lbl in mdl.primitive_op_labels:
        op = mdl.circuit_layer_operator(lbl, 'op')
        val += _np.sum(_np.abs(op.errorgen_coefficients_array()))

    #for lbl in mdl.primitive_povm_labels:
    #    op = mdl.circuit_layer_operator(lbl, 'povm')
    #    val += _np.sum(_np.abs(op.errorgen_coefficients_array()))

    return prefactor * _np.array([_np.sqrt(val)], 'd')


def _cptp_penalty_jac_fill(cp_penalty_vec_grad_to_fill, mdl, prefactor, op_basis, wrt_slice):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape (len(mdl.operations), n_params).
    """
    from .. import tools as _tools

    # d( sqrt(|chi|_Tr) ) = (0.5 / sqrt(|chi|_Tr)) * d( |chi|_Tr )
    for i, gate in enumerate(mdl.operations.values()):
        nparams = gate.num_params

        #get sgn(chi-matrix) == d(|chi|_Tr)/dchi in std basis
        # so sgnchi == d(|chi_std|_Tr)/dchi_std
        chi = _tools.fast_jamiolkowski_iso_std(gate, op_basis)
        assert(_np.linalg.norm(chi - chi.T.conjugate()) < 1e-4), \
            "chi should be Hermitian!"

        # Alt#1 way to compute sgnchi (evals) - works equally well to svd below
        #evals,U = _np.linalg.eig(chi)
        #sgnevals = [ ev/abs(ev) if (abs(ev) > 1e-7) else 0.0 for ev in evals]
        #sgnchi = _np.dot(U,_np.dot(_np.diag(sgnevals),_np.linalg.inv(U)))

        # Alt#2 way to compute sgnchi (sqrtm) - DOESN'T work well; sgnchi NOT very hermitian!
        #sgnchi = _np.dot(chi, _np.linalg.inv(
        #        _spl.sqrtm(_np.matrix(_np.dot(chi.T.conjugate(),chi)))))

        sgnchi = _tools.matrix_sign(chi)
        assert(_np.linalg.norm(sgnchi - sgnchi.T.conjugate()) < 1e-4), \
            "sgnchi should be Hermitian!"

        # get d(gate)/dp in op_basis [shape == (nP,dim,dim)]
        #OLD: dGdp = mdl.dproduct((gl,)) but wasteful
        dgate_dp = gate.deriv_wrt_params()  # shape (dim**2, nP)
        dgate_dp = _np.swapaxes(dgate_dp, 0, 1)  # shape (nP, dim**2, )
        dgate_dp.shape = (nparams, mdl.dim, mdl.dim)

        # Let M be the "shuffle" operation performed by fast_jamiolkowski_iso_std
        # which maps a gate onto the choi-jamiolkowsky "basis" (i.e. performs that C-J
        # transform).  This shuffle op commutes with the derivative, so that
        # dchi_std/dp := d(M(G))/dp = M(dG/dp), which we call "MdGdp_std" (the choi
        # mapping of dGdp in the std basis)
        m_dgate_dp_std = _np.empty((nparams, mdl.dim, mdl.dim), 'complex')
        for p in range(nparams):  # p indexes param
            m_dgate_dp_std[p] = _tools.fast_jamiolkowski_iso_std(dgate_dp[p], op_basis)  # now "M(dGdp_std)"
            assert(_np.linalg.norm(m_dgate_dp_std[p] - m_dgate_dp_std[p].T.conjugate()) < 1e-8)  # check hermitian

        m_dgate_dp_std = _np.conjugate(m_dgate_dp_std)  # so element-wise multiply
        # of complex number (einsum below) results in separately adding
        # Re and Im parts (also see NOTE in spam_penalty_jac_fill below)

        #contract to get (note contract along both mx indices b/c treat like a
        # mx basis): d(|chi_std|_Tr)/dp = d(|chi_std|_Tr)/dchi_std * dchi_std/dp
        #v =  _np.einsum("ij,aij->a",sgnchi,MdGdp_std)
        v = _np.tensordot(sgnchi, m_dgate_dp_std, ((0, 1), (1, 2)))
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(chi)))  # add 0.5/|chi|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        cp_penalty_vec_grad_to_fill[i, :] = 0.0

        gate_gpinds_subset, within_wrtslice, within_gpinds = _slct.intersect_within(wrt_slice, gate.gpindices)
        cp_penalty_vec_grad_to_fill[i, within_wrtslice] = v.real[within_gpinds]  # indexing w/array OR
        #slice works as expected in this case
        chi = sgnchi = dgate_dp = m_dgate_dp_std = v = None  # free mem

    return len(mdl.operations)  # the number of leading-dim indicies we filled in


def _spam_penalty_jac_fill(spam_penalty_vec_grad_to_fill, mdl, prefactor, op_basis, wrt_slice):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape ( _spam_penalty_size(mdl), n_params).
    """
    basis_mxs = op_basis.elements  # shape [mdl.dim, dmDim, dmDim]
    ddenmx_dv = demx_dv = basis_mxs.conjugate()  # b/c denMx = sum( spamvec[i] * Bmx[i] ) and "V" == spamvec
    #NOTE: conjugate() above is because ddenMxdV and dEMxdV will get *elementwise*
    # multiplied (einsum below) by another complex matrix (sgndm or sgnE) and summed
    # in order to gather the different components of the total derivative of the trace-norm
    # wrt some spam-vector change dV.  If left un-conjugated, we'd get A*B + A.C*B.C (just
    # taking the (i,j) and (j,i) elements of the sum, say) which would give us
    # 2*Re(A*B) = A.r*B.r - B.i*A.i when we *want* (b/c Re and Im parts are thought of as
    # separate, independent degrees of freedom) A.r*B.r + A.i*B.i = 2*Re(A*B.C) -- so
    # we need to conjugate the "B" matrix, which is ddenMxdV or dEMxdV below.

    # d( sqrt(|denMx|_Tr) ) = (0.5 / sqrt(|denMx|_Tr)) * d( |denMx|_Tr )
    for i, prepvec in enumerate(mdl.preps.values()):
        nparams = prepvec.num_params

        #get sgn(denMx) == d(|denMx|_Tr)/d(denMx) in std basis
        # dmDim = denMx.shape[0]
        denmx = _tools.vec_to_stdmx(prepvec, op_basis)
        assert(_np.linalg.norm(denmx - denmx.T.conjugate()) < 1e-4), \
            "denMx should be Hermitian!"

        sgndm = _tools.matrix_sign(denmx)
        assert(_np.linalg.norm(sgndm - sgndm.T.conjugate()) < 1e-4), \
            "sgndm should be Hermitian!"

        # get d(prepvec)/dp in op_basis [shape == (nP,dim)]
        dv_dp = prepvec.deriv_wrt_params()  # shape (dim, nP)
        assert(dv_dp.shape == (mdl.dim, nparams))

        # denMx = sum( spamvec[i] * Bmx[i] )

        #contract to get (note contrnact along both mx indices b/c treat like a mx basis):
        # d(|denMx|_Tr)/dp = d(|denMx|_Tr)/d(denMx) * d(denMx)/d(spamvec) * d(spamvec)/dp
        # [dmDim,dmDim] * [mdl.dim, dmDim,dmDim] * [mdl.dim, nP]
        #v =  _np.einsum("ij,aij,ab->b",sgndm,ddenMxdV,dVdp)
        v = _np.tensordot(_np.tensordot(sgndm, ddenmx_dv, ((0, 1), (1, 2))), dv_dp, (0, 0))
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(denmx)))  # add 0.5/|denMx|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        spam_penalty_vec_grad_to_fill[i, :] = 0.0
        gpinds_subset, within_wrtslice, within_gpinds = _slct.intersect_within(wrt_slice, prepvec.gpindices)
        spam_penalty_vec_grad_to_fill[i, within_wrtslice] = v.real[within_gpinds]  # slice or array index works!
        denmx = sgndm = dv_dp = v = None  # free mem

    #Compute derivatives for effect terms
    i = len(mdl.preps)
    for povmlbl, povm in mdl.povms.items():
        #Simplify effects of povm so we can take their derivatives
        # directly wrt parent Model parameters
        for _, effectvec in povm.simplify_effects(povmlbl).items():
            nparams = effectvec.num_params

            #get sgn(EMx) == d(|EMx|_Tr)/d(EMx) in std basis
            emx = _tools.vec_to_stdmx(effectvec, op_basis)
            # dmDim = EMx.shape[0]
            assert(_np.linalg.norm(emx - emx.T.conjugate()) < 1e-4), \
                "EMx should be Hermitian!"

            sgn_e = _tools.matrix_sign(emx)
            assert(_np.linalg.norm(sgn_e - sgn_e.T.conjugate()) < 1e-4), \
                "sgnE should be Hermitian!"

            # get d(prepvec)/dp in op_basis [shape == (nP,dim)]
            dv_dp = effectvec.deriv_wrt_params()  # shape (dim, nP)
            assert(dv_dp.shape == (mdl.dim, nparams))

            # emx = sum( spamvec[i] * basis_mx[i] )

            #contract to get (note contract along both mx indices b/c treat like a mx basis):
            # d(|EMx|_Tr)/dp = d(|EMx|_Tr)/d(EMx) * d(EMx)/d(spamvec) * d(spamvec)/dp
            # [dmDim,dmDim] * [mdl.dim, dmDim,dmDim] * [mdl.dim, nP]
            #v =  _np.einsum("ij,aij,ab->b",sgnE,dEMxdV,dVdp)
            v = _np.tensordot(_np.tensordot(sgn_e, demx_dv, ((0, 1), (1, 2))), dv_dp, (0, 0))
            v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(emx)))  # add 0.5/|EMx|_Tr factor
            assert(_np.linalg.norm(v.imag) < 1e-4)

            spam_penalty_vec_grad_to_fill[i, :] = 0.0
            gpinds_subset, within_wrtslice, within_gpinds = _slct.intersect_within(wrt_slice, effectvec.gpindices)
            spam_penalty_vec_grad_to_fill[i, within_wrtslice] = v.real[within_gpinds]
            i += 1

            sgn_e = dv_dp = v = None  # free mem

    #return the number of leading-dim indicies we filled in
    return len(mdl.preps) + sum([len(povm) for povm in mdl.povms.values()])


def _errorgen_penalty_jac_fill(errorgen_penalty_vec_grad_to_fill, mdl, prefactor, wrt_slice):

    def _get_local_deriv(op):
        z = op.errorgen_coefficients_array()  # val**2 term is abs(z) = sqrt(z*z.conj)
        dz = op.errorgen_coefficients_array_deriv_wrt_params()
        dterms = 0.5 * (z.conjugate()[:, None] * dz + z[:, None] * dz.conjugate()) / _np.abs(z)[:, None]
        # dterms = 0.5/sqrt(z*z.conj) * (dz * z.conj + z * dz.conj)  -- shape (nCoeffs, nOpParams)
        dsumterms = _np.sum(dterms, axis=0)  # sum over coefficients -- shape (nOpParams,)
        assert(_np.linalg.norm(dsumterms.imag) < 1e-8)
        return prefactor**2 * dsumterms.real  # (2 `prefactor` factors since deriv of val**2)
        #Add these lines if we treat different ops as different LS terms.
        # terms = _np.abs(z)
        # val = _np.sqrt(sum(_np.abs(z)))
        # dval = 0.5 * dsumterms / val[:, None]  # 0.5/sqrt(sum(terms)) * dsumterms
        # return dval

    val = _errorgen_penalty(mdl, prefactor)
    errorgen_penalty_vec_grad_to_fill[0, :] = 0.0

    #for lbl in mdl.primitive_prep_labels:
    #    op = mdl.circuit_layer_operator(lbl, 'prep')
    #    errorgen_penalty_vec_grad_to_fill[0, op.gpindices] += _get_local_deriv(op)

    for lbl in mdl.primitive_op_labels:
        op = mdl.circuit_layer_operator(lbl, 'op')
        gpinds_subset, within_wrtslice, within_gpinds = _slct.intersect_within(wrt_slice, op.gpindices)
        errorgen_penalty_vec_grad_to_fill[0, within_wrtslice] += _get_local_deriv(op)[within_gpinds]

    #for lbl in mdl.primitive_povm_labels:
    #    op = mdl.circuit_layer_operator(lbl, 'povm')
    #    errorgen_penalty_vec_grad_to_fill[0, op.gpindices] += _get_local_deriv(op)

    #Above fills derivative of val**2 = sum(terms), but we want deriv of val = sqrt(sum(terms)):
    errorgen_penalty_vec_grad_to_fill[0, :] *= 0.5 / val  # final == 1/sqrt(sum(terms)) * dsumterms

    #return the number of leading-dim indicies we filled in
    return 1


class LogLWildcardFunction(ObjectiveFunction):

    """
    A wildcard-budget bolt-on to an existing objective function.

    Currently, this existing function must be a log-likelihood type
    function because the computational logic assumes this.  The
    resulting object is an objective function over the space of
    wildcard budget parameter vectors (not model parameters).

    Parameters
    ----------
    logl_objective_fn : PoissonPicDeltaLogLFunction
        The bare log-likelihood function.

    base_pt : numpy.ndarray
        Unused.  The model-paramter vector where this objective function is based.

    wildcard : WildcardBudget
        The wildcard budget that adjusts the "bare" probabilities of
        `logl_objective_fn` before evaluating the rest of the objective function.
    """
    def __init__(self, logl_objective_fn, base_pt, wildcard):
        #TODO: remove base_pt -- it ends up not being needed (?)
        self.logl_objfn = logl_objective_fn
        self.basept = base_pt
        self.wildcard_budget = wildcard
        self.wildcard_budget_precomp = wildcard.precompute_for_same_circuits(self.logl_objfn.circuits)
        self.description = logl_objective_fn.description + " + wildcard budget"

        #assumes self.logl_objfn.fn(...) was called to initialize the members of self.logl_objfn
        self.logl_objfn.resource_alloc.add_tracked_memory(self.logl_objfn.probs.size)
        self.probs = self.logl_objfn.probs.copy()

    #def _default_evalpt(self):
    #    """The default point to evaluate functions at """
    #    return self.wildcard_budget.to_vector()

    #Mimic the underlying LogL objective
    def __getattr__(self, attr):
        return getattr(self.__dict__['logl_objfn'], attr)  # use __dict__ so no chance for recursive __getattr__

    def chi2k_distributed_qty(self, objective_function_value):
        """
        Convert a value of this objective function to one that is expected to be chi2_k distributed.

        Parameters
        ----------
        objective_function_value : float
            A value of this objective function, i.e. one returned from `self.fn(...)`.

        Returns
        -------
        float
        """
        return self.logl_objfn.chi2k_distributed_qty(objective_function_value)

    def fn(self, wvec=None):
        """
        Evaluate this objective function.

        Parameters
        ----------
        wvec : numpy.ndarray, optional
            The vector of (wildcard budget) parameters to evaluate the objective function at.
            If `None`, then the budget's current parameter vector is used (held internally).

        Returns
        -------
        float
        """
        return sum(self.terms(wvec))

    def terms(self, wvec=None):
        """
        Compute the terms of the objective function.

        The "terms" are the per-circuit-outcome values that get summed together
        to result in the objective function value.

        Parameters
        ----------
        wvec : numpy.ndarray, optional
            The vector of (wildcard budget) parameters to evaluate the objective function at.
            If `None`, then the budget's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        return self.lsvec(wvec)**2

    def lsvec(self, wvec=None):
        """
        Compute the least-squares vector of the objective function.

        This is the square-root of the terms-vector returned from :method:`terms`.
        This vector is the objective function value used by a least-squares
        optimizer when optimizing this objective function.  Note that the existence
        of this quantity requires that the terms be non-negative.  If this is not
        the case, an error is raised.

        Parameters
        ----------
        wvec : numpy.ndarray, optional
            The vector of (wildcard budget) parameters to evaluate the objective function at.
            If `None`, then the budget's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,)` where `nElements` is the number
            of circuit outcomes.
        """
        if wvec is not None: self.wildcard_budget.from_vector(wvec)
        self.wildcard_budget.update_probs(self.probs,
                                          self.logl_objfn.probs,
                                          self.logl_objfn.freqs,
                                          self.logl_objfn.layout,
                                          self.wildcard_budget_precomp)

        counts, N, freqs = self.logl_objfn.counts, self.logl_objfn.total_counts, self.logl_objfn.freqs
        return self.logl_objfn.raw_objfn.lsvec(self.logl_objfn.probs, counts, N, freqs)

    def dlsvec(self, wvec):
        """
        The derivative (jacobian) of the least-squares vector.

        Derivatives are taken with respect to wildcard budget parameters.

        Parameters
        ----------
        wvec : numpy.ndarray, optional
            The vector of (wildcard budget) parameters to evaluate the objective function at.
            If `None`, then the budget's current parameter vector is used (held internally).

        Returns
        -------
        numpy.ndarray
            An array of shape `(nElements,nParams)` where `nElements` is the number
            of circuit outcomes and `nParams` is the number of wildcard budget parameters.
        """
        raise NotImplementedError("No jacobian yet")


class CachedObjectiveFunction(_NicelySerializable):
    """
    Holds various values of an objective function at a particular point.

    This object is meant to be serializable, and not to facilitate computing the
    objective function anywhere else.  It doesn't contain or rely on comm objects.
    It does contain a *serial* or "global" layout that allows us to make sense of
    the elements of various probability vectors, etc., but we demand that this
    layout not depend on comm objects.

    The cache may only have values on the rank-0 proc (??)
    """
    collection_name = "pygsti_cached_objective_fns"

    @classmethod
    def from_dir(cls, dirname, quick_load=False):
        """
        Initialize a new CachedObjectiveFunction object from `dirname`.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Parameters
        ----------
        dirname : str
            The directory name.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Returns
        -------
        CachedObjectiveFunction
        """
        import pygsti.io as _io
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types', quick_load=quick_load))
        _NicelySerializable.__init__(ret)
        return ret

    @classmethod
    def _create_obj_from_doc_and_mongodb(cls, doc, mongodb, quick_load=False):
        import pygsti.io as _io
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.read_auxtree_from_mongodb_doc(mongodb, doc, 'auxfile_types', quick_load=quick_load))
        _NicelySerializable.__init__(ret)
        return ret

    def __init__(self, objective_function):
        super().__init__()
        self.layout = objective_function.layout.global_layout
        self.model_paramvec = objective_function.model.to_vector().copy()

        objfn_layout = objective_function.layout
        self.fn = objective_function.fn()  # internally gathers to all ranks
        self.chi2k_distributed_fn = objective_function.chi2k_distributed_qty(self.fn)

        local_terms = objective_function.terms()
        self.terms = objfn_layout.allgather_local_array('e', local_terms)
        self.chi2k_distributed_terms = objective_function.chi2k_distributed_qty(self.terms)
        self.num_elements = len(self.terms)

        #local_percircuit = objective_function.percircuit()
        #self.percircuit = objfn_layout.allgather_local_array('c', local_percircuit)
        self.num_circuits = len(self.layout.circuits)
        self.percircuit = _np.empty(self.num_circuits, 'd')
        for i in range(self.num_circuits):
            self.percircuit[i] = _np.sum(self.terms[self.layout.indices_for_index(i)], axis=0)
        self.chi2k_distributed_percircuit = objective_function.chi2k_distributed_qty(self.percircuit)

        if isinstance(objective_function, TimeIndependentMDCObjectiveFunction):
            self.probs = objfn_layout.allgather_local_array('e', objective_function.probs)
            self.freqs = objfn_layout.allgather_local_array('e', objective_function.freqs)
            self.counts = objfn_layout.allgather_local_array('e', objective_function.counts)
            self.total_counts = objfn_layout.allgather_local_array('e', objective_function.total_counts)
        else:
            self.probs = self.freqs = self.counts = self.total_counts = None

        self.auxfile_types = {'layout': 'serialized-object',
                              'model_paramvec': 'numpy-array',
                              'fn': 'numpy-array',
                              'chi2k_distributed_fn': 'numpy-array',
                              'terms': 'numpy-array',
                              'chi2k_distributed_terms': 'numpy-array',
                              'percircuit': 'numpy-array',
                              'chi2k_distributed_percircuit': 'numpy-array',
                              'probs': 'numpy-array',
                              'freqs': 'numpy-array',
                              'counts': 'numpy-array',
                              'total_counts': 'numpy-array'
                              }

    def write(self, dirname):
        """
        Write this CachedObjectiveFunction to a directory.

        Parameters
        ----------
        dirname : str
            The directory name to write.  This directory will be created
            if needed, and the files in an existing directory will be
            overwritten.

        Returns
        -------
        None
        """
        import pygsti.io as _io
        _io.write_obj_to_meta_based_dir(self, dirname, 'auxfile_types')

    def _add_auxiliary_write_ops_and_update_doc(self, doc, write_ops, mongodb, collection_name, overwrite_existing):
        import pygsti.io as _io
        _io.add_obj_auxtree_write_ops_and_update_doc(self, doc, write_ops, mongodb, collection_name,
                                                     'auxfile_types', overwrite_existing=overwrite_existing)

    @classmethod
    def _remove_from_mongodb(cls, mongodb, collection_name, doc_id, session, recursive):
        import pygsti.io as _io
        _io.remove_auxtree_from_mongodb(mongodb, collection_name, doc_id, 'auxfile_types', session,
                                        recursive=recursive)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'description': self.description,
                      'class_to_build': self.cls_to_build.__module__ + '.' + self.cls_to_build.__name__,
                      'regularization': self.regularization,
                      'penalties': self.penalties,
                      'additional_arguments': self.additional_args,
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.io.metadir import _class_for_name
        return cls(_class_for_name(state['class_to_build']), state['name'], state['description'],
                   state['regularization'], state['penalties'], *state['additional_arguments'])
