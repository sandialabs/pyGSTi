## user-exposure: low (EGN - using these objects directly is power-user functionality, though the ability of an OutcomeLabelDict to act like a normal dict is "high" exposure.)
"""
Defines the OutcomeLabelDict class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import collections as _collections
import copy as _copy


class OutcomeLabelDict(_collections.OrderedDict):
    """
    An ordered dictionary of outcome labels, whose keys are tuple-valued outcome labels.

    This class extends an ordinary OrderedDict by implements mapping
    string-values single-outcome labels to 1-tuples containing that
    label (and vice versa), allowing the use of strings as outcomes
    labels from the user's perspective.

    Parameters
    ----------
    items : list or dict, optional
        Initial values.  Should only be used as part of de-serialization.

    Attributes
    ----------
    _strict : bool
        Whether mapping from strings to 1-tuples is performed.
    """

    #Whether mapping from strings to 1-tuples is performed
    _strict = False

    @classmethod
    def to_outcome(cls, val):
        """
        Converts string outcomes like "0" to proper outcome tuples, like ("0",).

        (also converts non-tuples to tuples, e.g. `["0","1"]` to `("0","1")` )

        Parameters
        ----------
        val : str or tuple
            The value to convert into an outcome label (i.e. a tuple)

        Returns
        -------
        tuple
        """
        return (val,) if isinstance(val, str) else tuple(val)

    def __init__(self, items=None):
        """
        Creates a new OutcomeLabelDict.

        Parameters
        ----------
        items : list, optional
            Used by pickle and other serializations to initialize elements.
        """
        if items is None:
            items = []
        #** Note: if change __init__ signature, update __reduce__ below
        super(OutcomeLabelDict, self).__init__(items)

    def __getitem__(self, key):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        return super(OutcomeLabelDict, self).__getitem__(key)

    def __setitem__(self, key, val):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        super(OutcomeLabelDict, self).__setitem__(key, val)

    def get(self, key, default):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        return super(OutcomeLabelDict, self).get(key, default)

    def getitem_unsafe(self, key, defaultval):
        """
        Gets an item without checking that `key` is a properly formatted outcome tuple.

        Only use this method when you're sure `key` is an outcome tuple and not, e.g.,
        just a string.

        Parameters
        ----------
        key : object
            The key to retrieve

        defaultval : object
            The default value to use (if the key is absent).

        Returns
        -------
        object
        """
        return super(OutcomeLabelDict, self).get(key, defaultval)

    def setitem_unsafe(self, key, val):
        """
        Sets item without checking that the key is a properly formatted outcome tuple.

        Only use this method when you're sure `key` is an outcome tuple and not, e.g.,
        just a string.

        Parameters
        ----------
        key : object
            The key to retrieve.

        val : object
            the value to set.

        Returns
        -------
        None
        """
        super(OutcomeLabelDict, self).__setitem__(key, val)

    def __contains__(self, key):
        if not OutcomeLabelDict._strict:
            key = OutcomeLabelDict.to_outcome(key)
        return key in super(OutcomeLabelDict, self).keys()

    def contains_unsafe(self, key):
        """
        Checks for `key` without ensuring that it is a properly formatted outcome tuple.

        Only use this method when you're sure `key` is an outcome tuple and not, e.g.,
        just a string.

        Parameters
        ----------
        key : object
            The key to retrieve.

        Returns
        -------
        bool
        """
        return super(OutcomeLabelDict, self).__contains__(key)

    def copy(self):
        """
        Return a copy of this OutcomeLabelDict.

        Returns
        -------
        OutcomeLabelDict
        """
        return OutcomeLabelDict([(lbl, _copy.deepcopy(val))
                                 for lbl, val in self.items()])

    def __pygsti_reduce__(self):
        items = [(k, v) for k, v in self.items()]
        return (OutcomeLabelDict, (items,), None)

    def __reduce__(self):
        items = [(k, v) for k, v in self.items()]
        return (OutcomeLabelDict, (items,), None)
