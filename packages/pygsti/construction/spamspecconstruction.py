from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Construction routines for SPAM specifiers """

from ..tools import remove_duplicates as _remove_duplicates
from ..objects import spamspec as _ss


def build_spam_specs(fiducialGateStrings=None, prepStrs=None, effectStrs=None, prepSpecs=None, effectSpecs=None, prep_labels=('rho0',), effect_labels=('E0',)):
    """
    Computes rho and E specifiers based on optional arguments.  This function
      is used to generate the (prepSpecs,effectSpecs) tuple needed by many of the
      Core GST routines.

    Parameters
    ----------
    fiducialGateStrings : list of (tuples or GateStrings), optional
        Each tuple contains gate labels specifying a fiducial gate string, and it
        is *assumed* that the zeroth rhoVec and EVec are used with this string
        to form a prep-specifier and effect-specifier, respectively.
        e.g. [ (), ('Gx',), ('Gx','Gy') ]

    prepStrs : list of (tuples or GateStrings), optional
        Each tuple contains gate labels and it is *assumed* that the zeroth rhoVec
        is used with this string to form a prep-specifier.
        e.g. [ ('Gi',) , ('Gx','Gx') , ('Gx','Gi','Gy') ]

    effectStrs : list of (tuples or GateStrings), optional
        Each tuple contains gate labels and it is *assumed* that the zeroth EVec
        is used with this string to form a effect-specifier.
        e.g. [ ('Gi',) , ('Gi') , ('Gx','Gi','Gy') ]

    prepSpecs : list of tuples, optional
        Each tuple contains gate labels followed by an integer indexing a rhoVec.
        e.g. [ ('Gi',0) , ('Gx','Gx',0) , ('Gx','Gi','Gy',0) ]

    effectSpecs : list of tuples, optional
        Each tuple contains an integer EVec index followed by gate labels.
        e.g. [ (0,'Gi') , (1,'Gi') , (0,'Gx','Gi','Gy') ]

    prep_labels : tuple of strs, optional
        Labels to prepend to fiducial strings to create prepSpecs

    effect_labels : tuple of strs, optional
        Labels to append to fiducial strings to create effectSpecs

    Returns
    -------
      (prepSpecs, effectSpecs)
          each of the form of the optional parameters prepSpecs and effectSpecs above.
    """

    if prepSpecs is not None:
        if prepStrs is not None or fiducialGateStrings is not None:
            raise ValueError("Can only specify one of prepSpecs, prepStrs, or fiducialGateStrings")
    elif prepStrs is not None:
        prepSpecs = [ _ss.SpamSpec(rhoLbl,f) for f in prepStrs for rhoLbl in prep_labels ]
        if fiducialGateStrings is not None:
            raise ValueError("Can only specify one of prepSpecs, prepStrs, or fiducialGateStrings")
    elif fiducialGateStrings is not None:
        prepSpecs = [ _ss.SpamSpec(rhoLbl,f) for f in fiducialGateStrings for rhoLbl in prep_labels ]
    else:
        raise ValueError("Must specfiy one of prepSpecs, prepStrs, or fiducialGateStrings")


    if effectSpecs is not None:
        if effectStrs is not None or fiducialGateStrings is not None:
            raise ValueError("Can only specify one of effectSpecs, effectStrs, or fiducialGateStrings")
    elif effectStrs is not None:
        effectSpecs = [ _ss.SpamSpec(eLbl,f) for f in effectStrs for eLbl in effect_labels ]
        if fiducialGateStrings is not None:
            raise ValueError("Can only specify one of effectSpecs, effectStrs, or fiducialGateStrings")
    elif fiducialGateStrings is not None:
        effectSpecs = [ _ss.SpamSpec(eLbl,f) for f in fiducialGateStrings for eLbl in effect_labels ]
    else:
        raise ValueError("Must specfiy one of effectSpecs, effectStrs, or fiducialGateStrings")

    return prepSpecs, effectSpecs


def get_spam_strs(specs):
    """
    Get just the string portion of a pair of rho and E specifiers by
      stripping last element of prepSpecs and first element of effectSpecs
      to get prepStrs and effectStrs.  Duplicate strings are removed.

    Parameters
    ----------
    specs : tuple
        (prepSpecs, effectSpecs) to extract strings portions from.

    Returns
    -------
    prepStrs : list of strings
    effectStrs : list of strings
    """
    prepSpecs, effectSpecs = specs
    prepStrs = _remove_duplicates( [ prepSpec.str for prepSpec in prepSpecs ] )
    effectStrs   = _remove_duplicates( [ effectSpec.str for effectSpec in effectSpecs ] )
    return prepStrs, effectStrs
