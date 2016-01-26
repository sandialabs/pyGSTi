#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Construction routines for SPAM specifiers """

from ..objects import spamspec as _ss


def build_spam_specs(fiducialGateStrings=None, rhoStrs=None, EStrs=None, rhoSpecs=None, ESpecs=None, rhoVecInds=(0,), EVecInds=(0,) ):
    """
    Computes rho and E specifiers based on optional arguments.  This function
      is used to generate the (rhoSpecs,ESpecs) tuple needed by many of the 
      Core GST routines.

    Parameters
    ----------
    fiducialGateStrings : list of (tuples or GateStrings), optional
        Each tuple contains gate labels specifying a fiducial gate string, and it
        is *assumed* that the zeroth rhoVec and EVec are used with this string
        to form a rho-specifier and E-specifier, respectively.
        e.g. [ (), ('Gx',), ('Gx','Gy') ] 

    rhoStrs : list of (tuples or GateStrings), optional
        Each tuple contains gate labels and it is *assumed* that the zeroth rhoVec
        is used with this string to form a rho-specifier.
        e.g. [ ('Gi',) , ('Gx','Gx') , ('Gx','Gi','Gy') ]

    EStrs : list of (tuples or GateStrings), optional
        Each tuple contains gate labels and it is *assumed* that the zeroth EVec
        is used with this string to form a E-specifier.
        e.g. [ ('Gi',) , ('Gi') , ('Gx','Gi','Gy') ]

    rhoSpecs : list of tuples, optional
        Each tuple contains gate labels followed by an integer indexing a rhoVec.
        e.g. [ ('Gi',0) , ('Gx','Gx',0) , ('Gx','Gi','Gy',0) ]

    ESpecs : list of tuples, optional
        Each tuple contains an integer EVec index followed by gate labels.
        e.g. [ (0,'Gi') , (1,'Gi') , (0,'Gx','Gi','Gy') ]

    rhoVecInds : tuple of integers, optional
        Indices to prepend to fiducial strings to create rhoSpecs

    EVecInds : tuple of integers, optional
        Indices to append to fiducial strings to create ESpecs

    Returns
    -------
      (rhoSpecs, ESpecs) 
          each of the form of the optional parameters rhoSpecs and ESpecs above.
    """

    if rhoSpecs is not None:
        if rhoStrs is not None or fiducialGateStrings is not None:
           raise ValueError("Can only specify one of rhoSpecs, rhoStrs, or fiducialGateStrings")
    elif rhoStrs is not None:
        rhoSpecs = [ _ss.SpamSpec(iRho,f) for f in rhoStrs for iRho in rhoVecInds ]
        if fiducialGateStrings is not None:
           raise ValueError("Can only specify one of rhoSpecs, rhoStrs, or fiducialGateStrings")
    elif fiducialGateStrings is not None:
        rhoSpecs = [ _ss.SpamSpec(iRho,f) for f in fiducialGateStrings for iRho in rhoVecInds ]
    else:
        raise ValueError("Must specfiy one of rhoSpecs, rhoStrs, or fiducialGateStrings")
    

    if ESpecs is not None:
        if EStrs is not None or fiducialGateStrings is not None:
           raise ValueError("Can only specify one of ESpecs, EStrs, or fiducialGateStrings")
    elif EStrs is not None:
        ESpecs = [ _ss.SpamSpec(iEvec,f) for f in EStrs for iEvec in EVecInds ]
        if fiducialGateStrings is not None:
           raise ValueError("Can only specify one of ESpecs, EStrs, or fiducialGateStrings")
    elif fiducialGateStrings is not None:
        ESpecs = [ _ss.SpamSpec(iEvec,f) for f in fiducialGateStrings for iEvec in EVecInds ]
    else:
        raise ValueError("Must specfiy one of ESpecs, EStrs, or fiducialGateStrings")
        
    return rhoSpecs, ESpecs


def get_spam_strs(specs):
    """ 
    Get just the string portion of a pair of rho and E specifiers by
      stripping last element of rhoSpecs and first element of ESpecs
      to get rhoStrs and EStrs.

    Parameters
    ----------
    specs : tuple
        (rhoSpecs, ESpecs) to extract strings portions from.

    Returns
    -------
    rhoStrs : list of strings
    EStrs : list of strings
    """
    rhoSpecs, ESpecs = specs
    rhoStrs = [ rhoSpec.str for rhoSpec in rhoSpecs ]
    EStrs   = [ ESpec.str for ESpec in ESpecs ]
    return rhoStrs, EStrs
