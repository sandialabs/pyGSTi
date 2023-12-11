"""
Functions for the automatic construction of modelpacks.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import datetime as _dt
import subprocess as _sub

from pygsti.algorithms import fiducialselection as _fidsel
from pygsti.algorithms import germselection  as _germsel
from pygsti.algorithms import fiducialpairreduction as _fpr

def create_modelpack(model, outfile, fidsel_options=None, germsel_options=None, fpr_options=None,
                     existing_fids=None, existing_germs=None, existing_fpr=None,
                     clifford_compilation_options=None, existing_clifford_compilation=None,
                     date=None, commit=None, description=None):
    """Create a ModelPack from a model and given arguments for GST circuit generation or RB compilation.

    The ModelPack will inherit from GSTModelPack if fiducial selection (fidsel), germ selection (germsel),
    and fiducial pair reduction (FPR) options are given. Fiducials, germs, and fid pairs can either be
    generated from the given kwargs and model, or directly provided in cases where it has been run externally
    and is too computationally expensive to merit rerunning.
    
    The ModelPack will inherit from RBModelPack if a Clifford compilation is given, either as kwargs for a
    CliffordCompilationRules object or directly provided.

    Even in the case where modelpack options are directly provided, it is recommended to provide the generating
    kwargs for provenance when possible. Date, commit, and user-provided descriptions are also saved
    to facilitate being able to regenerate modelpacks in a reproducible way.

    Parameters
    ----------
    model: Model
        Target model used for fidsel, germsel, and FPR.
    
    outfile: str
        Filename for the generated modelpack.

    fidsel_options: dict of {dicts | tuple of dicts}
        Dictionary of fidsel options where keys will turn into allowed fiducial options in the modelpack.
        Values should be kwargs for running fiducial selection. The special key `fidsel_fn` allows the user to change 
        the function used for fiducial selection; by default, this will be `pygsti.algorithms.fiducialselection.find_fiducials`.
        In the case of fiducial selection,a two-tuple of dicts for prep and measure fiducials is allowed when different kwargs
        are desired (e.g. using 2Q gates in measure fiducials but not prep fiducials, etc.).
        If a key is given in both `fidsel_options` and `existing_fids`, then the value here should correspond
        to the kwargs that were *used* to generate the existing fiducials and is only provided for provenance,
        i.e. fiducial selection will not be rerun.
    
    germsel_options: dict of dicts
        Dictionary of germsel options where keys will turn into allowed germ options in the modelpack.
        Values should be kwargs for running germ selection. The special key `germsel_fn` allows the user to change
        the function used for germ selection; by default, this will be `pygsti.algorithms.germselection.find_germs`.
        If a key is given in both `germsel_options` and `existing_germs`, then the value here should correspond
        to the kwargs that were *used* to generate the existing germs and is only provided for provenance,
        i.e. germ selection will not be rerun.
    
    fpr_options: dict of dicts
        Dictionary of fpr options where keys will turn into allowed FPR options in the modelpack.
        Values should be kwargs for running FPR. The special key `fpr_fn` allows the user to change the function
        used for FPR; by default, this will be pygsti.algorithms.germselection.find_sufficient_fiducial_pairs_per_germ`.
        If a key is given in both `fpr_options` and `existing_fpr`, then the value here should correspond
        to the kwargs that were *used* to generate the existing FPR and is only provided for provenance,
        i.e. FPR will not be rerun.
    
    existing_fids: dict of tuples of lists
        Dictionary of fiducials where keys will turn into allowed fiducial options in the modelpack
        and values are two-tuples of lists for prep and measure fiducials, respectively.
    
    existing_germs: dict of lists
        Dictionary of germs where keys will turn into allowed germ options in the modelpack
        and values are two-tuples of lists for prep and measure fiducials, respectively.

    
    Returns
    -------
    modelpack: GSTModelPack
        The generated GSTModelPack
    """
