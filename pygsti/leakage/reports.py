#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import copy
import warnings
from typing import Any, Optional, Union, TYPE_CHECKING

from pygsti.leakage.gaugeopt import add_lago_models

if TYPE_CHECKING:
    from pygsti.protocols.gst import ModelEstimateResults


def _add_lago_estimates(mer: ModelEstimateResults, gaugeopt_verbosity: int = 0):
    # This is broken out into its own function in case someone wants to use
    # it in a report-generation function other than construct_leakage_report.
    for ek in mer.estimates:
        assert isinstance(ek, str)
        if ek == 'Target':
            # There's no need to gauge-optimize in this case.
            continue
        add_lago_models(mer, ek, verbosity=gaugeopt_verbosity)
    return


def construct_leakage_report(
        results : Union[ModelEstimateResults, dict[str,ModelEstimateResults]],
        title : str = 'auto',
        *, # no positional args past "title."
        confidence_level    : Optional[Union[int,float]] = None,
        kwargs_projhess     : Optional[dict[str,Any]]    = None,
        kwargs_stdreport    : Optional[dict[str,Any]]    = None,
        gaugeopt_verbosity  : int = 0,
    ):
    """
    This is a wrapper around construct_standard_report. It generates a Report object
    with leakage analysis, and returns that object along with a copy of ``results`` which
    contains gauge-optimized models created during leakage analysis.
    
    If provided arguments indicate a desire for confidence intervals in the report,
    then this function also computes the necessary likelihood function Hessians.

    Notes
    -----
    The special gauge optimization performed by this function uses the unitary gauge group,
    and uses a modified version of the Frobenius distance loss function. The modification
    reflects how the target gates in a leakage model are only _really_ defined on the
    computational subspace.
    """
    if kwargs_stdreport is None:
        kwargs_stdreport = dict()

    # Prep work. Pack title and confidence_level into kwargs_stdreport.
    clobbering_kwargs = {'title': title, 'confidence_level': confidence_level}
    for k, a in clobbering_kwargs.items():
        kwargs_stdreport[k] = kwargs_stdreport.get(k, a)
        if a != (clobberable_a := kwargs_stdreport[k]):
            msg  = f"Clobbering {k} in kwargs_stdreport ({clobberable_a}) "
            msg += f"with this function's {k} argument ({a})."
            warnings.warn(msg)
            kwargs_stdreport[k] = a

    # Actual work. Deep-copy results and mutate that object in-place.
    res_out  = copy.deepcopy(results)
    res_list = list(res_out.values()) if isinstance(res_out, dict) else [res_out]

    for r in res_list:
        _add_lago_estimates(r, gaugeopt_verbosity)
        if kwargs_stdreport['confidence_level'] is not None:
            r.add_hessians(project_hessian_kwargs=kwargs_projhess)

    # Wrap it up in a bow.
    from pygsti.report import construct_standard_report
    advanced_options = kwargs_stdreport.pop('advanced_options', dict())
    # No leakage display flag needed: construct_standard_report auto-detects leakage bases
    # and shows subspace metrics by default, with an interactive "Metrics" switch to view
    # full-space metrics. The LAGO/target models here carry the l2p1 (leakage) basis.
    report = construct_standard_report(
        res_out, advanced_options=advanced_options, **kwargs_stdreport
    )
    return report, res_out
