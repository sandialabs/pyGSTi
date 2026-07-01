""" Internal model of a section of a generated report """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.report.workspace import NotApplicable as _NotApplicable


def basis_aware_display(switchboard, name, ordinary, leakage):
    """
    Register (once, keyed by `name`) and return a per-cell display ``SwitchValue`` for a
    gates-vs-target table.

    Each cell's column tuple is chosen from that cell's model basis together with the
    interactive "Metrics" switch: the leakage (subspace-restricted) columns are used only
    when the cell's model basis implies leakage modeling *and* the reader has the "Metrics"
    switch in its 0-th ("Subspace") position; otherwise the ordinary full-space columns are
    used.  Because ``display`` is thereby a per-cell switched value, the metric *headers* a
    reader sees always match the metric *computation* used to fill them, and a report whose
    cells have different bases simply renders different columns per switch position.

    Parameters
    ----------
    switchboard : Switchboard
        The report's master switchboard.  Must have ``mdl_final`` populated and a
        ``metric_space_switch_index`` attribute (both set in ``_create_master_switchboard``).

    name : str
        Key under which the display ``SwitchValue`` is registered on ``switchboard``.  If a
        value is already registered under this key it is returned unchanged (idempotent
        across repeated / multi-brevity renders).

    ordinary : tuple
        The full-space column-name tuple (e.g. ``('inf', 'trace', 'diamond', ...)``).

    leakage : tuple
        The subspace/leakage column-name tuple (e.g. ``('sub-inf', 'sub-trace', ...)``).

    Returns
    -------
    SwitchValue
    """
    if name in switchboard:
        return switchboard[name]

    ms_idx = switchboard.metric_space_switch_index
    deps = tuple(switchboard.mdl_final.dependencies) + (ms_idx,)
    switchboard.add(name, deps)
    sv = switchboard[name]

    mdl_base = switchboard.mdl_final.base
    for idx in _np.ndindex(sv.base.shape):
        model_idx, ms = idx[:-1], idx[-1]
        mdl = mdl_base[model_idx]
        basis = getattr(mdl, 'basis', None)
        leaky = (not isinstance(mdl, _NotApplicable)) and basis is not None \
            and bool(getattr(basis, 'implies_leakage_modeling', False))
        sv.base[idx] = leakage if (ms == 0 and leaky) else ordinary
    return sv


class Section:
    """
    Abstract base class for report sections.

    Derived classes encapsulate the structure of data within the
    respective section of the report, and provide methods for
    rendering the section to various output formats.

    Parameters
    ----------
    `**kwargs`
        Computation of specific section elements can be configured at
        runtime by passing the name of a figure as a keyword argument
        set to ``False``.
    """
    _HTML_TEMPLATE = None

    @staticmethod
    def figure_factory(brevity_limit=None):
        """
        Decorator to designate a method as a figure factory.

        Parameters
        ----------
        brevity_limit : int or None, optional
            Mark that this figure should only be rendered for reports
            with brevity strictly less than this limit. Defaults to
            ``None``, indicating that the figure should always be
            rendered.
        """
        def decorator(fn):
            fn.__figure_brevity_limit__ = brevity_limit
            return fn
        return decorator

    def __init__(self, **kwargs):
        self._figure_factories = {}

        for name in dir(self.__class__):
            member = getattr(self.__class__, name)
            if hasattr(member, '__figure_brevity_limit__') and kwargs.get(name, True):
                self._figure_factories[name] = member

    def render(self, workspace, brevity=0, **kwargs):
        """
        Render this section's figures.

        Parameters
        ----------
        workspace : Workspace
            A ``Workspace`` used for caching figure computation.

        brevity : int, optional
            Level of brevity used when generating this section. At
            higher brevity levels, certain non-critical figures will
            not be rendered. Defaults to 0 (most verbose).

        `**kwargs`
            All additional reportable quantities used when computing
            the figures of this section.

        Returns
        -------
        `dict (str -> any)`
            Key-value map of report quantities used for this section.
        """
        return {
            k: v(workspace, brevity=brevity, **kwargs)
            for k, v in self._figure_factories.items()
            if v.__figure_brevity_limit__ is None or brevity < v.__figure_brevity_limit__
        }


from .datacomparison import DataComparisonSection
from .drift import DriftSection
from .gauge import (
    GaugeInvariantsGatesSection, GaugeInvariantsGermsSection,
    GaugeVariantSection, GaugeVariantsDecompSection,
    GaugeVariantsErrorGenSection, GaugeVariantsErrorGenNQubitSection,
    GaugeVariantsRawSection
)
from .goodness import GoodnessSection, GoodnessColorBoxPlotSection, GoodnessScalingSection, GoodnessUnmodeledSection
from .help import HelpSection
from .idle import IdleTomographySection
from .meta import InputSection, MetaSection
from .summary import SummarySection
