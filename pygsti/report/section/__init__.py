""" Internal model of a section of a generated report """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class Section:
    """
    Abstract base class for report sections.

    Derived classes encapsulate the structure of data within the
    respective section of the report, and provide methods for
    rendering the section to various output formats.

    Parameters
    ----------
    **kwargs
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

        **kwargs
            All additional reportable quantities used when computing
            the figures of this section.

        Returns
        -------
        dict (str -> any)
            Key-value map of report quantities used for this section.
        """
        return {
            k: v(workspace, brevity=brevity, **kwargs)
            for k, v in self._figure_factories.items()
            if v.__figure_brevity_limit__ is None or brevity < v.__figure_brevity_limit__
        }


from .summary import SummarySection
from .help import HelpSection
from .meta import InputSection, MetaSection
from .goodness import GoodnessSection, GoodnessColorBoxPlotSection, GoodnessScalingSection, GoodnessUnmodeledSection
from .gauge import (
    GaugeInvariantsGatesSection, GaugeInvariantsGermsSection,
    GaugeVariantSection, GaugeVariantsDecompSection,
    GaugeVariantsErrorGenSection, GaugeVariantsErrorGenNQubitSection,
    GaugeVariantsRawSection
)
from .idle import IdleTomographySection
from .datacomparison import DataComparisonSection
from .drift import DriftSection
