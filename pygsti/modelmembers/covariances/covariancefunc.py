"""
Base class for parameterized covariances.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelmembers import modelmember as _modelmember

class CovarianceFunction(_modelmember.ModelMember):

    """
    Base class for parameterized covariance functions.
    """

def __init__(self):
    super().__init__(None, None)

#TODO: Add handling to account for the fact that we don't actually need a state space


class DenseCovarianceFunction(CovarianceFunction):
    """
    Class for modeling densely parameterized covariance function.
    I.e. covariance function defined in a Piecewise constant fashion.
    """

    pass