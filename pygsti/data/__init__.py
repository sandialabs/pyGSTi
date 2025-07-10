"""
A sub-package holding data set objects and supporting analysis objects
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .dataset import DataSet
from .multidataset import MultiDataSet
from .datacomparator import DataComparator
from .freedataset import FreeformDataSet
from .hypothesistest import HypothesisTest

from .datasetconstruction import *
from .rpedata import make_rpe_data_set
