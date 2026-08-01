#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Wall-clock accuracy of ``opttools.timed_block``.

This asserts that the duration ``timed_block`` reports is close to the true
elapsed time, with a tolerance of a couple hundred milliseconds. On a loaded
machine a ``sleep(0.5)`` can overshoot that tolerance through no fault of the
code under test, so it lives here rather than in the unit suite.

The rest of ``timed_block``'s behavior -- that it writes the expected output,
honors ``pre_message``, and stores a numeric duration in the supplied dict or
defaultdict -- is load-insensitive and stays in
``test/unit/tools/test_opttools.py``.

    pytest test/performance/test_opttools_timing.py
"""

import unittest
from time import sleep

import pytest

from pygsti.tools import opttools as opt


@pytest.mark.slow
class TimedBlockAccuracyTester(unittest.TestCase):

    def test_timer(self):
        duration = 0.5
        timeDict = {}
        with opt.timed_block('time', timeDict):
            sleep(duration)
        lt_tol = 1e-3
        self.assertGreaterEqual(timeDict['time'], duration - lt_tol)  # sometimes sleeps last slightly less than specified duration.
        tolerance = 0.2  # this should deliberately be large, for repeatability
        self.assertLessEqual(
            timeDict['time'], duration + tolerance,
            "timed block result is greater than {} seconds off".format(tolerance))
