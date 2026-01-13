# This file is for classes that are more descriptive versions of the warnings available in python.

class PoorPerformanceRuntimeWarning(RuntimeWarning):
    """
    This warning indicates that the subsequent code is a slower code path then is available in a different path.
    One should not follow the slower code path in a performance critical code path.
    """
    pass
