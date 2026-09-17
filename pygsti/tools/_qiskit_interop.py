"""
Internal helpers for qiskit interoperability.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

from pygsti.tools.exceptions import (QiskitInteropWarning as _QiskitInteropWarning,
                                     MissingDependencyWarning as _MissingDependencyWarning)

# Half-open interval [min, max) of qiskit versions that pyGSTi's qiskit interop code is
# tested against, compared as tuples of release components. Keep this consistent with
# the 'ibmq' optional dependency in pyproject.toml.
TESTED_QISKIT_RANGE = ((1, 4), (3,))


def _parse_release(version_string):
    """Parse the leading numeric release components of a version string into a tuple of ints.

    Pre-release/dev suffixes are ignored: '3.0.0rc1' -> (3, 0), '2.5.0' -> (2, 5, 0).
    """
    release = []
    for part in version_string.split('.'):
        if part.isdigit():
            release.append(int(part))
        else:
            leading_digits = ''
            for char in part:
                if char.isdigit():
                    leading_digits += char
                else:
                    break
            if leading_digits:
                release.append(int(leading_digits))
            break
    return tuple(release)


def check_qiskit_version(context, required=True):
    """
    Import qiskit and warn if the installed version is outside the tested range.

    Parameters
    ----------
    context : str
        Name of the calling function/method, used in the warning and error messages.

    required : bool, optional
        If True (default), a missing qiskit installation raises a RuntimeError.
        If False, it emits a MissingDependencyWarning and returns None instead,
        for callers that can proceed without qiskit.

    Returns
    -------
    module or None
        The imported qiskit module, or None if qiskit is missing and `required` is False.
    """
    try:
        import qiskit
    except ImportError as err:
        if required:
            raise RuntimeError(f"Qiskit is required for {context}, and does not appear "
                               "to be installed.") from err
        _warnings.warn(f"Qiskit does not appear to be installed; {context} may not function "
                       "properly without it.", _MissingDependencyWarning, stacklevel=3)
        return None

    release = _parse_release(qiskit.__version__)
    minimum, maximum = TESTED_QISKIT_RANGE
    if not (minimum <= release < maximum):
        range_str = f"[{'.'.join(map(str, minimum))}, {'.'.join(map(str, maximum))})"
        _warnings.warn(f"{context} is tested against qiskit versions in the range {range_str} "
                       f"and may not function properly for your qiskit version, which is "
                       f"{qiskit.__version__}.", _QiskitInteropWarning, stacklevel=3)
    return qiskit
