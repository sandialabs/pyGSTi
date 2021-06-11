"""Implementations of common polynomial operations"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

warn_msg = """
An optimized Cython-based implementation of `{module}` is available as
an extension, but couldn't be imported. This might happen if the
extension has not been built. `pip install cython`, then reinstall
pyGSTi to build Cython extensions. Alternatively, setting the
environment variable `PYGSTI_NO_CYTHON_WARNING` will suppress this
message.
""".format(module=__name__)

try:
    # Import cython implementation if it's been built...
    from .fastopcalc import *
except ImportError:
    # ... If not, fall back to the python implementation, with a warning.
    import os as _os
    import warnings as _warnings

    if 'PYGSTI_NO_CYTHON_WARNING' not in _os.environ:
        _warnings.warn(warn_msg)

    from .slowopcalc import *


def bulk_eval_compact_polynomials(vtape, ctape, paramvec, dest_shape):
    """Typechecking wrapper for real- and complex-specific routines..

    The underlying method has two implementations: one for real-valued
    `ctape`, and one for complex-valued. This wrapper will dynamically
    dispatch to the appropriate implementation method based on the
    type of `ctape`. If the type of `ctape` is known prior to calling,
    it's slightly faster to call the appropriate implementation method
    directly; if not.
    """
    if _np.iscomplexobj(ctape):
        ret = bulk_eval_compact_polynomials_complex(vtape, ctape, paramvec, dest_shape)
        im_norm = _np.linalg.norm(_np.imag(ret))
        if im_norm > 1e-6:
            print("WARNING: norm(Im part) = {:g}".format(im_norm))
    else:
        ret = bulk_eval_compact_polynomials_real(vtape, ctape, paramvec, dest_shape)
    return _np.real(ret)


def bulk_eval_compact_polynomials_derivs(vtape, ctape, wrt_params, paramvec, dest_shape):
    """Typechecking wrapper for real- and complex-specific routines..

    The underlying method has two implementations: one for real-valued
    `ctape`, and one for complex-valued. This wrapper will dynamically
    dispatch to the appropriate implementation method based on the
    type of `ctape`. If the type of `ctape` is known prior to calling,
    it's slightly faster to call the appropriate implementation method
    directly; if not.
    """
    if _np.iscomplexobj(ctape):
        ret = bulk_eval_compact_polynomials_derivs_complex(vtape, ctape, wrt_params, paramvec, dest_shape)
        im_norm = _np.linalg.norm(_np.imag(ret))
        if im_norm > 1e-6:
            print("WARNING: norm(Im part) = {:g}".format(im_norm))
    else:
        ret = bulk_eval_compact_polynomials_derivs_real(vtape, ctape, wrt_params, paramvec, dest_shape)
    return _np.real(ret)
