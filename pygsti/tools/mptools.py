"""
Functions for working with Python multiprocessing (more than just map)
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import multiprocessing as _mp

# Modified from https://stackoverflow.com/a/53173433
def starmap_with_kwargs(fn, num_runs, num_processors, *args, **kwargs):
    # If only one processor, run serial (makes it easier to profile and avoids any Pool overhead)
    if num_processors == 1:
        r = []
        for _ in range(num_runs):
            r.append(fn(*args, **kwargs))
        return r

    with _mp.Pool(num_processors) as pool:
        args_for_starmap = zip(
            _itertools.repeat(fn, num_runs),
            _itertools.repeat(args, num_runs),
            _itertools.repeat(kwargs, num_runs))
        return pool.starmap(_apply_args_and_kwargs, args_for_starmap)

def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)