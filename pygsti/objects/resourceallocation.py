""" Resource allocation manager """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from pygsti.objects.objectivefns import _dummy_profiler


class ResourceAllocation(object):
    @classmethod
    def build_resource_allocation(cls, arg):
        if arg is None:
            return cls()
        elif isinstance(arg, ResourceAllocation):
            return arg
        else:  # assume argument is a dict of args
            return cls(arg.get('comm', None), arg.get('mem_limit', None),
                       arg.get('profiler', None), arg.get('distribute_method', 'default'))

    def __init__(self, comm=None, mem_limit=None, profiler=None, distribute_method="default"):
        self.comm = comm
        self.mem_limit = mem_limit
        if profiler is not None:
            self.profiler = profiler
        else:
            self.profiler = _dummy_profiler
        self.distribute_method = distribute_method

    def copy(self):
        return ResourceAllocation(self.comm, self.mem_limit, self.profiler, self.distribute_method)
