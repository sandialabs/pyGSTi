"""
A standard multi-qubit gate set module.

Variables for working with the a model containing X(pi/2) and Z(pi/2) gates.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelpacks._modelpack import GSTModelPack

#Set this flag to True if you want modelpack contents from pre-0.9.14 pyGSTi (for legacy purposes).
PRE_v0914_XZ = False

class _Module(GSTModelPack):
    description = "X(pi/2) and Z(pi/2) gates"

    gates = [('Gxpi2', 0), ('Gzpi2', 0)]

    _sslbls = (0,)


    if PRE_v0914_XZ:
        _germs = [(('Gxpi2', 0), ), (('Gzpi2', 0), ), (('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)), (('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0))]

        _germs_lite = [(('Gxpi2', 0), ), (('Gzpi2', 0), ), (('Gxpi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0))]

        _fiducials = None

        _prepfiducials = [(), (('Gxpi2', 0), ), (('Gxpi2', 0), ('Gzpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                    (('Gxpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0))]

        _measfiducials = [(), (('Gxpi2', 0), ), (('Gzpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0)), (('Gxpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)),
                    (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0))]

        global_fidpairs = [(0, 1), (1, 2), (4, 3), (4, 4)]

        _pergerm_fidpairsdict = {
            (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
            (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
            (('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0)): [(0, 3), (1, 2), (2, 5), (3, 1), (3, 3), (5, 3)],
            (('Gzpi2', 0), ('Gxpi2', 0), ('Gxpi2', 0)): [(0, 3), (0, 4), (1, 0), (1, 4), (2, 1), (4, 5)]
        }

        global_fidpairs_lite = [(0, 1), (1, 2), (4, 3), (4, 4)]

        _pergerm_fidpairsdict_lite = {
            (('Gxpi2', 0), ): [(1, 1), (3, 4), (4, 2), (5, 5)],
            (('Gzpi2', 0), ): [(0, 0), (2, 3), (5, 2), (5, 4)],
            (('Gxpi2', 0), ('Gzpi2', 0)): [(0, 3), (3, 2), (4, 0), (5, 3)],
            (('Gxpi2', 0), ('Gxpi2', 0), ('Gzpi2', 0)): [(0, 0), (0, 2), (1, 1), (4, 0), (4, 2), (5, 5)]
        }

    else:
        #find_germs(target_model, randomize=True, candidate_germ_counts={7:'all upto'}, algorithm='greedy', assume_real=True, float_type=np.double, seed=7252025)
        _germs = [(('Gxpi2',0),), (('Gzpi2',0),), (('Gxpi2',0), ('Gzpi2',0)), (('Gxpi2',0), ('Gzpi2',0), ('Gzpi2',0), ('Gzpi2',0)), 
                  (('Gxpi2',0), ('Gxpi2',0), ('Gzpi2',0), ('Gzpi2',0), ('Gzpi2',0), ('Gxpi2',0), ('Gzpi2',0))]
        #find_germs(target_model, randomize=False, candidate_germ_counts={3:'all upto'}, algorithm='greedy', assume_real=True, float_type=np.double, algorithm_kwargs={'op_penalty':1})
        _germs_lite = [(('Gxpi2',0),), (('Gzpi2',0),), (('Gxpi2',0), ('Gzpi2',0)), (('Gxpi2',0), ('Gxpi2',0), ('Gzpi2',0))]
        #Constructed by hand
        _prepfiducials = [(), # |0>
                        (('Gxpi2', 0),), #|-i> 
                        (('Gxpi2', 0), ('Gzpi2', 0)), #|+>
                        (('Gxpi2', 0), ('Gxpi2', 0)), #|1>
                        (('Gxpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0)), #|i>
                        (('Gxpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0))] #|->
        #Constructed by hand
        _measfiducials = [(), #<0|, <1|
                        (('Gxpi2', 0),), # <-i|, <i|
                        (('Gzpi2', 0), ('Gxpi2', 0)), #<+|, <-| 
                        (('Gxpi2', 0), ('Gxpi2', 0)), # <1|, <0|
                        (('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0)), #<i|, <-i|
                        (('Gzpi2', 0), ('Gzpi2', 0), ('Gzpi2', 0), ('Gxpi2', 0))] # <-|, <+|

        #global_fidpairs uses an outdated version of FPR which I don't think is worth regenerating
        #since this will eventually be removed whenever we finally overhaul modelpacks to bring them up-to-date.
        global_fidpairs = None
        #find_sufficient_fiducial_pairs_per_germ_greedy(smq1Q_XZ.target_model('full TP'), _prep_fiducial_circuits, _meas_fiducial_circuits, germs_robust, inv_trace_tol=10, initial_seed_mode='greedy')
        _pergerm_fidpairsdict ={(('Gxpi2',0),): [(5, 5), (0, 1), (1, 1), (2, 2)], 
                                (('Gzpi2',0),): [(5, 5), (0, 0), (4, 5), (1, 4)], 
                                (('Gxpi2',0), ('Gzpi2',0)): [(5, 5), (0, 5), (4, 5), (2, 5)], 
                                (('Gxpi2',0), ('Gzpi2',0), ('Gzpi2',0), ('Gzpi2',0)): [(5, 5), (0, 5), (1, 2), (2, 5)], 
                                (('Gxpi2',0), ('Gxpi2',0), ('Gzpi2',0), ('Gzpi2',0), ('Gzpi2',0), ('Gxpi2',0), ('Gzpi2',0)): [(5, 5), (4, 4), (1, 5), (0, 5), (3, 4), (0, 1)]}
        #find_sufficient_fiducial_pairs_per_germ_greedy(smq1Q_XZ.target_model('full TP'), _prep_fiducial_circuits, _meas_fiducial_circuits, germs_robust, inv_trace_tol=10, initial_seed_mode='greedy')
        _pergerm_fidpairsdict_lite = {(('Gxpi2',0),): [(5, 5), (0, 1), (1, 1), (2, 2)], 
                                      (('Gzpi2',0),): [(5, 5), (0, 0), (4, 5), (1, 4)], 
                                      (('Gxpi2',0), ('Gzpi2',0)): [(5, 5), (0, 5), (4, 5), (2, 5)], 
                                      (('Gxpi2',0), ('Gxpi2',0), ('Gzpi2',0)): [(5, 5), (3, 3), (1, 5), (1, 3), (0, 1), (3, 4)]}

    def _target_model(self, sslbls, **kwargs):
        return self._build_explicit_target_model(
            sslbls, [('Gxpi2', 0), ('Gzpi2', 0)], ['X(pi/2,{0})', 'Z(pi/2,{0})'], **kwargs)


import sys
sys.modules[__name__] = _Module()
