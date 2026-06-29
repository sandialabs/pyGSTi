import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import itertools
import pygsti
import numpy as np
import pickle
import os

from ..testutils import BaseTestCase, compare_files, temp_files
#from pygsti.forwardsims.mapforwardsim import MapForwardSimulator

#Note: calcs expect tuples (or Circuits) of *Labels*
from pygsti.baseobjs import Label as L

from pygsti.modelpacks.legacy import std1Q_XYI

def Ls(*args):
    """ Convert args to a tuple to Labels """
    return tuple([L(x) for x in args])

FD_JAC_PLACES = 5 # loose checking when computing finite difference derivatives (currently in map calcs)
FD_HESS_PLACES = 1 # looser checking when computing finite difference hessians (currently in map calcs)

SKIP_CVXPY = os.getenv('SKIP_CVXPY')

# This class is for unifying some models that get used in this file and in testGateSets2.py
class GateSetTestCase(BaseTestCase):

    def setUp(self):
        super(GateSetTestCase, self).setUp()

        #OK for these tests, since we test user interface?
        #Set Model objects to "strict" mode for testing
        pygsti.models.ExplicitOpModel._strict = False

        self.model = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])
        self.model.sim = 'matrix'
        self.tp_gateset = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            gate_type="full TP")
        self.tp_gateset.sim = 'matrix'
        self.static_gateset = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            gate_type="static")
        self.static_gateset.sim = 'matrix'

        self.mgateset = self.model.copy()
        self.mgateset.sim = 'map'


class TestGateSetMethods(GateSetTestCase):
    def test_bulk_multiplication(self):
        gatestring1 = ('Gx','Gy')
        gatestring2 = ('Gx','Gy','Gy')
        layout = self.model.sim.create_layout( [gatestring1,gatestring2] )

        p1 = np.dot( self.model['Gy'].to_dense(), self.model['Gx'].to_dense() )
        p2 = np.dot( self.model['Gy'].to_dense(), np.dot( self.model['Gy'].to_dense(), self.model['Gx'].to_dense() ))

        bulk_prods = self.model.sim.bulk_product([gatestring1,gatestring2])
        bulk_prods_scaled, scaleVals = self.model.sim.bulk_product([gatestring1,gatestring2], scale=True)
        bulk_prods2 = scaleVals[:,None,None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[ 0 ],p1)
        self.assertArraysAlmostEqual(bulk_prods[ 1 ],p2)
        self.assertArraysAlmostEqual(bulk_prods2[ 0 ],p1)
        self.assertArraysAlmostEqual(bulk_prods2[ 1 ],p2)

        #Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        PORIG = pygsti.forwardsims.matrixforwardsim._PSMALL; pygsti.forwardsims.matrixforwardsim._PSMALL = 10
        bulk_prods_scaled, scaleVals3 = self.model.sim.bulk_product([gatestring1,gatestring2], scale=True)
        bulk_prods3 = scaleVals3[:,None,None] * bulk_prods_scaled
        pygsti.forwardsims.matrixforwardsim._PSMALL = PORIG
        self.assertArraysAlmostEqual(bulk_prods3[0],p1)
        self.assertArraysAlmostEqual(bulk_prods3[1],p2)

    def test_hessians(self):
        gatestring0 = pygsti.circuits.Circuit(('Gi', 'Gx'))
        gatestring1 = pygsti.circuits.Circuit(('Gx', 'Gy'))
        gatestring2 = pygsti.circuits.Circuit(('Gx', 'Gy', 'Gy'))

        circuitList = pygsti.circuits.to_circuits([gatestring0, gatestring1, gatestring2])
        layout = self.model.sim.create_layout([gatestring0,gatestring1,gatestring2], array_types=('E','EPP'))
        mlayout = self.mgateset.sim.create_layout([gatestring0,gatestring1,gatestring2], array_types=('E','EPP'))

        nElements = layout.num_elements; nParams = self.model.num_params
        probs_to_fill = np.empty( nElements, 'd')
        dprobs_to_fill = np.empty( (nElements,nParams), 'd')
        hprobs_to_fill = np.empty( (nElements,nParams,nParams), 'd')
        self.assertNoWarnings(self.model.sim.bulk_fill_hprobs, hprobs_to_fill, layout,
                              pr_array_to_fill=probs_to_fill, deriv1_array_to_fill=dprobs_to_fill)

        nP = self.model.num_params

        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(nP) ]
        for s1,s2, hprobs_col, dprobs12_col in self.model.sim.iter_hprobs_by_rectangle(
            layout, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )
        dprobs12 = dprobs_to_fill[:,:,None] * dprobs_to_fill[:,None,:]

        #NOTE: Currently iter_hprobs_by_rectangle isn't implemented in map calculator - but it could
        # (and probably should) be later on, at which point the commented code here and
        # below would test it.

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill)
        self.assertArraysAlmostEqual(all_d12cols,dprobs12)


        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs_col, dprobs12_col in self.model.sim.iter_hprobs_by_rectangle(
            layout, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill[:,:,1:10])
        self.assertArraysAlmostEqual(all_d12cols,dprobs12[:,:,1:10])


        hcols = []
        d12cols = []
        slicesList = [ (slice(2,12),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs_col, dprobs12_col in self.model.sim.iter_hprobs_by_rectangle(
            layout, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill[:,2:12,1:10])
        self.assertArraysAlmostEqual(all_d12cols,dprobs12[:,2:12,1:10])


        hprobs_by_block = np.zeros(hprobs_to_fill.shape,'d')
        dprobs12_by_block = np.zeros(dprobs12.shape,'d')
        blocks1 = pygsti.tools.mpitools.slice_up_range(nP, 3)
        blocks2 = pygsti.tools.mpitools.slice_up_range(nP, 5)
        slicesList = list(itertools.product(blocks1,blocks2))
        for s1,s2, hprobs_blk, dprobs12_blk in self.model.sim.iter_hprobs_by_rectangle(
            layout, slicesList, True):
            hprobs_by_block[:,s1,s2] = hprobs_blk
            dprobs12_by_block[:,s1,s2] = dprobs12_blk

        #again, but no dprobs12
        hprobs_by_block2 = np.zeros(hprobs_to_fill.shape,'d')
        for s1,s2, hprobs_blk in self.model.sim.iter_hprobs_by_rectangle(
                layout, slicesList, False):
            hprobs_by_block2[:,s1,s2] = hprobs_blk

        self.assertArraysAlmostEqual(hprobs_by_block,hprobs_to_fill)
        self.assertArraysAlmostEqual(hprobs_by_block2,hprobs_to_fill)
        self.assertArraysAlmostEqual(dprobs12_by_block,dprobs12)


    def test_ondemand_probabilities(self):
        #First create a "sparse" dataset
        # # Columns = 0 count, 1 count
        dataset_txt = \
"""# Test Sparse format data set
{}    0:0  1:100
Gx    0:10 1:90 2:0
GxGy  0:40 1:60
Gx^4  0:100
"""
        with open(temp_files + "/SparseDataset.txt",'w') as f:
            f.write(dataset_txt)

        ds = pygsti.io.read_dataset(temp_files + "/SparseDataset.txt", record_zero_counts=False)
        self.assertEqual(ds.outcome_labels, [('0',), ('1',), ('2',)])
        self.assertEqual(ds[()].outcomes, [('1',)]) # only nonzero count is 1-count
        self.assertEqual(ds[()]['2'], 0) # but we can query '2' since it's a valid outcome label

        gstrs = list(ds.keys())
        model = std1Q_XYI.target_model()
        model.sim = 'map'
        layout = model.sim.create_layout(gstrs, dataset=ds)

        self.assertEqual(layout.outcomes(()), (('1',),) )
        self.assertTrue(layout.outcomes(('Gx',))==(('1',), ('0',)) or  layout.outcomes(('Gx',))==(('0',), ('1',)))
        self.assertTrue(layout.outcomes(('Gx','Gy'))==(('1',), ('0',)) or layout.outcomes(('Gx','Gy'))==(('0',), ('1',)))
        self.assertEqual(layout.outcomes(('Gx',)*4), (('0',),) )

        self.assertEqual(layout.indices(()), slice(0, 1, None))      
        self.assertEqual(layout.indices(('Gx',)), slice(1, 3, None))
        self.assertEqual(layout.indices(('Gx','Gy')), slice(3, 5, None))
        self.assertEqual(layout.indices(('Gx',)*4), slice(5, 6, None))

        self.assertEqual(layout.num_elements, 6)


        #A sparse dataset loading test using the more common format:
        dataset_txt2 = \
"""## Columns = 0 count, 1 count
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 100 0
"""

        with open(temp_files + "/SparseDataset2.txt",'w') as f:
            f.write(dataset_txt2)

        ds = pygsti.io.read_dataset(temp_files + "/SparseDataset2.txt", record_zero_counts=True)
        self.assertEqual(ds.outcome_labels, [('0',), ('1',)])
        self.assertEqual(ds[()].outcomes, [('0',),('1',)]) # both outcomes even though only nonzero count is 1-count
        with self.assertRaises(KeyError):
            ds[()]['2'] # *can't* query '2' b/c it's NOT a valid outcome label here





if __name__ == "__main__":
    unittest.main(verbosity=2)
