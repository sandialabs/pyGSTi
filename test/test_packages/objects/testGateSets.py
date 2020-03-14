import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import itertools
import collections
import pygsti
import numpy as np
import warnings
import pickle
import os

from ..testutils import BaseTestCase, compare_files, temp_files
#from pygsti.objects.mapforwardsim import MapForwardSimulator

#Note: calcs expect tuples (or Circuits) of *Labels*
from pygsti.objects import Label as L

from pygsti.modelpacks.legacy import std1Q_XYI
from pygsti.io import enable_old_object_unpickling
from pygsti.tools.compattools import patched_uuid

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
        pygsti.objects.ExplicitOpModel._strict = False

        self.model = pygsti.construction.build_explicit_model(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])

        self.tp_gateset = pygsti.construction.build_explicit_model(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization="TP")

        self.static_gateset = pygsti.construction.build_explicit_model(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization="static")

        self.mgateset = self.model.copy()
        #self.mgateset._calcClass = MapForwardSimulator
        self.mgateset.set_simtype('map')


class TestGateSetMethods(GateSetTestCase):
    def test_bulk_multiplication(self):
        gatestring1 = ('Gx','Gy')
        gatestring2 = ('Gx','Gy','Gy')
        evt,lookup,outcome_lookup = self.model.bulk_evaltree( [gatestring1,gatestring2] )

        p1 = np.dot( self.model['Gy'], self.model['Gx'] )
        p2 = np.dot( self.model['Gy'], np.dot( self.model['Gy'], self.model['Gx'] ))

        bulk_prods = self.model.bulk_product(evt)
        bulk_prods_scaled, scaleVals = self.model.bulk_product(evt, scale=True)
        bulk_prods2 = scaleVals[:,None,None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[ 0 ],p1)
        self.assertArraysAlmostEqual(bulk_prods[ 1 ],p2)
        self.assertArraysAlmostEqual(bulk_prods2[ 0 ],p1)
        self.assertArraysAlmostEqual(bulk_prods2[ 1 ],p2)

        #Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        PORIG = pygsti.objects.matrixforwardsim.PSMALL; pygsti.objects.matrixforwardsim.PSMALL = 10
        bulk_prods_scaled, scaleVals3 = self.model.bulk_product(evt, scale=True)
        bulk_prods3 = scaleVals3[:,None,None] * bulk_prods_scaled
        pygsti.objects.matrixforwardsim.PSMALL = PORIG
        self.assertArraysAlmostEqual(bulk_prods3[0],p1)
        self.assertArraysAlmostEqual(bulk_prods3[1],p2)


        #tag on a few extra EvalTree tests
        debug_stuff = evt.get_analysis_plot_infos()

    def test_hessians(self):
        gatestring0 = pygsti.obj.Circuit(('Gi','Gx'))
        gatestring1 = pygsti.obj.Circuit(('Gx','Gy'))
        gatestring2 = pygsti.obj.Circuit(('Gx','Gy','Gy'))

        circuitList = pygsti.construction.circuit_list([gatestring0,gatestring1,gatestring2])
        evt,lookup,outcome_lookup = self.model.bulk_evaltree( [gatestring0,gatestring1,gatestring2] )
        mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( [gatestring0,gatestring1,gatestring2] )



        nElements = evt.num_final_elements(); nParams = self.model.num_params()
        probs_to_fill = np.empty( nElements, 'd')
        dprobs_to_fill = np.empty( (nElements,nParams), 'd')
        hprobs_to_fill = np.empty( (nElements,nParams,nParams), 'd')
        self.assertNoWarnings(self.model.bulk_fill_hprobs, hprobs_to_fill, evt,
                              pr_mx_to_fill=probs_to_fill, deriv_mx_to_fill=dprobs_to_fill, check=True)


        nP = self.model.num_params()

        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(nP) ]
        for s1,s2, hprobs_col, dprobs12_col in self.model.bulk_hprobs_by_block(
            evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )
        dprobs12 = dprobs_to_fill[:,:,None] * dprobs_to_fill[:,None,:]

        #NOTE: Currently bulk_hprobs_by_block isn't implemented in map calculator - but it could
        # (and probably should) be later on, at which point the commented code here and
        # below would test it.

        #mhcols = []
        #md12cols = []
        #mslicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(nP) ]
        #for s1,s2, hprobs_col, dprobs12_col in self.mgateset.bulk_hprobs_by_block(
        #    mevt, mslicesList, True):
        #    mhcols.append(hprobs_col)
        #    md12cols.append(dprobs12_col)
        #mall_hcols = np.concatenate( mhcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        #mall_d12cols = np.concatenate( md12cols, axis=2 )
        #mdprobs12 = mdprobs_to_fill[:,:,None] * mdprobs_to_fill[:,None,:]

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill)
        self.assertArraysAlmostEqual(all_d12cols,dprobs12)
        #self.assertArraysAlmostEqual(mall_hcols,mhprobs_to_fill, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,mdprobs12, places=FD_HESS_PLACES)
        #
        #self.assertArraysAlmostEqual(mall_hcols,all_hcols, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,all_d12cols, places=FD_HESS_PLACES)


        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs_col, dprobs12_col in self.model.bulk_hprobs_by_block(
            evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )

        #mhcols = []
        #md12cols = []
        #mslicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(1,10) ]
        #for s1,s2, hprobs_col, dprobs12_col in self.mgateset.bulk_hprobs_by_block(
        #    spam_label_rows, mevt, mslicesList, True):
        #    mhcols.append(hprobs_col)
        #    md12cols.append(dprobs12_col)
        #mall_hcols = np.concatenate( mhcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        #mall_d12cols = np.concatenate( md12cols, axis=2 )

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill[:,:,1:10])
        self.assertArraysAlmostEqual(all_d12cols,dprobs12[:,:,1:10])
        #self.assertArraysAlmostEqual(mall_hcols,mhprobs_to_fill[:,:,1:10], places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,mdprobs12[:,:,1:10], places=FD_HESS_PLACES)
        #
        #self.assertArraysAlmostEqual(mall_hcols,all_hcols, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,all_d12cols, places=FD_HESS_PLACES)


        hcols = []
        d12cols = []
        slicesList = [ (slice(2,12),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs_col, dprobs12_col in self.model.bulk_hprobs_by_block(
            evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )

        #mhcols = []
        #md12cols = []
        #mslicesList = [ (slice(2,12),slice(i,i+1)) for i in range(1,10) ]
        #for s1,s2, hprobs_col, dprobs12_col in self.mgateset.bulk_hprobs_by_block(
        #    mevt, mslicesList, True):
        #    mhcols.append(hprobs_col)
        #    md12cols.append(dprobs12_col)
        #mall_hcols = np.concatenate( mhcols, axis=2 )  #axes = (spam+circuit, derivParam1, derivParam2)
        #mall_d12cols = np.concatenate( md12cols, axis=2 )

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill[:,2:12,1:10])
        self.assertArraysAlmostEqual(all_d12cols,dprobs12[:,2:12,1:10])
        #self.assertArraysAlmostEqual(mall_hcols,mhprobs_to_fill[:,2:12,1:10], places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,mdprobs12[:,2:12,1:10], places=FD_HESS_PLACES)
        #
        #self.assertArraysAlmostEqual(mall_hcols,all_hcols, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,all_d12cols, places=FD_HESS_PLACES)


        hprobs_by_block = np.zeros(hprobs_to_fill.shape,'d')
        dprobs12_by_block = np.zeros(dprobs12.shape,'d')
        #mhprobs_by_block = np.zeros(mhprobs_to_fill.shape,'d')
        #mdprobs12_by_block = np.zeros(mdprobs12.shape,'d')
        blocks1 = pygsti.tools.mpitools.slice_up_range(nP, 3)
        blocks2 = pygsti.tools.mpitools.slice_up_range(nP, 5)
        slicesList = list(itertools.product(blocks1,blocks2))
        for s1,s2, hprobs_blk, dprobs12_blk in self.model.bulk_hprobs_by_block(
            evt, slicesList, True):
            hprobs_by_block[:,s1,s2] = hprobs_blk
            dprobs12_by_block[:,s1,s2] = dprobs12_blk

        #again, but no dprobs12
        hprobs_by_block2 = np.zeros(hprobs_to_fill.shape,'d')
        for s1,s2, hprobs_blk in self.model.bulk_hprobs_by_block(
                evt, slicesList, False):
            hprobs_by_block2[:,s1,s2] = hprobs_blk

        #for s1,s2, hprobs_blk, dprobs12_blk in self.mgateset.bulk_hprobs_by_block(
        #    mevt, slicesList, True):
        #    mhprobs_by_block[:,s1,s2] = hprobs_blk
        #    mdprobs12_by_block[:,s1,s2] = dprobs12_blk

        self.assertArraysAlmostEqual(hprobs_by_block,hprobs_to_fill)
        self.assertArraysAlmostEqual(hprobs_by_block2,hprobs_to_fill)
        self.assertArraysAlmostEqual(dprobs12_by_block,dprobs12)
        #self.assertArraysAlmostEqual(mhprobs_by_block,hprobs_to_fill, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mdprobs12_by_block,dprobs12, places=FD_HESS_PLACES)


        #print("****DEBUG HESSIAN BY COL****")
        #print("shape = ",all_hcols.shape)
        #to_check = hprobs_to_fill[:,2:12,1:10]
        #for si in range(all_hcols.shape[0]):
        #    for stri in range(all_hcols.shape[1]):
        #        diff = np.linalg.norm(all_hcols[si,stri]-to_check[si,stri])
        #        print("[%d,%d] diff = %g" % (si,stri,diff))
        #        if diff > 1e-6:
        #            for i in range(all_hcols.shape[2]):
        #                for j in range(all_hcols.shape[3]):
        #                    x = all_hcols[si,stri,i,j]
        #                    y = to_check[si,stri,i,j]
        #                    if abs(x-y) > 1e-6:
        #                        print("  el(%d,%d):  %g - %g = %g" % (i,j,x,y,x-y))


    def test_tree_construction_mem_limit(self):
        circuits = pygsti.construction.circuit_list(
            [('Gx',),
             ('Gy',),
             ('Gx','Gy'),
             ('Gy','Gy'),
             ('Gy','Gx'),
             ('Gx','Gx','Gx'),
             ('Gx','Gy','Gx'),
             ('Gx','Gy','Gy'),
             ('Gy','Gy','Gy'),
             ('Gy','Gx','Gx') ] )
        ##Make a few-param model to better test mem limits
        mdl_few = self.model.copy()
        mdl_few.set_all_parameterizations("static")
        mdl_few.preps['rho0'] = self.model.preps['rho0'].copy()
        self.assertEqual(mdl_few.num_params(),4)

        #mdl_big = pygsti.construction.build_explicit_model(
        #    [('Q0','Q3','Q2')],['Gi'], [ "I(Q0)"])
        #mdl_big._calcClass = MapForwardSimulator

        class FakeComm(object):
            def __init__(self,size): self.size = size
            def Get_rank(self): return 0
            def Get_size(self): return self.size
            def bcast(self,obj, root=0): return obj

        for nprocs in (1,4,10,40,100):
            fake_comm = FakeComm(nprocs)
            for distributeMethod in ('deriv','circuits'):
                for memLimit in (-100, 1024, 10*1024, 100*1024, 1024**2, 10*1024**2):
                    print("Nprocs = %d, method = %s, memLim = %g" % (nprocs, distributeMethod, memLimit))
                    try:
                        evt,_,_,lookup,outcome_lookup = self.model.bulk_evaltree_from_resources(
                            circuits, mem_limit=memLimit, distribute_method=distributeMethod,
                            subcalls=['bulk_fill_hprobs'], comm=fake_comm)
                        evt,_,_,lookup,outcome_lookup = self.mgateset.bulk_evaltree_from_resources(
                            circuits, mem_limit=memLimit, distribute_method=distributeMethod,
                            subcalls=['bulk_fill_hprobs'], comm=fake_comm)
                        evt,_,_,lookup,outcome_lookup = mdl_few.bulk_evaltree_from_resources(
                            circuits, mem_limit=memLimit, distribute_method=distributeMethod,
                            subcalls=['bulk_fill_hprobs'], comm=fake_comm)
                        evt,_,_,lookup,outcome_lookup = mdl_few.bulk_evaltree_from_resources(
                            circuits, mem_limit=memLimit, distribute_method=distributeMethod,
                            subcalls=['bulk_fill_dprobs'], comm=fake_comm) #where bNp2Matters == False

                    except MemoryError:
                        pass #OK - when memlimit is too small and splitting is unproductive

        #balanced not implemented
        with self.assertRaises(NotImplementedError):
            evt,_,_,lookup,outcome_lookup = self.model.bulk_evaltree_from_resources(
                circuits, mem_limit=memLimit, distribute_method="balanced", subcalls=['bulk_fill_hprobs'])



    def test_tree_splitting(self):
        circuits = [('Gx',),
                       ('Gy',),
                       ('Gx','Gy'),
                       ('Gy','Gy'),
                       ('Gy','Gx'),
                       ('Gx','Gx','Gx'),
                       ('Gx','Gy','Gx'),
                       ('Gx','Gy','Gy'),
                       ('Gy','Gy','Gy'),
                       ('Gy','Gx','Gx') ]
        evtA,lookupA,outcome_lookupA = self.model.bulk_evaltree( circuits )

        evtB,lookupB,outcome_lookupB = self.model.bulk_evaltree( circuits )
        lookupB = evtB.split(lookupB, max_sub_tree_size=4)

        evtC,lookupC,outcome_lookupC = self.model.bulk_evaltree( circuits )
        lookupC = evtC.split(lookupC, num_sub_trees=3)

        with self.assertRaises(ValueError):
            evtBad,lkup,_ = self.model.bulk_evaltree( circuits )
            evtBad.split(lkup, num_sub_trees=3, max_sub_tree_size=4) #can't specify both

        self.assertFalse(evtA.is_split())
        self.assertTrue(evtB.is_split())
        self.assertTrue(evtC.is_split())
        self.assertEqual(len(evtA.get_sub_trees()), 1)
        self.assertEqual(len(evtB.get_sub_trees()), 5) #empirically
        self.assertEqual(len(evtC.get_sub_trees()), 3)
        self.assertLessEqual(max([len(subTree)
                             for subTree in evtB.get_sub_trees()]), 4)

        #print "Lenghts = ",len(evtA.get_sub_trees()),len(evtB.get_sub_trees()),len(evtC.get_sub_trees())
        #print "SubTree sizes = ",[len(subTree) for subTree in evtC.get_sub_trees()]

        bulk_probsA = np.empty( evtA.num_final_elements(), 'd')
        bulk_probsB = np.empty( evtB.num_final_elements(), 'd')
        bulk_probsC = np.empty( evtC.num_final_elements(), 'd')
        self.model.bulk_fill_probs(bulk_probsA, evtA)
        self.model.bulk_fill_probs(bulk_probsB, evtB)
        self.model.bulk_fill_probs(bulk_probsC, evtC)

        for i,opstr in enumerate(circuits):
            self.assertArraysAlmostEqual(bulk_probsA[ lookupA[i] ],
                                         bulk_probsB[ lookupB[i] ])
            self.assertArraysAlmostEqual(bulk_probsA[ lookupA[i] ],
                                         bulk_probsC[ lookupC[i] ])


    def test_load_old_gateset(self):
        #pygsti.obj.results.enable_old_python_results_unpickling()
        with enable_old_object_unpickling(), patched_uuid():
            with open(compare_files + "/pygsti0.9.6.gateset.pkl", 'rb') as f:
                mdl = pickle.load(f)
        #pygsti.obj.results.disable_old_python_results_unpickling()
        #pygsti.io.disable_old_object_unpickling()
        with open(temp_files + "/repickle_old_gateset.pkl", 'wb') as f:
            pickle.dump(mdl, f)

        with enable_old_object_unpickling("0.9.7"), patched_uuid():
            with open(compare_files + "/pygsti0.9.7.gateset.pkl", 'rb') as f:
                mdl = pickle.load(f)
        with open(temp_files + "/repickle_old_gateset.pkl", 'wb') as f:
            pickle.dump(mdl, f)

        #OLD: we don't do this anymore (_calcClass has been removed)
        #also test automatic setting of _calcClass
        #mdl = self.model.copy()
        #del mdl._calcClass
        #c = mdl._fwdsim() #automatically sets _calcClass
        #self.assertTrue(hasattr(mdl,'_calcClass'))

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

        ds = pygsti.io.load_dataset(temp_files + "/SparseDataset.txt", record_zero_counts=False)
        self.assertEqual(ds.get_outcome_labels(), [('0',), ('1',), ('2',)])
        self.assertEqual(ds[()].outcomes, [('1',)]) # only nonzero count is 1-count
        self.assertEqual(ds[()]['2'], 0) # but we can query '2' since it's a valid outcome label

        gstrs = list(ds.keys())
        raw_dict, elIndices, outcome_lookup, ntotal = std1Q_XYI.target_model().simplify_circuits(gstrs, ds)

        print("Raw circuit -> elabels dict:\n","\n".join(["%s: %s" % (str(k),str(v)) for k,v in raw_dict.items()]))
        print("\nElement indices lookup (orig opstr index -> element indices):\n",elIndices)
        print("\nOutcome lookup (orig opstr index -> list of outcome for each element):\n",outcome_lookup)
        print("\ntotal elements = ", ntotal)

        self.assertEqual(raw_dict[('rho0',)], [L('Mdefault_1')])
        self.assertEqual(raw_dict[('rho0','Gx',)], [L('Mdefault_0'), L('Mdefault_1')])
        self.assertEqual(raw_dict[('rho0','Gx','Gy')], [L('Mdefault_0'), L('Mdefault_1')])
        self.assertEqual(raw_dict[('rho0',) + ('Gx',)*4], [L('Mdefault_0')])

        self.assertEqual(elIndices, collections.OrderedDict(
            [(0, slice(0, 1, None)), (1, slice(1, 3, None)), (2, slice(3, 5, None)), (3, slice(5, 6, None))]) )

        self.assertEqual(outcome_lookup, collections.OrderedDict(
            [(0, [('1',)]), (1, [('0',), ('1',)]), (2, [('0',), ('1',)]), (3, [('0',)])]) )

        self.assertEqual(ntotal, 6)


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

        ds = pygsti.io.load_dataset(temp_files + "/SparseDataset2.txt", record_zero_counts=True)
        self.assertEqual(ds.get_outcome_labels(), [('0',), ('1',)])
        self.assertEqual(ds[()].outcomes, [('0',),('1',)]) # both outcomes even though only nonzero count is 1-count
        with self.assertRaises(KeyError):
            ds[()]['2'] # *can't* query '2' b/c it's NOT a valid outcome label here





if __name__ == "__main__":
    unittest.main(verbosity=2)
