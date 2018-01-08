import unittest
import itertools
import time
import sys
import numpy as np
from .mpinoseutils import *

import pygsti
from pygsti.construction import std1Q_XYI as std

g_maxLengths = [1,2,4,8]
g_numSubTrees = 3

def assertGatesetsInSync(gs, comm):
    if comm is not None:
        bc = gs if comm.Get_rank() == 0 else None
        gs_cmp = comm.bcast(bc, root=0)
        assert(gs.frobeniusdist(gs_cmp) < 1e-6)

        
def runAnalysis(obj, ds, prepStrs, effectStrs, gsTarget, lsgstStringsToUse,
                useFreqWeightedChiSq=False,
                minProbClipForWeighting=1e-4, fidPairList=None,
                comm=None, distributeMethod="gatestrings"):

    #Run LGST to get starting gate set
    assertGatesetsInSync(gsTarget, comm)
    gs_lgst = pygsti.do_lgst(ds, prepStrs, effectStrs, gsTarget,
                             svdTruncateTo=gsTarget.dim, verbosity=3)

    assertGatesetsInSync(gs_lgst, comm)
    gs_lgst_go = pygsti.gaugeopt_to_target(gs_lgst,gsTarget)
    
    assertGatesetsInSync(gs_lgst_go, comm)

    #Run full iterative LSGST
    tStart = time.time()
    if obj == "chi2":
        all_gs_lsgst = pygsti.do_iterative_mc2gst(
            ds, gs_lgst_go, lsgstStringsToUse,
            minProbClipForWeighting=minProbClipForWeighting,
            probClipInterval=(-1e5,1e5),
            verbosity=1, memLimit=3*(1024)**3, returnAll=True,
            useFreqWeightedChiSq=useFreqWeightedChiSq, comm=comm,
            distributeMethod=distributeMethod)
    elif obj == "logl":
        all_gs_lsgst = pygsti.do_iterative_mlgst(
            ds, gs_lgst_go, lsgstStringsToUse,
            minProbClip=minProbClipForWeighting,
            probClipInterval=(-1e5,1e5),
            verbosity=1, memLimit=3*(1024)**3, returnAll=True,
            useFreqWeightedChiSq=useFreqWeightedChiSq, comm=comm,
            distributeMethod=distributeMethod)

    tEnd = time.time()
    print("Time = ",(tEnd-tStart)/3600.0,"hours")

    return all_gs_lsgst


def runOneQubit(obj, ds, lsgstStrings, comm=None, distributeMethod="gatestrings"):
    #specs = pygsti.construction.build_spam_specs(
    #    std.fiducials, prep_labels=std.gs_target.get_prep_labels(),
    #    effect_labels=std.gs_target.get_effect_labels())

    return runAnalysis(obj, ds, std.fiducials, std.fiducials, std.gs_target,
                        lsgstStrings, comm=comm,
                        distributeMethod=distributeMethod)


def create_fake_dataset(comm):
    fidPairList = None
    maxLengths = [1,2,4,8,16]
    nSamples = 1000
    #specs = pygsti.construction.build_spam_specs(
    #    std.fiducials, prep_labels=std.gs_target.get_prep_labels(),
    #    effect_labels=std.gs_target.get_effect_labels())
    #rhoStrs, EStrs = pygsti.construction.get_spam_strs(specs)

    rhoStrs = EStrs = std.fiducials
    lgstStrings = pygsti.construction.list_lgst_gatestrings(
        rhoStrs, EStrs, list(std.gs_target.gates.keys()))
    lsgstStrings = pygsti.construction.make_lsgst_lists(
            list(std.gs_target.gates.keys()), rhoStrs, EStrs,
            std.germs, maxLengths, fidPairList )

    lsgstStringsToUse = lsgstStrings
    allRequiredStrs = pygsti.remove_duplicates(lgstStrings + lsgstStrings[-1])

    if comm is None or comm.Get_rank() == 0:
        gs_dataGen = std.gs_target.depolarize(gate_noise=0.1)
        dsFake = pygsti.construction.generate_fake_data(
            gs_dataGen, allRequiredStrs, nSamples, sampleError="multinomial",
            seed=1234)
        dsFake = comm.bcast(dsFake, root=0)
    else:
        dsFake = comm.bcast(None, root=0)

    #for gs in dsFake:
    #    if abs(dsFake[gs]['0']-dsFake_cmp[gs]['0']) > 0.5:
    #        print("DS DIFF: ",gs, dsFake[gs]['0'], "vs", dsFake_cmp[gs]['0'] )
    return dsFake, lsgstStrings


@mpitest(4)
def test_MPI_products(comm):

    #Create some gateset
    gs = std.gs_target.copy()

    #Remove spam elements so product calculations have element indices <=> product indices
    del gs.preps['rho0']
    del gs.povms['Mdefault']
    
    gs.kick(0.1,seed=1234)

    #Get some gate strings
    maxLengths = [1,2,4,8]
    gstrs = pygsti.construction.make_lsgst_experiment_list(
        list(std.gs_target.gates.keys()), std.fiducials, std.fiducials, std.germs, maxLengths)
    tree,lookup,outcome_lookup = gs.bulk_evaltree(gstrs)
    split_tree = tree.copy()
    split_lookup = split_tree.split(lookup,numSubTrees=g_numSubTrees)

    # Check wrtFilter functionality in dproduct
    some_wrtFilter = [0,2,3,5,10]
    for s in gstrs[0:20]:
        result = gs._calc().dproduct(s, wrtFilter=some_wrtFilter)
        chk_result = gs.dproduct(s) #no filtering
        for ii,i in enumerate(some_wrtFilter):
            assert(np.linalg.norm(chk_result[i]-result[ii]) < 1e-6)
        taken_chk_result = chk_result.take( some_wrtFilter, axis=0 )
        assert(np.linalg.norm(taken_chk_result-result) < 1e-6)


    #Check bulk products

      #bulk_product - no parallelization unless tree is split
    serial = gs.bulk_product(tree, bScale=False)
    parallel = gs.bulk_product(tree, bScale=False, comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial_scl, sscale = gs.bulk_product(tree, bScale=True)
    parallel, pscale = gs.bulk_product(tree, bScale=True, comm=comm)
    assert(np.linalg.norm(serial_scl*sscale[:,None,None] -
                          parallel*pscale[:,None,None]) < 1e-6)

      # will use a split tree to parallelize
    parallel = gs.bulk_product(split_tree, bScale=False, comm=comm)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(serial[lookup[i]]-parallel[split_lookup[i]]) < 1e-6)

    parallel, pscale = gs.bulk_product(split_tree, bScale=True, comm=comm)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(serial_scl[lookup[i]]*sscale[lookup[i],None,None] -
                              parallel[split_lookup[i]]*pscale[split_lookup[i],None,None]) < 1e-6)


      #bulk_dproduct - no split tree => parallel by col
    serial = gs.bulk_dproduct(tree, bScale=False)
    parallel = gs.bulk_dproduct(tree, bScale=False, comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial_scl, sscale = gs.bulk_dproduct(tree, bScale=True)
    parallel, pscale = gs.bulk_dproduct(tree, bScale=True, comm=comm)
    assert(np.linalg.norm(serial_scl*sscale[:,None,None,None] -
                          parallel*pscale[:,None,None,None]) < 1e-6)

      # will just ignore a split tree for now (just parallel by col)
    parallel = gs.bulk_dproduct(split_tree, bScale=False, comm=comm)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(serial[lookup[i]] - parallel[split_lookup[i]]) < 1e-6)

    parallel, pscale = gs.bulk_dproduct(split_tree, bScale=True, comm=comm)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(serial_scl[lookup[i]]*sscale[lookup[i],None,None,None] -
                              parallel[split_lookup[i]]*pscale[split_lookup[i],None,None,None]) < 1e-6)



      #bulk_hproduct - no split tree => parallel by col
    serial = gs.bulk_hproduct(tree, bScale=False)
    parallel = gs.bulk_hproduct(tree, bScale=False, comm=comm)
    assert(np.linalg.norm(serial-parallel) < 1e-6)

    serial_scl, sscale = gs.bulk_hproduct(tree, bScale=True)
    parallel, pscale = gs.bulk_hproduct(tree, bScale=True, comm=comm)
    assert(np.linalg.norm(serial_scl*sscale[:,None,None,None,None] -
                          parallel*pscale[:,None,None,None,None]) < 1e-6)

      # will just ignore a split tree for now (just parallel by col)
    parallel = gs.bulk_hproduct(split_tree, bScale=False, comm=comm)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(serial[lookup[i]] - parallel[split_lookup[i]]) < 1e-6)

    parallel, pscale = gs.bulk_hproduct(split_tree, bScale=True, comm=comm)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(serial_scl[lookup[i]]*sscale[lookup[i],None,None,None,None] -
                              parallel[split_lookup[i]]*pscale[split_lookup[i],None,None,None,None]) < 1e-6)



#OLD: pr functions deprecated
#@mpitest(4)
#def test_MPI_pr(comm):
#
#    #Create some gateset
#    gs = std.gs_target.copy()
#    gs.kick(0.1,seed=1234)
#
#    #Get some gate strings
#    maxLengths = g_maxLengths
#    gstrs = pygsti.construction.make_lsgst_experiment_list(
#        list(std.gs_target.gates.keys()), std.fiducials, std.fiducials, std.germs, maxLengths)
#    tree,lookup,outcome_lookup = gs.bulk_evaltree(gstrs)
#    split_tree = tree.copy()
#    lookup = split_tree.split(lookup,numSubTrees=g_numSubTrees)
#
#    #Check single-spam-label bulk probabilities
#
#    # non-split tree => automatically adjusts wrtBlockSize to accomodate
#    #                    the number of processors
#    serial = gs.bulk_pr('0', tree, clipTo=(-1e6,1e6))
#    parallel = gs.bulk_pr('0', tree, clipTo=(-1e6,1e6), comm=comm)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#
#    serial = gs.bulk_dpr('0', tree, clipTo=(-1e6,1e6))
#    parallel = gs.bulk_dpr('0', tree, clipTo=(-1e6,1e6), comm=comm)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#
#    serial, sp = gs.bulk_dpr('0', tree, returnPr=True, clipTo=(-1e6,1e6))
#    parallel, pp = gs.bulk_dpr('0', tree, returnPr=True, clipTo=(-1e6,1e6), comm=comm)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#    assert(np.linalg.norm(sp-pp) < 1e-6)
#
#    serial, sdp, sp = gs.bulk_hpr('0', tree, returnPr=True, returnDeriv=True,
#                             clipTo=(-1e6,1e6))
#    parallel, pdp, pp = gs.bulk_hpr('0', tree, returnPr=True,
#                                 returnDeriv=True, clipTo=(-1e6,1e6), comm=comm)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#    assert(np.linalg.norm(sdp-pdp) < 1e-6)
#    assert(np.linalg.norm(sp-pp) < 1e-6)
#
#
#    # split tree =>  distribures on sub-trees prior to adjusting
#    #                wrtBlockSize to accomodate remaining processors
#    serial = gs.bulk_pr('0', tree, clipTo=(-1e6,1e6))
#    parallel = gs.bulk_pr('0', split_tree, clipTo=(-1e6,1e6), comm=comm)
#    parallel = split_tree.permute_computation_to_original(parallel)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#
#    serial = gs.bulk_dpr('0', tree, clipTo=(-1e6,1e6))
#    parallel = gs.bulk_dpr('0', split_tree, clipTo=(-1e6,1e6), comm=comm)
#    parallel = split_tree.permute_computation_to_original(parallel)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#
#    serial, sp = gs.bulk_dpr('0', tree, returnPr=True, clipTo=(-1e6,1e6))
#    parallel, pp = gs.bulk_dpr('0', split_tree, returnPr=True, clipTo=(-1e6,1e6), comm=comm)
#    parallel = split_tree.permute_computation_to_original(parallel)
#    pp = split_tree.permute_computation_to_original(pp)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#    assert(np.linalg.norm(sp-pp) < 1e-6)
#
#    serial, sdp, sp = gs.bulk_hpr('0', tree, returnPr=True, returnDeriv=True,
#                             clipTo=(-1e6,1e6))
#    parallel, pdp, pp = gs.bulk_hpr('0', split_tree, returnPr=True,
#                                 returnDeriv=True, clipTo=(-1e6,1e6), comm=comm)
#    parallel = split_tree.permute_computation_to_original(parallel)
#    pdp = split_tree.permute_computation_to_original(pdp)
#    pp = split_tree.permute_computation_to_original(pp)
#    assert(np.linalg.norm(serial-parallel) < 1e-6)
#    assert(np.linalg.norm(sdp-pdp) < 1e-6)
#    assert(np.linalg.norm(sp-pp) < 1e-6)



@mpitest(4)
def test_MPI_probs(comm):

    #Create some gateset
    gs = std.gs_target.copy()
    gs.kick(0.1,seed=1234)

    #Get some gate strings
    maxLengths = g_maxLengths
    gstrs = pygsti.construction.make_lsgst_experiment_list(
        list(std.gs_target.gates.keys()), std.fiducials, std.fiducials, std.germs, maxLengths)
    #tree,lookup,outcome_lookup = gs.bulk_evaltree(gstrs)
    #split_tree = tree.copy()
    #lookup = split_tree.split(lookup, numSubTrees=g_numSubTrees)

    #Check all-spam-label bulk probabilities
    def compare_prob_dicts(a,b,indices=None):
        for gstr in gstrs:
            for outcome in a[gstr].keys():
                if indices is None:
                    assert(np.linalg.norm(a[gstr][outcome] -b[gstr][outcome]) < 1e-6)
                else:
                    for i in indices:
                        assert(np.linalg.norm(a[gstr][outcome][i] -b[gstr][outcome][i]) < 1e-6)

    # non-split tree => automatically adjusts wrtBlockSize to accomodate
    #                    the number of processors
    serial = gs.bulk_probs(gstrs, clipTo=(-1e6,1e6))
    parallel = gs.bulk_probs(gstrs, clipTo=(-1e6,1e6), comm=comm)
    compare_prob_dicts(serial,parallel)

    serial = gs.bulk_dprobs(gstrs, clipTo=(-1e6,1e6))
    parallel = gs.bulk_dprobs(gstrs, clipTo=(-1e6,1e6), comm=comm)
    compare_prob_dicts(serial,parallel)

    serial = gs.bulk_dprobs(gstrs, returnPr=True, clipTo=(-1e6,1e6))
    parallel = gs.bulk_dprobs(gstrs, returnPr=True, clipTo=(-1e6,1e6), comm=comm)
    compare_prob_dicts(serial,parallel,(0,1))

    serial = gs.bulk_hprobs(gstrs, returnPr=True, returnDeriv=True,
                             clipTo=(-1e6,1e6))
    parallel = gs.bulk_hprobs(gstrs, returnPr=True,
                                 returnDeriv=True, clipTo=(-1e6,1e6), comm=comm)
    compare_prob_dicts(serial,parallel,(0,1,2))

    ##OLD: cannot tell bulk_probs to use a split tree anymore (just give list)
    ## split tree =>  distribures on sub-trees prior to adjusting
    ##                wrtBlockSize to accomodate remaining processors
    #serial = gs.bulk_probs(tree, clipTo=(-1e6,1e6))
    #parallel = gs.bulk_probs(split_tree, clipTo=(-1e6,1e6), comm=comm)
    #for sl in serial:
    #    p = split_tree.permute_computation_to_original(parallel[sl])
    #    assert(np.linalg.norm(serial[sl]-p) < 1e-6)
    #
    #serial = gs.bulk_dprobs(tree, clipTo=(-1e6,1e6))
    #parallel = gs.bulk_dprobs(split_tree, clipTo=(-1e6,1e6), comm=comm)
    #for sl in serial:
    #    p = split_tree.permute_computation_to_original(parallel[sl])
    #    assert(np.linalg.norm(serial[sl]-p) < 1e-6)
    #
    #serial = gs.bulk_dprobs(tree, returnPr=True, clipTo=(-1e6,1e6))
    #parallel = gs.bulk_dprobs(split_tree, returnPr=True, clipTo=(-1e6,1e6), comm=comm)
    #for sl in serial:
    #    p0 = split_tree.permute_computation_to_original(parallel[sl][0])
    #    p1 = split_tree.permute_computation_to_original(parallel[sl][1])
    #    assert(np.linalg.norm(serial[sl][0]-p0) < 1e-6)
    #    assert(np.linalg.norm(serial[sl][1]-p1) < 1e-6)
    #
    #serial = gs.bulk_hprobs(tree, returnPr=True, returnDeriv=True,
    #                        clipTo=(-1e6,1e6))
    #parallel = gs.bulk_hprobs(split_tree, returnPr=True,
    #                          returnDeriv=True, clipTo=(-1e6,1e6), comm=comm)
    #for sl in serial:
    #    p0 = split_tree.permute_computation_to_original(parallel[sl][0])
    #    p1 = split_tree.permute_computation_to_original(parallel[sl][1])
    #    p2 = split_tree.permute_computation_to_original(parallel[sl][2])
    #    assert(np.linalg.norm(serial[sl][0]-p0) < 1e-6)
    #    assert(np.linalg.norm(serial[sl][1]-p1) < 1e-6)
    #    assert(np.linalg.norm(serial[sl][2]-p2) < 1e-6)



@mpitest(4)
def test_MPI_fills(comm):

    #Create some gateset
    gs = std.gs_target.copy()
    gs.kick(0.1,seed=1234)

    #Get some gate strings
    maxLengths = g_maxLengths
    gstrs = pygsti.construction.make_lsgst_experiment_list(
        list(std.gs_target.gates.keys()), std.fiducials, std.fiducials, std.germs, maxLengths)
    tree,lookup,outcome_lookup = gs.bulk_evaltree(gstrs)
    split_tree = tree.copy()
    split_lookup = split_tree.split(lookup,numSubTrees=g_numSubTrees)


    #Check fill probabilities
    nEls = tree.num_final_elements()
    nGateStrings = len(gstrs)
    nDerivCols = gs.num_params()

    #Get serial results
    vhp_serial = np.empty( (nEls,nDerivCols,nDerivCols),'d')
    vdp_serial = np.empty( (nEls,nDerivCols), 'd' )
    vp_serial  = np.empty(  nEls, 'd' )

    vhp_serial2 = np.empty( (nEls,nDerivCols,nDerivCols),'d')
    vdp_serial2 = np.empty( (nEls,nDerivCols), 'd' )
    vp_serial2  = np.empty(  nEls, 'd' )

    gs.bulk_fill_probs(vp_serial, tree,
                       (-1e6,1e6), comm=None)

    gs.bulk_fill_dprobs(vdp_serial, tree,
                        vp_serial2, (-1e6,1e6), comm=None,
                        wrtBlockSize=None)
    assert(np.linalg.norm(vp_serial2-vp_serial) < 1e-6)

    gs.bulk_fill_hprobs(vhp_serial, tree,
                        vp_serial2, vdp_serial2, (-1e6,1e6), comm=None,
                        wrtBlockSize1=None, wrtBlockSize2=None)
    assert(np.linalg.norm(vp_serial2-vp_serial) < 1e-6)
    assert(np.linalg.norm(vdp_serial2-vdp_serial) < 1e-6)


    #Check serial results with a split tree, just to be sure
    gs.bulk_fill_probs(vp_serial2, split_tree,
                       (-1e6,1e6), comm=None)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(vp_serial[ lookup[i] ] -
                              vp_serial2[ split_lookup[i] ]) < 1e-6)

    gs.bulk_fill_dprobs(vdp_serial2, split_tree,
                        vp_serial2, (-1e6,1e6), comm=None,
                        wrtBlockSize=None)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(vp_serial[ lookup[i] ] -
                              vp_serial2[ split_lookup[i] ]) < 1e-6)
        assert(np.linalg.norm(vdp_serial[ lookup[i] ] -
                              vdp_serial2[ split_lookup[i] ]) < 1e-6)

    gs.bulk_fill_hprobs(vhp_serial2, split_tree,
                        vp_serial2, vdp_serial2, (-1e6,1e6), comm=None,
                        wrtBlockSize1=None, wrtBlockSize2=None)
    for i,gstr in enumerate(gstrs):
        assert(np.linalg.norm(vp_serial[ lookup[i] ] -
                              vp_serial2[ split_lookup[i] ]) < 1e-6)
        assert(np.linalg.norm(vdp_serial[ lookup[i] ] -
                              vdp_serial2[ split_lookup[i] ]) < 1e-6)
        assert(np.linalg.norm(vhp_serial[ lookup[i] ] -
                              vhp_serial2[ split_lookup[i] ]) < 1e-6)

    #Get parallel results - with and without split tree
    vhp_parallel = np.empty( (nEls,nDerivCols,nDerivCols),'d')
    vdp_parallel = np.empty( (nEls,nDerivCols), 'd' )
    vp_parallel = np.empty( nEls, 'd' )

    for tstTree,tstLookup in zip([tree, split_tree],[lookup,split_lookup]):

        gs.bulk_fill_probs(vp_parallel, tstTree,
                           (-1e6,1e6), comm=comm)
        for i,gstr in enumerate(gstrs):
            assert(np.linalg.norm(vp_parallel[ tstLookup[i] ] -
                                  vp_serial[ lookup[i] ]) < 1e-6)

        for blkSize in [None, 4]:
            gs.bulk_fill_dprobs(vdp_parallel, tstTree,
                                vp_parallel, (-1e6,1e6), comm=comm,
                                wrtBlockSize=blkSize)
            for i,gstr in enumerate(gstrs):
                assert(np.linalg.norm(vp_parallel[ tstLookup[i] ] -
                                      vp_serial[ lookup[i] ]) < 1e-6)
                assert(np.linalg.norm(vdp_parallel[ tstLookup[i] ] -
                                      vdp_serial[ lookup[i] ]) < 1e-6)

            for blkSize2 in [None, 2, 4]:
                gs.bulk_fill_hprobs(vhp_parallel, tstTree,
                                    vp_parallel, vdp_parallel, (-1e6,1e6), comm=comm,
                                    wrtBlockSize1=blkSize, wrtBlockSize2=blkSize2)
                for i,gstr in enumerate(gstrs):
                    assert(np.linalg.norm(vp_parallel[ tstLookup[i] ] -
                                          vp_serial[ lookup[i] ]) < 1e-6)
                    assert(np.linalg.norm(vdp_parallel[ tstLookup[i] ] -
                                          vdp_serial[ lookup[i] ]) < 1e-6)
                    assert(np.linalg.norm(vhp_parallel[ tstLookup[i] ] -
                                          vhp_serial[ lookup[i] ]) < 1e-6)

    #Test Serial vs Parallel use of wrtFilter
    some_wrtFilter = [0,1,2,3,4,5,6,7] #must be contiguous now - not arbitraray
    some_wrtFilter2 = [6,7,8,9,10,11,12] #must be contiguous now - not arbitraray
    vhp_parallelF = np.empty( (nEls,nDerivCols,len(some_wrtFilter)),'d')
    vhp_parallelF2 = np.empty( (nEls,len(some_wrtFilter),len(some_wrtFilter2)),'d')
    vdp_parallelF = np.empty( (nEls,len(some_wrtFilter)), 'd' )

    for tstTree,tstLookup in zip([tree, split_tree],[lookup, split_lookup]):

        gs._calc().bulk_fill_dprobs(vdp_parallelF, tstTree,
                            None, (-1e6,1e6), comm=comm,
                            wrtFilter=some_wrtFilter, wrtBlockSize=None)
        for k,gstr in enumerate(gstrs):
            for ii,i in enumerate(some_wrtFilter):
                assert(np.linalg.norm(vdp_serial[lookup[k],i]-vdp_parallelF[tstLookup[k],ii]) < 1e-6)
        taken_result = vdp_serial.take( some_wrtFilter, axis=1 )
        for k,gstr in enumerate(gstrs):
            assert(np.linalg.norm(taken_result[lookup[k]]-vdp_parallelF[tstLookup[k]]) < 1e-6)

        gs._calc().bulk_fill_hprobs(vhp_parallelF, tstTree,
                        None, None,None, (-1e6,1e6), comm=comm,
                        wrtFilter2=some_wrtFilter, wrtBlockSize2=None)
        for k,gstr in enumerate(gstrs):
            for ii,i in enumerate(some_wrtFilter):
                assert(np.linalg.norm(vhp_serial[lookup[k],:,i]-vhp_parallelF[tstLookup[k],:,ii]) < 1e-6)
        taken_result = vhp_serial.take( some_wrtFilter, axis=2 )
        for k,gstr in enumerate(gstrs):
            assert(np.linalg.norm(taken_result[lookup[k]]-vhp_parallelF[tstLookup[k]]) < 1e-6)

        gs._calc().bulk_fill_hprobs(vhp_parallelF2, tstTree,
                        None, None,None, (-1e6,1e6), comm=comm,
                        wrtFilter1=some_wrtFilter, wrtFilter2=some_wrtFilter2)
        for k,gstr in enumerate(gstrs):
            for ii,i in enumerate(some_wrtFilter):
                for jj,j in enumerate(some_wrtFilter2):
                    assert(np.linalg.norm(vhp_serial[lookup[k],i,j]-vhp_parallelF2[tstLookup[k],ii,jj]) < 1e-6)
        taken_result = vhp_serial.take( some_wrtFilter, axis=1 ).take( some_wrtFilter2, axis=2)
        for k,gstr in enumerate(gstrs):
            assert(np.linalg.norm(taken_result[lookup[k]]-vhp_parallelF2[tstLookup[k]]) < 1e-6)


@mpitest(4)
def test_MPI_by_block(comm):

    #Create some gateset
    if comm is None or comm.Get_rank() == 0:
        gs = std.gs_target.copy()
        gs.kick(0.1,seed=1234)
        gs = comm.bcast(gs, root=0)
    else:
        gs = comm.bcast(None, root=0)

    #Get some gate strings
    maxLengths = g_maxLengths
    gstrs = pygsti.construction.make_lsgst_experiment_list(
        list(std.gs_target.gates.keys()), std.fiducials, std.fiducials, std.germs, maxLengths)
    tree,lookup,outcome_lookkup = gs.bulk_evaltree(gstrs)
    #split_tree = tree.copy()
    #split_lookup = split_tree.split(lookup,numSubTrees=g_numSubTrees)


    #Check that "by column" matches standard "at once" methods:
    nEls = tree.num_final_elements()
    nGateStrings = len(gstrs)
    nDerivCols = gs.num_params()

    #Get serial results
    vhp_serial = np.empty( (nEls,nDerivCols,nDerivCols),'d')
    vdp_serial = np.empty( (nEls,nDerivCols), 'd' )
    vp_serial = np.empty( nEls, 'd' )

    gs.bulk_fill_hprobs(vhp_serial, tree,
                        vp_serial, vdp_serial, (-1e6,1e6), comm=None)
    dprobs12_serial = vdp_serial[:,:,None] * vdp_serial[:,None,:]


    for tstTree,tstLookup in zip([tree],[lookup]): # currently no split trees allowed (ValueError), split_tree]:
        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nDerivCols),slice(i,i+1)) for i in range(nDerivCols) ]
        for s1,s2, hprobs, dprobs12 in gs.bulk_hprobs_by_block(
            tstTree, slicesList, True, comm):
            hcols.append(hprobs)
            d12cols.append(dprobs12)

        all_hcols = np.concatenate( hcols, axis=2 )
        all_d12cols = np.concatenate( d12cols, axis=2 )


        #print "SHAPES:"
        #print "hcols[0] = ",hcols[0].shape
        #print "all_hcols = ",all_hcols.shape
        #print "all_d12cols = ",all_d12cols.shape
        #print "vhp_serial = ",vhp_serial.shape
        #print "dprobs12_serial = ",dprobs12_serial.shape

        #for i in range(all_hcols.shape[3]):
        #    print "Diff(%d) = " % i, np.linalg.norm(all_hcols[0,:,8:,i]-vhp_serial[0,:,8:,i])
        #    if np.linalg.norm(all_hcols[0,:,8:,i]-vhp_serial[0,:,8:,i]) > 1e-6:
        #        for j in range(all_hcols.shape[3]):
        #            print "Diff(%d,%d) = " % (i,j), np.linalg.norm(all_hcols[0,:,8:,i]-vhp_serial[0,:,8:,j])
        #    assert(np.linalg.norm(all_hcols[0,:,8:,i]-vhp_serial[0,:,8:,i]) < 1e-6)

        for i,gstr in enumerate(gstrs):
            assert(np.linalg.norm(all_hcols[tstLookup[i]]-vhp_serial[lookup[i]]) < 1e-6)

        #for i in range(all_d12cols.shape[3]):
        #    print "Diff(%d) = " % i, np.linalg.norm(all_d12cols[0,:,8:,i]-dprobs12_serial[0,:,8:,i])
        #    if np.linalg.norm(all_d12cols[0,:,8:,i]-dprobs12_serial[0,:,8:,i]) > 1e-6:
        #        for j in range(all_d12cols.shape[3]):
        #            print "Diff(%d,%d) = " % (i,j), np.linalg.norm(all_d12cols[0,:,8:,i]-dprobs12_serial[0,:,8:,j])

        for i,gstr in enumerate(gstrs):
            assert(np.linalg.norm(all_d12cols[tstLookup[i]]-dprobs12_serial[lookup[i]]) < 1e-6)


        hcols = []
        d12cols = []
        slicesList = [ (slice(2,12),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs, dprobs12 in gs.bulk_hprobs_by_block(
            tstTree, slicesList, True, comm):
            hcols.append(hprobs)
            d12cols.append(dprobs12)

        all_hcols = np.concatenate( hcols, axis=2 )
        all_d12cols = np.concatenate( d12cols, axis=2 )
        for i,gstr in enumerate(gstrs):
            assert(np.linalg.norm(all_hcols[tstLookup[i]]-vhp_serial[lookup[i],2:12,1:10]) < 1e-6)
            assert(np.linalg.norm(all_d12cols[tstLookup[i]]-dprobs12_serial[lookup[i],2:12,1:10]) < 1e-6)


        hprobs_by_block = np.zeros(vhp_serial.shape,'d')
        dprobs12_by_block = np.zeros(dprobs12_serial.shape,'d')
        blocks1 = pygsti.tools.mpitools.slice_up_range(nDerivCols, 3)
        blocks2 = pygsti.tools.mpitools.slice_up_range(nDerivCols, 5)
        slicesList = list(itertools.product(blocks1,blocks2))
        for s1,s2, hprobs_blk, dprobs12_blk in gs.bulk_hprobs_by_block(
            tstTree, slicesList, True, comm):
            hprobs_by_block[:,s1,s2] = hprobs_blk
            dprobs12_by_block[:,s1,s2] = dprobs12_blk

        for i,gstr in enumerate(gstrs):
            assert(np.linalg.norm(hprobs_by_block[tstLookup[i]]-vhp_serial[lookup[i]]) < 1e-6)
            assert(np.linalg.norm(dprobs12_by_block[tstLookup[i]]-dprobs12_serial[lookup[i]]) < 1e-6)



#SCRATCH
#if np.linalg.norm(chk_ret[0]-dGs) >= 1e-6:
#    #if bScale:
#    #    print "SCALED"
#    #    print chk_ret[-1]
#
#    rank = comm.Get_rank()
#    if rank == 0:
#        print "DEBUG: parallel mismatch"
#        print "len(all_results) = ",len(all_results)
#        print "diff = ",np.linalg.norm(chk_ret[0]-dGs)
#        for row in range(dGs.shape[0]):
#            rowA = my_results[0][row,:].flatten()
#            rowB = all_results[rank][0][row,:].flatten()
#            rowC = dGs[row,:].flatten()
#            chk_C = chk_ret[0][row,:].flatten()
#
#            def sp(ar):
#                for i,x in enumerate(ar):
#                    if abs(x) > 1e-4:
#                        print i,":", x
#            def spc(ar1,ar2):
#                for i,x in enumerate(ar1):
#                    if (abs(x) > 1e-4 or abs(ar2[i]) > 1e-4): # and abs(x-ar2[i]) > 1e-6:
#                        print i,":", x, ar2[i], "(", (x-ar2[i]), ")", "[",x/ar2[i],"]"
#
#            assert( _np.linalg.norm(rowA-rowB) < 1e-6)
#            assert( _np.linalg.norm(rowC[0:len(rowA)]-rowA) < 1e-6)
#            #if _np.linalg.norm(rowA) > 1e-6:
#            if _np.linalg.norm(rowC - chk_C) > 1e-6:
#                print "SCALE for row%d = %g" % (row,rest_of_result[-1][row])
#                print "CHKSCALE for row%d = %g" % (row,chk_ret[-1][row])
#                print "row%d diff = " % row, _np.linalg.norm(rowC - chk_C)
#                print "row%d (rank%d)A = " % (row,rank)
#                sp(rowA)
#                print "row%d (all vs check) = " % row
#                spc(rowC, chk_C)
#
#                assert(False)
#    assert(False)





@mpitest(4)
def test_MPI_gatestrings_chi2(comm):
    #Create dataset for serial and parallel runs
    ds,lsgstStrings = create_fake_dataset(comm)

    #Individual processors
    my1ProcResults = runOneQubit("chi2",ds,lsgstStrings)

    #Using all processors
    myManyProcResults = runOneQubit("chi2",ds,lsgstStrings,comm,"gatestrings")

    for i,(gs1,gs2) in enumerate(zip(my1ProcResults,myManyProcResults)):
        assertGatesetsInSync(gs1, comm)
        assertGatesetsInSync(gs2, comm)

        gs2_go = pygsti.gaugeopt_to_target(gs2, gs1, {'gates': 1.0, 'spam': 1.0})
        print("Frobenius distance %d (rank %d) = " % (i,comm.Get_rank()), gs1.frobeniusdist(gs2_go))
        if gs1.frobeniusdist(gs2_go) >= 1e-5:
            print("DIFF (%d) = " % comm.Get_rank(), gs1.strdiff(gs2_go))
        assert(gs1.frobeniusdist(gs2_go) < 1e-5)
    return


@mpitest(4)
def test_MPI_gatestrings_logl(comm):
    #Create dataset for serial and parallel runs
    ds,lsgstStrings = create_fake_dataset(comm)

    #Individual processors
    my1ProcResults = runOneQubit("logl",ds,lsgstStrings)

    #Using all processors
    myManyProcResults = runOneQubit("logl",ds,lsgstStrings,comm,"gatestrings")

    for i,(gs1,gs2) in enumerate(zip(my1ProcResults,myManyProcResults)):
        assertGatesetsInSync(gs1, comm)
        assertGatesetsInSync(gs2, comm)

        gs2_go = pygsti.gaugeopt_to_target(gs2, gs1, {'gates': 1.0, 'spam': 1.0})
        print("Frobenius distance %d (rank %d) = " % (i,comm.Get_rank()), gs1.frobeniusdist(gs2_go))
        if gs1.frobeniusdist(gs2_go) >= 1e-5:
            print("DIFF (%d) = " % comm.Get_rank(), gs1.strdiff(gs2_go))
        assert(gs1.frobeniusdist(gs2_go) < 1e-5)
    return


@mpitest(4)
def test_MPI_derivcols(comm):
    #Create dataset for serial and parallel runs
    ds,lsgstStrings = create_fake_dataset(comm)

    #Individual processors
    my1ProcResults = runOneQubit("chi2",ds,lsgstStrings)

    #Using all processors
    myManyProcResults = runOneQubit("chi2",ds,lsgstStrings,comm,"deriv")

    for i,(gs1,gs2) in enumerate(zip(my1ProcResults,myManyProcResults)):
        assertGatesetsInSync(gs1, comm)
        assertGatesetsInSync(gs2, comm)

        gs2_go = pygsti.gaugeopt_to_target(gs2, gs1, {'gates': 1.0, 'spam': 1.0})
        print("Frobenius distance %d (rank %d) = " % (i,comm.Get_rank()), gs1.frobeniusdist(gs2_go))
        if gs1.frobeniusdist(gs2_go) >= 1e-5:
            print("DIFF (%d) = " % comm.Get_rank(), gs1.strdiff(gs2_go))
        assert(gs1.frobeniusdist(gs2_go) < 1e-5)
    return

@mpitest(4)
def test_run1Q_end2end(comm):
    from pygsti.construction import std1Q_XYI
    gs_target = std1Q_XYI.gs_target
    fiducials = std1Q_XYI.fiducials
    germs = std1Q_XYI.germs
    maxLengths = [1,2,4]
    
    gs_datagen = gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
    listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
        list(gs_target.gates.keys()), fiducials, fiducials, germs, maxLengths)
    ds = pygsti.construction.generate_fake_data(gs_datagen, listOfExperiments,
                                                nSamples=1000,
                                                sampleError="binomial",
                                                seed=1234, comm=comm)
    
    results = pygsti.do_long_sequence_gst(ds, gs_target, fiducials, fiducials,
                                          germs, maxLengths, comm=comm)
    
    pygsti.report.create_standard_report(results, "mpi_test_report",
                                         "MPI test report", confidenceLevel=95,
                                         verbosity=2, comm=comm)


@mpitest(4)
def test_MPI_germsel(comm):
    if comm is None or comm.Get_rank() == 0:
        gatesetNeighborhood = pygsti.alg.randomizeGatesetList(
            [std.gs_target], randomizationStrength=1e-3,
            numCopies=3, seed=2018)
        comm.bcast(gatesetNeighborhood, root=0)
    else:
        gatesetNeighborhood = comm.bcast(None, root=0)

    max_length   = 6
    gates        = std.gs_target.gates.keys()
    superGermSet = pygsti.construction.list_all_gatestrings_without_powers_and_cycles(gates, max_length)

    germs = pygsti.alg.build_up_breadth(gatesetNeighborhood, superGermSet,
                                        randomize=False, seed=2018, scoreFunc='all',
                                        threshold=1e6, verbosity=1, gatePenalty=1.0,
                                        memLimit=3*(1024**3), comm=comm)

@mpitest(4)
def test_MPI_profiler(comm):
    mem = pygsti.baseobjs.profiler._get_root_mem_usage(comm)
    mem = pygsti.baseobjs.profiler._get_max_mem_usage(comm)
    
    start_time = time.time()
    p = pygsti.baseobjs.Profiler(comm, default_print_memcheck=True)
    p.add_time("My Name", start_time, prefix=1)
    p.add_count("My Count", inc=1, prefix=1)
    p.add_count("My Count", inc=2, prefix=1)
    p.mem_check("My Memcheck", prefix=1)
    p.mem_check("My Memcheck", prefix=1)
    p.print_mem("My Memcheck just to print")
    p.print_mem("My Memcheck just to print", show_minmax=True)
    p.print_msg("My Message")
    p.print_msg("My Message", all_ranks=True)
    
    s = p.format_times(sortBy="name")
    s = p.format_times(sortBy="time")
    #with self.assertRaises(ValueError):
    #    p.format_times(sortBy="foobar")
        
    s = p.format_counts(sortBy="name")
    s = p.format_counts(sortBy="count")
    #with self.assertRaises(ValueError):
    #    p.format_counts(sortBy="foobar")

    s = p.format_memory(sortBy="name")
    s = p.format_memory(sortBy="usage")
    #with self.assertRaises(ValueError):
    #    p.format_memory(sortBy="foobar")
    #with self.assertRaises(NotImplementedError):
    #    p.format_memory(sortBy="timestamp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
