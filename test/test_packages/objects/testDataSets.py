import unittest
import collections
import pickle
import pygsti
import numpy as np
import warnings
import os
from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

class TestDataSetMethods(BaseTestCase):

    def test_from_scratch(self):
        # Create a dataset from scratch
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds[ ('Gx',) ] = {'0': 10, '1': 90}
        ds[ ('Gx',) ]['0'] = 10
        ds[ ('Gx',) ]['1'] = 90
        with self.assertRaises(NotImplementedError):
            ds[ ('Gx',) ]['new'] = 20 # assignment can't create *new* outcome labels (yet)
        #OLD ds.add_counts_1q( ('Gx','Gy'), 10, 40 )
        #OLD ds.add_counts_1q( ('Gx','Gy'), 40, 10 ) #freq much different from existing
        ds.add_count_dict( ('Gy','Gy'), {'FooBar': 10, '1': 90 }) # OK to add outcome labels on the fly
        ds.add_count_dict( ('Gy','Gy'), {'1': 90 }) # now all outcome labels OK now
        ds.add_count_dict( ('Gy','Gy'),pygsti.obj.labeldicts.OutcomeLabelDict([('0',10), ('1',90)]),
                           overwriteExisting=False) #adds counts at next available integer timestep
        ds.done_adding_data()

        #Test that we don't *need* to add anything
        dsEmpty = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        dsEmpty.done_adding_data()

        dsWritable = ds.copy_nonstatic()
        dsWritable[('Gy',)] = {'0': 20, '1': 80}

        dsWritable2 = dsWritable.copy_nonstatic()
         #test copy_nonstatic on already non-static dataset

        ds_str = str(ds)

        with self.assertRaises(ValueError):
            ds.add_count_dict( ('Gx',), {'0': 10, '1': 90 }) # done adding data
        #OLD with self.assertRaises(ValueError):
        #    ds.add_counts_1q( ('Gx',), 40,60) # done adding data

        self.assertEqual(ds[('Gx',)]['0'], 10)
        self.assertEqual(ds[('Gx',)]['1'], 90)
        print(ds)
        self.assertAlmostEqual(ds[('Gx',)].fraction('0'), 0.1)

        #Pickle and unpickle
        with open(temp_files + '/dataset.pickle', 'wb') as datasetfile:
            pickle.dump(ds, datasetfile)
        ds_from_pkl = None
        with open(temp_files + '/dataset.pickle', 'rb') as datasetfile:
            ds_from_pkl = pickle.load(datasetfile)
        self.assertEqual(ds_from_pkl[('Gx',)]['0'], 10)
        self.assertAlmostEqual(ds_from_pkl[('Gx',)].fraction('0'), 0.1)


        # Invoke the DataSet constructor other ways
        gstrs = [ ('Gx',), ('Gx','Gy'), ('Gy',) ]
        gstrInds = collections.OrderedDict( [ (('Gx',),0),  (('Gx','Gy'),1), (('Gy',),2) ] )
        gstrInds_static = collections.OrderedDict( [ (pygsti.obj.GateString(('Gx',)),slice(0,2)),
                                                     (pygsti.obj.GateString(('Gx','Gy')),slice(2,4)),
                                                     (pygsti.obj.GateString(('Gy',)),slice(4,6)) ] )
        olInds = collections.OrderedDict( [ ('0',0),  ('1',1) ] )

        oli = np.array([0,1],'i')
        oli_static = np.array( [0,1]*3, 'd' ) # 3 gate strings * 2 outcome labels each
        time_static = np.zeros( (6,), 'd' )
        reps_static = 10*np.ones( (6,), 'd' )

        oli_nonstc = [ oli, oli, oli ] # each item has num_outcomes elements
        time_nonstc = [ np.zeros(2,'d'), np.zeros(2,'d'), np.zeros(2,'d') ]
        reps_nonstc = [ 10*np.ones(2,'i'), 10*np.ones(2,'i'), 10*np.ones(2,'i') ]

        ds2 = pygsti.objects.DataSet(oli_nonstc, time_nonstc, reps_nonstc,
                                     gateStrings=gstrs, outcomeLabels=['0','1'])
        ds3 = pygsti.objects.DataSet(oli_nonstc[:], time_nonstc[:], reps_nonstc[:],
                                     gateStringIndices=gstrInds, outcomeLabelIndices=olInds)
        ds4 = pygsti.objects.DataSet(oli_static, time_static, reps_static,
                                     gateStringIndices=gstrInds_static, outcomeLabels=['0','1'], bStatic=True)
        ds5 = pygsti.objects.DataSet(oli_nonstc, time_nonstc, reps_nonstc, gateStrings=gstrs,
                                     outcomeLabels=['0','1'], bStatic=False)
        ds6 = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        ds6.done_adding_data() #ds6 = empty dataset

        ds2.add_counts_from_dataset(ds)
        ds3.add_counts_from_dataset(ds)
        with self.assertRaises(ValueError):
            ds4.add_counts_from_dataset(ds) #can't add to static DataSet

        with self.assertRaises(AssertionError):
            pygsti.objects.DataSet(gateStrings=gstrs) #no spam labels specified
        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(oli_static, time_static, reps_static,
                                   outcomeLabels=['0','1'], bStatic=True)
              #must specify gateLabels (or indices) when creating static DataSet
        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(gateStrings=gstrs, outcomeLabels=['0','1'], bStatic=True)
              #must specify counts when creating static DataSet

        #Test has_key methods
        self.assertTrue( ds2.has_key(('Gx',)) )
        self.assertTrue( ds2[('Gx',)].has_key('0'))

        #Test indexing methods
        cnt = 0
        for gstr in ds:

            if gstr in ds:
                if gstr in ds:
                    pass
                if pygsti.obj.GateString(gstr) in ds:
                    pass

            dsRow = ds[gstr]
            allLabels = list(dsRow.counts.keys())
            counts = dsRow.counts
            for spamLabel in counts:
                if spamLabel in counts: #we know to be true
                    cnt = counts[spamLabel]
                if spamLabel in counts:
                    cnt = counts[spamLabel]

        for dsRow in ds.values():
            for spamLabel,count in dsRow.counts.items():
                cnt += count

        #Check degrees of freedom
        ds.get_degrees_of_freedom()
        ds2.get_degrees_of_freedom()
        ds3.get_degrees_of_freedom()
        ds4.get_degrees_of_freedom()

        #String Manipulation
        ds.process_gate_strings( lambda s: pygsti.construction.manipulate_gatestring(s, [( ('Gx',), ('Gy',))]) )

        #Test truncation
        ds2.truncate( [('Gx',),('Gx','Gy')] ) #non-static
        ds4.truncate( [('Gx',),('Gx','Gy')] ) #static
        ds2.truncate( [('Gx',),('Gx','Gy'),('Gz',)], bThrowErrorIfStringIsMissing=False ) #non-static
        ds4.truncate( [('Gx',),('Gx','Gy'),('Gz',)], bThrowErrorIfStringIsMissing=False ) #static
        with self.assertRaises(ValueError):
            ds2.truncate( [('Gx',),('Gx','Gy'),('Gz',)], bThrowErrorIfStringIsMissing=True ) #Gz is missing
        with self.assertRaises(ValueError):
            ds4.truncate( [('Gx',),('Gx','Gy'),('Gz',)], bThrowErrorIfStringIsMissing=True ) #Gz is missing

        #test copy
        ds2_copy = ds2.copy() #non-static
        ds4_copy = ds4.copy() #static

        #Loading and saving
        ds2.save(temp_files + "/nonstatic_dataset.saved")
        ds2.save(temp_files + "/nonstatic_dataset.saved.gz")
        with open(temp_files + "/nonstatic_dataset.stream","wb") as streamfile:
            ds2.save(streamfile)

        ds4.save(temp_files + "/static_dataset.saved")
        ds4.save(temp_files + "/static_dataset.saved.gz")
        with open(temp_files + "/static_dataset.stream","wb") as streamfile:
            ds4.save(streamfile)

        ds2.load(temp_files + "/nonstatic_dataset.saved")
        ds2.load(temp_files + "/nonstatic_dataset.saved.gz")
        with open(temp_files + "/nonstatic_dataset.stream","rb") as streamfile:
            ds2.load(streamfile)

        ds4.load(temp_files + "/static_dataset.saved")
        ds4.load(temp_files + "/static_dataset.saved.gz")
        with open(temp_files + "/static_dataset.stream","rb") as streamfile:
            ds2.load(streamfile)

        #Test various other methods
        nStrs = len(ds)
        cntDict = ds[('Gy',)].as_dict()
        asStr = str(ds[('Gy',)])
        
        ds[('Gy',)].scale(2.0)
        self.assertEqual(ds[('Gy',)]['0'], 20)
        self.assertEqual(ds[('Gy',)]['1'], 180)
        

        #Test loading a deprecated dataset file
        #dsDeprecated = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/deprecated.dataset")



    def test_from_file(self):
        # creating and loading a text-format dataset file
        dataset_txt = \
"""## Columns = 0 count, 1 count
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 20 80
"""
        with open(temp_files + "/TinyDataset.txt","w") as output:
            output.write(dataset_txt)
        ds = pygsti.io.load_dataset(temp_files + "/TinyDataset.txt")
        self.assertEqual(ds[()][('0',)], 0)
        self.assertEqual(ds[('Gx','Gy')][('1',)], 60)

        dataset_txt2 = \
"""## Columns = 0 frequency, count total
{} 0 100
Gx 0.1 100
GxGy 0.4 100
Gx^4 0.2 100
"""
        with open(temp_files + "/TinyDataset2.txt","w") as output:
            output.write(dataset_txt2)
        ds2 = pygsti.io.load_dataset(temp_files + "/TinyDataset2.txt")
        self.assertEqualDatasets(ds, ds2)


    def test_generate_fake_data(self):

        gateset = pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy','Gz'],
                                                     [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)", "Z(pi/2,Q0)"])

        depol_gateset = gateset.depolarize(gate_noise=0.1,spam_noise=0)

        fids  = pygsti.construction.gatestring_list( [ (), ('Gx',), ('Gy'), ('Gx','Gx') ] )
        germs = pygsti.construction.gatestring_list( [ ('Gi',), ('Gx',), ('Gy'), ('Gi','Gi','Gi')] )
        gateStrings = pygsti.construction.create_gatestring_list(
            "f0+T(germ,N)+f1", f0=fids, f1=fids, germ=germs, N=3,
            T=pygsti.construction.repeat_with_max_length,
            order=["germ","f0","f1"])
        pygsti.remove_duplicates_in_place(gateStrings)

        ds_none = pygsti.construction.generate_fake_data(depol_gateset, gateStrings,
                                                        nSamples=1000, sampleError='none')
        ds_round = pygsti.construction.generate_fake_data(depol_gateset, gateStrings,
                                                          nSamples=1000, sampleError='round')
        ds_binom = pygsti.construction.generate_fake_data(depol_gateset, gateStrings, nSamples=1000,
                                                          sampleError='binomial', seed=100)
        ds_multi = pygsti.construction.generate_fake_data(depol_gateset, gateStrings,
                                                          nSamples=1000, sampleError='multinomial', seed=100)
        ds_otherds = pygsti.construction.generate_fake_data(ds_none, gateStrings,
                                                             nSamples=None, sampleError='none')

        weightedStrings = [ pygsti.obj.WeightedGateString( gs.tup, weight=1.0 ) for gs in gateStrings ]
        ds_fromwts = pygsti.construction.generate_fake_data(depol_gateset, weightedStrings,
                                                            nSamples=1000, sampleError='none')

        with self.assertRaises(ValueError):
            pygsti.construction.generate_fake_data(depol_gateset, weightedStrings,
                                                   nSamples=1000, sampleError='FooBar') #invalid sampleError



        # TO SEED SAVED FILE, RUN BELOW LINES:
        #pygsti.io.write_dataset(compare_files + "/Fake_Dataset_none.txt", ds_none,  gateStrings)
        #pygsti.io.write_dataset(compare_files + "/Fake_Dataset_round.txt", ds_round, gateStrings)
        #pygsti.io.write_dataset(compare_files + "/Fake_Dataset_binom.txt", ds_binom, gateStrings)
        #pygsti.io.write_dataset(compare_files + "/Fake_Dataset_multi.txt", ds_multi, gateStrings)

        bDeepTesting = bool( 'PYGSTI_DEEP_TESTING' in os.environ and
                             os.environ['PYGSTI_DEEP_TESTING'].lower() in ("yes","1","true") )
          #Do not test *random* datasets for equality unless "deep testing", since different
          # versions/installs of numpy give different random numbers and we don't expect
          # datasets will be equal.


        saved_ds = pygsti.io.load_dataset(compare_files + "/Fake_Dataset_none.txt", cache=True)
        #print("SAVED = ",saved_ds)
        #print("NONE = ",ds_none)
        self.assertEqualDatasets(ds_none, saved_ds)

        saved_ds = pygsti.io.load_dataset(compare_files + "/Fake_Dataset_round.txt")
        self.assertEqualDatasets(ds_round, saved_ds)

        saved_ds = pygsti.io.load_dataset(compare_files + "/Fake_Dataset_binom.txt")
        if bDeepTesting and self.isPython2(): self.assertEqualDatasets(ds_binom, saved_ds)

        saved_ds = pygsti.io.load_dataset(compare_files + "/Fake_Dataset_multi.txt")
        if bDeepTesting and self.isPython2(): self.assertEqualDatasets(ds_multi, saved_ds)


    def test_gram(self):
        ds = pygsti.objects.DataSet(outcomeLabels=[('0',),('1',)])
        ds.add_count_dict( ('Gx','Gx'), {('0',): 40, ('1',): 60} )
        ds.add_count_dict( ('Gx','Gy'), {('0',): 40, ('1',): 60} )
        ds.add_count_dict( ('Gy','Gx'), {('0',): 40, ('1',): 60} )
        ds.add_count_dict( ('Gy','Gy'), {('0',): 40, ('1',): 60} )
        ds.done_adding_data()

        basis = pygsti.get_max_gram_basis( ('Gx','Gy'), ds)
        self.assertEqual(basis, [ ('Gx',), ('Gy',) ] )

        gateset = pygsti.construction.build_gateset( [2], [('Q0',)],['Gx','Gy'],
                                                     [ "X(pi/4,Q0)", "Y(pi/4,Q0)"])
        rank, evals, tgt_evals = pygsti.max_gram_rank_and_evals(ds, gateset)
        self.assertEqual(rank, 1)


    def test_multi_dataset(self):
        multi_dataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total
{} 0 100 0 100
Gx 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""
        with open(temp_files + "/TinyMultiDataset.txt","w") as output:
            output.write(multi_dataset_txt)
        multiDS = pygsti.io.load_multidataset(temp_files + "/TinyMultiDataset.txt", cache=True)

        bad_multi_dataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total
{} 0 100 0 100
FooBar 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""
        with open(temp_files + "/BadTinyMultiDataset.txt","w") as output:
            output.write(bad_multi_dataset_txt)
        with self.assertRaises(ValueError):
            pygsti.io.load_multidataset(temp_files + "/BadTinyMultiDataset.txt")

        gstrInds = collections.OrderedDict( [ (pygsti.obj.GateString(('Gx',)),slice(0,2)),
                                              (pygsti.obj.GateString(('Gx','Gy')),slice(2,4)),
                                              (pygsti.obj.GateString(('Gy',)),slice(4,6)) ] )
        olInds = collections.OrderedDict( [ ('0',0),  ('1',1) ] )

        ds1_oli = np.array( [0,1]*3, 'i' ) # 3 gate strings * 2 outcome labels
        ds1_time = np.zeros(6,'d')
        ds1_rep = 10*np.ones(6,'i')

        ds2_oli = np.array( [0,1]*3, 'i' ) # 3 gate strings * 2 outcome labels
        ds2_time = np.zeros(6,'d')
        ds2_rep = 5*np.ones(6,'i')

        mds_oli = collections.OrderedDict( [ ('ds1', ds1_oli), ('ds2', ds2_oli) ] )
        mds_time = collections.OrderedDict( [ ('ds1', ds1_time), ('ds2', ds2_time) ] )
        mds_rep = collections.OrderedDict( [ ('ds1', ds1_rep), ('ds2', ds2_rep) ] )

        mds2 = pygsti.objects.MultiDataSet(mds_oli, mds_time, mds_rep, gateStringIndices=gstrInds,
                                           outcomeLabels=['0','1'])
        mds3 = pygsti.objects.MultiDataSet(mds_oli, mds_time, mds_rep, gateStringIndices=gstrInds,
                                           outcomeLabelIndices=olInds)
        mds4 = pygsti.objects.MultiDataSet(outcomeLabels=['0','1'])
        mds5 = pygsti.objects.MultiDataSet()

        #Create a multidataset with time dependence and no rep counts
        ds1_oli = np.array( [0,1]*3, 'i' ) # 3 gate strings * 2 outcome labels
        ds1_time = np.array(np.arange(0,6),'d')

        ds2_oli = np.array( [0,1]*3, 'i' ) # 3 gate strings * 2 outcome labels
        ds2_time = np.array(np.arange(2,8),'d')

        mds_oli = collections.OrderedDict( [ ('ds1', ds1_oli), ('ds2', ds2_oli) ] )
        mds_time = collections.OrderedDict( [ ('ds1', ds1_time), ('ds2', ds2_time) ] )
        mdsNoReps = pygsti.objects.MultiDataSet(mds_oli, mds_time, None, gateStringIndices=gstrInds,
                                                outcomeLabels=['0','1'])


        #mds2.add_dataset_counts("new_ds1", ds1_cnts)
        sl_none = mds5.get_outcome_labels()

        #Create some datasets to test adding datasets to multidataset
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        ds.add_count_dict( (), {'0': 10, '1': 90} )
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds.add_count_dict( ('Gx','Gy'), {'0': 20, '1':80} )
        ds.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 20, '1':80} )
        ds.done_adding_data()

        ds2 = pygsti.objects.DataSet(outcomeLabels=['0','foobar']) #different spam labels than multids
        ds2.add_count_dict( (), {'0': 10, 'foobar': 90} )
        ds2.add_count_dict( ('Gx',), {'0': 10, 'foobar': 90} )
        ds2.add_count_dict( ('Gx','Gy'), {'0': 10, 'foobar':90} )
        ds2.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 10, 'foobar':90} )
        ds2.done_adding_data()

        ds3 = pygsti.objects.DataSet(outcomeLabels=['0','1']) #different gate strings
        ds3.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds3.done_adding_data()

        ds4 = pygsti.objects.DataSet(outcomeLabels=['0','1']) #non-static dataset
        ds4.add_count_dict( ('Gx',), {'0': 10, '1': 90} )

        multiDS['myDS'] = ds
        with self.assertRaises(ValueError):
            multiDS['badDS'] = ds2 # different spam labels
        with self.assertRaises(ValueError):
            multiDS['badDS'] = ds3 # different gates
        with self.assertRaises(ValueError):
            multiDS['badDS'] = ds4 # not static

        nStrs = len(multiDS)
        labels = list(multiDS.keys())
        self.assertEqual(labels, ['DS0', 'DS1', 'myDS'])
        self.assertTrue( multiDS.has_key('DS0') )

        for label in multiDS:
            DS = multiDS[label]
            if label in multiDS:
                pass

        for DS in multiDS.values():
            pass

        for label,DS in multiDS.items():
            pass

        #iteration over MultiDataSet without reps (slightly different logic)
        for label in mdsNoReps: 
            pass
        for label,ds in mdsNoReps.items():
            pass
        for ds in mdsNoReps.values():
            pass
        

        sumDS = multiDS.get_datasets_aggregate('DS0','DS1')
        sumDS_noReps = mdsNoReps.get_datasets_aggregate('ds1','ds2')
        multiDS_str = str(multiDS)
        multiDS_copy = multiDS.copy()

        with self.assertRaises(ValueError):
            sumDS = multiDS.get_datasets_aggregate('DS0','foobar') #bad dataset name


        #Pickle and unpickle
        with open(temp_files + '/multidataset.pickle', 'wb') as picklefile:
            pickle.dump(multiDS, picklefile)
        mds_from_pkl = None
        with open(temp_files + '/multidataset.pickle', 'rb') as picklefile:
            mds_from_pkl = pickle.load(picklefile)
        self.assertEqual(mds_from_pkl['DS0'][('Gx',)]['0'], 10)

        #Loading and saving
        multiDS.save(temp_files + "/multidataset.saved")
        multiDS.save(temp_files + "/multidataset.saved.gz")
        mdsNoReps.save(temp_files + "/multidataset_noreps.saved")
        with open(temp_files + "/multidataset.stream","wb") as streamfile:
            multiDS.save(streamfile)

        multiDS.load(temp_files + "/multidataset.saved")
        multiDS.load(temp_files + "/multidataset.saved.gz")
        mdsNoReps.load(temp_files + "/multidataset_noreps.saved")
        with open(temp_files + "/multidataset.stream","rb") as streamfile:
            multiDS.load(streamfile)
        multiDS2 = pygsti.obj.MultiDataSet(fileToLoadFrom=temp_files + "/multidataset.saved")

        #Finally, add a dataset w/reps to a multidataset without them
        mdsNoReps.add_dataset('DSwReps', mds2['ds1'])

    def test_collisionAction(self):
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'], collisionAction="keepseparate")
        ds.add_count_dict( ('Gx','Gx'), {'0':10, '1':90} )
        ds.add_count_dict( ('Gx','Gy'), {'0':20, '1':80} )
        ds.add_count_dict( ('Gx','Gx'), {'0':30, '1':70} ) # a duplicate
        self.assertEqual( ds.keys(), [ ('Gx','Gx'), ('Gx','Gy'), ('Gx','Gx','#1') ] )
        self.assertEqual( ds.keys(stripOccurrenceTags=True), [ ('Gx','Gx'), ('Gx','Gy'), ('Gx','Gx') ] )

        ds.set_row( ('Gx','Gx'), {'0': 5, '1': 95}, occurrence=1 ) #test set_row with occurrence arg


    def test_tddataset_construction(self):

        #Create an empty dataset
        #(Tests done_adding_data without adding any data)
        dsEmpty = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        dsEmpty.done_adding_data()
        
        #Create an empty dataset and add data
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        ds.add_raw_series_data( ('Gx',), #gate sequence
                            ['0','0','1','0','1','0','1','1','1','0'], #spam labels
                            [0.0, 0.2, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.35, 1.5], #time stamps
                            None) #no repeats

        with self.assertRaises(ValueError):
            ds[('Gx',)].scale(2.0) # can't scale a dataset without repeat counts

        oli = collections.OrderedDict([('0',0), ('1',1)])
        ds2 = pygsti.objects.DataSet(outcomeLabelIndices=oli)
        ds2.add_raw_series_data( ('Gy',),  #gate sequence
                             ['0','1'], #spam labels
                             [0.0, 1.0], #time stamps
                             [3,7]) #repeats


        #Create a non-static already initialized dataset
        gatestrings = pygsti.construction.gatestring_list([('Gx',), ('Gy','Gx')])
        gatestringIndices = collections.OrderedDict([ (gs,i) for i,gs in enumerate(gatestrings)])
        oliData = [ np.array([0,1,0]), np.array([1,1,0]) ]
        timeData = [ np.array([1.0,2.0,3.0]), np.array([4.0,5.0,6.0]) ]
        repData = [ np.array([1,1,1]), np.array([2,2,2]) ]
        ds = pygsti.objects.DataSet(oliData, timeData, repData, gatestrings, None,
                                      ['0','1'], None,  bStatic=False)
        ds = pygsti.objects.DataSet(oliData, timeData, repData, None, gatestringIndices,
                                      None, oli, bStatic=False) #provide gate string & spam label index dicts instead of lists
        ds = pygsti.objects.DataSet(oliData, timeData, None, None, gatestringIndices,
                                      None, oli) #no rep data is OK - just assumes 1; bStatic=False is default

        #Test loading a non-static set from a saved file
        ds.save(temp_files + "/test_tddataset.saved")
        ds3 = pygsti.objects.DataSet(fileToLoadFrom=temp_files + "/test_tddataset.saved")


        #Create an static already initialized dataset
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        GS = pygsti.objects.GateString #no auto-convert to GateStrings when using gateStringIndices
        gatestringIndices = collections.OrderedDict([ #always need this when creating a static dataset
            ( GS(('Gx',)) , slice(0,3) ),                 # (now a dict of *slices* into flattened 1D 
            ( GS(('Gy','Gx')), slice(3,6) ) ])            #  data arrays)
        oliData = np.array([0,1,0,1,1,0])
        timeData = np.array([1.0,2.0,3.0,4.0,5.0,6.0]) 
        repData = np.array([1,1,1,2,2,2])
        ds = pygsti.objects.DataSet(oliData, timeData, repData, None, gatestringIndices,
                                      ['0','1'], None,  bStatic=True)
        ds = pygsti.objects.DataSet(oliData, timeData, repData, None, gatestringIndices,
                                      None, oli, bStatic=True) #provide spam label index dict instead of list
        ds = pygsti.objects.DataSet(oliData, timeData, None, None, gatestringIndices,
                                      None, oli, bStatic=True) #no rep data is OK - just assumes 1

        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(oliData, timeData, repData, gatestrings, None,
                                     ['0','1'], None,  bStatic=True) # NEEDS gatestringIndices b/c static

        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(gateStringIndices=gatestringIndices,
                                     outcomeLabelIndices=oli, bStatic=True) #must specify data when creating a static dataset
            
        #with self.assertRaises(ValueError):
        pygsti.objects.DataSet() #OK now: no longer need at least outcomeLabels or outcomeLabelIndices
        
        
        #Test loading a static set from a saved file
        ds.save(temp_files + "/test_tddataset.saved")
        ds3 = pygsti.objects.DataSet(fileToLoadFrom=temp_files + "/test_tddataset.saved")
        
    def test_tddataset_methods(self):
        # Create a dataset from scratch

        def printInfo(ds, gstr):
            print( "*** %s info ***" % str(gstr))
            print( ds[gstr] )
            print( ds[gstr].oli )
            print( ds[gstr].time )
            print( ds[gstr].reps )
            print( ds[gstr].outcomes )
            print( ds[gstr].get_expanded_ol() )
            print( ds[gstr].get_expanded_oli() )
            print( ds[gstr].get_expanded_times() )
            print( ds[gstr].counts )
            print( ds[gstr].fractions )
            print( ds[gstr].total )
            print( ds[gstr].fraction('0') )
            print( "[0] (int) = ",ds[gstr][0] ) # integer index
            print( "[0.0] (float) = ",ds[gstr][0.0] ) # time index
            print( "['0'] (str) = ",ds[gstr]['0'] ) # outcome-label index
            print( "[('0',)] (tuple) = ",ds[gstr][('0',)] ) # outcome-label index            
            print( "at time 0 = ", ds[gstr].counts_at_time(0.0) )
            all_times, _ = ds[gstr].timeseries('all')
            print( "series('all') = ", ds[gstr].timeseries('all') )
            print( "series('0') = ",ds[gstr].timeseries('0') )
            print( "series('1') = ",ds[gstr].timeseries('1') )            
            print( "series('0',alltimes) = ",ds[gstr].timeseries('0', all_times) )
            print( len(ds[gstr]) )
            print("\n")
        
        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'])
        ds.add_raw_series_data( ('Gx',),
                            ['0','0','1','0','1','0','1','1','1','0'],
                            [0.0, 0.2, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.35, 1.5], None)

        printInfo(ds, ('Gx',) )
        
        ds[('Gy','Gy')] = (['0','1'], [0.0, 1.0]) #add via spam-labels, times
        dsNoReps = ds.copy() #tests copy() before any rep-data is added

        ds.add_raw_series_data( ('Gy',),['0','1'],[0.0, 1.0], [3,7]) #using repetitions
        ds.add_series_data( ('Gy','Gy'), [ {'0': 2, '1': 8}, {'0': 6, '1': 4}, {'1': 10} ],
                            [0.0, 1.2, 2.4])
        OD = collections.OrderedDict
        ds.add_series_data( ('Gy','Gy','Gy'), [ OD([('0',2),('1',8)]), OD([('0',6),('1',4)]), OD([('1',10)]) ],
                            [0.0, 1.2, 2.4]) # add with ordered dicts

        ds[('Gx','Gx')] = (['0','1'], [0.0, 1.0], [10,10]) #add via spam-labels, times, reps
        ds[('Gx','Gy')] = (['0','1'], [0.0, 1.0]) #add via spam-labels, times *after* we've added rep data

        printInfo(ds, ('Gy',) )
        printInfo(ds, ('Gy','Gy') )

        ds.add_raw_series_data( ('Gx','Gx'),['0','1'],[0.0, 1.0], [6,14], overwriteExisting=True) #the default
        ds.add_raw_series_data( ('Gx','Gx'),['0','1'],[1.0, 2.0], [5,10], overwriteExisting=False)

        #Setting (spamlabel,time,count) data
        ds[('Gx',)][0] = ('1',0.1,1)
        ds[('Gy',)][1] = ('0',0.4,3)
        dsNoReps[('Gx',)][0]     = ('1',0.1,1) # reps must == 1
        dsNoReps[('Gy','Gy')][1] = ('0',0.4)    # or be omitted
        printInfo(ds, ('Gx',) )
        printInfo(ds, ('Gy',) )
        
        with self.assertRaises(ValueError):
            ds[('Gx',)].outcomes = ['x','x'] #can't assign outcomes

        dsScaled = ds.copy()
        for row in dsScaled.values():
            row.scale(3.141592) # so counts are no longer intergers
        printInfo(dsScaled, ('Gx',) ) #triggers rounding warnings
        dsScaled.done_adding_data()
        printInfo(dsScaled, ('Gx',) ) # (static case)
        
        ds.done_adding_data()
        dsNoReps.done_adding_data()

        #Setting data while static is not allowed
        #with self.assertRaises(ValueError):
        #    ds[('Gx',)][0] = ('1',0.1,1) #this is OK b/c doesn't add data...
        with self.assertRaises(ValueError):
            ds.add_raw_series_data( ('Gy','Gx'),['0','1'],[0.0, 1.0], [2,2])
        with self.assertRaises(ValueError):
            ds.add_series_from_dataset(ds) #can't add to a static dataset
        with self.assertRaises(ValueError):
            dsNoReps.build_repetition_counts() #not allowed on static dataset

        #test contents
        self.assertTrue( ('Gx',) in ds)
        self.assertTrue( ('Gx',) in ds.keys())
        self.assertTrue( ds.has_key(('Gx',)) )
        self.assertEqual( list(ds.get_outcome_labels()), [('0',),('1',)] )
        self.assertEqual( list(ds.get_gate_labels()), ['Gx','Gy'] )

        #Check degrees of freedom
        ds.get_degrees_of_freedom()
        ds.get_degrees_of_freedom( [('Gx',)] )
        dsNoReps.get_degrees_of_freedom()
        dsNoReps.get_degrees_of_freedom( [('Gx',)] )
        dsScaled.get_degrees_of_freedom()
        dsScaled.get_degrees_of_freedom( [('Gx',)] )


        #test iteration
        for gstr,dsRow in ds.items():
            print(gstr, dsRow)
            dsRow2 = ds[gstr]
            spamLblIndex, timestamp, reps = dsRow[0] #can index as 3-array
            for spamLblIndex, timestamp, reps in dsRow: # or iterate over
                print(spamLblIndex, timestamp, reps)
        for dsRow in ds.values():
            print(dsRow)

        for gstr,dsRow in dsNoReps.items():
            print(gstr, dsRow)
            dsRow2 = dsNoReps[gstr]
            spamLblIndex, timestamp, reps = dsRow[0] #can index as 3-array
            for spamLblIndex, timestamp, reps in dsRow: # or iterate over
                print(spamLblIndex, timestamp, reps)                
        for dsRow in dsNoReps.values():
            print(dsRow)

            
        #Later: add_series_from_dataset(otherTDDataSet)

        print("Whole thing:")
        print(ds)

        dsWritable = ds.copy_nonstatic()
        dsWritable[('Gx',)][0] = ('1',0.1,1)
        dsWritable.add_raw_series_data( ('Gy','Gx'),['0','1'],[0.0, 1.0], [2,2])
        dsWritable.add_series_from_dataset(ds)
        
        
        dsWritable2 = dsWritable.copy_nonstatic()
         #test copy_nonstatic on already non-static dataset

        #Pickle and unpickle
        with open(temp_files + '/tddataset.pickle', 'wb') as datasetfile:
            pickle.dump(ds, datasetfile)
        ds_from_pkl = None
        with open(temp_files + '/tddataset.pickle', 'rb') as datasetfile:
            ds_from_pkl = pickle.load(datasetfile)

        # LATER: Invoke the DataSet constructor other ways

        #Test truncation
        dsWritable.truncate( [('Gx',),('Gy',)] ) #non-static
        ds.truncate( [('Gx',),('Gy',)] ) #static
        dsWritable.truncate( [('Gx',),('Gy',),('Gz',)], bThrowErrorIfStringIsMissing=False ) #non-static
        ds.truncate( [('Gx',),('Gy',),('Gz',)], bThrowErrorIfStringIsMissing=False ) #static
        with self.assertRaises(ValueError):
            dsWritable.truncate( [('Gx',),('Gy',),('Gz',)], bThrowErrorIfStringIsMissing=True ) #Gz is missing
        with self.assertRaises(ValueError):
            ds.truncate( [('Gx',),('Gy',),('Gz',)], bThrowErrorIfStringIsMissing=True ) #Gz is missing

        #Test time slicing
        print("Before [1,2) time slice")
        print(ds)
        ds_slice = ds.time_slice(1.0,2.0)
        ds_empty_slice = ds.time_slice(100.0,101.0)
        ds_slice2 = dsNoReps.time_slice(1.0,2.0)
        print("Time slice:")
        print(ds_slice)
        ds_slice = ds.time_slice(1.0,2.0,aggregateToTime=0.0)
        print("Time slice (aggregated to t=0):")
        print(ds_slice)
        
        #test copy
        dsWritable_copy = dsWritable.copy() #non-static
        ds_copy = ds.copy() #static

        #Loading and saving
        ds.save(temp_files + "/nonstatic_tddataset.saved")
        ds.save(temp_files + "/nonstatic_tddataset.saved.gz")
        with open(temp_files + "/nonstatic_tddataset.stream","wb") as streamfile:
            ds.save(streamfile)

        dsWritable.save(temp_files + "/static_tddataset.saved")
        dsWritable.save(temp_files + "/static_tddataset.saved.gz")
        with open(temp_files + "/static_tddataset.stream","wb") as streamfile:
            dsWritable.save(streamfile)

        ds.load(temp_files + "/nonstatic_tddataset.saved")
        ds.load(temp_files + "/nonstatic_tddataset.saved.gz")
        with open(temp_files + "/nonstatic_tddataset.stream","rb") as streamfile:
            ds.load(streamfile)

        dsWritable.load(temp_files + "/static_tddataset.saved")
        dsWritable.load(temp_files + "/static_tddataset.saved.gz")
        with open(temp_files + "/static_tddataset.stream","rb") as streamfile:
            dsWritable.load(streamfile)

        #Test various other methods
        nStrs = len(ds)
        #Remove these test for now since TravisCI scipy doesn't like to interpolate
        #ds.compute_fourier_filtering(verbosity=5)
        #dsT = ds.create_dataset_at_time(0.2)
        #dsT2 = ds.create_dataset_from_time_range(0,0.3)

    def test_deprecated_dataset(self):
        with open(compare_files + '/deprecated.dataset', 'rb') as datasetfile:
            ds_from_pkl = pickle.load(datasetfile)

        
    def test_tddataset_from_file(self):
        # creating and loading a text-format dataset file
        # NOTE: left of = sign is letter alias, right of = sign is spam label
        dataset_txt = \
"""## 0 = 0
## 1 = 1
{} 011001
Gx 111000111
Gy 11001100
"""
        with open(temp_files + "/TDDataset.txt","w") as output:
            output.write(dataset_txt)
        ds = pygsti.io.load_tddataset(temp_files + "/TDDataset.txt")
        self.assertEqual(ds[()].fraction('1'), 0.5)
        self.assertEqual(ds[('Gy',)].fraction('1'), 0.5)
        self.assertEqual(ds[('Gx',)].total, 9)

        bad_dataset_txt = \
"""## 0 = 0
## 1 = 1
Foobar 011001
Gx 111000111
Gy 11001100
"""
        with open(temp_files + "/BadTDDataset.txt","w") as output:
            output.write(bad_dataset_txt)
        with self.assertRaises(ValueError):
            pygsti.io.load_tddataset(temp_files + "/BadTDDataset.txt")
        

    def test_load_old_dataset(self):
        vs = "v2" if self.versionsuffix == "" else "v3"
        #pygsti.obj.results.enable_old_python_results_unpickling()
        with open(compare_files + "/pygsti0.9.3.dataset.pkl.%s" % vs,'rb') as f:
            ds = pickle.load(f)
        #pygsti.obj.results.disable_old_python_results_unpickling()
        with open(temp_files + "/repickle_old_dataset.pkl.%s" % vs,'wb') as f:
            pickle.dump(ds, f)



#OLD
#    def test_intermediate_measurements(self):
#        gs = std.gs_target.depolarize(gate_noise=0.05, spam_noise=0.1)
#        E = gs.povms['Mdefault']['0']
#        Erem = gs.povms['Mdefault']['1']
#        gs.gates['Gmz_0'] = np.dot(E,E.T)
#        gs.gates['Gmz_1'] = np.dot(Erem,Erem.T)
#        #print(gs['Gmz_0'] + gs['Gmz_1'])
#
#        gatestring_list = pygsti.construction.gatestring_list([ 
#            (),
#            ('Zmeas',),
#            ('Gx','Zmeas') 
#        ])
#        
#        ds_gen = pygsti.construction.generate_fake_data(gs, gatestring_list, nSamples=100,
#                                                        sampleError="multinomial", seed=0,
#                                                        measurementGates={'Zmeas': ['Gmz_0', 'Gmz_1']})
#        #Test copy operations
#        ds_gen2 = ds_gen.copy()
#        ds_gen3 = ds_gen.copy_nonstatic()
#
#        #create manually so no randomness
#        ds = pygsti.objects.DataSet(outcomeLabels=['0','1'],
#                                    measurementGates={'Zmeas': ['Gmz_0', 'Gmz_1']})
#        ds.add_count_list( (), [10,90] )
#        ds.add_count_list( ('Gmz_0',), [9,1] )
#        ds.add_count_list( ('Gmz_1',), [9,81] )
#        ds.add_count_list( ('Gx','Gmz_0'), [37,4] )
#        ds.add_count_list( ('Gx','Gmz_1'), [5,54] )
#        ds.done_adding_data()
#        
#        self.assertAlmostEqual( ds[('Gmz_0',)].fraction('0'), 9.0 / (9.0 + 1.0 + 9.0 + 81.0) )
#        self.assertAlmostEqual( ds[('Gx','Gmz_1')].fraction('1'), 54.0 / (37.0 + 4.0 + 5.0 + 54.0) )
#
#        ds[('Gmz_0',)]['0'] = 20
#        self.assertEqual(ds[('Gmz_0',)]['0'], 20)
#        self.assertEqual(ds[('Gmz_0',)].total, (20.0 + 1.0 + 9.0 + 81.0) )
#        ds[('Gmz_0',)].scale(0.5)
#        self.assertEqual(ds[('Gmz_0',)]['0'], 10)
#        self.assertEqual(ds[('Gmz_0',)].total, (10.0 + 0.5 + 9.0 + 81.0) )

        


if __name__ == "__main__":
    unittest.main(verbosity=2)
