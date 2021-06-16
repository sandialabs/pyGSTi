import collections
import os
import pickle
import unittest

import numpy as np

import pygsti
from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references


class TestDataSetMethods(BaseTestCase):

    def test_from_scratch(self):
        # Create a dataset from scratch
        ds = pygsti.objects.DataSet(outcome_labels=['0', '1'])
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds[ ('Gx',) ] = {'0': 10, '1': 90}
        ds[ ('Gx',) ]['0'] = 10
        ds[ ('Gx',) ]['1'] = 90
        with self.assertRaises(NotImplementedError):
            ds[ ('Gx',) ]['new'] = 20 # assignment can't create *new* outcome labels (yet)
        ds.add_count_dict( ('Gy','Gy'), {'FooBar': 10, '1': 90 }) # OK to add outcome labels on the fly
        ds.add_count_dict( ('Gy','Gy'), {'1': 90 }) # now all outcome labels OK now
        ds.add_count_dict(('Gy','Gy'), pygsti.obj.labeldicts.OutcomeLabelDict([('0', 10), ('1', 90)]))
        ds.done_adding_data()

        #Pickle and unpickle
        with open(temp_files + '/dataset.pickle', 'wb') as datasetfile:
            pickle.dump(ds, datasetfile)
        ds_from_pkl = None
        with open(temp_files + '/dataset.pickle', 'rb') as datasetfile:
            ds_from_pkl = pickle.load(datasetfile)
        self.assertEqual(ds_from_pkl[('Gx',)]['0'], 10)
        self.assertAlmostEqual(ds_from_pkl[('Gx',)].fractions['0'], 0.1)


        # Invoke the DataSet constructor other ways
        gstrs = [ ('Gx',), ('Gx','Gy'), ('Gy',) ]
        gstrInds = collections.OrderedDict( [ (('Gx',),0),  (('Gx','Gy'),1), (('Gy',),2) ] )
        gstrInds_static = collections.OrderedDict([(pygsti.obj.Circuit(('Gx',)), slice(0, 2)),
                                                   (pygsti.obj.Circuit(('Gx', 'Gy')), slice(2, 4)),
                                                   (pygsti.obj.Circuit(('Gy',)), slice(4, 6))])
        olInds = collections.OrderedDict( [ ('0',0),  ('1',1) ] )

        oli = np.array([0,1],'i')
        oli_static = np.array( [0,1]*3, 'd' ) # 3 operation sequences * 2 outcome labels each
        time_static = np.zeros( (6,), 'd' )
        reps_static = 10*np.ones( (6,), 'd' )

        oli_nonstc = [ oli, oli, oli ] # each item has num_outcomes elements
        time_nonstc = [ np.zeros(2,'d'), np.zeros(2,'d'), np.zeros(2,'d') ]
        reps_nonstc = [ 10*np.ones(2,'i'), 10*np.ones(2,'i'), 10*np.ones(2,'i') ]

        ds2 = pygsti.objects.DataSet(oli_nonstc, time_nonstc, reps_nonstc,
                                     circuits=gstrs, outcome_labels=['0','1'])
        ds4 = pygsti.objects.DataSet(oli_static, time_static, reps_static,
                                     circuit_indices=gstrInds_static, outcome_labels=['0','1'], static=True)

        ds2.add_counts_from_dataset(ds)

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

        #Test loading a deprecated dataset file
        #dsDeprecated = pygsti.objects.DataSet(file_to_load_from=compare_files + "/deprecated.dataset")



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
        print(ds.cirIndex.keys())
        print(('Gx','Gy') in ds)
        print(('Gx','Gy') in ds.keys())
        self.assertEqual(ds[('Gx','Gy')][('1',)], 60)

        dataset_txt2 = \
"""## Columns = 0 count, 1 count
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 20 80
"""
        with open(temp_files + "/TinyDataset2.txt","w") as output:
            output.write(dataset_txt2)
        ds2 = pygsti.io.load_dataset(temp_files + "/TinyDataset2.txt")
        self.assertEqualDatasets(ds, ds2)


    def test_generate_fake_data(self):

        model = pygsti.construction.create_explicit_model([('Q0',)], ['Gi', 'Gx', 'Gy', 'Gz'],
                                                          [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)", "Z(pi/2,Q0)"])

        depol_gateset = model.depolarize(op_noise=0.1,spam_noise=0)

        fids  = pygsti.construction.to_circuits([(), ('Gx',), ('Gy'), ('Gx', 'Gx')])
        germs = pygsti.construction.to_circuits([('Gi',), ('Gx',), ('Gy'), ('Gi', 'Gi', 'Gi')])
        circuits = pygsti.construction.create_circuits(
            "f0+T(germ,N)+f1", f0=fids, f1=fids, germ=germs, N=3,
            T=pygsti.construction.repeat_with_max_length,
            order=["germ","f0","f1"])
        pygsti.remove_duplicates_in_place(circuits)

        ds_none = pygsti.construction.simulate_data(depol_gateset, circuits,
                                                    num_samples=1000, sample_error='none')
        ds_round = pygsti.construction.simulate_data(depol_gateset, circuits,
                                                     num_samples=1000, sample_error='round')
        ds_otherds = pygsti.construction.simulate_data(ds_none, circuits,
                                                       num_samples=None, sample_error='none')

        # TO SEED SAVED FILE, RUN BELOW LINES:
        if regenerate_references():
            pygsti.io.write_dataset(compare_files + "/Fake_Dataset_none.txt", ds_none, circuits)
            pygsti.io.write_dataset(compare_files + "/Fake_Dataset_round.txt", ds_round, circuits)

        bDeepTesting = bool( 'PYGSTI_DEEP_TESTING' in os.environ and
                             os.environ['PYGSTI_DEEP_TESTING'].lower() in ("yes","1","true") )
          #Do not test *random* data for equality unless "deep testing", since different
          # versions/installs of numpy give different random numbers and we don't expect
          # data will be equal.


        saved_ds = pygsti.io.load_dataset(compare_files + "/Fake_Dataset_none.txt", cache=True)
        #print("SAVED = ",saved_ds)
        #print("NONE = ",ds_none)
        self.assertEqualDatasets(ds_none, saved_ds)

        saved_ds = pygsti.io.load_dataset(compare_files + "/Fake_Dataset_round.txt")
        self.assertEqualDatasets(ds_round, saved_ds)


    def test_multi_dataset(self):
        multi_dataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 count, DS1 1 count
{} 0 100 0 100
Gx 10 90 10 90
GxGy 40 60 40 60
Gx^4 20 80 20 80
"""
        with open(temp_files + "/TinyMultiDataset.txt","w") as output:
            output.write(multi_dataset_txt)
        multiDS = pygsti.io.load_multidataset(temp_files + "/TinyMultiDataset.txt", cache=True)

        bad_multi_dataset_txt = \
"""## Columns = DS0 0 count, DS0 1 count, DS1 0 count, DS1 1 count
{} 0 100 0 100
FooBar 10 90 10 90
GxGy 40 60 40 60
Gx^4 20 80 20 80
"""
        with open(temp_files + "/BadTinyMultiDataset.txt","w") as output:
            output.write(bad_multi_dataset_txt)
        with self.assertRaises(ValueError):
            pygsti.io.load_multidataset(temp_files + "/BadTinyMultiDataset.txt")

        gstrInds = collections.OrderedDict([(pygsti.obj.Circuit(('Gx',)), slice(0, 2)),
                                            (pygsti.obj.Circuit(('Gx', 'Gy')), slice(2, 4)),
                                            (pygsti.obj.Circuit(('Gy',)), slice(4, 6))])
        olInds = collections.OrderedDict( [ ('0',0),  ('1',1) ] )

        ds1_oli = np.array( [0,1]*3, 'i' ) # 3 operation sequences * 2 outcome labels
        ds1_time = np.zeros(6,'d')
        ds1_rep = 10*np.ones(6,'i')

        ds2_oli = np.array( [0,1]*3, 'i' ) # 3 operation sequences * 2 outcome labels
        ds2_time = np.zeros(6,'d')
        ds2_rep = 5*np.ones(6,'i')

        mds_oli = collections.OrderedDict( [ ('ds1', ds1_oli), ('ds2', ds2_oli) ] )
        mds_time = collections.OrderedDict( [ ('ds1', ds1_time), ('ds2', ds2_time) ] )
        mds_rep = collections.OrderedDict( [ ('ds1', ds1_rep), ('ds2', ds2_rep) ] )

        mds2 = pygsti.objects.MultiDataSet(mds_oli, mds_time, mds_rep, circuit_indices=gstrInds,
                                           outcome_labels=['0','1'])
        mds3 = pygsti.objects.MultiDataSet(mds_oli, mds_time, mds_rep, circuit_indices=gstrInds,
                                           outcome_label_indices=olInds)
        mds4 = pygsti.objects.MultiDataSet(outcome_labels=['0', '1'])
        mds5 = pygsti.objects.MultiDataSet()

        #Create a multidataset with time dependence and no rep counts

        ds1_time = np.array(np.arange(0,6),'d')

        ds2_oli = np.array( [0,1]*3, 'i' ) # 3 operation sequences * 2 outcome labels
        ds2_time = np.array(np.arange(2,8),'d')

        mds_oli = collections.OrderedDict( [ ('ds1', ds1_oli), ('ds2', ds2_oli) ] )
        mds_time = collections.OrderedDict( [ ('ds1', ds1_time), ('ds2', ds2_time) ] )
        mdsNoReps = pygsti.objects.MultiDataSet(mds_oli, mds_time, None, circuit_indices=gstrInds,
                                                outcome_labels=['0','1'])


        #Create some data to test adding data to multidataset
        ds = pygsti.objects.DataSet(outcome_labels=['0', '1'])
        ds.add_count_dict( (), {'0': 10, '1': 90} )
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds.add_count_dict( ('Gx','Gy'), {'0': 20, '1':80} )
        ds.add_count_dict( ('Gx','Gx','Gx','Gx'), {'0': 20, '1':80} )
        ds.done_adding_data()

        multiDS['myDS'] = ds

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
        multiDS2 = pygsti.obj.MultiDataSet(file_to_load_from=temp_files + "/multidataset.saved")

        #Finally, add a dataset w/reps to a multidataset without them
        mdsNoReps.add_dataset('DSwReps', mds2['ds1'])

    def test_tddataset_construction(self):
        #Create a non-static already initialized dataset
        circuits = pygsti.construction.to_circuits([('Gx',), ('Gy', 'Gx')])
        gatestringIndices = collections.OrderedDict([ (mdl,i) for i,mdl in enumerate(circuits)])
        oliData = [ np.array([0,1,0]), np.array([1,1,0]) ]
        timeData = [ np.array([1.0,2.0,3.0]), np.array([4.0,5.0,6.0]) ]
        repData = [ np.array([1,1,1]), np.array([2,2,2]) ]
        oli = collections.OrderedDict( [(('0',),0), (('1',),1)] )
        ds = pygsti.objects.DataSet(oliData, timeData, repData, circuits, None,
                                    ['0','1'], None, static=False)
        ds = pygsti.objects.DataSet(oliData, timeData, repData, None, gatestringIndices,
                                    None, oli, static=False) #provide operation sequence & spam label index dicts instead of lists
        ds = pygsti.objects.DataSet(oliData, timeData, None, None, gatestringIndices,
                                    None, oli) #no rep data is OK - just assumes 1; bStatic=False is default

        #Test loading a non-static set from a saved file
        ds.save(temp_files + "/test_tddataset.saved")
        ds3 = pygsti.objects.DataSet(file_to_load_from=temp_files + "/test_tddataset.saved")


        #Create an static already initialized dataset
        ds = pygsti.objects.DataSet(outcome_labels=['0', '1'])
        CIR = pygsti.objects.Circuit #no auto-convert to Circuits when using circuit_indices
        gatestringIndices = collections.OrderedDict([ #always need this when creating a static dataset
            ( CIR(('Gx',)) , slice(0,3) ),                 # (now a dict of *slices* into flattened 1D
            ( CIR(('Gy','Gx')), slice(3,6) ) ])            #  data arrays)
        oliData = np.array([0,1,0,1,1,0])
        timeData = np.array([1.0,2.0,3.0,4.0,5.0,6.0])
        repData = np.array([1,1,1,2,2,2])
        oli = collections.OrderedDict( [(('0',),0), (('1',),1)] )
        ds = pygsti.objects.DataSet(oliData, timeData, repData, None, gatestringIndices,
                                    ['0','1'], None, static=True)
        ds = pygsti.objects.DataSet(oliData, timeData, repData, None, gatestringIndices,
                                    None, oli, static=True) #provide spam label index dict instead of list
        ds = pygsti.objects.DataSet(oliData, timeData, None, None, gatestringIndices,
                                    None, oli, static=True) #no rep data is OK - just assumes 1

        #Test loading a static set from a saved file
        ds.save(temp_files + "/test_tddataset.saved")
        ds3 = pygsti.objects.DataSet(file_to_load_from=temp_files + "/test_tddataset.saved")

    def test_tddataset_methods(self):
        # Create a dataset from scratch

        def printInfo(ds, opstr):
            print( "*** %s info ***" % str(opstr))
            print( ds[opstr] )
            print( ds[opstr].oli )
            print( ds[opstr].time )
            print( ds[opstr].reps )
            print( ds[opstr].outcomes )
            print( ds[opstr].expanded_ol )
            print( ds[opstr].expanded_oli )
            print( ds[opstr].expanded_times )
            print( ds[opstr].counts )
            print( ds[opstr].fractions )
            print( ds[opstr].total )
            print( ds[opstr].fractions['0'] )
            print( "[0] (int) = ",ds[opstr][0] ) # integer index
            print( "[0.0] (float) = ",ds[opstr][0.0] ) # time index
            print( "['0'] (str) = ",ds[opstr]['0'] ) # outcome-label index
            print( "[('0',)] (tuple) = ",ds[opstr][('0',)] ) # outcome-label index
            print( "at time 0 = ", ds[opstr].counts_at_time(0.0) )
            all_times, _ = ds[opstr].timeseries('all')
            print( "series('all') = ", ds[opstr].timeseries('all') )
            print( "series('0') = ",ds[opstr].timeseries('0') )
            print( "series('1') = ",ds[opstr].timeseries('1') )
            print( "series('0',alltimes) = ",ds[opstr].timeseries('0', all_times) )
            print( len(ds[opstr]) )
            print("\n")

        ds = pygsti.objects.DataSet(outcome_labels=['0', '1'])
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

        ds.add_raw_series_data( ('Gx','Gx'),['0','1'],[0.0, 1.0], [6,14], overwrite_existing=True) #the default
        ds.add_raw_series_data( ('Gx','Gx'),['0','1'],[1.0, 2.0], [5,10], overwrite_existing=False)

        #Setting (spamlabel,time,count) data
        ds[('Gx',)][0] = ('1',0.1,1)
        ds[('Gy',)][1] = ('0',0.4,3)
        dsNoReps[('Gx',)][0]     = ('1',0.1,1) # reps must == 1
        dsNoReps[('Gy','Gy')][1] = ('0',0.4)    # or be omitted
        printInfo(ds, ('Gx',) )
        printInfo(ds, ('Gy',) )

        ds.done_adding_data()
        dsNoReps.done_adding_data()

        print("Whole thing:")
        print(ds)

        dsWritable = ds.copy_nonstatic()
        dsWritable[('Gx',)][0] = ('1',0.1,1)
        dsWritable.add_raw_series_data( ('Gy','Gx'),['0','1'],[0.0, 1.0], [2,2])
        dsWritable.add_series_from_dataset(ds)

        #Pickle and unpickle
        with open(temp_files + '/tddataset.pickle', 'wb') as datasetfile:
            pickle.dump(ds, datasetfile)
        ds_from_pkl = None
        with open(temp_files + '/tddataset.pickle', 'rb') as datasetfile:
            ds_from_pkl = pickle.load(datasetfile)


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
        ds = pygsti.io.load_time_dependent_dataset(temp_files + "/TDDataset.txt")
        self.assertEqual(ds[()].fractions['1'], 0.5)
        self.assertEqual(ds[('Gy',)].fractions['1'], 0.5)
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
            pygsti.io.load_time_dependent_dataset(temp_files + "/BadTDDataset.txt")


    def test_load_old_dataset(self):
        #pygsti.obj.results.enable_old_python_results_unpickling()
        with pygsti.io.enable_old_object_unpickling():
            with open(compare_files + "/pygsti0.9.6.dataset.pkl", 'rb') as f:
                ds = pickle.load(f)
        #pygsti.obj.results.disable_old_python_results_unpickling()
        #pygsti.io.disable_old_object_unpickling()
        with open(temp_files + "/repickle_old_dataset.pkl", 'wb') as f:
            pickle.dump(ds, f)

        with pygsti.io.enable_old_object_unpickling("0.9.7"):
            with open(compare_files + "/pygsti0.9.7.dataset.pkl", 'rb') as f:
                ds = pickle.load(f)
        with open(temp_files + "/repickle_old_dataset.pkl", 'wb') as f:
            pickle.dump(ds, f)


    def test_auxinfo(self):
        # creating and loading a text-format dataset file w/auxiliary info
        dataset_txt = \
"""## Columns = 0 count, 1 count
{} 0 100 # 'test':45
Gx 10 90 # (3,4): "value"
GxGy 40 60 # "can be": "anything", "allowed in": "a python dict", 4: {"example": "this"}
Gx^4 20 80
"""
        with open(temp_files + "/AuxDataset.txt","w") as output:
            output.write(dataset_txt)
        ds = pygsti.io.load_dataset(temp_files + "/AuxDataset.txt")
        self.assertEqual(ds[()][('0',)], 0)
        self.assertEqual(ds[('Gx','Gy')][('1',)], 60)

        self.assertEqual(ds[()].aux, {"test":45})
        self.assertEqual(ds[('Gx','Gy')].aux, {"can be": "anything", "allowed in": "a python dict", 4: {"example": "this"}})
        self.assertEqual(ds[('Gx',)].aux, { (3,4): "value" })
        self.assertEqual(ds[('Gx','Gx','Gx','Gx')].aux, {})




if __name__ == "__main__":
    unittest.main(verbosity=2)
