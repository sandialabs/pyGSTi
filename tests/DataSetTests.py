import unittest
import collections
import pickle
import pygsti
import numpy as np

class DataSetTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def assertEqualDatasets(self, ds1, ds2):
        self.assertEqual(len(ds1),len(ds2))
        for gatestring in ds1:
            self.assertAlmostEqual( ds1[gatestring]['plus'], ds2[gatestring]['plus'], places=3 )
            self.assertAlmostEqual( ds1[gatestring]['minus'], ds2[gatestring]['minus'], places=3 )


class TestDataSetMethods(DataSetTestCase):

    def test_from_scratch(self):
        # Create a dataset from scratch
        ds = pygsti.objects.DataSet(spamLabels=['plus','minus'])
        ds.add_count_dict( ('Gx',), {'plus': 10, 'minus': 90} )
        ds.add_counts_1q( ('Gx','Gy'), 10, 40 )
        ds.add_counts_1q( ('Gx','Gy'), 40, 10 )
        with self.assertRaises(ValueError):
            ds.add_count_dict( ('Gx',), {'FooBar': 10, 'minus': 90 }) #bad spam label
        with self.assertRaises(ValueError):
            ds.add_count_dict( ('Gx',), {'minus': 90 }) # not all spam labels
        with self.assertRaises(ValueError):
            ds[('Gx',)] = { 'plus':10, 'minus': 90 } #can't set counts directly -- use addData functions
        with self.assertRaises(ValueError):
            ds[('Gx',)]['plus'] = 10 #can't set counts directly -- use addData functions
        ds.done_adding_data()
        
        ds_str = str(ds)

        with self.assertRaises(ValueError):
            ds.add_count_dict( ('Gx',), {'plus': 10, 'minus': 90 }) # done adding data
        with self.assertRaises(ValueError):
            ds.add_counts_1q( ('Gx',), 40,60) # done adding data

        self.assertEquals(ds[('Gx',)]['plus'], 10)
        self.assertAlmostEqual(ds[('Gx',)].fraction('plus'), 0.1)
        
        #Pickle and unpickle
        pickle.dump(ds, open("temp_test_files/dataset.pickle","w"))
        ds_from_pkl = pickle.load(open("temp_test_files/dataset.pickle","r"))
        self.assertEquals(ds_from_pkl[('Gx',)]['plus'], 10)
        self.assertAlmostEqual(ds_from_pkl[('Gx',)].fraction('plus'), 0.1)


        # Invoke the DataSet constructor other ways
        gstrs = [ ('Gx',), ('Gx','Gy'), ('Gy',) ]
        gstrInds = collections.OrderedDict( [ (('Gx',),0),  (('Gx','Gy'),1), (('Gy',),2) ] )
        slInds = collections.OrderedDict( [ ('plus',0),  ('minus',1) ] )
        cnts_static = np.ones( (3,2), 'd' ) # 3 gate strings, 2 spam labels
        cnts_nonstc = [ np.ones(2,'d'), np.ones(2,'d'), np.ones(2,'d') ]

        ds2 = pygsti.objects.DataSet(cnts_nonstc, gateStrings=gstrs, spamLabels=['plus','minus'])
        ds3 = pygsti.objects.DataSet(cnts_nonstc, gateStringIndices=gstrInds, spamLabelIndices=slInds)
        ds4 = pygsti.objects.DataSet(cnts_static, gateStrings=gstrs,
                                     spamLabels=['plus','minus'], bStatic=True)
        ds5 = pygsti.objects.DataSet(cnts_nonstc, gateStrings=gstrs,
                                     spamLabels=['plus','minus'], bStatic=False)
        ds6 = pygsti.objects.DataSet(spamLabels=['plus','minus'])
        ds6.done_adding_data() #ds6 = empty dataset

        ds2.add_counts_from_dataset(ds)
        ds3.add_counts_from_dataset(ds)
        with self.assertRaises(ValueError):
            ds4.add_counts_from_dataset(ds) #can't add to static DataSet

        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(gateStrings=gstrs) #no spam labels specified
        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(cnts_static, spamLabels=['plus','minus'], bStatic=True)
              #must specify gateLabels (or indices) when creating static DataSet
        with self.assertRaises(ValueError):
            pygsti.objects.DataSet(gateStrings=gstrs, spamLabels=['plus','minus'], bStatic=True)
              #must specify counts when creating static DataSet


        #Test indexing methods
        cnt = 0
        for gstr in ds:

            if gstr in ds:
                if ds.has_key(gstr):
                    pass
                if ds.has_key(pygsti.obj.GateString(gstr)):
                    pass

            dsRow = ds[gstr]
            allLabels = dsRow.keys()
            for spamLabel in dsRow:
                if spamLabel in dsRow: #we know to be true
                    cnt = dsRow[spamLabel]
                if dsRow.has_key(spamLabel):
                    cnt = dsRow[spamLabel]
            
        for dsRow in ds.itervalues():
            for spamLabel,count in dsRow.iteritems():
                cnt += count

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
        ds2.save("temp_test_files/nonstatic_dataset.saved")
        ds2.save("temp_test_files/nonstatic_dataset.saved.gz")
        ds2.save(open("temp_test_files/nonstatic_dataset.stream","w"))

        ds4.save("temp_test_files/static_dataset.saved")
        ds4.save("temp_test_files/static_dataset.saved.gz")
        ds4.save(open("temp_test_files/static_dataset.stream","w"))

        ds2.load("temp_test_files/nonstatic_dataset.saved")
        ds2.load("temp_test_files/nonstatic_dataset.saved.gz")
        ds2.load(open("temp_test_files/nonstatic_dataset.stream","r"))

        ds4.load("temp_test_files/static_dataset.saved")
        ds4.load("temp_test_files/static_dataset.saved.gz")
        ds4.load(open("temp_test_files/static_dataset.stream","r"))

        #Test various other methods
        nStrs = len(ds)
        cntDict = ds[('Gx',)].as_dict()
        asStr = str(ds[('Gx',)])

                

    def test_from_file(self):
        # creating and loading a text-format dataset file
        dataset_txt = \
"""## Columns = plus count, minus count
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 20 80
"""
        open("temp_test_files/TinyDataset.txt","w").write(dataset_txt)
        ds = pygsti.io.load_dataset("temp_test_files/TinyDataset.txt")
        self.assertEquals(ds[()]['plus'], 0)
        self.assertEquals(ds[('Gx','Gy')]['minus'], 60)

        dataset_txt2 = \
"""## Columns = plus frequency, count total
{} 0 100
Gx 0.1 100
GxGy 0.4 100
Gx^4 0.2 100
"""
        open("temp_test_files/TinyDataset2.txt","w").write(dataset_txt2)
        ds2 = pygsti.io.load_dataset("temp_test_files/TinyDataset2.txt")
        self.assertEqualDatasets(ds, ds2)


    def test_generate_fake_data(self):

        gateset = pygsti.construction.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                                     [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                                     rhoExpressions=["0"], EExpressions=["1"], 
                                                     spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

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

        

        # TO SEED SAVED FILE, RUN THIS: 
        #pygsti.io.write_dataset("cmp_chk_files/Fake_Dataset_none.txt", gateStrings, ds_none) 
        #pygsti.io.write_dataset("cmp_chk_files/Fake_Dataset_round.txt", gateStrings, ds_round) 
        #pygsti.io.write_dataset("cmp_chk_files/Fake_Dataset_binom.txt", gateStrings, ds_binom) 
        #pygsti.io.write_dataset("cmp_chk_files/Fake_Dataset_multi.txt", gateStrings, ds_multi) 

        saved_ds = pygsti.io.load_dataset("cmp_chk_files/Fake_Dataset_none.txt", cache=True)
        self.assertEqualDatasets(ds_none, saved_ds)

        saved_ds = pygsti.io.load_dataset("cmp_chk_files/Fake_Dataset_round.txt")
        self.assertEqualDatasets(ds_round, saved_ds)

        saved_ds = pygsti.io.load_dataset("cmp_chk_files/Fake_Dataset_binom.txt")
        self.assertEqualDatasets(ds_binom, saved_ds)

        saved_ds = pygsti.io.load_dataset("cmp_chk_files/Fake_Dataset_multi.txt")
        self.assertEqualDatasets(ds_multi, saved_ds)

    def test_gram(self):
        ds = pygsti.objects.DataSet(spamLabels=['plus','minus'])
        ds.add_count_dict( ('Gx','Gx'), {'plus': 40, 'minus': 60} )
        ds.add_count_dict( ('Gx','Gy'), {'plus': 40, 'minus': 60} )
        ds.add_count_dict( ('Gy','Gx'), {'plus': 40, 'minus': 60} )
        ds.add_count_dict( ('Gy','Gy'), {'plus': 40, 'minus': 60} )
        ds.done_adding_data()

        basis = pygsti.get_max_gram_basis( ('Gx','Gy'), ds)
        self.assertEqual(basis, [ ('Gx',), ('Gy',) ] )

    def test_multi_dataset(self):
        multi_dataset_txt = \
"""## Columns = DS0 plus count, DS0 minus count, DS1 plus frequency, DS1 count total
{} 0 100 0 100
Gx 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""
        open("temp_test_files/TinyMultiDataset.txt","w").write(multi_dataset_txt)
        multiDS = pygsti.io.load_multidataset("temp_test_files/TinyMultiDataset.txt", cache=True)

        bad_multi_dataset_txt = \
"""## Columns = DS0 plus count, DS0 minus count, DS1 plus frequency, DS1 count total
{} 0 100 0 100
FooBar 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""
        open("temp_test_files/BadTinyMultiDataset.txt","w").write(bad_multi_dataset_txt)
        with self.assertRaises(ValueError):
            pygsti.io.load_multidataset("temp_test_files/BadTinyMultiDataset.txt")

        gstrs = [ ('Gx',), ('Gx','Gy'), ('Gy',) ]
        gstrInds = collections.OrderedDict( [ (('Gx',),0),  (('Gx','Gy'),1), (('Gy',),2) ] )
        slInds = collections.OrderedDict( [ ('plus',0),  ('minus',1) ] )
        ds1_cnts = np.ones( (3,2), 'd' ) # 3 gate strings, 2 spam labels
        ds2_cnts = 10*np.ones( (3,2), 'd' ) # 3 gate strings, 2 spam labels
        cnts = collections.OrderedDict( [ ('ds1', ds1_cnts), ('ds2', ds2_cnts) ] )

        mds2 = pygsti.objects.MultiDataSet(cnts, gateStrings=gstrs, spamLabels=['plus','minus'])
        mds3 = pygsti.objects.MultiDataSet(cnts, gateStringIndices=gstrInds, spamLabelIndices=slInds)
        mds4 = pygsti.objects.MultiDataSet(spamLabels=['plus','minus'])
        mds5 = pygsti.objects.MultiDataSet()

        mds2.add_dataset_counts("new_ds1", ds1_cnts)

        #Create some datasets to test adding datasets to multidataset
        ds = pygsti.objects.DataSet(spamLabels=['plus','minus'])
        ds.add_count_dict( (), {'plus': 10, 'minus': 90} )
        ds.add_count_dict( ('Gx',), {'plus': 10, 'minus': 90} )
        ds.add_counts_1q( ('Gx','Gy'), 20, 80 )
        ds.add_counts_1q( ('Gx','Gx','Gx','Gx'), 20, 80 )
        ds.done_adding_data()

        ds2 = pygsti.objects.DataSet(spamLabels=['plus','foobar']) #different spam labels than multids
        ds2.add_count_dict( (), {'plus': 10, 'foobar': 90} )
        ds2.add_count_dict( ('Gx',), {'plus': 10, 'foobar': 90} )
        ds2.add_count_dict( ('Gx','Gy'), {'plus': 10, 'foobar': 90} )
        ds2.add_count_dict( ('Gx','Gx','Gx','Gx'), {'plus': 10, 'foobar': 90} )
        ds2.done_adding_data()

        ds3 = pygsti.objects.DataSet(spamLabels=['plus','minus']) #different gate strings
        ds3.add_count_dict( ('Gx',), {'plus': 10, 'minus': 90} )
        ds3.done_adding_data()

        ds4 = pygsti.objects.DataSet(spamLabels=['plus','minus']) #non-static dataset
        ds4.add_count_dict( ('Gx',), {'plus': 10, 'minus': 90} )

        multiDS['myDS'] = ds
        with self.assertRaises(ValueError):
            multiDS['badDS'] = ds2 # different spam labels
        with self.assertRaises(ValueError):
            multiDS['badDS'] = ds3 # different gates
        with self.assertRaises(ValueError):
            multiDS['badDS'] = ds4 # not static

        nStrs = len(multiDS)
        labels = multiDS.keys()
        self.assertEqual(labels, ['DS0', 'DS1', 'myDS'])

        for label in multiDS:
            DS = multiDS[label]
            if label in multiDS:
                pass
            if multiDS.has_key(label):
                pass

        for DS in multiDS.itervalues():
            pass

        for label,DS in multiDS.iteritems():
            pass

        sumDS = multiDS.get_datasets_sum('DS0','DS1')
        multiDS_str = str(multiDS)
        multiDS_copy = multiDS.copy()


        #Pickle and unpickle
        pickle.dump(multiDS, open("temp_test_files/multidataset.pickle","w"))
        mds_from_pkl = pickle.load(open("temp_test_files/multidataset.pickle","r"))
        self.assertEquals(mds_from_pkl['DS0'][('Gx',)]['plus'], 10)

        #Loading and saving
        multiDS.save("temp_test_files/multidataset.saved")
        multiDS.save("temp_test_files/multidataset.saved.gz")
        multiDS.save(open("temp_test_files/multidataset.stream","w"))

        multiDS.load("temp_test_files/multidataset.saved")
        multiDS.load("temp_test_files/multidataset.saved.gz")
        multiDS.load(open("temp_test_files/multidataset.stream","r"))
        multiDS2 = pygsti.obj.MultiDataSet(fileToLoadFrom="temp_test_files/multidataset.saved")
        


        

        
        
      
if __name__ == "__main__":
    unittest.main(verbosity=2)
