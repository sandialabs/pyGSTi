import unittest
import GST
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
        ds = GST.DataSet(spamLabels=['plus','minus'])
        ds.add_count_dict( ('Gx',), {'plus': 10, 'minus': 90} )
        ds.add_count_dict( ('Gx','Gy'), {'plus': 40, 'minus': 60} )
        ds.done_adding_data()

        self.assertEquals(ds[('Gx',)]['plus'], 10)
        self.assertAlmostEqual(ds[('Gx',)].fraction('plus'), 0.1)

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
        ds = GST.load_dataset("temp_test_files/TinyDataset.txt")
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
        ds2 = GST.load_dataset("temp_test_files/TinyDataset2.txt")
        self.assertEqualDatasets(ds, ds2)


    def test_generate_fake_data(self):

        gateset = GST.build_gateset( [2], [('Q0',)],['Gi','Gx','Gy'], 
                                    [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
                                    rhoExpressions=["0"], EExpressions=["1"], 
                                    spamLabelDict={'plus': (0,0), 'minus': (0,-1) })

        depol_gateset = GST.GateSetTools.depolarize_gateset(gateset, noise=0.1)

        fids  = GST.gatestring_list( [ (), ('Gx',), ('Gy'), ('Gx','Gx') ] )
        germs = GST.gatestring_list( [ ('Gi',), ('Gx',), ('Gy'), ('Gi','Gi','Gi')] )        
        gateStrings = GST.create_gatestring_list("f0+T(germ,N)+f1", f0=fids, f1=fids,
                                                germ=germs, N=3, T=GST.GateStringTools.repeat_with_max_length,
                                                order=["germ","f0","f1"])
        GST.ListTools.remove_duplicates_in_place(gateStrings)

        ds_none = GST.generate_fake_data(depol_gateset, gateStrings, nSamples=1000, sampleError='none')
        ds_round = GST.generate_fake_data(depol_gateset, gateStrings, nSamples=1000, sampleError='round')
        ds_binom = GST.generate_fake_data(depol_gateset, gateStrings, nSamples=1000, sampleError='binomial', seed=100)
        ds_multi = GST.generate_fake_data(depol_gateset, gateStrings, nSamples=1000, sampleError='multinomial', seed=100)

        # TO SEED SAVED FILE, RUN THIS: 
        #GST.write_dataset_file("cmp_chk_files/Fake_Dataset_none.txt", gateStrings, ds_none) 
        #GST.write_dataset_file("cmp_chk_files/Fake_Dataset_round.txt", gateStrings, ds_round) 
        #GST.write_dataset_file("cmp_chk_files/Fake_Dataset_binom.txt", gateStrings, ds_binom) 
        #GST.write_dataset_file("cmp_chk_files/Fake_Dataset_multi.txt", gateStrings, ds_multi) 

        saved_ds = GST.load_dataset("cmp_chk_files/Fake_Dataset_none.txt", cache=True)
        self.assertEqualDatasets(ds_none, saved_ds)

        saved_ds = GST.load_dataset("cmp_chk_files/Fake_Dataset_round.txt")
        self.assertEqualDatasets(ds_round, saved_ds)

        saved_ds = GST.load_dataset("cmp_chk_files/Fake_Dataset_binom.txt")
        self.assertEqualDatasets(ds_binom, saved_ds)

        saved_ds = GST.load_dataset("cmp_chk_files/Fake_Dataset_multi.txt")
        self.assertEqualDatasets(ds_multi, saved_ds)

    def test_gram(self):
        ds = GST.DataSet(spamLabels=['plus','minus'])
        ds.add_count_dict( ('Gx','Gx'), {'plus': 40, 'minus': 60} )
        ds.add_count_dict( ('Gx','Gy'), {'plus': 40, 'minus': 60} )
        ds.add_count_dict( ('Gy','Gx'), {'plus': 40, 'minus': 60} )
        ds.add_count_dict( ('Gy','Gy'), {'plus': 40, 'minus': 60} )
        ds.done_adding_data()

        basis = GST.GramMatrix.get_max_gram_basis( ('Gx','Gy'), ds)
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
        multiDS = GST.load_multidataset("temp_test_files/TinyMultiDataset.txt", cache=True)
        


        
        
      
if __name__ == "__main__":
    unittest.main(verbosity=2)
