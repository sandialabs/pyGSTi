import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
import numpy as np
import os, time
from ..testutils import BaseTestCase, compare_files, temp_files

class TestWriteAndLoad(BaseTestCase):

    def test_dataset_file(self):

        strList = pygsti.construction.circuit_list( [(), ('Gx',), ('Gx','Gy') ] )
        #weighted_strList = [ pygsti.obj.WeightedOpString((), weight=0.1),
        #                     pygsti.obj.WeightedOpString(('Gx',), weight=2.0),
        #                     pygsti.obj.WeightedOpString(('Gx','Gy'), weight=1.5) ]
        pygsti.io.write_empty_dataset(temp_files + "/emptyDataset.txt", strList, numZeroCols=2, appendWeightsColumn=False)
        #pygsti.io.write_empty_dataset(temp_files + "/emptyDataset2.txt", weighted_strList,
        #                          headerString='## Columns = myplus count, myminus count', appendWeightsColumn=True)

        with self.assertRaises(ValueError):
            pygsti.io.write_empty_dataset(temp_files + "/emptyDataset.txt", [ ('Gx',) ], numZeroCols=2) #must be Circuits
        with self.assertRaises(ValueError):
            pygsti.io.write_empty_dataset(temp_files + "/emptyDataset.txt", strList, headerString="# Nothing ")
              #must give numZeroCols or meaningful header string (default => 2 cols)


        ds = pygsti.obj.DataSet(outcomeLabels=['0','1'], comment="Hello")
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
        ds.done_adding_data()

        pygsti.io.write_dataset(temp_files + "/dataset_loadwrite.txt",
                                ds, pygsti.construction.circuit_list(list(ds.keys()))[0:10]) #write only first 10 strings
        ds2 = pygsti.io.load_dataset(temp_files + "/dataset_loadwrite.txt")
        ds3 = pygsti.io.load_dataset(temp_files + "/dataset_loadwrite.txt", cache=True) #creates cache file
        ds4 = pygsti.io.load_dataset(temp_files + "/dataset_loadwrite.txt", cache=True) #loads from cache file

        ds.comment = "# Hello" # comment character doesn't get doubled...
        pygsti.io.write_dataset(temp_files + "/dataset_loadwrite.txt", ds,
                                outcomeLabelOrder=['0','1'])
        ds5 = pygsti.io.load_dataset(temp_files + "/dataset_loadwrite.txt", cache=True) #rewrites cache file

        for s in ds:
            self.assertEqual(ds[s]['0'],ds5[s][('0',)])
            self.assertEqual(ds[s]['1'],ds5[s][('1',)])

        with self.assertRaises(ValueError):
            pygsti.io.write_dataset(temp_files + "/dataset_loadwrite.txt",ds, [('Gx',)] ) #must be Circuits

    def test_sparse_dataset_files(self):
        ds = pygsti.objects.DataSet()

        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds[ ('Gy',) ] = {'0': 20, '1': 80}
        ds[ ('Gx','Gy') ] = {('0','0'): 30, ('1','1'): 70}
        
        ds.done_adding_data()
        print("ORIG DS:"); print(ds)

        ordering = [('0',), ('1',), ('0','0'), ('1','1')]
        pygsti.io.write_dataset(temp_files + "/sparse_dataset1.txt", ds, outcomeLabelOrder=None, fixedColumnMode=True)
        pygsti.io.write_dataset(temp_files + "/sparse_dataset2.txt", ds, outcomeLabelOrder=None, fixedColumnMode=False)
        pygsti.io.write_dataset(temp_files + "/sparse_dataset1.txt", ds, outcomeLabelOrder=ordering, fixedColumnMode=True)
        pygsti.io.write_dataset(temp_files + "/sparse_dataset2.txt", ds, outcomeLabelOrder=ordering, fixedColumnMode=False)

        ds1 = pygsti.io.load_dataset(temp_files + "/sparse_dataset1.txt")
        ds2 = pygsti.io.load_dataset(temp_files + "/sparse_dataset2.txt")

        print("\nDS1:"); print(ds1)
        print("\nDS2:"); print(ds2)

        for s in ds:
            self.assertEqual(ds[s].counts,ds1[s].counts)
            self.assertEqual(ds[s].counts,ds2[s].counts)

            
    def test_multidataset_file(self):
        strList = pygsti.construction.circuit_list( [(), ('Gx',), ('Gx','Gy') ] )
        pygsti.io.write_empty_dataset(temp_files + "/emptyMultiDataset.txt", strList,
                                           headerString='## Columns = ds1 0 count, ds1 1 count, ds2 0 count, ds2 1 count')

        multi_dataset_txt = \
"""# My Comment
## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total
# My Comment2
{} 0 100 0 100
Gx 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""
        with open(temp_files + "/TestMultiDataset.txt","w") as output:
            output.write(multi_dataset_txt)
        time.sleep(3) #so cache file is created strictly *after* dataset is created (above)

        ds = pygsti.io.load_multidataset(temp_files + "/TestMultiDataset.txt")
        ds2 = pygsti.io.load_multidataset(temp_files + "/TestMultiDataset.txt", cache=True) #creates cache file
        ds3 = pygsti.io.load_multidataset(temp_files + "/TestMultiDataset.txt", cache=True) #load from cache

        pygsti.io.write_multidataset(temp_files + "/TestMultiDataset2.txt", ds, strList)
        ds_copy = pygsti.io.load_multidataset(temp_files + "/TestMultiDataset2.txt")

        self.assertEqual(ds_copy['DS0'][('Gx',)]['0'], ds['DS0'][('Gx',)]['0'] )
        self.assertEqual(ds_copy['DS0'][('Gx','Gy')]['1'], ds['DS1'][('Gx','Gy')]['1'] )

        #write all strings in ds to file with given spam label ordering
        ds.comment = "# Hello" # comment character doesn't get doubled...
        pygsti.io.write_multidataset(temp_files + "/TestMultiDataset3.txt",
                                     ds, outcomeLabelOrder=('0','1'))

        with self.assertRaises(ValueError):
            pygsti.io.write_multidataset(
                temp_files + "/TestMultiDatasetErr.txt",ds, [('Gx',)])
              # operation sequence list must be Circuit objects




    def test_circuit_list_file(self):
        strList = pygsti.construction.circuit_list( [(), ('Gx',), ('Gx','Gy') ] )
        pygsti.io.write_circuit_list(temp_files + "/gatestringlist_loadwrite.txt", strList, "My Header")
        strList2 = pygsti.io.load_circuit_list(temp_files + "/gatestringlist_loadwrite.txt")

        pythonStrList = pygsti.io.load_circuit_list(temp_files + "/gatestringlist_loadwrite.txt",
                                                       readRawStrings=True)
        self.assertEqual(strList, strList2)
        self.assertEqual(pythonStrList[2], 'GxGy')

        with self.assertRaises(ValueError):
            pygsti.io.write_circuit_list(
                temp_files + "/gatestringlist_bad.txt",
                [ ('Gx',)], "My Header") #Must be Circuits


    def test_gateset_file(self):
        pygsti.io.write_model(std.target_model(), temp_files + "/gateset_loadwrite.txt", "My title")

        mdl_no_identity = std.target_model()
        mdl_no_identity.povm_identity = None
        pygsti.io.write_model(mdl_no_identity,
                                temp_files + "/gateset_noidentity.txt")

        mdl = pygsti.io.load_model(temp_files + "/gateset_loadwrite.txt")
        self.assertAlmostEqual(mdl.frobeniusdist(std.target_model()), 0)

        #OLD (remainder model type)
        #gateset_m1m1 = pygsti.construction.build_explicit_model([('Q0',)],['Gi','Gx','Gy'],
        #                                                 [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
        #                                                 prepLabels=['rho0'], prepExpressions=["0"],
        #                                                 effectLabels=['E0'], effectExpressions=["0"],
        #                                                 spamdefs={'0': ('rho0','E0'),
        #                                                           '1': ('remainder','remainder') })
        #pygsti.io.write_model(gateset_m1m1, temp_files + "/gateset_m1m1_loadwrite.txt", "My title m1m1")
        #mdl_m1m1 = pygsti.io.load_model(temp_files + "/gateset_m1m1_loadwrite.txt")
        #self.assertAlmostEqual(mdl_m1m1.frobeniusdist(gateset_m1m1), 0)

        gateset_txt = """# Model file using other allowed formats
PREP: rho0
StateVec
1 0

PREP: rho1
DensityMx
0 0
0 1

POVM: Mdefault

EFFECT: 00
StateVec
1 0

END POVM

GATE: Gi
UnitaryMx
 1 0
 0 1

GATE: Gx
UnitaryMxExp
 0    pi/4
pi/4   0

GATE: Gy
UnitaryMxExp
 0       -1j*pi/4
1j*pi/4    0

GATE: Gx2
UnitaryMx
 0  1
 1  0

GATE: Gy2
UnitaryMx
 0   -1j
1j    0

BASIS: pp 4
GAUGEGROUP: Full
"""
        with open(temp_files + "/formatExample.model","w") as output:
            output.write(gateset_txt)
        mdl_formats = pygsti.io.load_model(temp_files + "/formatExample.model")
        #print mdl_formats

        rotXPi   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "X(pi,Q0)")
        rotYPi   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "Y(pi,Q0)")
        rotXPiOv2   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2   = pygsti.construction.build_operation( [(4,)],[('Q0',)], "Y(pi/2,Q0)")

        self.assertArraysAlmostEqual(mdl_formats.operations['Gi'], np.identity(4,'d'))
        self.assertArraysAlmostEqual(mdl_formats.operations['Gx'], rotXPiOv2)
        self.assertArraysAlmostEqual(mdl_formats.operations['Gy'], rotYPiOv2)
        self.assertArraysAlmostEqual(mdl_formats.operations['Gx2'], rotXPi)
        self.assertArraysAlmostEqual(mdl_formats.operations['Gy2'], rotYPi)

        self.assertArraysAlmostEqual(mdl_formats.preps['rho0'], 1/np.sqrt(2)*np.array([[1],[0],[0],[1]],'d'))
        self.assertArraysAlmostEqual(mdl_formats.preps['rho1'], 1/np.sqrt(2)*np.array([[1],[0],[0],[-1]],'d'))
        self.assertArraysAlmostEqual(mdl_formats.povms['Mdefault']['00'], 1/np.sqrt(2)*np.array([[1],[0],[0],[1]],'d'))

        #pygsti.print_mx( rotXPi )
        #pygsti.print_mx( rotYPi )
        #pygsti.print_mx( rotXPiOv2 )
        #pygsti.print_mx( rotYPiOv2 )




    def test_circuit_dict_file(self):
        file_txt = "# LinearOperator string dictionary\nF1 GxGx\nF2 GxGy"  #TODO: make a Writers function for operation sequence dicts
        with open(temp_files + "/gatestringdict_loadwrite.txt","w") as output:
            output.write(file_txt)

        d = pygsti.io.load_circuit_dict(temp_files + "/gatestringdict_loadwrite.txt")
        self.assertEqual( tuple(d['F1']), ('Gx','Gx'))


    def test_gateset_writeload(self):
        mdl = std.target_model()

        for param in ('full','TP','CPTP','static'):
            print("Param: ",param)
            mdl.set_all_parameterizations(param)
            filename = temp_files + "/gateset_%s.txt" % param
            pygsti.io.write_model(mdl, filename)
            gs2 = pygsti.io.read_model(filename)
            self.assertAlmostEqual( mdl.frobeniusdist(gs2), 0.0 )
            for lbl in mdl.operations:
                self.assertEqual( type(mdl.operations[lbl]), type(gs2.operations[lbl]))
            for lbl in mdl.preps:
                self.assertEqual( type(mdl.preps[lbl]), type(gs2.preps[lbl]))
            for lbl in mdl.povms:
                self.assertEqual( type(mdl.povms[lbl]), type(gs2.povms[lbl]))
            for lbl in mdl.instruments:
                self.assertEqual( type(mdl.instruments[lbl]), type(gs2.instruments[lbl]))

        #TODO: create unknown derived classes and write this model.
        

if __name__ == "__main__":
    unittest.main(verbosity=2)
