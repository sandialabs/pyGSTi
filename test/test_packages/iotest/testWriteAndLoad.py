import unittest
import pygsti
from pygsti.construction import std1Q_XYI as std
import numpy as np
import os, time
from ..testutils import BaseTestCase, compare_files, temp_files

class TestWriteAndLoad(BaseTestCase):

    def test_dataset_file(self):

        strList = pygsti.construction.gatestring_list( [(), ('Gx',), ('Gx','Gy') ] )
        weighted_strList = [ pygsti.obj.WeightedGateString((), weight=0.1),
                             pygsti.obj.WeightedGateString(('Gx',), weight=2.0),
                             pygsti.obj.WeightedGateString(('Gx','Gy'), weight=1.5) ]
        pygsti.io.write_empty_dataset(temp_files + "/emptyDataset.txt", strList, numZeroCols=2, appendWeightsColumn=False)
        pygsti.io.write_empty_dataset(temp_files + "/emptyDataset2.txt", weighted_strList,
                                  headerString='## Columns = myplus count, myminus count', appendWeightsColumn=True)

        with self.assertRaises(ValueError):
            pygsti.io.write_empty_dataset(temp_files + "/emptyDataset.txt", [ ('Gx',) ], numZeroCols=2) #must be GateStrings
        with self.assertRaises(ValueError):
            pygsti.io.write_empty_dataset(temp_files + "/emptyDataset.txt", strList, headerString="# Nothing ")
              #must give numZeroCols or meaningful header string (default => 2 cols)


        ds = pygsti.obj.DataSet(outcomeLabels=['0','1'], comment="Hello")
        ds.add_count_dict( ('Gx',), {'0': 10, '1': 90} )
        ds.add_count_dict( ('Gx','Gy'), {'0': 40, '1': 60} )
        ds.done_adding_data()

        pygsti.io.write_dataset(temp_files + "/dataset_loadwrite.txt",
                                ds, pygsti.construction.gatestring_list(list(ds.keys()))[0:10]) #write only first 10 strings
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
            pygsti.io.write_dataset(temp_files + "/dataset_loadwrite.txt",ds, [('Gx',)] ) #must be GateStrings

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
        strList = pygsti.construction.gatestring_list( [(), ('Gx',), ('Gx','Gy') ] )
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
              # gate string list must be GateString objects




    def test_gatestring_list_file(self):
        strList = pygsti.construction.gatestring_list( [(), ('Gx',), ('Gx','Gy') ] )
        pygsti.io.write_gatestring_list(temp_files + "/gatestringlist_loadwrite.txt", strList, "My Header")
        strList2 = pygsti.io.load_gatestring_list(temp_files + "/gatestringlist_loadwrite.txt")

        pythonStrList = pygsti.io.load_gatestring_list(temp_files + "/gatestringlist_loadwrite.txt",
                                                       readRawStrings=True)
        self.assertEqual(strList, strList2)
        self.assertEqual(pythonStrList[2], 'GxGy')

        with self.assertRaises(ValueError):
            pygsti.io.write_gatestring_list(
                temp_files + "/gatestringlist_bad.txt",
                [ ('Gx',)], "My Header") #Must be GateStrings


    def test_gateset_file(self):
        pygsti.io.write_gateset(std.gs_target, temp_files + "/gateset_loadwrite.txt", "My title")

        gs_no_identity = std.gs_target.copy()
        gs_no_identity.povm_identity = None
        pygsti.io.write_gateset(gs_no_identity,
                                temp_files + "/gateset_noidentity.txt")

        gs = pygsti.io.load_gateset(temp_files + "/gateset_loadwrite.txt")
        self.assertAlmostEqual(gs.frobeniusdist(std.gs_target), 0)

        #OLD (remainder gateset type)
        #gateset_m1m1 = pygsti.construction.build_gateset([2], [('Q0',)],['Gi','Gx','Gy'],
        #                                                 [ "I(Q0)","X(pi/2,Q0)", "Y(pi/2,Q0)"],
        #                                                 prepLabels=['rho0'], prepExpressions=["0"],
        #                                                 effectLabels=['E0'], effectExpressions=["0"],
        #                                                 spamdefs={'0': ('rho0','E0'),
        #                                                           '1': ('remainder','remainder') })
        #pygsti.io.write_gateset(gateset_m1m1, temp_files + "/gateset_m1m1_loadwrite.txt", "My title m1m1")
        #gs_m1m1 = pygsti.io.load_gateset(temp_files + "/gateset_m1m1_loadwrite.txt")
        #self.assertAlmostEqual(gs_m1m1.frobeniusdist(gateset_m1m1), 0)

        gateset_txt = """# Gateset file using other allowed formats
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

BASIS: pp 2
GAUGEGROUP: Full
"""
        with open(temp_files + "/formatExample.gateset","w") as output:
            output.write(gateset_txt)
        gs_formats = pygsti.io.load_gateset(temp_files + "/formatExample.gateset")
        #print gs_formats

        rotXPi   = pygsti.construction.build_gate( [2],[('Q0',)], "X(pi,Q0)")
        rotYPi   = pygsti.construction.build_gate( [2],[('Q0',)], "Y(pi,Q0)")
        rotXPiOv2   = pygsti.construction.build_gate( [2],[('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2   = pygsti.construction.build_gate( [2],[('Q0',)], "Y(pi/2,Q0)")

        self.assertArraysAlmostEqual(gs_formats.gates['Gi'], np.identity(4,'d'))
        self.assertArraysAlmostEqual(gs_formats.gates['Gx'], rotXPiOv2)
        self.assertArraysAlmostEqual(gs_formats.gates['Gy'], rotYPiOv2)
        self.assertArraysAlmostEqual(gs_formats.gates['Gx2'], rotXPi)
        self.assertArraysAlmostEqual(gs_formats.gates['Gy2'], rotYPi)

        self.assertArraysAlmostEqual(gs_formats.preps['rho0'], 1/np.sqrt(2)*np.array([[1],[0],[0],[1]],'d'))
        self.assertArraysAlmostEqual(gs_formats.preps['rho1'], 1/np.sqrt(2)*np.array([[1],[0],[0],[-1]],'d'))
        self.assertArraysAlmostEqual(gs_formats.povms['Mdefault']['00'], 1/np.sqrt(2)*np.array([[1],[0],[0],[1]],'d'))

        #pygsti.print_mx( rotXPi )
        #pygsti.print_mx( rotYPi )
        #pygsti.print_mx( rotXPiOv2 )
        #pygsti.print_mx( rotYPiOv2 )




    def test_gatestring_dict_file(self):
        file_txt = "# Gate string dictionary\nF1 GxGx\nF2 GxGy"  #TODO: make a Writers function for gate string dicts
        with open(temp_files + "/gatestringdict_loadwrite.txt","w") as output:
            output.write(file_txt)

        d = pygsti.io.load_gatestring_dict(temp_files + "/gatestringdict_loadwrite.txt")
        self.assertEqual( tuple(d['F1']), ('Gx','Gx'))


    def test_gateset_writeload(self):
        gs = std.gs_target.copy()

        for param in ('full','TP','CPTP','static'):
            print("Param: ",param)
            gs.set_all_parameterizations(param)
            filename = temp_files + "/gateset_%s.txt" % param
            pygsti.io.write_gateset(gs, filename)
            gs2 = pygsti.io.read_gateset(filename)
            self.assertAlmostEqual( gs.frobeniusdist(gs2), 0.0 )
            for lbl in gs.gates:
                self.assertEqual( type(gs.gates[lbl]), type(gs2.gates[lbl]))
            for lbl in gs.preps:
                self.assertEqual( type(gs.preps[lbl]), type(gs2.preps[lbl]))
            for lbl in gs.povms:
                self.assertEqual( type(gs.povms[lbl]), type(gs2.povms[lbl]))
            for lbl in gs.instruments:
                self.assertEqual( type(gs.instruments[lbl]), type(gs2.instruments[lbl]))

        #TODO: create unknown derived classes and write this gateset.
        

if __name__ == "__main__":
    unittest.main(verbosity=2)
