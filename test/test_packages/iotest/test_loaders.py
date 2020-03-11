import numpy as np

from . import IOBase, with_temp_path
from .references import generator as io_gen

from pygsti import io
import pygsti.construction as pc
from pygsti.io import loaders


class LoadersTester(IOBase):
    def test_load_dataset(self):
        ds2 = loaders.load_dataset(str(self.reference_path('dataset_loadwrite.txt')))
        ds = io_gen.ds
        for s in ds:
            self.assertEqual(ds[s]['0'], ds2[s][('0',)])
            self.assertEqual(ds[s]['1'], ds2[s][('1',)])

    @with_temp_path
    def test_load_dataset_from_cache(self, tmp_path):
        tmp_path = str(tmp_path)
        ds = io_gen.ds
        io.write_dataset(tmp_path, ds)
        ds3 = loaders.load_dataset(tmp_path, cache=True)  # creates cache file
        ds4 = loaders.load_dataset(tmp_path, cache=True)  # loads from cache file
        # TODO assert cache behavior

        ds.comment = "# Hello"  # comment character doesn't get doubled...
        io.write_dataset(tmp_path, ds)
        ds5 = loaders.load_dataset(tmp_path, cache=True)  # rewrites cache file
        for s in ds:
            self.assertEqual(ds[s]['0'], ds5[s][('0',)])
            self.assertEqual(ds[s]['1'], ds5[s][('1',)])

    def test_load_sparse_dataset(self):
        ds1a = loaders.load_dataset(str(self.reference_path('sparse_dataset1a.txt')))
        ds2a = loaders.load_dataset(str(self.reference_path('sparse_dataset2a.txt')))
        ds1b = loaders.load_dataset(str(self.reference_path('sparse_dataset1b.txt')))
        ds2b = loaders.load_dataset(str(self.reference_path('sparse_dataset2b.txt')))
        ds = io_gen.sparse_ds
        for s in ds:
            self.assertEqual(ds[s].counts, ds1a[s].counts)
            self.assertEqual(ds[s].counts, ds2a[s].counts)
            self.assertEqual(ds[s].counts, ds1b[s].counts)
            self.assertEqual(ds[s].counts, ds2b[s].counts)

    def test_load_multidataset(self):
        mds = loaders.load_multidataset(str(self.reference_path('TestMultiDataset.txt')))
        # TODO assert correctness

    @with_temp_path
    def test_load_multidataset_from_cache(self, tmp_path):
        tmp_path = str(tmp_path)
        mds = loaders.load_multidataset(str(self.reference_path('TestMultiDataset.txt')))
        io.write_multidataset(tmp_path, mds)
        mds2 = loaders.load_multidataset(tmp_path, cache=True)
        mds3 = loaders.load_multidataset(tmp_path, cache=True)
        # TODO assert cache behavior
        # TODO assert correctness

        mds.comment = "# Hello"
        io.write_multidataset(tmp_path, mds)
        mds4 = loaders.load_multidataset(tmp_path, cache=True)  # invalidate cache
        # TODO assert correctness

    def test_load_circuit_list(self):
        path = str(self.reference_path('gatestringlist_loadwrite.txt'))
        circuit_list2 = loaders.load_circuit_list(path)
        python_circuit_list = loaders.load_circuit_list(path, read_raw_strings=True)
        circuit_list = io_gen.circuit_list

        self.assertEqual(circuit_list, circuit_list2)
        self.assertEqual(python_circuit_list[2], 'GxGy')

    def test_load_model(self):
        model = io_gen.std_model
        model2 = loaders.load_model(str(self.reference_path('gateset_loadwrite.txt')))
        self.assertAlmostEqual(model2.frobeniusdist(model), 0)

    def test_load_model_alt_format(self):
        mdl_formats = loaders.load_model(str(self.reference_path('formatExample.model')))

        rotXPi = pc.build_operation([(4,)], [('Q0',)], "X(pi,Q0)")
        rotYPi = pc.build_operation([(4,)], [('Q0',)], "Y(pi,Q0)")
        rotXPiOv2 = pc.build_operation([(4,)], [('Q0',)], "X(pi/2,Q0)")
        rotYPiOv2 = pc.build_operation([(4,)], [('Q0',)], "Y(pi/2,Q0)")

        self.assertArraysAlmostEqual(mdl_formats.operations['Gi'], np.identity(4, 'd'))
        self.assertArraysAlmostEqual(mdl_formats.operations['Gx'], rotXPiOv2)
        self.assertArraysAlmostEqual(mdl_formats.operations['Gy'], rotYPiOv2)
        self.assertArraysAlmostEqual(mdl_formats.operations['Gx2'], rotXPi)
        self.assertArraysAlmostEqual(mdl_formats.operations['Gy2'], rotYPi)

        self.assertArraysAlmostEqual(mdl_formats.preps['rho0'], 1 / np.sqrt(2) * np.array([[1], [0], [0], [1]], 'd'))
        self.assertArraysAlmostEqual(mdl_formats.preps['rho1'], 1 / np.sqrt(2) * np.array([[1], [0], [0], [-1]], 'd'))
        self.assertArraysAlmostEqual(mdl_formats.povms['Mdefault']['00'],
                                     1 / np.sqrt(2) * np.array([[1], [0], [0], [1]], 'd'))

    def test_load_circuit_dict(self):
        d = loaders.load_circuit_dict(str(self.reference_path('gatestringdict_loadwrite.txt')))
        self.assertEqual(tuple(d['F1']), ('Gx', 'Gx'))
