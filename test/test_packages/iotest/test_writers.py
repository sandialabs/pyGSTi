import numpy as np

from . import IOBase, with_temp_path
# from ..reference_gen import io_gen
from .references import generator as io_gen

from pygsti import io
import pygsti.construction as pc
from pygsti.io import writers


class WriteDatasetTester(IOBase):
    def setUp(self):
        self.circuit_list = io_gen.circuit_list
        self.ds = io_gen.ds
        self.reference_path_ref = self.reference_path('dataset_loadwrite.txt')

    @with_temp_path
    def test_write_empty_dataset(self, tmp_path):
        writers.write_empty_dataset(tmp_path, self.circuit_list, num_zero_cols=2, append_weights_column=False)
        # TODO assert correctness

    @with_temp_path
    def test_write_empty_dataset_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_empty_dataset(tmp_path, [('Gx',)], num_zero_cols=2)

    @with_temp_path
    def test_write_empty_dataset_raises_on_need_header(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_empty_dataset(tmp_path, self.circuit_list, header_string="# Nothing ")

    @with_temp_path
    def test_write_dataset(self, tmp_path):
        writers.write_dataset(tmp_path, self.ds)
        self.assertFilesEquivalent(tmp_path, self.reference_path('dataset_loadwrite.txt'))

    @with_temp_path
    def test_write_dataset_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_dataset(tmp_path, self.ds, [('Gx',)])


class WriteSparseDatasetTester(IOBase):
    def setUp(self):
        self.ds = io_gen.sparse_ds

    @with_temp_path
    def test_write_sparse_dataset(self, tmp_path):
        writers.write_dataset(tmp_path, self.ds, outcome_label_order=None, fixed_column_mode=True)
        self.assertFilesEquivalent(tmp_path, self.reference_path('sparse_dataset1a.txt'))
        writers.write_dataset(tmp_path, self.ds, outcome_label_order=None, fixed_column_mode=False)
        self.assertFilesEquivalent(tmp_path, self.reference_path('sparse_dataset2a.txt'))

    @with_temp_path
    def test_write_sparse_dataset_ordered(self, tmp_path):
        ordering = io_gen.ordering
        writers.write_dataset(tmp_path, self.ds, outcome_label_order=ordering, fixed_column_mode=True)
        self.assertFilesEquivalent(tmp_path, self.reference_path('sparse_dataset1b.txt'))
        writers.write_dataset(tmp_path, self.ds, outcome_label_order=ordering, fixed_column_mode=False)
        self.assertFilesEquivalent(tmp_path, self.reference_path('sparse_dataset2b.txt'))


class WriteMultidatasetTester(IOBase):
    def setUp(self):
        self.circuit_list = io_gen.circuit_list
        self.reference_path_ref = self.reference_path('TestMultiDataset.txt')
        # TODO generate dynamically
        self.mds = io.load_multidataset(str(self.reference_path_ref))

    @with_temp_path
    def test_write_empty_multidataset(self, tmp_path):
        writers.write_empty_dataset(tmp_path, self.circuit_list,
                                    header_string='## Columns = ds1 0 count, ds1 1 count, ds2 0 count, ds2 1 count')
        # TODO assert correctness

    @with_temp_path
    def test_write_multidataset(self, tmp_path):
        writers.write_multidataset(tmp_path, self.mds, self.circuit_list)
        # self.assertFilesEquivalent(tmp_path, self.reference_path_ref)
        # TODO fix failing

    @with_temp_path
    def test_write_multidataset_with_spam_label_ordering(self, tmp_path):
        writers.write_multidataset(tmp_path, self.mds, outcome_label_order=('0', '1'))
        # TODO assert correctness

    @with_temp_path
    def test_write_multidataset_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_multidataset(tmp_path, self.mds, [('Gx',)])


class WriteCircuitListTester(IOBase):
    def setUp(self):
        self.circuit_list = io_gen.circuit_list
        self.header = io_gen.circuit_list_header
        self.reference_path_ref = self.reference_path('gatestringlist_loadwrite.txt')

    @with_temp_path
    def test_write_circuit_list(self, tmp_path):
        writers.write_circuit_list(tmp_path, self.circuit_list, self.header)
        self.assertFilesEquivalent(tmp_path, self.reference_path_ref)

    @with_temp_path
    def test_write_circuit_list_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_circuit_list(tmp_path, [('Gx',)], self.header)


class WriteModelTester(IOBase):
    def setUp(self):
        self.mdl = io_gen.std_model
        self.title = io_gen.gateset_title
        self.reference_path_ref = self.reference_path('gateset_loadwrite.txt')

    @with_temp_path
    def test_write_model_no_identity(self, tmp_path):
        writers.write_model(io_gen.std_model_no_identity, tmp_path)
        self.assertFilesEquivalent(tmp_path, self.reference_path('gateset_noidentity.txt'))

    @with_temp_path
    def test_write_model(self, tmp_path):
        writers.write_model(self.mdl, tmp_path, self.title)
        self.assertFilesEquivalent(tmp_path, self.reference_path_ref)
