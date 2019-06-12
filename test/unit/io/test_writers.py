import numpy as np

from ..util import BaseCase, with_temp_path
from ..fixture_gen import io_gen

from pygsti import io
import pygsti.construction as pc
from pygsti.io import writers


def _is_comment(line):
    return line.startswith('#') and not line.startswith('##')


def _next_semantic(f):
    while True:
        line = f.readline()
        if line == '':
            return None
        if not _is_comment(line):
            return line.rstrip()


class WritersBase(BaseCase):
    def assertFilesEquivalent(self, path_a, path_b, mode='r'):
        """Helper method to assert that the contents of two files are equivalent."""
        with open(path_a, mode) as f_a:
            with open(path_b, mode) as f_b:
                while True:
                    line_a = _next_semantic(f_a)
                    line_b = _next_semantic(f_b)
                    if line_a is None or line_b is None:
                        if line_a is None and line_b is None:
                            break
                        else:
                            self.fail("Early end-of-file")
                    self.assertEqual(line_a, line_b)


class WriteDatasetTester(WritersBase):
    def setUp(self):
        self.circuit_list = io_gen._circuit_list
        self.ds = io_gen._ds
        self.fixture_path_ref = self.fixture_path('dataset_loadwrite.txt')

    @with_temp_path
    def test_write_empty_dataset(self, tmp_path):
        writers.write_empty_dataset(tmp_path, self.circuit_list, numZeroCols=2, appendWeightsColumn=False)
        # TODO assert correctness

    @with_temp_path
    def test_write_empty_dataset_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_empty_dataset(tmp_path, [('Gx',)], numZeroCols=2)

    @with_temp_path
    def test_write_empty_dataset_raises_on_need_header(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_empty_dataset(tmp_path, self.circuit_list, headerString="# Nothing ")

    @with_temp_path
    def test_write_dataset(self, tmp_path):
        writers.write_dataset(tmp_path, self.ds)
        self.assertFilesEquivalent(tmp_path, self.fixture_path('dataset_loadwrite.txt'))

    @with_temp_path
    def test_write_dataset_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_dataset(tmp_path, self.ds, [('Gx',)])


class WriteSparseDatasetTester(WritersBase):
    def setUp(self):
        self.ds = io_gen._sparse_ds

    @with_temp_path
    def test_write_sparse_dataset(self, tmp_path):
        writers.write_dataset(tmp_path, self.ds, outcomeLabelOrder=None, fixedColumnMode=True)
        self.assertFilesEquivalent(tmp_path, self.fixture_path('sparse_dataset1a.txt'))
        writers.write_dataset(tmp_path, self.ds, outcomeLabelOrder=None, fixedColumnMode=False)
        self.assertFilesEquivalent(tmp_path, self.fixture_path('sparse_dataset2a.txt'))

    @with_temp_path
    def test_write_sparse_dataset_ordered(self, tmp_path):
        ordering = io_gen._ordering
        writers.write_dataset(tmp_path, self.ds, outcomeLabelOrder=ordering, fixedColumnMode=True)
        self.assertFilesEquivalent(tmp_path, self.fixture_path('sparse_dataset1b.txt'))
        writers.write_dataset(tmp_path, self.ds, outcomeLabelOrder=ordering, fixedColumnMode=False)
        self.assertFilesEquivalent(tmp_path, self.fixture_path('sparse_dataset2b.txt'))


class WriteMultidatasetTester(WritersBase):
    def setUp(self):
        self.circuit_list = io_gen._circuit_list
        self.fixture_path_ref = self.fixture_path('TestMultiDataset.txt')
        # TODO generate dynamically
        self.mds = io.load_multidataset(str(self.fixture_path_ref))

    @with_temp_path
    def test_write_empty_multidataset(self, tmp_path):
        writers.write_empty_dataset(tmp_path, self.circuit_list,
                                    headerString='## Columns = ds1 0 count, ds1 1 count, ds2 0 count, ds2 1 count')
        # TODO assert correctness

    @with_temp_path
    def test_write_multidataset(self, tmp_path):
        writers.write_multidataset(tmp_path, self.mds, self.circuit_list)
        self.assertFilesEquivalent(tmp_path, self.fixture_path_ref)
        # TODO fix failing

    @with_temp_path
    def test_write_multidataset_with_spam_label_ordering(self, tmp_path):
        writers.write_multidataset(tmp_path, self.mds, outcomeLabelOrder=('0', '1'))
        # TODO assert correctness

    @with_temp_path
    def test_write_multidataset_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_multidataset(tmp_path, self.mds, [('Gx',)])


class WriteCircuitListTester(WritersBase):
    def setUp(self):
        self.circuit_list = io_gen._circuit_list
        self.header = io_gen._circuit_list_header
        self.fixture_path_ref = self.fixture_path('gatestringlist_loadwrite.txt')

    @with_temp_path
    def test_write_circuit_list(self, tmp_path):
        writers.write_circuit_list(tmp_path, self.circuit_list, self.header)
        self.assertFilesEquivalent(tmp_path, self.fixture_path_ref)

    @with_temp_path
    def test_write_circuit_list_raises_on_bad_type(self, tmp_path):
        with self.assertRaises(ValueError):
            writers.write_circuit_list(tmp_path, [('Gx',)], self.header)


class WriteModelTester(WritersBase):
    def setUp(self):
        self.mdl = io_gen._std_model
        self.title = io_gen._gateset_title
        self.fixture_path_ref = self.fixture_path('gateset_loadwrite.txt')

    @with_temp_path
    def test_write_model_no_identity(self, tmp_path):
        writers.write_model(io_gen._std_model_no_identity, tmp_path)
        self.assertFilesEquivalent(tmp_path, self.fixture_path('gateset_noidentity.txt'))

    @with_temp_path
    def test_write_model(self, tmp_path):
        writers.write_model(self.mdl, tmp_path, self.title)
        self.assertFilesEquivalent(tmp_path, self.fixture_path_ref)
