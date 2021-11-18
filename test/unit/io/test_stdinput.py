import unittest

from ..util import BaseCase, with_temp_path
import pygsti.io as io
from pygsti.circuits import Circuit
from pygsti.data import DataSet

class DataSetLoaderTester(BaseCase):

    @with_temp_path
    def test_load_ignore_zero_count_lines1(self, pth):
        contents = ("## Outcomes = 0, 1\n"
                    "Gc0 0:10 1:23  # {'test': 1}\n"
                    "Gc1 0:1 1:1 # {'test': 1}\n"
                    "Gc2 0:43 1:23  # {'test': 1}\n")
        with open(pth, 'w') as f:
            f.write(contents)

        ds = io.read_dataset(pth, ignore_zero_count_lines=False)
        self.assertEqual(ds[Circuit('Gc0')]['0'], 10)
        self.assertEqual(ds[Circuit('Gc1')]['0'], 1)
        self.assertEqual(ds[Circuit('Gc2')]['1'], 23)
        
        self.assertEqual(ds[Circuit('Gc0')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc1')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc2')].aux['test'], 1)

    @with_temp_path
    def test_load_ignore_zero_count_lines2(self, pth):
        contents = ("## Outcomes = 0, 1\n"
                    "Gc0 # {'test': 1}\n"
                    "Gc1 0:1 1:1 # {'test': 1}\n"
                    "Gc2   # {'test': 1}\n")
        with open(pth, 'w') as f:
            f.write(contents)

        ds = io.read_dataset(pth, ignore_zero_count_lines=False)
        self.assertEqual(ds[Circuit('Gc0')]['0'], 0)
        self.assertEqual(ds[Circuit('Gc1')]['0'], 1)
        self.assertEqual(ds[Circuit('Gc2')]['0'], 0)

        self.assertEqual(ds[Circuit('Gc0')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc1')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc2')].aux['test'], 1)


    @with_temp_path
    def test_load_ignore_zero_count_lines3(self, pth):
        contents = ("## Outcomes = 0, 1\n"
                    "Gc1 0:1 1:1 # {'test': 1}\n"
                    "Gc2  # {'test': 1}\n")
        with open(pth, 'w') as f:
            f.write(contents)

        ds = io.read_dataset(pth, ignore_zero_count_lines=False)
        self.assertEqual(ds[Circuit('Gc1')]['0'], 1)
        self.assertEqual(ds[Circuit('Gc2')]['0'], 0)

        self.assertEqual(ds[Circuit('Gc1')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc2')].aux['test'], 1)

    @with_temp_path
    def test_load_ignore_zero_count_lines4(self, pth):
        c1 = Circuit('Gc1')
        c2 = Circuit('Gc2')
        c3 = Circuit('Gc3')

        ds = DataSet()

        ds.add_count_dict(c1, {}, aux={'test':1})
        ds.add_count_dict(c2, {'0':1}, aux={'test':1})
        ds.add_count_dict(c3, {}, aux={'test':1})
        #print(ds)

        io.write_dataset(pth, ds, fixed_column_mode=False)
        ds = io.read_dataset(pth, ignore_zero_count_lines=False)

        self.assertEqual(ds[c1]['0'], 0)
        self.assertEqual(ds[c2]['0'], 1)
        self.assertEqual(ds[c3]['0'], 0)  # especially make sure last line is read in properly!

        self.assertEqual(ds[c1].aux['test'], 1)
        self.assertEqual(ds[c2].aux['test'], 1)
        self.assertEqual(ds[c3].aux['test'], 1)

    @with_temp_path
    def test_load_ignore_BAD_count_lines1(self, pth):
        contents = ("## Outcomes = 0, 1\n"
                    "Gc0 BAD  # {'test': 1}\n"
                    "Gc1 0:1 1:1 # {'test': 1}\n"
                    "Gc2 BAD  # {'test': 1}\n")
        with open(pth, 'w') as f:
            f.write(contents)

        ds = io.read_dataset(pth, ignore_zero_count_lines=False)
        self.assertEqual(ds[Circuit('Gc0')]['0'], 0)
        self.assertEqual(ds[Circuit('Gc1')]['0'], 1)
        self.assertEqual(ds[Circuit('Gc2')]['0'], 0)

        self.assertEqual(ds[Circuit('Gc0')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc1')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc2')].aux['test'], 1)

    @with_temp_path
    def test_load_ignore_BAD_count_lines2(self, pth):
        contents = ("## Outcomes = 0, 1\n"
                    "Gc1 0:1 1:1 # {'test': 1}\n"
                    "Gc2 BAD  # {'test': 1}\n"
                    "Gc3 0:1 1:1 # {'test': 1}\n"
                    "Gc4 BAD  # {'test': 1}\n")
        with open(pth, 'w') as f:
            f.write(contents)

        ds = io.read_dataset(pth, ignore_zero_count_lines=False)
        self.assertEqual(ds[Circuit('Gc1')]['0'], 1)
        self.assertEqual(ds[Circuit('Gc2')]['0'], 0)
        self.assertEqual(ds[Circuit('Gc3')]['0'], 1)
        self.assertEqual(ds[Circuit('Gc4')]['0'], 0)

        self.assertEqual(ds[Circuit('Gc1')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc2')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc3')].aux['test'], 1)
        self.assertEqual(ds[Circuit('Gc4')].aux['test'], 1)
