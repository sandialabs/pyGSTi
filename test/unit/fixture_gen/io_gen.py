"""IO test fixture generation"""
import sys

import pygsti

from . import _memo, _write, _versioned, _FixtureGenABC, _instantiate


class _M(_FixtureGenABC):

    def ds(self):
        ds = pygsti.obj.DataSet(outcomeLabels=['0', '1'], comment="Hello")
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds.add_count_dict(('Gx', 'Gy'), {'0': 40, '1': 60})
        ds.done_adding_data()
        return ds

    @_write
    def build_loadwrite_dataset(self):
        ds = self.ds()

        def writer(path):
            pygsti.io.write_dataset(path,
                                    ds,
                                    pygsti.construction.circuit_list(list(ds.keys()))[0:10]) #write only first 10 strings
        return 'dataset_loadwrite.txt', writer

    @_write
    def build_loadwrite_dataset_2(self):
        ds = self.ds()
        ds.comment = "# Hello"

        def writer(path):
            pygsti.io.write_dataset(path,
                                    ds,
                                    outcomeLabelOrder=['0','1'])
        return 'dataset_loadwrite_2.txt', writer


_instantiate(__name__, _M)
