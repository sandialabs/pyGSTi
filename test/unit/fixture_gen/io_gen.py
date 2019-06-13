"""IO test fixture generation"""
from __future__ import unicode_literals
import sys

import pygsti
from pygsti import io
from pygsti.construction import std1Q_XYI as std

from . import _memo, _write, _versioned, _FixtureGenABC, _instantiate


class IOFixtureGen(_FixtureGenABC):
    @_memo
    def _ds(self):
        ds = pygsti.obj.DataSet(outcomeLabels=['0', '1'], comment="Hello")
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds.add_count_dict(('Gx', 'Gy'), {'0': 40, '1': 60})
        ds.done_adding_data()
        return ds

    @_write
    def build_loadwrite_dataset(self):
        return 'dataset_loadwrite.txt', lambda path: io.write_dataset(str(path), self._ds)

    @_memo
    def _sparse_ds(self):
        ds = pygsti.objects.DataSet()
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds[('Gy',)] = {'0': 20, '1': 80}
        ds[('Gx', 'Gy')] = {('0', '0'): 30, ('1', '1'): 70}
        ds.done_adding_data()
        return ds

    @_write
    def build_sparse_dataset_1a(self):
        def writer(path):
            io.write_dataset(str(path), self._sparse_ds, outcomeLabelOrder=None, fixedColumnMode=True)
        return 'sparse_dataset1a.txt', writer

    @_write
    def build_sparse_dataset_2a(self):
        def writer(path):
            io.write_dataset(str(path), self._sparse_ds, outcomeLabelOrder=None, fixedColumnMode=False)
        return 'sparse_dataset2a.txt', writer

    @_memo
    def _ordering(self):
        return [('0',), ('1',), ('0', '0'), ('1', '1')]

    @_write
    def build_sparse_dataset_1b(self):
        def writer(path):
            io.write_dataset(str(path), self._sparse_ds,
                             outcomeLabelOrder=self._ordering, fixedColumnMode=True)
        return 'sparse_dataset1b.txt', writer

    @_write
    def build_sparse_dataset_2b(self):
        def writer(path):
            io.write_dataset(str(path), self._sparse_ds,
                             outcomeLabelOrder=self._ordering, fixedColumnMode=False)
        return 'sparse_dataset2b.txt', writer

    @_write
    def build_multidataset(self):
        content = """# My Comment
## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total
# My Comment2
{} 0 100 0 100
Gx 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""

        def writer(path):
            with open(str(path), 'w') as f:
                f.write(content)
        return 'TestMultiDataset.txt', writer

    @_memo
    def _circuit_list(self):
        return pygsti.construction.circuit_list([(), ('Gx',), ('Gx', 'Gy')])

    @_memo
    def _circuit_list_header(self):
        return "My Header"

    @_write
    def build_gatestringlist(self):
        return 'gatestringlist_loadwrite.txt', lambda path: io.write_circuit_list(str(path), self._circuit_list,
                                                                                  self._circuit_list_header)

    @_memo
    def _std_model(self):
        return std.target_model()

    @_memo
    def _gateset_title(self):
        return "My Title"

    @_write
    def build_gateset(self):
        return 'gateset_loadwrite.txt', lambda path: io.write_model(self._std_model, str(path), self._gateset_title)

    @_memo
    def _std_model_no_identity(self):
        mdl = std.target_model()
        mdl.povm_identity = None
        return mdl

    @_write
    def build_gateset_no_identity(self):
        return 'gateset_noidentity.txt', lambda path: io.write_model(self._std_model_no_identity, str(path))

    @_write
    def build_model_format_example(self):
        content = """# Model file using other allowed formats
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

        def writer(path):
            with open(str(path), 'w') as f:
                f.write(content)
        return 'formatExample.model', writer

    @_write
    def build_gatestringdict(self):
        content = """# LinearOperator string dictionary
F1 GxGx
F2 GxGy
"""

        def writer(path):
            with open(str(path), 'w') as f:
                f.write(content)
        return 'gatestringdict_loadwrite.txt', writer


_instantiate(__name__, IOFixtureGen)
