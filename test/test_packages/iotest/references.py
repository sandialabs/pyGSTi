"""IO test reference generation"""
import functools
from pathlib import Path

import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as std
from . import TEMP_FILE_PATH


def _write(filename):
    def decorator(fn):
        fn.__ref_file__ = filename

        @functools.wraps(fn)
        def wrapper(self, root):
            return fn(self, Path(root) / filename)
        return wrapper
    return decorator


def _memo(fn):
    return property(functools.lru_cache(maxsize=1)(fn))


class IOGen:
    def __init__(self):
        super(IOGen, self).__init__()
        self._writer_map = {}
        for name in dir(self):
            member = getattr(self, name)
            if hasattr(member, '__ref_file__'):
                self._writer_map[member.__ref_file__] = member

    @_memo
    def ds(self):
        ds = pygsti.data.DataSet(outcome_labels=['0', '1'], comment="Hello")
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds.add_count_dict(('Gx', 'Gy'), {'0': 40, '1': 60})
        ds.done_adding_data()
        return ds

    @_write('dataset_loadwrite.txt')
    def loadwrite_dataset(self, path):
        pygsti.io.write_dataset(path, self.ds)

    @_memo
    def sparse_ds(self):
        ds = pygsti.data.DataSet()
        ds.add_count_dict(('Gx',), {'0': 10, '1': 90})
        ds[('Gy',)] = {'0': 20, '1': 80}
        ds[('Gx', 'Gy')] = {('0', '0'): 30, ('1', '1'): 70}
        ds.done_adding_data()
        return ds

    @_write('sparse_dataset1a.txt')
    def sparse_dataset_1a(self, path):
        pygsti.io.write_dataset(
            path, self.sparse_ds, outcome_label_order=None,
            fixed_column_mode=True
        )

    @_write('sparse_dataset2a.txt')
    def sparse_dataset_2a(self, path):
        pygsti.io.write_dataset(
            path, self.sparse_ds, outcome_label_order=None,
            fixed_column_mode=False
        )

    @_memo
    def ordering(self):
        return [('0',), ('1',), ('0', '0'), ('1', '1')]

    @_write('sparse_dataset1b.txt')
    def sparse_dataset_1b(self, path):
        pygsti.io.write_dataset(
            path, self.sparse_ds, outcome_label_order=self.ordering,
            fixed_column_mode=True
        )

    @_write('sparse_dataset2b.txt')
    def sparse_dataset_2b(self, path):
        pygsti.io.write_dataset(
            path, self.sparse_ds, outcome_label_order=self.ordering,
            fixed_column_mode=False
        )

    @_write('TestMultiDataset.txt')
    def multidataset(self, path):
        content = """# My Comment
## Columns = DS0 0 count, DS0 1 count, DS1 0 frequency, DS1 count total
# My Comment2
{} 0 100 0 100
Gx 10 90 0.1 100
GxGy 40 60 0.4 100
Gx^4 20 80 0.2 100
"""

        with open(str(path), 'w') as f:
            f.write(content)

    @_memo
    def circuit_list(self):
        return pygsti.circuits.to_circuits([(), ('Gx',), ('Gx', 'Gy')])

    @_memo
    def circuit_list_header(self):
        return "My Header"

    @_write('gatestringlist_loadwrite.txt')
    def gatestringlist_loadwrite(self, path):
        pygsti.io.write_circuit_list(path, self.circuit_list, self.circuit_list_header)

    @_memo
    def std_model(self):
        return std.target_model()

    @_memo
    def gateset_title(self):
        return "My Title"

    @_write('gateset_loadwrite.txt')
    def gateset(self, path):
        pygsti.io.write_model(self.std_model, path, self.gateset_title)

    @_memo
    def std_model_no_identity(self):
        mdl = std.target_model()
        mdl.povm_identity = None
        return mdl

    @_write('gateset_noidentity.txt')
    def gateset_noidentity(self, path):
        pygsti.io.write_model(self.std_model_no_identity, path)

    @_write('formatExample.model')
    def model_format_example(self, path):
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

        with open(str(path), 'w') as f:
            f.write(content)

    @_write('gatestringdict_loadwrite.txt')
    def gatestringdict_loadwrite(self, path):
        content = """# LinearOperator string dictionary
F1 GxGx
F2 GxGy
"""

        with open(str(path), 'w') as f:
            f.write(content)

    def write(self, file_path):
        file_path.parent.mkdir(exist_ok=True, parents=True)
        self._writer_map[file_path.name](file_path.parent)

    def write_all(self, root):
        for writer in self._writer_map.values():
            writer(root)


# Singleton instance
generator = IOGen()


if __name__ == '__main__':
    TEMP_FILE_PATH.mkdir(exist_ok=True, parents=True)
    generator.write_all(TEMP_FILE_PATH)
