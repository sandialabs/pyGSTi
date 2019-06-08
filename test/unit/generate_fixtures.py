"""Build or rebuild test fixtures on the disk"""

import functools
from warnings import warn

from pygsti.construction import std1Q_XYI as std
import pygsti

from .util import Path, version_label, _TEST_DATA_PATH

__builders__ = []


def _write(fn):
    """Helper wrapper for fixture generators.

    The underlying function must return two objects, a relative
    filename and a function that will write to a given path. This
    wrapper will check if the file exists and raise if it does, unless
    called with ``force=True``.
    """
    @functools.wraps(fn)
    def inner(*args, force=False, **kwargs):
        filename, write_fn = fn(*args, **kwargs)
        filepath = _TEST_DATA_PATH / filename

        if not force and filepath.exists():
            raise FileExistsError(str(filepath))
        else:
            write_fn(filepath)
    __builders__.append(inner)
    return inner


def _versioned(fn):
    """Indicates the generated data is python-version-specific"""
    fn.__versioned__ = True
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        filename, write_fn = fn(*args, **kwargs)
        return "{}/{}".format(version_label(), filename), write_fn
    return inner


def _memo(fn):
    fn.__memo__ = None
    @functools.wraps(fn)
    def inner(self):
        if fn.__memo__ is None:
            fn.__memo__ = fn(self)
        return fn.__memo__
    return property(inner)


class _M:
    def __init__(self):
        self._model = std.target_model()
        self._fiducials = std.fiducials
        self._germs = std.germs
        self._maxLengthList = [0, 1, 2, 4, 8]
        self._CM = pygsti.baseobjs.profiler._get_mem_usage()

    @_memo
    def _datagen_gateset(self):
        return self._model.depolarize(op_noise=0.05, spam_noise=0.1)

    @_memo
    def _opLabels(self):
        return list(self._model.operations.keys())

    @_memo
    def _expList(self):
        return pygsti.construction.make_lsgst_experiment_list(
            self._opLabels, self._fiducials, self._fiducials, self._germs, self._maxLengthList)

    @_memo
    def _ds(self):
        return pygsti.construction.generate_fake_data(
            self._datagen_gateset, self._expList, nSamples=10000, sampleError='binomial', seed=100)

    @_memo
    def _lgstStrings(self):
        return pygsti.construction.list_lgst_circuits(self._fiducials, self._fiducials, self._opLabels)

    @_memo
    def _lsgstStrings(self):
        return pygsti.construction.make_lsgst_lists(
            self._opLabels, self._fiducials, self._fiducials, self._germs, self._maxLengthList)

    @_memo
    def _ds_lgst(self):
        return pygsti.construction.generate_fake_data(
            self._datagen_gateset, self._lgstStrings, nSamples=10000, sampleError='binomial', seed=100)

    @_memo
    def _mdl_lgst(self):
        return pygsti.do_lgst(self._ds, self._fiducials, self._fiducials, self._model, svdTruncateTo=4, verbosity=0)

    @_memo
    def _mdl_lgst_go(self):
        return pygsti.gaugeopt_to_target(self._mdl_lgst, self._model, {'spam': 1.0, 'gates': 1.0}, checkJac=True)

    @_memo
    def _mdl_clgst(self):
        return pygsti.contract(self._mdl_lgst_go, "CPTP")

    @_memo
    def _mdl_lsgst(self):
        return pygsti.do_iterative_mc2gst(self._ds, self._mdl_clgst, self._lsgstStrings, verbosity=0,
                                          minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
                                          memLimit=self._CM + 1024**3)

    @_memo
    def _mdl_lsgst_go(self):
        return pygsti.gaugeopt_to_target(self._mdl_lsgst, self._model, {'spam': 1.0})

    @_write
    @_versioned
    def build_analysis_dataset(self):
        return 'analysis.dataset', lambda path: self._ds.save(str(path))

    @_write
    @_versioned
    def build_lgst_analysis_dataset(self):
        return 'analysis_lgst.dataset', lambda path: self._ds_lgst.save(str(path))

    @_write
    def build_lsgst_analysis_model(self):
        return 'analysis.model', lambda path: pygsti.io.write_model(self._mdl_lsgst_go, str(path), "Saved LSGST Analysis Model")


def _build(builders, *args, **kwargs):
    _m = _M()
    for fn in builders:
        try:
            fn(_m, *args, **kwargs)
        except FileExistsError as e:
            warn("File already exists: {} (hint: use \u001b[31m--force\u001b[0m to overwrite)".format(e))


def generate_versioned(force=False):
    """Generate and write all python-version-specific test fixture data"""
    _build([f for f in __builders__ if hasattr(f, '__versioned__')], force=force)


def generate_nonversioned(force=False):
    """Generate and write all non-python-version-specific test fixture data"""
    _build([f for f in __builders__ if not hasattr(f, '__versioned__')], force=force)


def generate_all(force=False):
    """Generate and write all test fixture data"""
    _build(__builders__, force=force)


# Can be run as a script: `python -m test.unit.generate_fixtures -h'
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--force', action='store_true', help="overwrite existing test fixtures")
    parser.add_argument('-p', '--only-versioned', action='store_true', help="only build python-version-specific fixtures")
    parser.add_argument('-n', '--only-nonversioned', action='store_true', help="only build non-python-version-specific fixtures")
    args = parser.parse_args()

    if args.only_versioned:
        gen = generate_versioned
    elif args.only_nonversioned:
        gen = generate_nonversioned
    else:
        gen = generate_all

    gen(force=args.force)
