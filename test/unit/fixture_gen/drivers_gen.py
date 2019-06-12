"""Driver test model & dataset fixture generation"""
import sys

from pygsti.construction import std1Q_XYI as std
import pygsti

from . import _memo, _write, _versioned, _FixtureGenABC, _instantiate


class DriverFixtureGen(_FixtureGenABC):
    def __init__(self):
        super().__init__()
        self._model = std.target_model()
        self._fiducials = std.fiducials
        self._germs = std.germs
        self._maxLengthList = [1, 2, 4]

    @_memo
    def _opLabels(self):
        return list(self._model.operations.keys())

    @_memo
    def _lsgstStrings(self):
        return pygsti.construction.make_lsgst_lists(
            self._opLabels, self._fiducials, self._fiducials, self._germs, self._maxLengthList
        )

    @_memo
    def _lsgstStrings_tgp(self):
        return pygsti.construction.make_lsgst_lists(
            self._opLabels, self._fiducials, self._fiducials, self._germs, self._maxLengthList,
            truncScheme="truncated germ powers"
        )

    @_memo
    def _lsgstStrings_lae(self):
        return pygsti.construction.make_lsgst_lists(
            self._opLabels, self._fiducials, self._fiducials, self._germs, self._maxLengthList,
            truncScheme="length as exponent"
        )

    @_memo
    def _datagen_gateset(self):
        return self._model.depolarize(op_noise=0.05, spam_noise=0.1)

    @_memo
    def _datagen_gateset2(self):
        return self._model.depolarize(op_noise=0.1, spam_noise=0.03).rotate((0.05, 0.13, 0.02))

    @_memo
    def _ds(self):
        return pygsti.construction.generate_fake_data(
            self._datagen_gateset, self._lsgstStrings[-1], nSamples=1000, sampleError='binomial', seed=100
        )

    @_memo
    def _ds2(self):
        ds2 = pygsti.construction.generate_fake_data(
            self._datagen_gateset2, self._lsgstStrings[-1], nSamples=1000, sampleError='binomial', seed=100
        ).copy_nonstatic()
        ds2.add_counts_from_dataset(self._ds)
        ds2.done_adding_data()
        return ds2

    @_memo
    def _ds_tgp(self):
        return pygsti.construction.generate_fake_data(
            self._datagen_gateset, self._lsgstStrings_tgp[-1],
            nSamples=1000, sampleError='binomial', seed=100
        )

    @_memo
    def _ds_lae(self):
        return pygsti.construction.generate_fake_data(
            self._datagen_gateset, self._lsgstStrings_lae[-1],
            nSamples=1000, sampleError='binomial', seed=100
        )

    @_write
    @_versioned
    def build_drivers_dataset(self):
        return 'drivers.dataset', lambda path: self._ds.save(str(path))

    @_write
    @_versioned
    def build_drivers_dataset_non_markovian(self):
        return 'drivers2.dataset', lambda path: self._ds2.save(str(path))

    @_write
    @_versioned
    def build_drivers_dataset_tgp(self):
        return 'drivers_tgp.dataset', lambda path: self._ds_tgp.save(str(path))

    @_write
    @_versioned
    def build_drivers_dataset_lae(self):
        return 'drivers_lae.dataset', lambda path: self._ds_lae.save(str(path))


_instantiate(__name__, DriverFixtureGen)
