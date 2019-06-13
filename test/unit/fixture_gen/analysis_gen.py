"""Analysis model & dataset fixture generation"""
import sys

import pygsti
from pygsti.construction import std1Q_XYI as std

from . import _memo, _write, _versioned, _FixtureGenABC, _instantiate


class AnalysisFixtureGen(_FixtureGenABC):
    def __init__(self):
        super(AnalysisFixtureGen, self).__init__()
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


_instantiate(__name__, AnalysisFixtureGen)
