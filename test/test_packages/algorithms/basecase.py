import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti.objects import Basis

from ..testutils import BaseTestCase, compare_files, regenerate_references

class AlgorithmsBase(BaseTestCase):
    def setUp(self):
        super(AlgorithmsBase, self).setUp()

        self.model = std.target_model()
        self.datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0.1)

        self.fiducials = std.fiducials
        self.germs = std.germs
        #OLD self.specs = pygsti.construction.build_spam_specs(self.fiducials, effect_labels=['E0']) #only use the first EVec

        self.op_labels = list(self.model.operations.keys()) # also == std.gates
        self.lgstStrings = pygsti.construction.create_lgst_circuits(self.fiducials, self.fiducials, self.op_labels)

        self.maxLengthList = [0,1,2,4,8]

        self.elgstStrings = pygsti.construction.create_elgst_lists(
            self.op_labels, self.germs, self.maxLengthList )

        self.lsgstStrings = pygsti.construction.create_lsgst_circuit_lists(
            self.op_labels, self.fiducials, self.fiducials, self.germs, self.maxLengthList )

        ## RUN BELOW LINES to create analysis dataset (SAVE)
        if regenerate_references():
            expList = pygsti.construction.create_lsgst_circuits(
                self.op_labels, self.fiducials, self.fiducials, self.germs, self.maxLengthList )
            ds = pygsti.construction.simulate_data(self.datagen_gateset, expList,
                                                   num_samples=10000, sample_error='binomial', seed=100)
            ds.save(compare_files + "/analysis.dataset")

        self.ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/analysis.dataset")

        ## RUN BELOW LINES to create LGST analysis dataset (SAVE)
        if regenerate_references():
            ds_lgst = pygsti.construction.simulate_data(self.datagen_gateset, self.lgstStrings,
                                                        num_samples=10000, sample_error='binomial', seed=100)
            ds_lgst.save(compare_files + "/analysis_lgst.dataset")

        self.ds_lgst = pygsti.objects.DataSet(file_to_load_from=compare_files + "/analysis_lgst.dataset")
