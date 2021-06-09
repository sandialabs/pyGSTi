"""Shared test fixtures for pygsti.objects unit tests"""
import pygsti
from pygsti.modelpacks import smq1Q_XYI as smq
from pygsti.objects import Label, Circuit, CircuitList
from ..util import Namespace

ns = Namespace()
ns.model = smq.target_model('TP')
ns.max_max_length = 2
ns.aliases = {Label(('GA1', 0)): Circuit([('Gxpi2', 0)])}


@ns.memo
def datagen_model(self):
    return self.model.depolarize(op_noise=0.05, spam_noise=0.1)


@ns.memo
def circuits(self):
    return smq.get_gst_circuits(max_max_length=self.max_max_length)


@ns.memo
def dataset(self):
    return pygsti.construction.simulate_data(
        self.datagen_model, self.circuits, 1000, seed=2020)


@ns.memo
def sparse_dataset(self):
    return pygsti.construction.simulate_data(
        self.datagen_model, self.circuits, 50, seed=2020, record_zero_counts=False)


@ns.memo
def perfect_dataset(self):
    return pygsti.construction.simulate_data(
        self.datagen_model, self.circuits, 1000, sample_error='none')


@ns.memo
def alias_circuits(self):
    alias_list = [c.replace_layer(Label(("Gxpi2", 0)), Label(("GA1", 0))) for c in self.circuits]
    return CircuitList(alias_list, self.aliases)


@ns.memo
def alias_model(self):
    aliased_model = self.model.copy()
    aliased_model.operations[('GA1', 0)] = self.model.operations[('Gxpi2', 0)]
    aliased_model.operations.pop(('Gxpi2', 0))
    return aliased_model


@ns.memo
def alias_datagen_model(self):
    aliased_model = self.datagen_model.copy()
    aliased_model.operations[('GA1', 0)] = self.datagen_model.operations[('Gxpi2', 0)]
    aliased_model.operations.pop(('Gxpi2', 0))
    return aliased_model
