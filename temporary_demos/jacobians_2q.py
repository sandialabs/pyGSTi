
import numpy as np

from pygsti.forwardsims import ForwardSimulator, MapForwardSimulator, TorchForwardSimulator
from pygsti.models import Model
from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout
from pygsti.circuits import Circuit
from pygsti.modelpacks import smq2Q_XYICNOT, ModelPack
import time

from typing import Sequence

BulkFillDProbsArgs = tuple[np.ndarray, CircuitOutcomeProbabilityArrayLayout]


def get_sim_name(sim):
    tstr = str(type(sim)).lower()
    if 'map' in  tstr:
        simname = 'Map  ' 
    elif 'torch' in tstr:
        simname = 'Torch'
    elif 'matrix' in tstr:
        simname = 'Mat  '
    else:
        raise ValueError()
    if 'simple' in tstr:
        raise ValueError()
    return simname


class Benchmarker:

    def __init__(self, model: Model, sims: list[ForwardSimulator], circuits : Sequence[Circuit]):
        assert isinstance(sims, list)
        # assert isinstance(sims[0], ForwardSimulator)
        self.sims = sims
        self.base_model = model
        self.models = [model.copy() for _ in range(len(self.sims))]
        for i,m in enumerate(self.models):
            m.sim = sims[i]
        assert len(circuits) > 0
        assert isinstance(circuits[0], Circuit)
        self.circuits = np.array(circuits, dtype=object)
        return
    
    @staticmethod
    def from_modelpack(modelpack: ModelPack, sims: list[ForwardSimulator], gst_maxlen: int):
        model_ideal = modelpack.target_model()
        model_ideal.convert_members_inplace(to_type='full TP')
        # ^ TorchFowardSimulator can only work with TP modelmembers.
        model_noisy = model_ideal.depolarize(op_noise=0.05, spam_noise=0.025)
        exp_design = modelpack.create_gst_experiment_design(max_max_length=gst_maxlen, fpr=True)
        circuits = exp_design.circuit_lists[-1]._circuits
        fsth = Benchmarker(model_noisy, sims, circuits)
        return fsth
    
    def prep_parallelizable_args(self, num_groups: int) -> list[list[BulkFillDProbsArgs]]:
        # Step 1: naively divide circuits into num_groups blocks that could be processed
        # in parallel by any given simulator. If MapForwardSimulator is run with MPI
        # then it would figure out its own blocking in a clever way. 
        num_circuits = self.circuits.size
        block_size = int(np.ceil(num_circuits / num_groups)) 
        circuit_blocks : list[Circuit] = []
        for i in range(num_groups):
            start = block_size*i
            stop  = min(block_size*(i + 1), num_circuits)
            if start >= num_circuits or start == stop:
                break
            block = self.circuits[start:stop].tolist()
            circuit_blocks.append(block)

        # Step 2: for each model and and block of circuits, prepare some datastructures
        # that the model's simulator needs in order to evaluate bulk_fill_dprobs (the
        # bottleneck function we'd like to benchmark and eventually accelerate).
        argblocks_by_model = []
        for m in self.models:
            argblocks = []
            for b in circuit_blocks:
                layout = m.sim.create_layout(b, array_types=('ep',)) # type: ignore
                array  = layout.allocate_local_array('ep', 'd')
                args   = (array, layout)
                argblocks.append(args)
            argblocks_by_model.append(argblocks)

        return argblocks_by_model

    def time_parallelizable_dprobs(self, model, argblocks : list[BulkFillDProbsArgs]):
        # The for-loop below is embarassingly parallel.
        tic = time.time()
        for array, layout in argblocks:
            model.sim.bulk_fill_dprobs(array, layout)
        toc = time.time()
        t = toc - tic
        return t

    def compare_dprob_times(self, chunks=1):
        print('-'*80)
        argblocks_by_model = self.prep_parallelizable_args(chunks)
        print('-'*27 + f' {len(self.circuits)} circuits, {chunks} chunks ' + '-'*27)
        times = np.zeros(len(self.models))
        for i, (m, argblocks) in enumerate(zip(self.models, argblocks_by_model)):
            tic = time.time()
            self.time_parallelizable_dprobs(m, argblocks)
            toc = time.time()
            t = toc - tic
            simname = get_sim_name(m.sim)
            print(f'Time for {simname} : {t}')
            times[i] = t
        print('-'*80)
        return times


def demo_benchmarker_on_modelpack(torch_use_gpu: bool, gst_maxlen=32):
    sims = [TorchForwardSimulator(use_gpu=torch_use_gpu), MapForwardSimulator]
    mp = smq2Q_XYICNOT
    print()
    b = Benchmarker.from_modelpack(mp, sims, gst_maxlen)
    b.compare_dprob_times()
    return

if __name__ == '__main__':
    trials = 2

    print('\n' + '*'*80)
    print( ' '*30 + 'WITH GPU')
    print('*'*80)
    for i in range(trials):
        demo_benchmarker_on_modelpack(True)

    print('\n' + '*'*80)
    print( ' '*30 + 'WITHOUT GPU')
    print('*'*80)
    for i in range(trials):
        demo_benchmarker_on_modelpack(False)


    print(0)
