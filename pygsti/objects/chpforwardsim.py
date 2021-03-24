"""
Defines the CHPForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import os as _os
from pathlib import Path as _Path
import re as _re
import subprocess as _sp
import tempfile as _tf

from .profiler import DummyProfiler as _DummyProfiler
from .label import Label as _Label
from .labeldicts import OutcomeLabelDict as _OutcomeLabelDict
from .weakforwardsim import WeakForwardSimulator as _WeakForwardSimulator
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
_dummy_profiler = _DummyProfiler()


class CHPForwardSimulator(_WeakForwardSimulator):
    """
    A WeakForwardSimulator returning probabilities with Scott Aaronson's CHP code
    """
    def __init__(self, chpexe, shots, model=None):
        """
        Construct a new CHPForwardSimulator.

        Parameters
        ----------
        chpexe: str or Path
            Path to CHP executable
        shots: int
            Number of times to run each circuit to obtain an approximate probability
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self.chpexe = _Path(chpexe)
        assert self.chpexe.is_file(), "A valid CHP executable must be passed to CHPForwardSimulator"

        super().__init__(shots, model)

    def _compute_circuit_outcome_for_shot(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        """
        Compute probabilities of a multiple "outcomes" for a single circuit for a single shot.

        Parameters
        ----------

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        time : float, optional
            The *start* time at which `circuit` is evaluated.
        """
        assert(time is None), "CHPForwardSimulator cannot be used to simulate time-dependent circuits yet"
        
        # Use temporary file as per https://stackoverflow.com/a/8577225
        fd, path = _tf.mkstemp()
        try:
            # TODO: Handle prep/povm properly? Not that there's a lot we can do...
            # Also the expand_instruments bit can probably be pulled into base class

            # Substitute sslbls for names in gates
            # TODO: Don't like this coupling... how to avoid?
            labels = self.model.state_space_labels.labels[0]
            label_dict = {l:i for i,l in enumerate(labels)}

            with _os.fdopen(fd, 'w') as tmp:
                # Prep placeholder...
                tmp.write('#\n')

                for op_label in circuit:
                    op = self.model.circuit_layer_operator(op_label, 'op')
                    tmp.write(op.get_chp_str(**label_dict))
                
                # PVM placeholder...
                for i in range(len(labels)):
                    tmp.write(f'm {i}\n')

            # Run CHP
            process = _sp.Popen([f'{self.chpexe.resolve()}', f'{path}'], stdout=_sp.PIPE, stderr=_sp.PIPE)
            out, err = process.communicate()
        finally:
            _os.remove(path)

        # Extract outputs
        pattern = _re.compile('Outcome of measuring qubit (\d): (\d)')
        qubit_outcomes = []
        for match in pattern.finditer(out.decode('utf-8')):
            qubit_outcomes.append((int(match.group(1)), match.group(2)))

        outcome = ''.join([qo[1] for qo in sorted(qubit_outcomes)])
        outcome_label = _OutcomeLabelDict.to_outcome(outcome)
        index = outcomes.index(outcome_label)

        array_to_fill[index] += 1.0
