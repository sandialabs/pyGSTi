---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Your first benchmark

Randomized benchmarking turns a device into a number. You describe your qubits and their native gates, pyGSTi samples random circuits that should return you to a known bitstring, and the rate at which they stop doing so is the error rate $r$. This page is the shortest honest path from a device description to that number, using Clifford RB on two qubits. It runs in a few seconds and it cuts every option you can live without. When you want the options back, read [how RB works](../guides/rb/HowRBWorks), which covers the same three steps at full width, and [choosing a protocol](ChooseAProtocol) if you are not sure Clifford RB is the flavour you want.

```{code-cell} ipython3
import pygsti
from pygsti.processors import QubitProcessorSpec
from pygsti.processors import CliffordCompilationRules as CCR
```

## Describe the device

RB circuits have to be made of gates your hardware actually has, laid out on connections it actually has. So the first object you build is a `QubitProcessorSpec`: how many qubits, what they are called, which gate names exist, and on which qubit tuples each gate is available. Here it's three qubits in a line, with $\pm\pi/2$ rotations about $x$ and $y$ as the single-qubit gates and controlled-Z between neighbours.

```{code-cell} ipython3
pspec = QubitProcessorSpec(
    num_qubits=3,
    gate_names=['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'],
    availability={'Gcphase': [('Q0', 'Q1'), ('Q1', 'Q2')]},
    qubit_labels=['Q0', 'Q1', 'Q2'])
```

Clifford RB needs one more thing, because the circuits it samples are written in Cliffords and your device does not run Cliffords: it runs `Gxpi2` and friends. The compilation rules say how to turn one into the other. Build them with `CCR.create_standard` and stop thinking about them.

```{code-cell} ipython3
compilations = {
    'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
    'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)}
```

If your device's gate names are not pyGSTi's, or your connectivity is more interesting than a line, see [describing your device](../guides/workflow/DescribeYourDevice).

## Build the circuits

A `CliffordRBDesign` is a list of circuits plus the bookkeeping that says which bitstring each one should return. You choose three things: the benchmark depths, the number of random circuits $k$ at each depth, and which qubits to benchmark. Depth $m$ here means a circuit of $m+1$ uniformly random Cliffords followed by the unique Clifford that inverts them, so $m+2$ in total and $m=0$ is the shortest circuit the protocol permits, not an empty circuit.

```{code-cell} ipython3
depths = [0, 1, 2, 4, 8, 16]
k = 10
exp_design = pygsti.protocols.CliffordRBDesign(
    pspec, compilations, depths, k, qubit_labels=('Q0', 'Q1'),
    seed=20250817, verbosity=0)

print(len(exp_design.all_circuits_needing_data), "circuits")
```

Two dials control cost. Depth range sets how far you can see: if the deepest circuits still succeed most of the time you have not measured a decay, and if they have all decayed to chance you have wasted them. $k$ sets how much of the spread between random circuits you average away. Ten is small for publication and fine for a first look.

## Get data

Write the design to a directory and pyGSTi leaves you a `data/dataset.txt` with every circuit listed and every count blank. That file is the seam between pyGSTi and your lab: run the circuits, fill in the counts, read the directory back. [Getting your own data in](YourOwnData) describes the file format line by line.

We have no device here, so we fill the template from a simulator instead: a crosstalk-free model with 0.5% depolarization after each native gate.

```{code-cell} ipython3
pygsti.io.write_empty_protocol_data('../../tutorial_files/first_benchmark', exp_design, clobber_ok=True)

# -- REPLACE these two lines with real counts in .../first_benchmark/data/dataset.txt --
noise_model = pygsti.models.create_crosstalk_free_model(
    pspec, depolarization_strengths={g: 0.005 for g in pspec.gate_names})
pygsti.io.fill_in_empty_dataset_with_fake_data(
    '../../tutorial_files/first_benchmark/data/dataset.txt', noise_model,
    num_samples=1000, seed=1234)

data = pygsti.io.read_data_from_dir('../../tutorial_files/first_benchmark')
```

`clobber_ok=True` lets this page be re-run without deleting the directory by hand. In your own scripts leave it at its default of `False`, so that a stray re-run cannot overwrite data you spent device time on.

## Fit, and read off r

The protocol converts counts to success probabilities, averages them at each depth, and fits $P_m = A + Bp^m$. Two fits come back: one with $A$ free, and one with $A$ pinned to $1/2^n$, the success probability of guessing. Bootstrapped error bars come with them.

```{code-cell} ipython3
results = pygsti.protocols.RB().run(data)

for fitkey in ('full', 'A-fixed'):
    r = results.fits[fitkey].estimates['r']
    rstd = results.fits[fitkey].stds['r']
    print("{:8s}: r = {:1.3e}  (2 sigma: +/- {:1.3e})".format(fitkey, r, 2 * rstd))
```

That is the benchmark. `results.plot()` draws the decay curve with the fit over it, which you should look at before you believe any $r$: a fit to six averaged points can be confidently wrong, and the two fits disagreeing by much more than their error bars is the usual first sign of it.

Note that $r$ is far larger than the 0.5% per-gate depolarization we simulated, by more than an order of magnitude. That is not a bug. Clifford RB reports error per benchmarked *Clifford*, and a random two-qubit Clifford compiles into roughly sixteen native gates in this gate set, so the per-Clifford error is around sixteen times the per-gate error. The design will tell you the exact count:

```{code-cell} ipython3
counts = exp_design.average_native_gates_per_clifford_for_circuit(0, 0)
print(counts)
```

If you want a number per native gate, either divide by that count or run [Direct RB](../guides/rb/DirectRB), which benchmarks native layers without the Clifford detour.

## Say which convention you used

There are two conventions for turning the decay constant $p$ into an error rate, and on few qubits they differ by enough to matter:

$$ r_{\rm EI} = \frac{(4^n - 1)(1 - p)}{4^n}, \qquad r_{\rm AGI} = \frac{(2^n - 1)(1 - p)}{2^n}. $$

The first approximates mean entanglement infidelity, the second mean average gate infidelity. pyGSTi defaults to EI, because entanglement fidelities of a tensor product multiply, which makes EI-type rates from different numbers of qubits comparable. Much of the Clifford RB literature uses AGI. Pass `rtype='AGI'` to `pygsti.protocols.RB()` to switch.

The ratio is $r_{\rm EI}/r_{\rm AGI} = (2^n+1)/2^n$: 50% on one qubit, 25% on two, shrinking as $n$ grows. An $r$ reported without its convention is ambiguous by that much, so report which one you used, every time.

## Where to go next

The obvious next move is to benchmark more qubits, and Clifford RB will not take you there: the compilation cost of an $n$-qubit Clifford makes it impractical beyond one or two qubits, which is a property of the protocol and not of pyGSTi's implementation. [Direct RB](../guides/rb/DirectRB), [mirror RB](../guides/rb/MirrorRB) and [binary RB](../guides/rb/BinaryRB) exist for exactly this, and they reuse the `QubitProcessorSpec` and the analysis you already have here. Mirror RB is the one that reaches tens to hundreds of qubits.

Staying at two qubits, [Clifford RB in detail](../guides/rb/CliffordRB) covers randomized measurement outcomes, interleaving a specific gate to benchmark it against a baseline, and what to do when the standard compilations do not fit your gate set. And if a single number turns out not to answer your question, [choosing a protocol](ChooseAProtocol) points at the tomographic methods that tell you what kind of error you have rather than how much.
