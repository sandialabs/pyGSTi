# Choosing a protocol

pyGSTi implements a dozen characterization and benchmarking protocols, and they answer different questions. Picking the wrong one wastes an experiment, so this page is about the choice itself. Every protocol named here has its own chapter; the point of this one is to get you to the right chapter.

Start from what you want to learn.

**I want to know what my gates actually do.** You want gate set tomography. GST estimates a full description of every gate, state preparation and measurement in your gate set, self-consistently, without assuming any of them are already good. That completeness is what makes it expensive: it is practical on one or two qubits, and the circuit count and fit cost climb steeply past that. See [your first GST run](FirstGST).

**I want one number for how good my gates are.** You want randomized benchmarking. RB reports an error rate per operation, averaged over a random ensemble, and it is robust to state-preparation and measurement error in a way that tomography is not. It tells you much less than GST, and it tells you that much for far more qubits. See [how RB works](../guides/rb/HowRBWorks).

**I want to know how large a circuit my device can run.** You want volumetric benchmarks, which sweep circuit width against circuit depth and report where the device stops producing usable output. This is the closest thing to a capability map. See [volumetric benchmarks](../guides/benchmarks/VolumetricBenchmarks).

**I want to know whether my device is stable.** You want drift characterization, which looks for time-dependence in data you have already taken. It needs timestamped data, and around 100 or more time-stamps per circuit. See [drift characterization](../guides/drift/DriftCharacterization).

**I want a very precise value for a rotation angle.** You want robust phase estimation, which spends its circuits on precision for a small number of angles rather than on breadth. See [robust phase estimation](../advanced/specialist/RobustPhaseEstimation).

## The RB family

RB is not one protocol. For the multi-qubit flavours the main difference is how many qubits you can benchmark at once; Interleaved RB and SU(2) qudit RB differ instead in what they report, isolating a single gate and resolving noise per irreducible component respectively.

| Protocol | Qubit range | What it benchmarks |
|---|---|---|
| [Clifford RB](../guides/rb/CliffordRB) | 1 or 2 | the $n$-qubit Clifford group directly, native gates only indirectly |
| [Interleaved RB](../guides/rb/InterleavedRB) | 1 or 2 | one specific gate, against a Clifford RB baseline |
| [Direct RB](../guides/rb/DirectRB) | more than Clifford RB | your native gates, directly |
| [Mirror RB](../guides/rb/MirrorRB) | tens to hundreds | your native gates, via mirror circuits |
| [Binary RB](../guides/rb/BinaryRB) | tens to hundreds | your native gates, via a gate-efficient SPAM method |
| [SU(2) qudit RB](../guides/rb/SU2QuditRB) | one spin-$j$ qudit | global SU(2) rotations, per noise component |

Clifford RB is the one most people mean by "RB." It stops being practical at one or two qubits because each sampled $n$-qubit Clifford must be compiled into native gates, and both the compilation and the resulting circuit grow quickly with $n$. Direct RB drops the Clifford layer and benchmarks native gates directly. Mirror RB streamlines further, which is what buys it the qubit count.

[Simultaneous RB](../guides/rb/SimultaneousRB) is not a protocol of its own but an add-on: run Clifford, Direct or Mirror RB (pyGSTi's integrated support covers those three) on several disjoint sets of qubits at once, and compare against running them in isolation. The difference between the two is a crosstalk signal.

## What each protocol demands of you

The choice is also constrained by what you can actually run.

**GST needs its own circuits.** You cannot run GST on circuits you already have; the protocol derives its guarantees from a specific structure of preparation fiducials, germs and measurement fiducials. Build the experiment design first, then take data. See [getting your own data in](YourOwnData).

**RB needs a processor specification and compilation rules.** You describe your device's native gates and connectivity, and pyGSTi samples circuits that respect them.

**Drift needs timestamps, and a lot of them.** Aggregated counts will not do; the analysis works on when each shot happened, and it wants around 100 or more time-stamps per circuit (fewer if each time-stamp carries multiple outcomes).

**Model testing needs almost nothing.** If you have counts for arbitrary circuits and a model you want to check, [model testing](../guides/analysis/ModelTesting) will score that model against that data without requiring any particular circuit structure. It is the fallback when nothing above fits.

## If you are still unsure

Run GST on one qubit. It is the cheapest way to see the whole pyGSTi workflow end to end, the report it produces is the most informative thing the package makes, and one-qubit GST is fast enough to finish while you read [your first GST run](FirstGST).
