# Which guide do I need?

This part is the practitioner layer. It assumes you have been through [Start here](../start/Index), or that you already know what you want to run, and it covers each protocol at full width — the options, the failure modes, and the variants that the guided path leaves out.

The chapters split into two kinds, and knowing which is which saves a lot of scrolling.

## Pick the chapter for your protocol

- [Gate set tomography](gst/RunningGST) — a complete, self-consistent description of every gate, preparation and measurement in a small gate set. The longest chapter here, because GST has the most ways to go wrong: designing the circuits, cutting their number, judging the fit, and what to do when the fit is bad.
- [Randomized benchmarking](rb/HowRBWorks) — error rates from random circuits, in several flavours. Start with this chapter rather than a specific flavour; it explains the shared workflow and then points at Clifford, direct, mirror and binary RB.
- [Volumetric benchmarking](benchmarks/VolumetricBenchmarks) — how wide and how deep your circuits can get before the signal dies, as a map over circuit shapes rather than a single number.
- [Drift characterization](drift/DriftCharacterization) — detecting and characterizing instability from time-stamped data, on any circuits and any number of qubits.

## Chapters every protocol draws on

- [Running QCVV protocols](workflow/Workflow) — the experiment design → data → protocol → results pattern that all of the above share. Read it once and the rest of this part reads faster.
- [Explicit models](models/Models) — building and modifying the models you feed to a protocol, including multi-qubit models and the noise you put on them.
- [Results](analysis/Results) — what a protocol hands back, and how to get numbers, error bars and reports out of it. Also where [troubleshooting](analysis/Troubleshooting) and [gauge freedom in practice](analysis/GaugeFreedom) live.

## Getting off simulated data

[Running experiments on IBM Q processors](hardware/IBMQ) covers submitting an experiment design to real hardware and getting the counts back. For hardware pyGSTi does not talk to directly, [getting your own data in](../start/YourOwnData) is the general route.

If none of this fits — you are extending pyGSTi, working with a system that is not a set of qubits, or reaching beneath the `Protocol` layer — see [advanced topics and internals](../advanced/Index).
