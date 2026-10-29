# Objects

These tutorials introduce the essential objects in `pyGSTi` and how they fit together. If you're taking the bottom-up route, read them alongside the [essential-objects overview](overview/01-Essential-Objects).

- [Matrix bases](objects/MatrixBases) — the operator bases (Pauli-product, Gell-Mann, standard) pyGSTi uses to represent operations.
- [Circuits](objects/Circuit) — building and manipulating quantum circuits, and working with [circuit lists](objects/CircuitLists).
- [Data sets](objects/DataSet) — storing experimental counts, including [multi-](objects/MultiDataSet) and [time-stamped](objects/TimestampedDataSets) datasets.
- [Experiment designs](objects/ExperimentDesign) — specifying which circuits an experiment should run.
- [Models](objects/Models) — the central object representing a QIP, with explicit vs. implicit construction, parameterizations, and noise.
- [Model packs](objects/ModelPacks) — ready-made target models and experiment designs for common gate sets.
- [Operators](objects/Operators) — states, POVMs, gates, instruments, and operation factories.
- [Processor specifications](objects/ProcessorSpec) — describing a processor's qubits, available gates, and connectivity.
- [Results](objects/Results) — the result objects that protocols produce.
