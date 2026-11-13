# Measurement workflow

A pyGSTi analysis starts with an experiment design, pairs that design with measured counts, and runs a protocol to produce results. [Running QCVV protocols](ProtocolExamples) demonstrates this sequence for GST, randomized benchmarking, model testing, robust phase estimation, and drift analysis.

Choose the notebook for the object or step you're working on:

- [Describing your device](DescribeYourDevice) specifies qubits, available gates, and connectivity.
- [Circuits](Circuits) constructs and manipulates the sequences you'll run.
- [Experiment designs](ExperimentDesigns) organizes circuits for a protocol, including combined experiments.
- [Data sets](DataSets) stores outcome counts and works with individual circuit records.
- [Simulating data](SimulatingData) generates counts from a model when you don't yet have experimental data.
- [File input and output](FilesAndDirectories) reads and writes circuits, models, data, and protocol directories.

For counts collected outside pyGSTi, start with [Getting your own data in](../../start/YourOwnData).
