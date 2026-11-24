# Circuits, designs, and data

A pyGSTi protocol takes an experiment design (the circuits to run) paired with a data set (their measured counts). These pages cover those objects directly: building and editing them, and saving and loading them. [Your first GST run](../../start/FirstGST) and [Your first benchmark](../../start/FirstBenchmark) show them in use end to end.

Choose the notebook for the object or step you're working on:

- [Circuits](Circuits) constructs and manipulates the sequences you'll run.
- [Experiment designs](ExperimentDesigns) organizes circuits for a protocol, including combined experiments.
- [Data sets](DataSets) stores outcome counts and works with individual circuit records.
- [File input and output](FileIO) reads and writes circuits, models, data, and protocol directories.

Two pages connect these objects to other platforms:

- [Cirq integration](Cirq) converts circuits between Cirq and pyGSTi and loads experiment results into a pyGSTi data set.
- [IBM Q integration](IBMQ) submits an experiment design to an IBM backend through Qiskit and collects its counts for analysis. It requires Qiskit and an IBM Q account.

For counts collected any other way, start with [Getting your own data in](../../start/YourOwnData).
