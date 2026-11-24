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

# Experiment designs
In `pygsti`, an experiment design object specifies how to produce the data needed by a characterization protocol.  In its simplest form, it consists entirely of a list of `Circuit` objects, but for some protocols (e.g. GST) additional meta-data is needed regarding the structure and intent for these circuits (e.g. the germ and fiducial circuits and maximum lengths used to build a set of final GST circuits).

```{code-cell} ipython3
import pygsti
from pygsti.circuits import Circuit as C
```

## Simple experiment designs
Some protocols, such as volumetric benchmarks, don't require much or any structure among the circuits they use.  In these cases you can just build a simple experiment design consisting of one or more lists of circuits.  The `ExperimentDesign` class is the base class of all experiment design in pyGSTi and contains just a simple list of cicuits along with the qubit labels these circuits act upon.  The slightly more complicated `CircuitListsDesign` structures its circuits into several (often nested) lists.  These circuit lists often describe the circuits used by successive stages of an a protocol, e.g. circuits with different numbers of Clifford gates or germ repetitions in RB or GST respectively.  A third type of design, a `FreeformDesign`, contains just a single list (like `ExperimentDesign`) but allows you to associate arbitrary meta-data with each circuit in the list. 

Creating these types of experiment designs is easy.  First we create a list of circuits:

```{code-cell} ipython3
circuits = [C("Gxpi2:0"), C("Gypi2:0"), C("Gxpi2:0^2"), C("Gypi2:0^2")]
circuits  # a list of Circuits
```

And from that list we can construct an experiment design:

```{code-cell} ipython3
edesign_plain = pygsti.protocols.ExperimentDesign(circuits)
edesign_plain.all_circuits_needing_data
```

All experiment designs hold what qubits their circuits act upon:

```{code-cell} ipython3
edesign_plain.qubit_labels
```

We could also assign the circuits to different sub-lists (here based on their depth), and create a `CircuitListsDesign` that has the same overall circuit list as the design above but with the additional structure of the sub-lists.  The `CircuitListsDesign` object just contains this structure, it doesn't dictate what the lists must be used for within a protocol.  A `Protocol` object can decide this for itself, or a more specific (derived) experiment design class may impose further presumptions on the purposes of the lists.

```{code-cell} ipython3
circuits1 = [C("Gxpi2:0"), C("Gypi2:0")]
circuits2 = [C("Gxpi2:0^2"), C("Gypi2:0^2")]
edesign_lists = pygsti.protocols.CircuitListsDesign([circuits1, circuits2])
edesign_lists.all_circuits_needing_data  # all of the experiment design's circuits
```

```{code-cell} ipython3
edesign_lists.circuit_lists[0]  # just the first sub-list
```

If we wanted to associate some meta-data, let's say a name and ID number, with each circuit in an experiment design, we could use a `FreeformDesign`:

```{code-cell} ipython3
ff_design = pygsti.protocols.FreeformDesign({C("Gxpi2:0"): {'id': 0, 'name': "dog"},
                                             C("Gypi2:0"): {'id': 1, 'name': "cat"},
                                             C("Gxpi2:0^2"): {'id': 2, 'name': "trouble"}})
ff_design.aux_info
```

## Protocol-specific experiment designs
Some protocols need more than a list of circuits: they need to know how the circuits were built. Their experiment designs derive from `CircuitListsDesign` and add that information.

- **Gate set tomography.** `GateSetTomographyDesign` adds a target processor specification to nested circuit lists of increasing depth. `StandardGSTDesign` derives from it and also keeps the preparation fiducials, measurement fiducials, germs and maximum lengths the circuits were built from, which reports use to organize their plots. The [GST experiment designs](../gst/designs/Index) chapter covers building them, from [model packs](../gst/designs/ModelpackDesigns) or from [your own fiducials and germs](../gst/designs/GSTCircuits), and [GST protocol options](../gst/running/ProtocolOptions) builds a `GateSetTomographyDesign` from arbitrary circuit lists.
- **Randomized benchmarking.** `CliffordRBDesign`, `DirectRBDesign` and `MirrorRBDesign` add what the RB protocol needs about each random circuit, chiefly its correct outcome. [Running an RB experiment](../rb/RBWorkflow) builds one from a processor specification, compilation rules, depths and circuits per depth.

The rest of this page uses one GST design, from a model pack:

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI as std  # pack for the 1-qubit X(pi/2), Y(pi/2), Idle model
gst_design = std.create_gst_experiment_design(max_max_length=4)
print(len(gst_design.all_circuits_needing_data), "GST circuits")
```

## Simultaneous and combined experiment designs
It is not uncommon to run multiple characterization protocols on a single processor, and it can be desirable to run circuits for multiple protocols in a single round of data-taking.  The `CombinedExperimentDesign` and `SimultaneousExperimentDesign` classes allow you to build a *hierarchy* of nested experiment designs that facilitate a multiple-protocols-on-one-processor paradigm.

- A `CombinedExperimentDesign` acts as a folder for any number of other experiment designs.  The combined design's circuits are just the concatenation (but without duplicates) of the circuits of its sub-designs.  A combined design can be used to create an "umbrella" experiment design containing the circuits for multiple protocols whose circuits are disjoint, in which case the combined design acts as a folder of circuit lists.  A combined design could also hold multiple sub-designs that share the same overall circuit list but supply different structure (meta-data) about these circuits.  In this case the combined design acts as a folder of circuit structures, all for the same set of circuits.

- A `SimultaneousExperimentDesign` describes the concurrent application of one or more experiment designs on different disjoint subsets of the simultaneou design's qubits.  The sub-designs of a simultaneous design must therefore act (i.e. have circuits that act) on different sets of qubits.  The circuits of the simultaneous design are the "tensor-product" of the circuits on its sub-designs.  When one sub-design has more circuits than another, the shorter design is "padded" with idle circuits.

The examples below show how to create combined and simultaneous experiment designs using simple `ExperimentDesign` objects as their members. [Simultaneous RB](../rb/SimultaneousRB) and the [volumetric benchmark examples](../benchmarks/VolumetricBenchmarks) use them with protocol-specific designs.

### Combined experiment designs

```{code-cell} ipython3
edesignA = pygsti.protocols.ExperimentDesign([C("Gxpi2:0"), C("Gypi2:0"), C("Gxpi2:0^2"), C("Gypi2:0^2")])
edesignB = pygsti.protocols.ExperimentDesign([C("Gxpi2:0"), C("Gypi2:0"), C("Gxpi2:0^4"), C("Gypi2:0^4")])

combined_design = pygsti.protocols.CombinedExperimentDesign({'A': edesignA, 'B': edesignB})
combined_design.all_circuits_needing_data
```

Note that circuits common to `edesignA` and `edesignB` are not repeated in the combined design's circuits.  To access one of the sub-designs, you simply index the combined design:

```{code-cell} ipython3
combined_design['A'].all_circuits_needing_data
```

It is possible to add sub-designs to a combined design as long as the overall list of circuits doesn't change.  (This is required so that the `.all_circuits_needing_data` list of any existing experiment can be treated as immutable and used as a reliable index, e.g. for data.)  For example, a design containing just the `*^4` circuits could be added:

```{code-cell} ipython3
edesignC = pygsti.protocols.ExperimentDesign([C("Gxpi2:0^4"), C("Gypi2:0^4")])
combined_design['C'] = edesignC
```

But a design containing *new* circuits cannot be:

```{code-cell} ipython3
edesignD = pygsti.protocols.ExperimentDesign([C("Gxpi2:0^8"), C("Gypi2:0^8")])
try:
    combined_design['D'] = edesignD
except Exception as e:
    print("ERROR: ", str(e))
```

Finally, if you have a non-`CombinedDesign` experiment design that you decide later on you'd like to be a combined design, it is possible to *promote* many of the other experiment design types in pyGSTi to a combined design. For example, after creating an experiment design you may decide there is additional/different structure you wish to store about the circuits, or that you'd like to single out a subset of the circuits for special processing.  Here is an example of how to do this for a standard `ExperimentDesign`.

```{code-cell} ipython3
combined_designA = edesignA.promote_to_combined(name='entire')
edesign_singlegates = pygsti.protocols.ExperimentDesign([C("Gxpi2:0"), C("Gypi2:0")])
combined_designA['single_gates'] = edesign_singlegates
combined_designA.keys()
```

```{code-cell} ipython3
combined_designA.all_circuits_needing_data
```

```{code-cell} ipython3
combined_designA['entire'].all_circuits_needing_data
```

```{code-cell} ipython3
combined_designA['single_gates'].all_circuits_needing_data
```

### Simultaneous experiment designs

```{code-cell} ipython3
edesign_on_0 = pygsti.protocols.ExperimentDesign([C("Gxpi2:0"), C("Gypi2:0"), C("Gxpi2:0^2"), C("Gypi2:0^2")])
edesign_on_1 = pygsti.protocols.ExperimentDesign([C("Gxpi2:1"), C("Gypi2:1"), C("Gxpi2:1^2")])

sim_design = pygsti.protocols.SimultaneousExperimentDesign([edesign_on_0, edesign_on_1])
sim_design.all_circuits_needing_data
```

Notice that in the final circuit the `"Gypi2:0^2"` circuit was padded with idle operations to become `"Gypi2:(0,1)^2"` since there is no fourth circuit in the `edesign_on_1` design.  The `.qubit_labels` of a simultaneous design are, by default, the union of the qubits acted on by its sub-designs:

```{code-cell} ipython3
sim_design.qubit_labels
```

But these qubit labels can also be given directly, in which case all the circuits in the design are always constructed to act on these qubits (this is equivalent to padding the un-specified qubits with idles):

```{code-cell} ipython3
sim_design2 = pygsti.protocols.SimultaneousExperimentDesign([edesign_on_0], qubit_labels=(0,1,2))
sim_design2.all_circuits_needing_data
```

Note that all of the above circuits have qubit labels `(0,1,2)` even though the simultaneous design was only given a single sub-design with circuits on qubit `0`.

## Loading and saving experiment designs
An experiment design writes itself to the `edesign` subdirectory of a root directory, and `pygsti.io.read_edesign_from_dir` reads it back. Combined and simultaneous designs write one subdirectory per sub-design, mirroring their structure. [File input and output](FileIO) describes the full directory layout shared with data and results.

```{code-cell} ipython3
combined_design.write("../../../tutorial_files/example_combined_edesign_root")
combined_design_loaded = pygsti.io.read_edesign_from_dir("../../../tutorial_files/example_combined_edesign_root")
list(combined_design_loaded.keys())
```

A simple experiment design can also be created by reading text files containing circuit lists from a directory.  The `create_edesign_from_dir` attempts to read in all the files beneath a root directory's `edesign` subdirectory as circuit list files (text files with one circuit per line).  If a single loadable file is found, an `ExperimentDesign` is created.  If multiple loadable files are found, a `CircuitListsDesign` is created.

The cell below demonstrates the single-file case.

```{code-cell} ipython3
import pathlib

circuits = """# Example text file of circuits
{}@(0)
[]@(0)
Gxpi2:0@(0)
Gypi2:0@(0)
"""

# Write a text file with the above circuit strings to a root/edesign directory
root = pathlib.Path("../../../tutorial_files/example_edesign_root")  # the root directory
root.mkdir(exist_ok=True); (root / 'edesign').mkdir(exist_ok=True)  # create a `edesign` subdirectory
with open(str(root / 'edesign' / 'circuits.txt'), 'w') as f:
    f.write(circuits)

edesign_from_dir = pygsti.io.create_edesign_from_dir(str(root))
edesign_from_dir.all_circuits_needing_data
```

## Estimating runtime of experiment designs

One might be interested in estimating the time needed for an experiment design on real hardware. There is a a simple tool in PyGSTi to estimate the runtime using a rough model involving gate times, the number of needed batches and rounds, and measurement/reset and classical upload time between batches. We show this here for the generated GST experiment above, using values that might be relevant to trapped ion devices.

```{code-cell} ipython3
from pygsti.tools import edesigntools as edtools
```

```{code-cell} ipython3
# Tool does not specify time units; here, we use seconds
edtools.calculate_edesign_estimated_runtime(gst_design,
    gate_time_1Q=10e-6,
    gate_time_2Q=100e-6,
    total_shots_per_circuit=1000,
    measure_reset_time=500e-6,
    interbatch_latency=0.1,
    shots_per_circuit_per_batch=100,
    circuits_per_batch=200)
```
