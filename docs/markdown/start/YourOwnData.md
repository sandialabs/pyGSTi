---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Getting your own data in

Almost every other page in this documentation simulates its data. That keeps the examples self-contained and runnable, but it skips the step you actually care about: you ran circuits on your own hardware, you have counts, and you need pyGSTi to analyze them.

This page covers that step and nothing else. Which route you take depends on where your circuits came from.

**Your circuits came from a pyGSTi experiment design.** You built a `StandardGSTDesign` or an RB design, exported the circuit list, ran it on your device, and now you have counts for those specific circuits. Use the template route below. This is the common case, and it is the one pyGSTi is built around.

**Your circuits are your own.** You have counts for circuits pyGSTi did not choose. You can still fit and test models against them, but protocols that assume a particular circuit structure (GST, RB) will not apply. Skip to [bringing your own circuits](#bringing-your-own-circuits).

```{code-cell} ipython3
import pygsti
from pygsti.modelpacks import smq1Q_XYI
```

## Write the template, fill it in, read it back

pyGSTi's preferred shape for an experiment on disk is a directory holding the experiment design and the data side by side. You create that directory *before* taking data, with the counts left blank:

```{code-cell} ipython3
edesign = smq1Q_XYI.create_gst_experiment_design(max_max_length=2)
pygsti.io.write_empty_protocol_data("../../tutorial_files/MyExperiment", edesign, clobber_ok=True)
```

That writes two subdirectories. `edesign/` holds the circuit list, the processor specification and the fiducials and germs, so the directory remembers what experiment it is. `data/dataset.txt` is the part you fill in:

```{code-cell} ipython3
with open("../../tutorial_files/MyExperiment/data/dataset.txt") as f:
    print("".join(f.readlines()[:8]))
```

The first line is a column directive naming the outcomes, in order. Every line after it is one circuit followed by one count per outcome. So `({})Gxpi2:0@(0)  0  0` means a circuit consisting of `Gxpi2` on qubit 0, over qubit line 0, with zero counts recorded for outcome `0` and zero for outcome `1`. Replace those zeros with what your device measured and the file is done.

A few things about the circuit syntax are worth knowing, because you will be matching your own records against these strings:

- `Gxpi2:0` is a gate label: gate name `Gxpi2`, acting on qubit `0`.
- `{}` is the parser's no-op. It occupies no layer, so `({})Gxpi2:0` is the same circuit as `Gxpi2:0`. It appears here because the fiducial slot is empty for this row, not because anything idles.
- `[]` *is* an empty layer, and that is how an idle appears. The idle germ shows up in this file as `([])^2@(0)`.
- `^2` after a subcircuit means that subcircuit repeated twice; germ powers show up this way.
- `@(0)` declares the circuit's qubit *line labels*. It does not set the count columns: those come from the `## Columns` header, which pyGSTi builds from your processor specification.

```{warning}
The column-per-outcome format above is what `write_empty_protocol_data` writes for designs of **three qubits or fewer**. Above that it switches to a sparse format with no `## Columns` line, where each row carries comma-separated `<outcome>:<count>` items instead. Check the header of the file you actually got before writing a parser against it.
```

Here we stand in for a lab by writing plausible counts into every row. Substitute your own and the rest of this page is unchanged:

```{code-cell} ipython3
path = "../../tutorial_files/MyExperiment/data/dataset.txt"
lines = open(path).read().splitlines()

filled = []
for line in lines:
    if line.startswith('#') or not line.strip():
        filled.append(line)
        continue
    circuit = line.rsplit('  ', 2)[0]
    filled.append(f"{circuit}  512  512")     # <-- your counts go here

open(path, 'w').write("\n".join(filled) + "\n")
print(filled[1])
```

Now read the whole directory back. You get a `ProtocolData`: the experiment design and the counts, paired.

```{code-cell} ipython3
data = pygsti.io.read_data_from_dir("../../tutorial_files/MyExperiment")
print(f"{len(data.edesign.all_circuits_needing_data)} circuits in the design")
print(f"{len(data.dataset)} circuits with counts")
print(f"{data.dataset[data.edesign.all_circuits_needing_data[1]]}")
```

That object is what every pyGSTi protocol consumes. From here you are on the ordinary path, so continue with [your first GST run](FirstGST) or whichever protocol matches your experiment.

### If your counts are already in a file

You do not have to go through the template. If you already have a text file in the format above, or you can write one, read it directly and pair it with the design yourself:

```{code-cell} ipython3
ds = pygsti.io.read_dataset("../../tutorial_files/MyExperiment/data/dataset.txt")
data = pygsti.protocols.ProtocolData(edesign, ds)
print(f"{len(ds)} circuits, {sum(ds[c].total for c in list(ds)[:5])} counts in the first five")
```

Two details bite people here. The column directive fixes the outcome *order*, so if your file lists `1 count` before `0 count` you must say so in that line rather than reordering the numbers.

The second is worth being blunt about, because pyGSTi handles it poorly. A circuit present in the design but absent from your data is missing, not zero. Pairing them anyway succeeds silently: `ProtocolData(edesign, ds)` issues no warning. You find out when you run a protocol and it raises a bare `KeyError` naming one circuit, with no list and no count of how many others are also absent. So check the pairing yourself before you fit:

```{code-cell} ipython3
missing = [c for c in edesign.all_circuits_needing_data if c not in ds]
print(f"{len(missing)} of {len(edesign.all_circuits_needing_data)} circuits have no data")
```

## Bringing your own circuits

If your circuits did not come from a pyGSTi experiment design, build a `DataSet` directly. Add one count dictionary per circuit and declare when you are done:

```{code-cell} ipython3
ds = pygsti.data.DataSet(outcome_labels=['0', '1'])
ds.add_count_dict(pygsti.circuits.Circuit([('Gxpi2', 0)]), {'0': 400, '1': 600})
ds.add_count_dict(pygsti.circuits.Circuit([('Gxpi2', 0), ('Gxpi2', 0)]), {'0': 950, '1': 50})
ds.done_adding_data()
print(ds)
```

A `ProtocolData` still wants an experiment design, but you can pass `None` and pyGSTi will infer a bare one from the circuits present:

```{code-cell} ipython3
data = pygsti.protocols.ProtocolData(None, ds)
print("inferred design:", type(data.edesign).__name__,
      "over", len(data.edesign.all_circuits_needing_data), "circuits")
```

What you can do with that is narrower than the template route. GST needs its specific fiducial-germ-fiducial circuits to identify a gate set, and the RB protocols need the random circuit families their designs generate, so neither will run on a list they did not ask for. What does still work is testing a model you already have against the data you collected:

```{code-cell} ipython3
results = pygsti.protocols.ModelTest(smq1Q_XYI.target_model()).run(data)
print(type(results).__name__)
```

See [model testing](../guides/analysis/ModelTesting) for what to do with that result, and [judging the fit](../guides/gst/JudgingTheFit) for how to read the statistics it reports.

## Where to go next

- [Your first GST run](FirstGST) — the full protocol path, now that your data is loaded.
- [Files and directories](../guides/workflow/FilesAndDirectories) — the directory convention in more detail, including where results get written.
- [Data sets](../guides/workflow/DataSets) — the `DataSet` object itself: timestamps, multiple passes, and outcome bookkeeping.
