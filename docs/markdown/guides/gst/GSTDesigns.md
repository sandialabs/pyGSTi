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

# GST experiment designs

A GST experiment design is a list of circuits to run on your device, and the quality of your estimate is capped by that list before you collect a single count. Choose it badly and no amount of careful fitting will recover what you did not measure. This chapter is about choosing it well.

The list is built from three ingredients: *fiducials*, which prepare and measure in enough different bases to see the whole state space; *germs*, short circuits whose repetition amplifies particular errors; and *maximum lengths*, which say how many times to repeat each germ. Everything downstream is a consequence of those three.

Before taking them apart, it is worth seeing what they do. pyGSTi ships **model packs** for a handful of standard gate sets, and each one carries precomputed fiducials, germs and fiducial-pair-reduction results. That is a crutch, and this page leans on it shamelessly: it lets you turn every knob in one line and watch the effect, long before you know how any of the ingredients are made. The rest of the chapter shows you how to produce them for a device that has no pack.

```{code-cell} ipython3
from pygsti.modelpacks import smq1Q_XYI

design = smq1Q_XYI.create_gst_experiment_design(max_max_length=8)
len(design.all_circuits_needing_data)
```

That number is the whole game. It is what your experiment costs, and every knob below moves it.

## How deep to go: `max_max_length`

`max_max_length` sets how many times germs get repeated. Pass a number and you get maximum lengths at powers of two up to it; pass a list and it is used as-is.

```{code-cell} ipython3
for L in [1, 2, 4, 8, 16, 32, 64]:
    d = smq1Q_XYI.create_gst_experiment_design(max_max_length=L)
    print(f"max_max_length={L:3d}   {len(d.all_circuits_needing_data):4d} circuits")
```

Two things are visible here. The count climbs slowly at first and then settles into a constant 168 circuits per doubling. That is because a germ cannot appear until the maximum length is at least as long as the germ itself, so the early doublings are still bringing germs into play; once all of them are active, each doubling adds one repetition block per germ per fiducial pair, and the pack has 6 preparation fiducials, 6 measurement fiducials and 5 germs.

The other thing is what you buy. Accuracy in GST improves roughly as $1/L$ at the largest maximum length, so doubling `max_max_length` buys you about a factor of two in precision for a fixed additive cost in circuits. That is the trade that makes long-sequence GST worth doing, and it is why the default advice is to go as deep as your device's coherence will support rather than as wide.

## Which germs: `lite`

Model packs carry two germ sets. The `lite` set is found without randomizing around the target model; the full set is found by randomizing, which makes it more pessimistic and considerably larger.

```{code-cell} ipython3
for lite in [True, False]:
    germs = smq1Q_XYI.germs(lite=lite)
    d = smq1Q_XYI.create_gst_experiment_design(max_max_length=8, lite=lite)
    print(f"lite={lite!s:5s}  {len(germs):2d} germs   {len(d.all_circuits_needing_data):4d} circuits")
```

The full set nearly doubles the experiment. It buys robustness: a germ set chosen without randomization can be *amplificationally complete* for the target model and lose that property for the noisy model you actually have. The docstring's advice is to leave `lite=True` unless you know you need otherwise, and that is right for most people most of the time.

## Cutting fiducial pairs: `fpr`

Most of those circuits are redundant. Fiducial pair reduction finds, for each germ, a subset of fiducial pairs that still sees everything that germ amplifies.

```{code-cell} ipython3
for lite in [True, False]:
    for fpr in [False, True]:
        d = smq1Q_XYI.create_gst_experiment_design(max_max_length=8, lite=lite, fpr=fpr)
        print(f"lite={lite!s:5s} fpr={fpr!s:5s}  {len(d.all_circuits_needing_data):4d} circuits")
```

Note what those four numbers say together. The full germ set with reduction costs 195 circuits; the lite germ set without it costs 448. You can have the more pessimistic germs *and* less than half the experiment, which is not the trade-off most people expect to be available.

That is a claim about circuit counts, not about what you can learn, and the two are not the same thing. [Checking your design](CheckYourDesign) is how you tell the difference, by computing the Fisher information of a design before you spend anything running it.

## Throwing circuits away at random: `keep_fraction`

`keep_fraction` drops circuits from the repeated-germ blocks at random, with `keep_seed` for reproducibility. It is blunter than fiducial pair reduction and it makes no attempt to preserve what any germ amplifies.

```{code-cell} ipython3
for kf in [1.0, 0.5, 0.25]:
    d = smq1Q_XYI.create_gst_experiment_design(max_max_length=8, keep_fraction=kf, keep_seed=42)
    print(f"keep_fraction={kf:<5}  {len(d.all_circuits_needing_data):4d} circuits")
```

Reach for this when you want a quick sanity run rather than an estimate you intend to quote.

## The remaining knobs

`create_gst_experiment_design` takes several more arguments, passed through to `StandardGSTDesign`:

| argument | what it does |
|---|---|
| `germs` | override the pack's germ list entirely |
| `germ_length_limits` | cap repetition for individual germs, when some germ is unreliable at depth |
| `include_lgst` | prepend the circuits LGST needs for its initial estimate |
| `nest` | whether each maximum-length list contains all the shorter ones |
| `qubit_labels` | relabel the pack's qubits to match your device |
| `circuit_rules`, `op_label_aliases` | rewrite circuits on the way out |
| `dscheck`, `action_if_missing` | validate the design against a dataset you already hold |

`include_lgst` is worth a note because its cost is not what the name suggests: at these settings it accounts for a single circuit, since almost everything LGST wants is already in the list.

## Where this chapter goes next

Each knob above hands off to a page that explains the machinery underneath it.

[GST circuit construction](GSTCircuits) covers `max_max_length` and the nested-list structure it produces, and how to build such lists yourself when you are not starting from a pack.

[Fiducial and germ selection](FiducialsAndGerms) covers where `germs` and the fiducials come from, and how to run the selection algorithms for a device with no precomputed set.

[Fiducial pair reduction](FewerCircuits) covers what `fpr` is doing, and the several reduction strategies that the single boolean is hiding.

[Checking your design](CheckYourDesign) closes the loop, by measuring what a design can actually tell you before you run it.
