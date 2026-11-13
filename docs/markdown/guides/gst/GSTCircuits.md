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

# GST circuit construction

GST algorithms work best on circuits with a particular shape. Each circuit begins with a "preparation fiducial" and ends with a "measurement fiducial", short subcircuits whose job is to extend the native preparation and measurement operations into informationally complete sets. Between the fiducials sits a "germ", a short subcircuit repeated some number of times. Repeating a germ amplifies whatever errors that germ is sensitive to. Take a whole set of germs chosen to be *amplificationally complete* and every type of error gets amplified, which is what gives GST its sensitivity. More repetitions means more amplification means more sensitivity.

If you're not sure what a `Circuit` is, read the [circuit tutorial](../workflow/Circuits) first. This page covers how to build the circuit lists that GST consumes, and the general-purpose circuit-list machinery underneath them.

So why not use arbitrarily long circuits? Two reasons.

1. The gates are imperfect. At some length (which can depend on the germ) the state is depolarized past the point where it can be distinguished from the maximally mixed state. That sets an upper bound on useful circuit depth.
2. GST's optimizers are local and gradient-based. Starting from the longest circuits tends to land you in a local optimum. Instead the algorithms iterate over a sequence of circuit lists, seeding each fit with the previous one's result. A well-chosen sequence of lists sharpens the objective function gradually and steers the optimization toward a global (or near-global) optimum.

In practice you build the lists like this. Pick an increasing sequence of **maximum lengths**, usually powers of two such as `[1,2,4,8,16,...]`, ending at the longest useful depth. Pick preparation- and measurement-fiducial lists and a germ list, chosen so the fiducials give informationally complete preparations and measurements and the germs are amplificationally complete. For each maximum length $L$, take every circuit of the form

`preparation_fiducial + repeated_germ + measurement_fiducial`

where `repeated_germ` is a germ repeated as many whole times as fit within length $L$. For numerical robustness, the list for a given $L$ also contains everything from the smaller values of $L$, so the lists are nested.

Fiducial and germ lists come out of pyGSTi's fiducial-selection and germ-selection algorithms, described in the [fiducial and germ selection tutorial](FiducialsAndGerms). Here we use one of the built-in [model packs](../../start/TargetModels), which ship with pre-computed fiducials and germs.

## Setup

```{code-cell} ipython3
import pygsti
import pygsti.circuits as pc
from pygsti.modelpacks import smq1Q_XY  # the standard X(pi/2), Y(pi/2) model info

prep_fiducials = smq1Q_XY.prep_fiducials()
meas_fiducials = smq1Q_XY.meas_fiducials()
germs = smq1Q_XY.germs()

print("Prep fiducials:\n", prep_fiducials)
print("Meas fiducials:\n", meas_fiducials)
print("Germs:\n", germs)
```

## The nested lists a long-sequence fit consumes

`create_lsgst_circuit_lists` takes an operation-label source (a list of labels or a `Model`), the fiducials, the germs, and the maximum lengths, and returns one list per maximum length. By default it also folds in the circuits Linear GST needs (every preparation fiducial paired with every measurement fiducial, and every such pair with a single gate between them); pass `include_lgst=False` to leave them out.

```{code-cell} ipython3
max_lengths = [1, 2, 4]
target_model = smq1Q_XY.target_model()
lsgst_lists = pc.create_lsgst_circuit_lists(
    target_model, prep_fiducials, meas_fiducials, germs, max_lengths)

# Larger-L lists contain everything in the smaller-L lists.  A germ only shows up once
# L is at least its length: smq1Q_XY's length-3 germ first appears at L=4, repeated once.
for i, lst in enumerate(lsgst_lists):
    print("List %d (max-length L=%d): %d circuits" % (i, max_lengths[i], len(lst)))

print()
print('\n'.join([c.str for c in lsgst_lists[0][:10]]))  # ".str" gives a single-line representation
```

These aren't plain lists. `create_lsgst_circuit_lists` hands back `PlaquetteGridCircuitStructure` objects, which index and iterate like a list of `Circuit`s but also remember how each circuit decomposes into fiducials, germ, and $L$. That decomposition is what lets the report generator lay per-circuit quantities out on a grid later. The grid axes are `xs` (the maximum lengths) and `ys` (the germs), and `iter_plaquettes` walks the occupied cells.

```{code-cell} ipython3
struct = lsgst_lists[-1]
print(type(struct).__name__, "with", len(struct), "circuits")
print("x axis (%s):" % struct.xlabel, struct.xs)
print("y axis (%s):" % struct.ylabel, [y.str for y in struct.ys])

for (x, y), plaq in list(struct.iter_plaquettes())[:3]:
    print("\nL=%d, germ=%s -> %d circuits, base=%s" % (x, y.str, len(plaq), plaq.base.str))
    for row, col, circuit in list(plaq)[:3]:
        print("   row %d, col %d: %s" % (row, col, circuit.str))
```

The `y` axis has one more entry than the germ list, and the first plaquette printed above has an empty circuit where a germ should be. That row is the LGST block. Its fiducial-pair circuits have no germ between the fiducials, so the structure files them under the empty circuit and the report grid gets an extra row at the top. Turn `include_lgst` off and that row disappears.

## The experiment list

If you're taking data rather than running a fit, you want a *single* list of every circuit the fit will ask for. Because the lists are usually nested, the last element of `lsgst_lists` is usually that list, but advanced usages break the nesting. Use `create_lsgst_circuits` instead and don't think about it.

```{code-cell} ipython3
lsgst_experiment_list = pc.create_lsgst_circuits(
    target_model, prep_fiducials, meas_fiducials, germs, max_lengths)
print("%d experiments to do..." % len(lsgst_experiment_list))
```

## Building circuit lists from scratch

Everything above is built on general-purpose list-construction functions in `pygsti.circuits`. You'll want these whenever you're constructing experiments that aren't standard GST. They lean on the fact that `Circuit` objects behave and compose as tuples of layer labels.

### `create_circuits`, the nested-loop workhorse

`create_circuits` evaluates its positional arguments inside a nested loop over its list- or tuple-valued keyword arguments. That description is a mouthful; the examples are clearer.

```{code-cell} ipython3
As = [('a1',), ('a2',)]
Bs = [('b1', 'b2'), ('b3', 'b4')]

def rep2(x):
    return x + x

list1 = pc.create_circuits("a", a=As)
list2 = pc.create_circuits("a+b", a=As, b=Bs, order=['a', 'b'])
list3 = pc.create_circuits("R(a)+c", a=As, c=[('c',)], R=rep2)

print("list1 = %s" % list(map(tuple, list1)))
print("list2 = %s" % list2)
print("list3 =\n%s" % "".join(map(str, list3)))
```

The rule for what becomes a loop variable is narrow: a keyword argument whose value is a `list` or `tuple` is looped over, and anything else is passed straight through to the evaluation as a plain name. So `c=[('c',)]` above is a loop variable that happens to have one value, while `R=rep2` is a pass-through, and any callable you pass through that way can be applied to the loop variables. Watch the `tuple` half of that rule: a keyword whose value is a single circuit written as a tuple of labels gets looped over label by label rather than held fixed, and since concatenating a bare label onto a `Circuit` trips an assertion that `create_circuits` swallows, the usual symptom is an empty result list rather than an error. Wrap it in a one-element list.

### Repeating a germ

The `repeat_`*xxx* functions give you the several ways a germ can be stretched to a target length. Modern GST always repeats a germ an *integer* number of times rather than truncating mid-germ, so `repeat_with_max_length` is the one that gets used; `repeat_and_truncate` is kept for reproducing older experiment designs.

```{code-cell} ipython3
# args (x, N): repeat x until it is exactly length N
print(pc.repeat_and_truncate(('A', 'B', 'C'), 5))

# args (x, N): repeat x the largest whole number of times with len <= N
print(pc.repeat_with_max_length(('A', 'B', 'C'), 5))

# args (x, N): that largest whole number itself
print(pc.repeat_count_with_max_length(('A', 'B', 'C'), 5))
```

Combining a repeated germ with fiducials on either side is a nested loop, so `create_circuits` does it. `to_circuits` is the bulk converter from tuples to `Circuit` objects.

```{code-cell} ipython3
fids = pc.to_circuits([('Gf0',), ('Gf1',)])              # fiducial circuits
demo_germs = pc.to_circuits([('G0',), ('G1a', 'G1b')])   # germ circuits

circuits1 = pc.create_circuits("f0+germ*e+f1", f0=fids, f1=fids,
                               germ=demo_germs, e=2, order=["germ", "f0", "f1"])
print("circuits1 = \n", "\n".join(map(str, circuits1)), "\n")

circuits2 = pc.create_circuits("f0+T(germ,N)+f1", f0=fids, f1=fids,
                               germ=demo_germs, N=3, T=pc.repeat_and_truncate,
                               order=["germ", "f0", "f1"])
print("circuits2 = \n", "\n".join(map(str, circuits2)), "\n")

circuits3 = pc.create_circuits("f0+T(germ,N)+f1", f0=fids, f1=fids,
                               germ=demo_germs, N=3, T=pc.repeat_with_max_length,
                               order=["germ", "f0", "f1"])
print("circuits3 = \n", "\n".join(map(str, circuits3)), "\n")
```

### Enumerating circuits and LGST lists

The `list_`*xxx* and `create_`*xxx* functions cover the common cases directly. `list_all_circuits` enumerates every circuit in a length range, and `create_lgst_circuits` builds the set of circuits needed to run Linear GST from a pair of fiducial lists.

```{code-cell} ipython3
my_gates = ['Gx', 'Gy']  # operation labels -- often just model.operations.keys()
all_strings = pc.list_all_circuits(my_gates, minlength=0, maxlength=2)
print("All circuits over %s up to length 2 =\n" % my_gates, "\n".join(map(str, all_strings)))
```

```{code-cell} ipython3
my_fiducials = pc.to_circuits([('Gf1',), ('Gf2',)])
lgst_strings = pc.create_lgst_circuits(my_fiducials, my_fiducials, my_gates)
print("%d LGST circuits:\n" % len(lgst_strings), "\n".join(map(str, lgst_strings)))
```

### Putting it together by hand

Here's `create_lsgst_circuit_lists` rebuilt from those pieces, simplified by assuming the preparation and measurement fiducials are the same list. Read it if you want to know exactly what the library function is doing, or as a starting point for a nonstandard design.

```{code-cell} ipython3
def my_make_lsgst_lists(op_labels, fiducials, germ_list, max_length_list):
    lgst_strings = pc.create_lgst_circuits(fiducials, fiducials, op_labels)
    lsgst_list = pc.to_circuits([()])  # running list of everything so far

    if max_length_list[0] == 0:
        list_of_lists = [lgst_strings]
        max_length_list = max_length_list[1:]
    else:
        list_of_lists = []

    for max_len in max_length_list:
        lsgst_list += pc.create_circuits("f0+R(germ,N)+f1", f0=fiducials,
                                         f1=fiducials, germ=germ_list, N=max_len,
                                         R=pc.repeat_with_max_length,
                                         order=('germ', 'f0', 'f1'))
        list_of_lists.append(pygsti.remove_duplicates(lgst_strings + lsgst_list))

    return list_of_lists

my_lsgst_lists = my_make_lsgst_lists(['Gx', 'Gy'], prep_fiducials, germs, max_lengths)
print('\n'.join(['%d circuits' % len(l) for l in my_lsgst_lists]))
```

The counts won't match `lsgst_lists` above, and the whole gap comes from two things. This version seeds the running list with the empty circuit, which is one extra. And it builds its LGST block from the bare labels `'Gx'` and `'Gy'` rather than the model's actual `Gxpi2:0`/`Gypi2:0` labels. That second one matters more than it looks: the fiducials are themselves made of `Gxpi2:0` and `Gypi2:0`, so an LGST block over the real labels is already contained in the germ-loop output and dedupes away almost entirely, while a block over invented labels `'Gx'`/`'Gy'` mostly does not. Swap both back and the counts land exactly on the library's. The library function also handles fiducial-pair reduction, per-germ length limits, label aliasing, and truncation schemes other than whole germ powers, none of which appear here.

## Find and replace on circuits

`manipulate_circuit` and `manipulate_circuits` rewrite circuits according to a list of replacement rules. Each rule is a pair of layer-label tuples: find the first, substitute the second. Take these four rules:

- AB $\rightarrow$ AB' (if B follows A, prime B)
- BA $\rightarrow$ B''A (if B precedes A, double-prime B)
- CA $\rightarrow$ CA' (if A follows C, prime A)
- BC $\rightarrow$ BC' (if C follows B, prime C)

```{code-cell} ipython3
sequence_rules = [
    (("A", "B"), ("A", "B'")),
    (("B", "A"), ("B''", "A")),
    (("C", "A"), ("C", "A'")),
    (("B", "C"), ("B", "C'"))]
```

Applying them gives BAB $\rightarrow$ B''AB', ABA $\rightarrow$ AB'A, CAB $\rightarrow$ CA'B', ABC $\rightarrow$ AB'C'. The ABA case is worth noticing. Matches are collected by scanning left to right, and a label one rule has already claimed cannot be modified by a later rule, so the AB rule fires first and blocks the BA rule from touching the same B. Overlapping rules frustrate each other, and the order you list them in decides who wins.

```{code-cell} ipython3
from pygsti.circuits import Circuit, manipulate_circuit

for s in ['BAB', 'ABA', 'CAB', 'ABC']:
    print(manipulate_circuit(Circuit(tuple(s)), sequence_rules).str)
```

```{code-cell} ipython3
# manipulate_circuits does the same thing to a whole list at once
orig_lst = pc.to_circuits([tuple('BAB'), tuple('ABA'), tuple('CAB'), tuple('ABC')])
lst = pc.manipulate_circuits(orig_lst, sequence_rules)
print('\n'.join([c.str for c in lst]))
```

## Operation label aliases

Aliasing is a narrower version of the same idea, and it applies **only to `DataSet` lookups**. An alias maps one operation label to a circuit, so the labels a `Model` uses need not match the labels the data was recorded under. This is what lets several models with different (often simpler) gate labellings be fit against the same data.

Unlike `manipulate_circuit`, the thing being found is a single operation label, never a longer pattern.

```{code-cell} ipython3
aliases = {'Gx': Circuit([('Gxpi2', 0)]), 'Gy': Circuit([('Gypi2', 0)])}
some_circuits = pc.to_circuits([('Gx', 'Gy'), ('Gx', 'Gx', 'Gy')])
print('\n'.join([c.str for c in pygsti.tools.apply_aliases_to_circuits(some_circuits, aliases)]))
```

Most of the time you won't call this yourself. The GST construction and protocol functions take an `op_label_aliases` argument and apply it where it's needed.
