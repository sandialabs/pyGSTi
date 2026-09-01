# What should I read first?

This part is a short guided path from a fresh install to a characterization result you can actually read. It is the whole of what most people need, and the pages are meant to be taken in order.

**The path.** [Install pyGSTi](Install), then [choose a protocol](ChooseAProtocol). That second page is a decision rather than a tutorial: the protocols answer different questions, and picking the wrong one wastes an experiment. What you pick decides which of the next two pages you read — [your first GST run](FirstGST) if you want a full description of what a small gate set actually does, [your first benchmark](FirstBenchmark) if you want a single error rate for a larger device. Both run end to end on simulated data, so you can follow along before you have any of your own.

Once one of those works, [getting your own data in](YourOwnData) swaps the simulated counts for real ones, and [reading your results](ReadingResults) covers what comes back and which number to look at before the others.

**Reference, not steps.** Two pages here are meant to be consulted rather than worked through. [Key concepts](KeyConcepts) states gauge freedom, error generators, parameterizations and the fit statistics once, in prose, so the rest of the documentation can assume them. [Model packs](TargetModels) is the catalogue of gate sets that ship with pyGSTi, which is where most people's target model comes from.

**When you outgrow this.** A device the model packs do not describe, a protocol variant, a fit that does not look right, or anything involving more than a couple of qubits — the [characterization guides](../guides/Index) pick up from there.
