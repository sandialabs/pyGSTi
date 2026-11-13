# GST experiment designs

The circuits in a GST experiment determine which errors the fit can estimate. Start with [GST designs from model packs](ModelpackDesigns) if your gate set has a model pack; the notebook shows how maximum lengths, germ sets, and fiducial-pair reduction change the circuit count.

- [GST circuit construction](GSTCircuits) builds the repeated-germ circuit lists and combines them with preparation and measurement fiducials.
- [Fiducial and germ selection](FiducialsAndGerms) selects these ingredients for a model without precomputed choices.
- [Fiducial pair reduction](FewerCircuits) compares ways to reduce the number of circuits.
- [Evaluating experiment designs using Fisher information](CheckYourDesign) checks a design's sensitivity before you collect data.

Once you have a design and its outcome counts, continue to [Running GST](RunningGST).
