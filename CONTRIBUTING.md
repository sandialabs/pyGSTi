# Contributing to pyGSTi

Thanks for taking the time to contribute to pyGSTi! Open-source projects like
ours couldn't exist without contributors like you.

This document contains a set of guidelines for contributing to pyGSTi and
related packages hosted under the
[pyGSTio group](https://github.com/pyGSTio). We ask that contributors make an
earnest effort to follow these guidelines, but no one's keeping score here --
just use your best judgement, and by all means, feel free to propose changes to
these guidelines in a pull request.

### Reporting Bugs

Found a bug in pyGSTi? We'd appreciate letting us know!

* First, **see if the bug has already been reported** by searching on Github
  under [Issues](https://github.com/pyGSTio/pyGSTi/issues).

* If you can't find an open issue about the problem,
  [open a new one](https://github.com/pyGSTio/pyGSTi/issues/new)! Be sure to
  include a **title**, a **clear description** of what went wrong, and **as much
  relevant information as possible.** If you can, try to include detailed steps
  to reproduce the problem.

* Once you open an issue, please **keep an eye on it** -- you may be asked to
  provide more details.

### Suggesting Features & Enhancements

Want to add a new feature or change an existing one?

* First, **see if the feature or enhancement has already been suggested** by
  searching on Github under [Issues](https://github.com/pyGSTio/pyGSTi/issues).

* If you can't find a similar suggestion in the issue tracker, [open a new
  issue](https://github.com/pyGSTio/pyGSTi/issues/new) for your feature
  suggestion.

### Making a Contribution

Want to fix a bug, add a feature, or change an existing feature?

Because pyGSTi is a project of the Quantum Performance Lab at Sandia National
Labs, the process of contributing to pyGSTi is different for contributors
working at Sandia.

#### For non-Sandians

Unfortunately, **we can't currently accept pull requests from contributors
outside of SNL.** We're working on setting up a contributor license agreement,
so, someday, this may change. Check back later!

#### For Sandians

* **Contact the authors** at [pygsti@sandia.gov](mailto:pygsti@sandia.gov) to
  request an invite to the [pyGSTio group](https://github.com/pyGSTio).

* If there is an **open issue** in the
  [issue tracker](https://github.com/pyGSTio/pyGSTi/issues) for the bug,
  feature, or enhancement, **assign yourself** to it.

* In your local repository, make a **feature branch** off of `develop` for your
  contribution. The names of branches for *features* and *enhancements* should
  begin with `feature-`. Branch names for *bug fixes* should begin with
  `bugfix-`.

* Write and commit your patch in this branch. Try and make **frequent commits
  containing only related work.** While you should **push** your changes to
  Github often, please try to avoid changing history on publicly-visible
  branches.

* Once you're finished, **pull** the latest refs on `develop` from Github and
  **merge** `develop` into your feature branch. Resolve any merge conflicts that
  arise.

* On Github under [Pull Requests](https://github.com/pyGSTio/pyGSTi/pulls),
  create a new pull request to merge your feature branch into `develop`.

* If your feature passes linting and automated tests, it will be reviewed by a
  pyGSTi core developer, who may have suggestions for changes or fixes which you
  should make before your patch can be merged.
