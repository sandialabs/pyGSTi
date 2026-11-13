# Contributing to the documentation

The tutorial sources live in `docs/markdown/` in the [pyGSTi repository](https://github.com/sandialabs/pyGSTi). Each executable page is MyST Markdown; its paired notebook is generated for execution and publication. Edit the Markdown source, then preview it locally before opening a pull request.

## Preview your changes

With the repository's documentation dependencies installed, run this from its root:

```bash
python docs/build-prose.py
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/prose/_build/html
```

Open `http://localhost:8000` in your browser. Re-run the build after an edit and refresh the page. This preview renders the tutorials without executing their code or generating the API reference. Example report links need separately generated reports. See `docs/README.md` in the repository for remote access, notebook execution, and a full build.

## Useful projects

These are starting points for a contribution. Check existing issues or discuss your proposal on [GitHub](https://github.com/sandialabs/pyGSTi/issues) before undertaking a larger change.

- **Qiskit circuit and data interchange.** Add a self-contained tutorial for converting circuits and counts without requiring an IBM hardware account. The existing IBM Q tutorial covers hardware submission.
- **More realistic input data.** Add small, reproducible examples of loading counts from common experiment formats to the data-loading guide.
- **Accessible reports and figures.** Check keyboard navigation, dark-mode contrast, and descriptions of plots; include a small example when reporting a problem.
- **Focused reference pages.** Separate long reference material from worked examples in the circuit-label, model-noise, results, and file-format tutorials.
- **Reliable section links.** Investigate cross-page heading links in the rendered book and add a check that catches broken anchors.

For a tutorial change, check that its cells still run in order and that the explanation agrees with their output. Report the Python and pyGSTi versions used; runtime observations depend on the machine and should not be presented as promises.
