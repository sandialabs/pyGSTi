# Building and previewing the documentation

Use an environment with pyGSTi's documentation dependencies (the `docs` extra). From a source checkout, build the extensions in place and set `PYTHONPATH` to that checkout so notebook kernels and Sphinx import the code you are reviewing.

## Routing pages and notebooks

Pages with children help readers choose a notebook: explain what each child covers and when to use it. Put worked code and output in leaf notebooks, keeping Jupytext metadata on the notebooks and off the routing pages. The `markdown/` folders mirror `_toc.yml`: one folder per part and chapter, each landing page named `Index.md`. When a page changes chapter, move its file to match, then update its incoming links, the relative `tutorial_files`, `example_files` and report paths in its own cells (their depth changes with the folder), and any entry in `execute-notebooks.py`'s shared-file execution order.

## Fast preview

From the repository root:

```bash
export PYTHONPATH="$PWD"
python docs/build-prose.py
python -m http.server 8950 --bind 127.0.0.1 --directory docs/_build/prose/_build/html
```

Open <http://localhost:8950>. Re-run the build after editing Markdown and refresh the browser; unchanged pages are cached. `--all` forces a full re-read. Stop the server with Ctrl-C.

This build renders the tutorials without executing notebooks or generating the API reference. A source checkout therefore shows code without execution output. Example report links require reports generated separately. The ReadTheDocs version selector is supplied by ReadTheDocs and does not appear in a local preview.

For a remote checkout, forward port 8950 from the environment running the server. In VS Code connected directly to that environment, use **Ports → Forward a Port → 8950**. If the server runs directly on an SSH host, run this on your own computer:

```bash
ssh -N -L 8950:127.0.0.1:8950 user@host
```

Then open the same localhost URL. A rootless container has its own network namespace: forwarding the host's loopback port will not reach a container-only server unless its port was published. In that case, use the editor's container connection, or copy the rendered directory to your computer and serve it there.

To make a portable bundle, run this in the checkout:

```bash
python -m zipfile -c docs/_build/docs-preview.zip docs/_build/prose/_build/html
```

Download and extract the ZIP, then run `python -m http.server 8950 --bind 127.0.0.1 --directory html` from the extracted directory. Open <http://localhost:8950>. The bundle contains the rendered pages and local assets; externally hosted resources such as MathJax still need an internet connection.

## Notebook output and example reports

To preview the site with executed notebooks and the example reports:

```bash
export PYTHONPATH="$PWD"
python docs/local-preview.py full      # execute, collect reports, build (~35 minutes on 8 cores)
python docs/local-preview.py update    # after editing pages: rebuild in seconds
python -m http.server 8950 --bind 127.0.0.1 --directory docs/_build/executed/_build/prose/_build/html
```

`full` passes arguments after `--` to `execute-notebooks.py`, for example `-- --jobs 4`. `update` keeps the outputs already in each notebook, so a page whose code you changed shows stale output until you run `full` again. The script's docstring has the details.

The steps `full` runs, for doing them by hand:

```bash
export PYTHONPATH="$PWD"
grep -rl --include='*.md' '^jupytext:' docs/markdown | xargs jupytext --to notebook --update
python -c "import sys; sys.path.insert(0, 'docs'); import conftest; conftest._generate_shared_fixtures()"
python docs/execute-notebooks.py --jobs 4 --timeout 1800 --allow-errors
python docs/collect-reports.py collect
```

Run these commands in order and stop if one fails. Set the job count to suit the available CPU and memory. The driver retains partial outputs and exits nonzero if a cell fails. Some tutorials require optional packages, an MPI launcher, or hardware credentials; use `--exclude-also` with paths relative to `docs/markdown` without a suffix to deliberately skip unavailable examples, and record what was skipped.

The `grep` selects the paired notebook pages; landing and prose pages carry no Jupytext header and must not become notebooks. Use `jupytext --to notebook --update`, rather than bidirectional `--sync`, after editing Markdown: freshly executed notebooks have newer timestamps and must not overwrite your edits. Generated `.ipynb` files and reports should not be committed to source branches.

`docs/stage-rtd-branch.py` assembles executed notebooks and collected reports into a generated publishing branch. Its module documentation explains the staging options. The production workflow still uses `develop-with-notebooks`; do not delete that branch while the workflow and ReadTheDocs configuration depend on it. The staging tools also support future documentation reviews, so they remain useful after the restructuring PR merges.

## Full reference build

```bash
export PYTHONPATH="$PWD"
jb build docs
```

This also generates the API reference and takes substantially longer. It does not execute notebooks; execution and publishing are separate steps above.
