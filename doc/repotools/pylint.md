# Pylint

### By default, ./lint.py lints for items given by positional arguments:

`./lint.py filename item` -> `./lint.py docs missing-docstring empty-docstring`
  (generate a file `output/pylint/docs.out` that logs places that have empty or missing docstrings) 

A full list of items can be found [here](https://docs.pylint.org/features.html#general-options)  
  (Or in `pylint_message.txt`)

For simplicity, a filename can be left off if only one item is being looked for:

`./lint.py unused-import` generates a file `unused-import` logging the occurances of `unused-import`  

Some examples:
 - `./lint.py duplicate-code`
 - `./lint.py todos fixme`
 - `./lint.py unused unused-variable unused-import unused-argument`

### If no positional arguments are given, pylint lints for `all`:
  - All warnings except those blacklisted in `pylint_config`
  - All errors except those blacklisted in `pylint_config`
  - No refactors except those whitelisted in `pylint_config`
  - No conventions, ever.
  
Branches `beta` and `master`:  
  If `--score` is specified, `lint.py` exits with `1` if the code scores lower than the latest run (visible in pylint_config)

### Optional flags:

| Flag            | Description                                            |
|-----------------|:-------------------------------------------------------|
| `--score`       | compares repo score to latest score, exits 0 if lower  |
| `--errors`      | generate pylint report for errors                      |
| `--warnings`    | generate pylint report for warnings                    |
| `--adjustables` | generate pylint report for adjustable refactors        |
| `--noerrors`    | return 1 if any errors are found in pygsti (not tests) |

(Multiple flags can be provided in a call to lint.py, as well as the positional arguments)

The command `./lint.py --noerrors` is called whenever a push is made to any branch. There aren't any 'errors' in the repository right now, so this will catch any missing parentheses or other syntax errors. It is definitely worth the time to run, but can be bypassed with the flag `--no-verify`.

The command `./lint.py --score` is run in the background after a commit to develop. (A report will be generated within a minute)

### Config file:

Found in `test/config/pylint_config.json`. Allows setting default values for refactoring issues (`i.e. max-statements=100` limits functions/methods to 100 lines when running `./lint.py --adjustables`). Also allows specifying pylint version (Set to pylint3 by default) and a threshold score for branches like beta and master (which require travis CI to pass, anyways..)

#### `clonedigger`:
While pylint has capability for finding duplicated code (`./lint.py duplicate-code`), I've found the tool `clonedigger` to be better.  

Installation:
`pip install clonedigger`

Usage:
`clonedigger tools/ -o duplicated_tools.html` would generate a file outlining duplicated code in a tools package. 

The script `duplicated.py` will automatically generate these files for each sub package in pygsti! Use it!
