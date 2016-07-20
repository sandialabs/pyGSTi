# Pylint

### By default, ./lint.py lints for items given by positional arguments:

`./lint.py filename item` -> `./lint.py docs missing-docstring empty-docstring`
  (generate a file `output/pylint/docs.out` that logs places that have empty or missing docstrings) 

A full list of items can be found [here](https://docs.pylint.org/features.html#general-options)  
  (Or in `pylint_message.txt`)

For simplicity, a filename can be left off if only one item is being looked for:

`./lint.py unused-import` generates a file `unused-import` logging the occurances of `unused-import`  

Some examples:
 `./lookFor.py duplicate-code`
 `./lookFor.py todos fixme`
 `./lookFor.py unused unused-variable unused-import unused-argument`

### If no positional arguments are given, pylint lints for `all`:
  - All warnings except those blacklisted in `pylint_config`
  - All errors except those blacklisted in `pylint_config`
  - No refactors except those whitelisted in `pylint_config`
  - No conventions, ever.
  
  If `--score` is specified, `lint.py` exits with `1` if the code scores lower than the latest run (visible in pylint_config)

### Optional flags:

| Flag            | Description                                           |
|-----------------|:------------------------------------------------------|
| `--score`       | compares repo score to latest score, exits 0 if lower |
| `--errors`      | generate pylint report for errors                     |
| `--warnings`    | generate pylint report for warnings                   |
| `--adjustables` | generate pylint report for adjustable refactors       |

(Multiple flags can be provided in a call to lint.py, as well as the positional arguments)


