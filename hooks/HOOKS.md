# Hooks 
### (and how to use them):

#### Setup:  

Go to `pyGSTi/hooks` directory, and run `./setup_hooks.sh`  
This copies the contents of `pyGSTi/hooks/git` into `pyGSTi/.git/hooks`  
After hooks have been configured, any pull or merge will update them again  
(If the `post-merge` hook breaks, and stops automatically updating hooks, they will need to be re-updated by another call to `setup_hooks.sh`, after a fix is made)

#### Hook operations, by branch:
(See hooks/hooksettings.py for succinct settings)

##### all

`prepare-commit-msg` - Adds `[ci skip]` to commit message if only `.md` and `.txt` files have changed

##### master

`pre-commit`   -  requires pylint score to be higher than that in `pyGSTi/test/pylint_config.yml` (currently `9.10`) - creates report

`post-commit`  -  generates html **locally** for `gh-pages`

`pre-push`     -  updates `gh-pages`

##### beta

`pre-commit`   -  requires pylint score to be higher than that in `pyGSTi/test/pylint_config.yml` (currently `9.10`) - creates report

##### develop

`post-commit`  -  generates pylint report `pyGSTi/test/output/pylint/all.out` - doesn't check score

### Important note:

git hooks can be bypassed with the flag `--no-verify`, for example, in the case that something needs to be pushed to `develop` as a hotfix, but lowers the pylint score. (The latest pylint score in `pyGSTi/test/pylint_config.yml` can also be lowered to something more reasonable, if required(It would be nice to have it above `9.0`?))
