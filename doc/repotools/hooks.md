# Hooks 
## How to use them:

#### Setup:  

Go to `pyGSTi/hooks` directory, and run `./setup_hooks.sh`  
This copies the contents of `pyGSTi/hooks/git` into `pyGSTi/.git/hooks`  
After hooks have been configured, any pull or merge will update them again  
(If the `post-merge` hook breaks, and stops automatically updating hooks, they will need to be re-updated by another call to `setup_hooks.sh`, after a fix is made)

#### Hook operations, by branch:
(See hooks/hooksettings.py for succinct settings)

##### all

`prepare-commit-msg` - Adds `[ci skip]` to commit message if only `.md` and `.txt` files have changed

`pre-commit` reindents pygsti. DOES add changes to the commit but will abort if there are unchanged files (So that they aren't accidentally added!).

##### master

`pre-commit`   -  requires pylint score to be higher than that in `pyGSTi/test/pylint_config.json` (currently `9.10`) - creates report

`pre-push`     -  updates `gh-pages`

##### beta

`pre-commit`   -  requires pylint score to be higher than that in `pyGSTi/test/pylint_config.json` (currently `9.10`) - creates report

##### travis

`after_success` - (Commented-out section would update beta automatically)

#### Important note:

git hooks can be bypassed with the flag `--no-verify`, for example, in the case that something needs to be pushed to `develop` as a hotfix, but lowers the pylint score. (The latest pylint score in `pyGSTi/test/pylint_config.json` can also be lowered to something more reasonable, if required(It would be nice to have it above `9.0`?))

## How to maintain them:

- The setup script outlined above is pretty important in keepings hooks updated. If something breaks, you might want to check that and potentially the post-merge script that calls it. This also means that you'll have to call the setup script **manually** when working on the hooks.

- Some hooks rely on the `automation_tools` package. This gets copied from `test/helpers` whenever the setup script is called. (So, the setup script would have to be called to update the version that the hooks use)

- The hooks can be run manually (for testing) by first calling the setup script, and then moving the the top-level pygsti directory (pyGSTi), and issuing a command like `./.git/hooks/pre-commit` (if in the hooks/git directory, try the command `cd ..; ./setup_hooks.sh; cd ..; ./.git/hooks/hook; cd hooks/git`). Note that the travis hooks are run from the hooks/travis directory, but the git hooks are run from `pyGSTi`

- Use the `--no-verify` flag whenever anything breaks

- Be careful with issuing git commands from the hooks (I did this with the `post-commit` hook, and even though I used `--no-verify`, I ended up with in infinite loop of hooks calling hooks, since the `post-commit` hook runs even when `--no-verify` is used)


