# Hooks 
## How to use them:

#### Setup:  

Go to `pyGSTi/hooks` directory, and run `./setup_hooks.sh`  
This copies the contents of `pyGSTi/hooks/git` into `pyGSTi/.git/hooks`  
It also copies `pyGSTi/test/helpers/automation_tools` into the hooks directory
After hooks have been configured, any pull or merge will update them again  
(If the `post-merge` hook breaks, and stops automatically updating hooks, they will need to be re-updated by another call to `setup_hooks.sh`, after a fix is made)

#### Hook operations, by branch:
(See hooks/hooksettings.py for succinct settings)

##### all

`prepare-commit-msg` - Adds `[ci skip]` to commit message if only `.md` and `.txt` files have changed

`post-commit` reindents pygsti. DOES add changes to the commit but will abort if there are unchanged files (So that they aren't accidentally added!).

`post-commit` starts linting in a background process - output will be generated in pyGSTi/test/output/pylint/all.out

`pre-push` lints for errors in pygsti and fails if any are found. This takes a bit of time, but it will prevent any syntax errors or other minor mistakes from being pushed to the repository

##### master

`post-commit`  -  updates `gh-pages`

##### travis

`deploy` - Merges develop into beta if there are no merge conflicts

#### Important note:

restrictive git hooks (i.e. `pre-push`) can be bypassed with the flag `--no-verify`

## How to maintain them:

- The setup script outlined above is pretty important in keepings hooks updated. If something breaks, you might want to check that and potentially the post-merge script that calls it. This also means that you'll have to call the setup script **manually** when working on the hooks.

- Some hooks rely on the `automation_tools` package. This gets copied from `test/helpers` whenever the setup script is called. (So, the setup script would have to be called to update the version that the hooks use)

- The hooks can be run manually (for testing) by first calling the setup script, and then moving the the top-level pygsti directory (pyGSTi), and issuing a command like `./.git/hooks/pre-commit` (if in the hooks/git directory, try the command `cd ..; ./setup_hooks.sh; cd ..; ./.git/hooks/hook; cd hooks/git`). Note that the travis hooks are run from the hooks/travis directory, but the git hooks are run from `pyGSTi`

- Use the `--no-verify` flag whenever anything breaks

- Be careful with issuing git commands from the hooks (I did this with the `post-commit` hook, and even though I used `--no-verify`, I ended up with in infinite loop of hooks calling hooks, since the `post-commit` hook runs even when `--no-verify` is used)


