# Hooks 
### (and how to use them):

#### Setup:  

Go to `pyGSTi/hooks` directory, and run `./setup_hooks.sh`  
This copies the contents of `pyGSTi/hooks` into `pyGSTi/.git/hooks`  
After hooks have been configured, any pull or merge will update them again  
(If the `post-merge` hook breaks, and stops automatically updating hooks, they will need to be re-updated by another call to `setup_hooks.sh`, after a fix is made)

#### Hook operations, by branch:
(See hooks/hooksettings.py for succinct settings)

##### master

`pre-commit`   -  requires pylint score to be higher than that in `pyGSTi/pylint/config.yml` (currently `9.72`) - creates report

`post-commit`  -  generates html **locally** for `gh-pages`

`pre-push`     -  updates `gh-pages`

##### beta

`pre-commit`   -  requires pylint score to be higher than that in `pyGSTi/pylint/config.yml` (currently `9.72`) - creates report

##### develop

`post-commit`  -  generates pylint report `pyGSTi/pylint/output/all.out` - doesn't check score
