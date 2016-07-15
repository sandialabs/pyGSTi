from __future__ import print_function

def get_branchname():
    try:
        branchname = subprocess.check_output(['bash', '.git/hooks/.get_branch'])
    except subprocess.CalledProcessError:
        branchname = 'unnamed_branch'
    branchname = os.path.basename(branchname)
    branchname = branchname.replace('\n', '')
    return branchname


