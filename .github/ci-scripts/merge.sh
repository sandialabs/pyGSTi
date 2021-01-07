#! /usr/bin/env bash
# This script is run by a GitHub Action
# This merges passing builds on the `deploy' branch to `beta'

GIT_USER="GitHub Action"
GIT_EMAIL="pygsti@noreply.github.com"

# we should only be run during Action
if [ -z "$GITHUB_ACTIONS" ]; then
    echo "This script is run automatically by GitHub Actions."
    echo "Please don't run it manually!"
    echo '... But if you really have to, set $GITHUB_ACTIONS to "true" to bypass this safety check.'
    exit 1
fi

# trigger only on `develop' by default (configurable)
if [ -z "$TRIGGER_REF" ]; then
    exit 1 # For now, do not allow default while debugging
    TRIGGER_REF="/refs/heads/develop"
fi

# push to `beta' by default (configurable)
if [ -z "$MERGE_BRANCH" ]; then
    exit 1 # For now, do not allow default while debugging
    MERGE_BRANCH="beta"
fi

echo "GITHUB_REPOSITORY = $GITHUB_REPOSITORY"
echo "GITHUB_REF = $GITHUB_REF"
echo "PUSH_BRANCH = $PUSH_BRANCH"

# Following should only be set for pull requests
PULL_REQUEST=false
if [ -z "$GITHUB_HEAD_REF" ] || [ -z "$GITHUB_BASE_REF" ]; then
    PULL_REQUEST=true
fi
echo "PULL_REQUEST = $PULL_REQUEST"

if [ "$GITHUB_REF" = "$TRIGGER_REF" ] && [ "$PULL_REQUEST" = "false" ]; then
    # setup git user
    git config user.email "$GIT_EMAIL"
    git config user.name "$GIT_USER"

    # branch develop head to beta
    git checkout -b "$MERGE_BRANCH"

    # if branch exists upstream, apply it on top of this one
    UPSTREAM_URI="git@github.com:$GITHUB_REPOSITORY.git"
    UPSTREAM_BRANCH=$(git ls-remote --heads "$UPSTREAM_URI" "$MERGE_BRANCH")
    if [ "$?" -eq 0 ] && [ -n "$UPSTREAM_BRANCH" ]; then
        git pull --ff-only "$UPSTREAM_URI" "$MERGE_BRANCH"
        if [ "$?" -eq 0 ]; then
            echo "Fast-forwarded $MERGE_BRANCH to $TRIGGER_REF."
        else
            echo "ERROR: couldn't fast-forward $MERGE_BRANCH to $TRIGGER_REF!"
            echo "These branches must be merged manually."
            echo "hint: If you wish to merge these branches automatically in the future,"
            echo "hint: add the conflicting refs from $PUSH BRANCH to $DEVELOP BRANCH."
            exit 2
        fi
    fi

    # SS 2020-01-06: Should not need this part in this script

    # push branch to remote repo
    # this requires the Travis CI pubkey be added as a write-access
    # deployment key to the repo
    #git push "$UPSTREAM_URI" "$MERGE_BRANCH"
fi
