#! /usr/bin/env bash
# This script is run automatically by TravisCI
# This pushes passing builds on the `deploy' branch to `beta'

GIT_USER="Travis CI"
GIT_EMAIL="travis@travis-ci.org"

# we should only be run by Travis CI
if [ -z "$TRAVIS" ]; then
    echo "This script is run automatically by Travis CI."
    echo "Please don't run it manually!"
    echo '... But if you really have to, set $TRAVIS to "true" to bypass this safety check.'
    exit 1
fi

# trigger only on `develop' by default (configurable)
if [ -z "$TRIGGER_BRANCH" ]; then
    TRIGGER_BRANCH="develop"
fi

# push to `beta' by default (configurable)
if [ -z "$PUSH_BRANCH" ]; then
    PUSH_BRANCH="beta"
fi

echo "TRAVIS_REPO_SLUG = $TRAVIS_REPO_SLUG"
echo "TRAVIS_BRANCH = $TRAVIS_BRANCH"
echo "PUSH_BRANCH = $PUSH_BRANCH"
echo "TRAVIS_PULL_REQUEST = $TRAVIS_PULL_REQUEST"

if [ "$TRAVIS_BRANCH" = "$TRIGGER_BRANCH" ] && [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
    # setup git user
    git config user.email "$GIT_EMAIL"
    git config user.name "$GIT_USER"

    # branch develop head to beta
    git checkout -b "$PUSH_BRANCH"

    # if branch exists upstream, apply it on top of this one
    UPSTREAM_URI="git@github.com:$TRAVIS_REPO_SLUG.git"
    UPSTREAM_BRANCH=$(git ls-remote --heads "$UPSTREAM_URI" "$PUSH_BRANCH")
    if [ "$?" -eq 0 ] && [ -n "$UPSTREAM_BRANCH" ]; then
        git pull --ff-only "$UPSTREAM_URI" "$PUSH_BRANCH"
        if [ "$?" -eq 0 ]; then
            echo "Fast-forwarded $PUSH_BRANCH to $TRIGGER_BRANCH."
        else
            echo "ERROR: couldn't fast-forward $PUSH_BRANCH to $TRIGGER_BRANCH!"
            echo "These branches must be merged manually."
            echo "hint: If you wish to merge these branches automatically in the future,"
            echo "hint: add the conflicting refs from $PUSH BRANCH to $DEVELOP BRANCH."
            exit 2
        fi
    fi

    # push branch to remote repo
    # this requires the Travis CI pubkey be added as a write-access
    # deployment key to the repo
    git push "$UPSTREAM_URI" "$PUSH_BRANCH"
fi
