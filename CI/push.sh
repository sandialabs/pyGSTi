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
    git config --global user.email "$GIT_EMAIL"
    git config --global user.name "$GIT_USER"

    # branch develop head to beta
    git checkout -b "$PUSH_BRANCH"

    # # TODO create deployment token
    # PUSH_URI="https://$GH_TOKEN@github.com/$TRAVIS_REPO_SLUG.git"
    PUSH_URI="https://$USER:$PUSHKEY@github.com/$TRAVIS_REPO_SLUG.git"

    # suppress output to avoid leaking token
    echo "Pushing to $TRAVIS_REPO_SLUG:$PUSH_BRANCH..."
    git push --quiet "$PUSH_URI" "$PUSH_BRANCH" > /dev/null 2>&1
    check=$?
    if [ "$check" -eq 0 ]; then
        echo "Done."
    else
        echo "Push failed with exit status $check!"
        echo "You will most likely need to manually merge $TRIGGER_BRANCH into $PUSH_BRANCH"
        exit $check
    fi
fi
