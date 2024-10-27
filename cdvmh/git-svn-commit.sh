#!/bin/bash

do_help() {
  echo "NAME"
  echo "  git-svn-commit - a helper tool to preserve SVN and GIT repositories in a consistent state"
  echo
  echo "SYNOPSIS"
  echo "  git-svn-commit [options] [<master> <trunk>]"
  echo
  echo "DESCRIPTION"
  echo "  Update the SVN repository and <trunk> branch according to the changes in a <master> branch. If no arguments are specified then 'master' and 'trunk' branches are used."
  echo
  echo "OPTIONS"
  echo "  --svn-username      Set username for SVN repository"
  echo "  --help              Display this information"
}

on_error() {
  retcode=$?
  echo "error: unable to update SVN repository: $1";
  exit $retcode
}

trunk=
master=

svn_username=

while [ $# -gt 0 ]; do
  if [ $1 = "--help"  ] || [ $1 = "-h" ]; then
    do_help
    exit 0
  fi
  if [ "$1" = "--svn-username" ]; then
    current_opt=$1
    shift
    if [ $# -gt 0 ]; then
      svn_username="--username $1"
    else
      on_error "missing argument for $current_opt option"
    fi
  else
    if [ -z "$master" ]; then
      master=$1
    elif [ -z "$trunk" ]; then
      trunk=$1
    else
      on_error "too many arguments are specified"
    fi
  fi
  shift
done

if [ -z "$master" ]; then
   master="master"
fi
if [ -z "$trunk" ]; then
   trunk="trunk"
fi

echo "info: synchronise '$trunk' branch with changes in '$master' branch"

current=$(git rev-parse --abbrev-ref HEAD)
test $? -eq 0 || on_error "unable to get the current branch name"

# Just check that $master branch is exist.
git checkout $master || on_error "unable to switch to '$master' branch"

git checkout $trunk || on_error "unable to switch to '$trunk' branch"
git svn rebase $svn_username || on_error "unable to upated working copy according to the SVN repository state"

tags=(`git tag -l "svn-r*" | sed s/svn-r// | sort -rn`)
test ${#tags[@]} -ne 0 || on_error "unable to find tag for the last revision"
echo "info: the last commit stored in the SVN repository is found"
git log -n1 svn-r${tags[0]}

echo "info: pick commits from svn-r${tags[0]} to the HEAD of the $master branch"
git cherry-pick $master ^svn-r${tags[0]}

git svn dcommit $svn_username || on_error "unable  to update the SVN repository"

revision=$(git svn info | grep Revision | sed s/[a-zA-Z:\ \t]*//)
test $? -eq 0 || on_error "unable to get the SVN revision number"
echo "info: set the 'svn-r$revision' tag for the the HEAD of the $master branch"
git checkout $master
git tag svn-r$revision

git checkout $current
