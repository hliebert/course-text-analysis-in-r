#!/usr/bin/env bash
set -euo pipefail

# basic user/editor setup
git config --global user.email "helge.liebert@gmail.com"
git config --global user.name "Helge Liebert"
git config --global core.editor "vim"
git config --global core.editor "emacsclient -nw -a \'\'"
cat ~/.gitconfig

## clone course file to home directory
## (alternatively, initialize a new repository using 'git init')
cd ~/
git clone https://github.com/hliebert/course-unstructured-data.git
cd course-unstructured-data/
ll

# check readme
cat Readme.md

## basic info about repository
## HEAD is a pointer/label to the most recent commit of the branch you are currently on.
## master is the default branch created when you initialize a git repository
git status
git log
git branch
git blame Readme.md
git blame PC-lab-1/install.r

## Diff last/second to last commit
git diff HEAD~1 HEAD
git diff HEAD~2 HEAD

## create a new file
echo "This is a new readme file." > newreadme.md
diff newreadme.md Readme.md
git status
git diff

## stage file
git add newreadme.md
git status
git diff --staged

## remove from staging area again
git restore --staged newreadme.md
git status
git diff
git diff --staged
git add newreadme.md

## commit changes
git commit -m "Test file commit"
git status
git diff origin HEAD

## more changes
echo "With even more content." >> newreadme.md
cat newreadme.md
touch andanotherfile.md
git diff
git add .
git diff --staged
git commit -m "Added more information to readme"
git log -5

## push to remote
git push origin

## more changes
git touch newfile
git clean -f
git rm newreadme.sh
git restore newreadme.sh
git rm newreadme.sh
git rm Readme.sh
git add .
git commit -m "Removed readme to try without it."

## go back
git log -5
git reset HEAD~3
git status
git log -5
git clean -f

## new branch
git branch develop
git branch
git checkout develop
git branch

## some modifications
echo "Some new experimental feature." PC-lab-1/newfeature
git add newfeature
git commit -m "Added new experimental feature X for testing."
git log -5

## switch back to master
git checkout master
git log -5
git diff master develop
git diff origin/master develop
git diff origin/master origin/develop ## does not exist!

## merge branches and delete develop branch
git merge develop
git log -5
git branch -d develop
git branch

## setup .gitignore if you want to ignore certain filetypes
# nano gitignore
