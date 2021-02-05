#!/usr/bin/env bash
set -euo pipefail


# ======================== convert and filter pdf files ========================

echo $PWD
mkdir txts

# convert all pdfs to text
# for file in pdf/*.pdf; do pdftotext "$file"; done

# more readible if you write a shell script to execute
# for file in pdf/*.pdf
#     do pdftotext "$file"
# done

# move all the txt files to a different folder
# mv pdf/*.txt txt/

# pdfs too large to upload, unzip already converted txts
unzip txt.zip

# search for patterns: with matches highlighted
grep -i --color "causal" txt/*.txt

# with line number
grep -in "causal" txt/*.txt
# only count of matches
grep -ic "causal" txt/*.txt

# only matching files, only count of matches
grep -ilc "causal" txt/*.txt

# extend pattern with regex
grep -ilc -E "(causal|missing data)" txt/*.txt

# look at first and last
grep -ilc -E "(causal|missing data)" txt/*.txt | head -n 10
grep -ilc -E "(causal|missing data)" txt/*.txt | tail -n 10

# pipe all non-matches (-L) to rm for deletion, careful with rm!
grep -Z -L -E "(causal|missing data)" txt/*.txt | xargs -0 rm


# ====================== convert example txt file to a csv =====================

# check example file
grep -E --color '^32[0-9]' example.txt
grep -E --color '^32[0-9]{12}' example.txt

# doesn't recognize line ending
grep -E --color '^32[0-9]{12}$' example.txt

# use sed (Stream EDitor) to substitute some patterns within the file
echo "These linux tools are boring" | sed 's/boring/exciting/'

echo "This is a test." > test.txt
cat test.txt
sed 's/a test/another test/' test.txt
sed -e 's/This/That/; s/a test/yet another test/' test.txt

# convert dos to unix line ending
# (carriage return ^M typed as CTRL-SHIFT-V CTRL-M, or CTRL-q CTRL-M on Windows)
sed 's/^M$//' example.txt > example-unix.txt
# tr -d '\r' < example.txt > example-unix.txt.txt


# works
grep -P --color '^32[0-9]{12}$' example-unix.txt
# doesn't - grep matches lines, cannot match over lines. most editors (or sed) can.
# may work on WSL, doesn't work on Git Bash
grep -P --color '^32[0-9]{12}\n\n' example-unix.txt

# substitute pattern is 's/pattern/replacement/[option]', \0 repeats whole match
sed -e 's/^32[0-9]\{12\}$/\0/' example-unix.txt
# sed -n -e 's/^32[0-9]\{12\}\n/\0/' example-unix.txt
sed -e 's/^32[0-9]\{12\}$/HIER-STEHT-DIE-ID/' example-unix.txt

# crude way of inserting a separator
sed -e 's/^32[0-9]\{12\}$/";\n\0;"/' example-unix.txt > example.csv


# check and fix first/last row (could also do this in an editor)
# first row
head example.csv
## -i operates on the file directly, 1d deletes the first line
sed -i '1d' example.csv

# last row, $ selects last row, a appends the following characters
tail example.csv
sed -i '$a"' example.csv


