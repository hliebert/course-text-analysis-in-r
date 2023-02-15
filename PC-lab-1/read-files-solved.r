################################################################################
## Filename: read.r
## Description: 
## Author: Helge Liebert
## Created: Mi. Aug 26 14:44:17 2020
## Last-Updated: Mo Jan 24 17:07:42 2022
################################################################################


#========================== reading all the txt files ==========================

library(stringr)
library(readr)
## library(data.table)

# set encoding (may be necessary on Windows)
## options(encoding = "UTF-8")
## options(max.print = 10000)

## you will need to extract the source files to a folder

## use txt-utf-8.zip if you are on MacOS or Linux
## unzip("txt-utf-8.zip")

## use txt-latin-1.zip if you are on Windows
## unzip("txt-latin-1.zip")

## get file names, w/ and w/o paths
files <- list.files(path = "txt/", pattern = "*.txt", full.names = TRUE)
names <- list.files(path = "txt/", pattern = "*.txt")

## read all
## content <- lapply(files, readr::read_file)

## read only sample of 100 documents, faster on cluster
## content <- lapply(files[sample(length(files), 10)], readr::read_file)

## alternative using base, may need tweaking with encoding based on locale and OS
## content <- lapply(files, function(f) readChar(f, nchars = file.info(f)$size))

## read only first 200 bytes, to preserve memory
content <- lapply(files, function(f) readChar(f, nchars = 5000))

# read as data
data <- as.data.frame(cbind(names, content))
names(data)
str(data)

## regex to get author names
data$names <- gsub("\\.txt$", "", data$names)
data$author <- gsub(" - .*$", "", data$names)
head(data)

## cleaner, no false positives (check first obs)
data$author <- str_extract(data$names, "^.*?( - )")
data$author <- gsub(" - ", "", data$author)
## head(data)

## same for year
data$year <- str_extract(data$names, " - (20|19)[0-9][0-9] - ")
data$year <- gsub(" - ", "", data$year)
head(data)

## same for title
data$title <- str_extract(data$names, " - .*$") ## not good, title may contain hyphen
data$title <- str_extract(data$names, " - (20|19)[0-9][0-9] - .*$")
data$title <- gsub("^ - (20|19)[0-9][0-9] - ", "", data$title)
head(data)

## trim whitespace everywhere
data$author <- trimws(data$author)
data$year <- trimws(data$year)
data$title <- trimws(data$title)
head(data)

#============================= Filter/clean content ============================

## remove supplementary material
data <- data[!grepl("^Supplemental", data$content), ]

## check initial metadata
head(data$content)
data$content[5]

## remove JSTOR metadata page
data$content <- gsub("^.* are collaborating with JSTOR to digitize.*?\\.", "", data$content)
head(data$content)

## More here
## ...


#========================= reading the single txt file =========================

jobs <- read_file("example-unix.txt")
jobs <- read_file("example.txt")

## TASK: Create a data frame with ids in one column and job ad text in another
## ...

ids <- str_extract_all(jobs, "32[0-9]{12}")
ids <- unlist(ids)
ids

posts <- str_split(jobs, "32[0-9]{12}")[[1]]
posts <- unlist(posts)[-1]
posts

posts <- trimws(posts)
posts

jobs.df <- data.frame(ids, posts)
str(jobs.df)


#==================== reading the single txt file, using sed ===================

## The above will read the complete text file into memory, which may be infeasible
## (or very slow). Other command line tools like `sed` can do text manipulations
## much faster. I included this for you to explore and for self study.

## I recommend doing this directly in a shell (e.g., `bash` or `zsh`), not in R.
## Escaping is tedious in R. Note that in the notebook the output of `system()`
## calls is not visible. You can check it in Rstudio, or better yet, directly from
## a shell instead of calling `system()` in R.

## advanced, for self study. this works on very large files.
## crude way of creating a readable file quickly using shell programs.
## uses sed to insert a ';' separator and line break based on a regex pattern,
## such that ID and Text field can be read as a csv. you could also do this from
## the command line.  note: this requires sed to be installed on your system.
## also, R requires double backslash escaping, and escaping nested quotations --
## overall easier to do this directly in a shell.

## These three lines are all you need to execute (see also convert-pdfs.sh).
## sed -e 's/^32[0-9]\{12\}$/"\n\0;"/' example-unix.txt > example.csv
## sed -i '1d' example.csv
## sed -i '$a"' example.csv

## check structure
system("head example-unix.txt -n 100")

## match id, then replace with separator (;), linebreak (\n), matched id (\0), separator (;)
system("sed -e 's/^32[0-9]\\{12\\}$/\"\\n\\0;\"/' example-unix.txt")

## same, but redirect stream output to file
system("sed -e 's/^32[0-9]\\{12\\}$/\"\\n\\0;\"/' example-unix.txt > example.csv")

# check and fix first/last row (could also do this in an editor)
system("head example.csv")
system("tail example.csv")

## -i operates on the file directly, 1d deletes the first line
system("sed -i '1d' example.csv")
# last row, $ selects last row, a appends the following characters
system("sed -i '$a\"' example.csv")

# check
system("head example.csv")
system("tail example.csv")


## read csv
example <- read.table("example.csv", sep = ";")

options(scipen = 9999)
head(example)
dim(example)
stopifnot(ncol(example) == 2)

names(example) <- c("id", "ad")
example$ad <- trimws(example$ad)
head(example)
head(example$ad)

# fix character encoding
Encoding(example$ad) <- "UTF-8"
head(example$ad)
