################################################################################
## Filename: read.r
## Description: 
## Author: Helge Liebert
## Created: Mi. Aug 26 14:44:17 2020
## Last-Updated: Do Feb  4 16:30:16 2021
################################################################################


#========================== reading all the txt files ==========================

library(stringr)
library(readr)
## library(data.table)

# set encoding (may be necessary on Windows)
## options(encoding = "UTF-8")
## options(max.print = 10000)

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


#========================= reading the single txt file =========================

## crude way of creating a readable file quickly using shell programs
## using sed to insert a ';' separator and line break such that ID and Text
## field can be read as a csv. you could also do this from the command line.
## note: this requires sed to be installed on your system.
## also, R requires double backslash escaping, and escaping nested quotations
## -- easier to do this in a shell
## sed -e 's/^32[0-9]\{12\}$/"\n\0;"/' example-unix.txt > example.csv

## check structure
system("head example-unix.txt -n 100")

## match id, then replace with separator, linebreak, matched id, separator
system("sed -e 's/^32[0-9]\\{12\\}$/\"\\n\\0;\"/' example-unix.txt > example.csv")

# check and fix first/last row (could also do this in an editor)
system("head example.csv -n 110")
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
