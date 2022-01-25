################################################################################
## Filename: read.r
## Description: 
## Author: Helge Liebert
## Created: Mi. Aug 26 14:44:17 2020
## Last-Updated: Mo Jan 24 16:29:49 2022
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

