################################################################################
## Filename: regex.r
## Description: 
## Author: Helge Liebert
## Created: Fr Jan 21 14:29:15 2022
## Last-Updated: Mi Feb 16 13:03:56 2022
################################################################################

library("gutenbergr")
library("stringr")
library("data.table")

capitals <- c(
  "The Hague",
  "Andorra la Vella",
  "Athens",
  "Belgrade",
  "Berlin",
  "Bern",
  "Bratislava",
  "Brussels",
  "Bucharest",
  "Budapest",
  "Chisinau",
  "Copenhagen",
  "Dublin",
  "Helsinki",
  "Kiev",
  "Lisbon",
  "Ljubljana",
  "London",
  "Luxembourg",
  "Madrid",
  "Minsk",
  "Monaco",
  "Moscow",
  "Nicosia",
  "Oslo",
  "Paris",
  "Podgorica",
  "Prague",
  "Reykjavik",
  "Riga",
  "Rome",
  "San Marino",
  "Sarajevo",
  "Skopje",
  "Sofia",
  "Stockholm",
  "Tallinn",
  "Tirana",
  "Vaduz",
  "Valletta",
  "Vatican City",
  "Vienna",
  "Vilnius",
  "Warsaw",
  "Zagreb"
)

countries <- c(
  "Netherlands",
  "Andorra",
  "Greece",
  "Serbia",
  "Germany",
  "Switzerland",
  "Slovakia",
  "Belgium",
  "Romania",
  "Hungary",
  "Moldova",
  "Denmark",
  "Ireland",
  "Finland",
  "Ukraine",
  "Portugal",
  "Slovenia",
  "United Kingdom",
  "Luxembourg",
  "Spain",
  "Belarus",
  "Monaco",
  "Russia",
  "Cyprus",
  "Norway",
  "France",
  "Montenegro",
  "Czech Republic",
  "Iceland",
  "Latvia",
  "Italy",
  "San Marino",
  "Bosnia & Herzegovina",
  "North Macedonia",
  "Bulgaria",
  "Sweden",
  "Estonia",
  "Albania",
  "Liechtenstein",
  "Malta",
  "Holy See",
  "Austria",
  "Lithuania",
  "Poland",
  "Croatia"
)


## testing and indexing
grep("Rome", capitals)

## using boolean indexing
grepl("Rome", capitals)

## variations
grep("^R", capitals)
grep("^R", capitals, value = TRUE)
grep("^R", capitals, value = TRUE, invert = TRUE)
grep("^R.*a$", capitals, value = TRUE)

grepl("^R", capitals)
!grepl("^R", capitals)

countries[!grepl("^R", capitals)]

## data.table %like% operator is just like grepl()
capitals %like% "^R"
!(capitals %like% "^R")

## other binary operators are also helpful
capitals == "Riga"
capitals %in% "Riga"
capitals == "Riga" | capitals == "Madrid"
capitals %in% c("Riga", "Madrid")

## substitution
gsub("something", "something else", "something here")

## backreferences are possible
gsub("(something).*(else)", "\\2", "something else", perl = TRUE)

## also other transformations
gsub("(something).*(else)", "\\U\\2", "something else", perl = TRUE)

## fixing the capital of the Netherlands
gsub("The Hague", "Amsterdam", capitals)

## the stringr library has more dedicated string functions, though many are
## duplicates or can easily be derived from base functions.
str_detect(countries, "tia")
str_count(countries, "land")

## these are the same
countries[grepl("land", countries)]
countries[str_detect(countries, "land")]


## TASK: Find all capitals in countries beginning with a vowel and not ending with "land".
capitals[grepl("^[AEIOU].*", countries) & !grepl("land$", countries)]

## TASK: Find all countries that contain exactly two words in the title and swap them.
grep("([A-Z][a-z]+) ([A-Z][a-z]+)", countries, value = TRUE)
gsub("([A-Z][a-z]+) ([A-Z][a-z]+)", "\\2 \\1", countries)

## accounting for Bosnia and Herzegovina
grep("([A-Z][a-z]+)( (& )*)([A-Z][a-z]+)", countries, value = TRUE)
gsub("([A-Z][a-z]+)( | & )([A-Z][a-z]+)", "\\3\\2\\1", countries)


#================================ More exercises ===============================

## Project Gutenberg offers lots of free classic literature
gutenberg_works(languages = "en")

## gutenberg_works(author == "London, Jack", title == "Call of the Wild")
## gutenberg_works(author == "Twain, Mark", title == "Adventures of Huckleberry Finn")
## gutenberg_works(author == "Goethe, Johann Wolfgang von", title == "Faust")
## gutenberg_works(author == "Dostoyevsky, Fyodor", title == "Crime and Punishment")
## gutenberg_works(author == "Conrad, Joseph", title == "Heart of Darkness")
## gutenberg_works(author == "Defoe, Daniel", title == "The Life and Adventures of Robinson Crusoe")


#=============================== Robinson Crusoe ===============================

## Robinson Crusoe
gutenberg_works(author == "Defoe, Daniel", title == "The Life and Adventures of Robinson Crusoe")
robinson <- gutenberg_download(521, mirror = "http://mirrors.xmission.com/gutenberg/")
robinson <- robinson$text
robinson

## TASK: How many lines in the book mention Friday?
sum(grepl("Friday", robinson))

## TASK: How many lines in the book mention Friday or goats?
sum(grepl("Friday|goat", robinson))

## TASK: On which line does he first mention finding another man's footprint on the beach?
##       What does the paragraph say?
grep("foot", robinson)
grep("print", robinson)

robinson[grep("foot", robinson)]
robinson[grep("print", robinson)]

# lines which mention foot and print
grepl("foot", robinson) & grepl("print", robinson)
robinson[grepl("foot", robinson) & grepl("print", robinson)]

## Alternative ways of doing the same

## capturing groups and logical OR
grep("(print.*foot)|(foot.*print)", robinson)

## Positive lookahead assertion, non-consuming
grep("(?=.*print)(?=.*foot)", robinson, perl = TRUE)

# these are the lines we are looking for
robinson[grep("(?=.*print)(?=.*foot)", robinson, perl = TRUE)]

# alternative
robinson[grep("man.s.*foot", robinson)]
robinson[grep("man\\Ss.*foot", robinson)]

line <- grep("print of a man.s naked foot", robinson)
line

robinson[(line - 1):(line + 19)]


#==================================== Faust ====================================

## Faust, english
gutenberg_works(author == "Goethe, Johann Wolfgang von", title == "Faust")
faust <- gutenberg_download(3023, mirror = "http://mirrors.xmission.com/gutenberg/")
faust <- faust$text

## Faust, german
gutenberg_works(author == "Goethe, Johann Wolfgang von", title == "Faust: Der Tragödie erster Teil", languages = "de")
faust.de <- gutenberg_download(2229, mirror = "http://mirrors.xmission.com/gutenberg/")
faust.de <- faust.de$text

## TASK: Find the paragraph with the famous citation where Mephistopheles
## introduces himself to Faust in his study.

## Hint: Places and actors are in full caps.
## Hint: Faust asks devil who he is.
## Hint: Devil speaks in riddles.
## Hint: "part of power", "spirit which denies"
## Hint: "gutes will und böses schafft", "geist der verneint"


faust[grep("will.*evil", faust, ignore.case = TRUE)]
faust[grep("work.*good", faust, ignore.case = TRUE)]
faust[1000:1200]

## none of these are informative
faust[grep("will", faust, ignore.case = TRUE)]
faust[grep("good", faust, ignore.case = TRUE)]
faust[grep("eternal", faust, ignore.case = TRUE)]
faust[grep("power", faust, ignore.case = TRUE)]

## if you remember parts of the citation correctly, this is it
faust[grep("who.*you", faust, ignore.case = TRUE)]
faust[grep("who.*thou", faust, ignore.case = TRUE)]
faust[grep("spirit.*denies", faust, ignore.case = TRUE)]

## Alternatively, identify the chapter in the study and the speakers (Faust and Mephistopheles)
grep("MEPHISTOPHELES", faust)
grep("STUDY", faust)
grep("study", faust, ignore.case = TRUE)

begin.chapter <- grep("STUDY", faust)[1]
devil.speaks <- grep("MEPHISTOPHELES", faust)
faust.speaks <- grep("FAUST", faust)
other.speaks <- grep("[A-Z][A-Z]+", faust)

devil.speaks.instudy <- devil.speaks[devil.speaks > begin.chapter]
faust.speaks.instudy <- faust.speaks[faust.speaks > begin.chapter]

devil.speaks.instudy <- devil.speaks.instudy[devil.speaks.instudy > min(faust.speaks.instudy)]
faust.speaks.instudy <- faust.speaks.instudy[faust.speaks.instudy > min(devil.speaks.instudy)]

devil.speaks.instudy <- devil.speaks.instudy[1:10]
faust.speaks.instudy <- faust.speaks.instudy[1:10]

devil.talks <- unlist(Map(`:`, devil.speaks.instudy, faust.speaks.instudy - 1))

faust[devil.talks]
faust[1996:2021]

## German, easier if you remember the citation
grep("geist.*verneint", faust.de, value = TRUE, ignore.case = TRUE)
line <- grep("geist.*verneint", faust.de, ignore.case = TRUE)
faust.de[(line - 10):(line + 7)]
