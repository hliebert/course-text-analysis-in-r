################################################################################
## Filename: regex.r
## Description: 
## Author: Helge Liebert
## Created: Fr Jan 21 14:29:15 2022
## Last-Updated: Mo Jan 24 13:01:04 2022
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
input <- "something"
gsub("something", "something else", input)

## backreferences are possible
input <- "something else"
gsub("(something).*(else)", "\\2", input, perl = TRUE)

## also other transformations
gsub("(something).*(else)", "\\U\\2", input, perl = TRUE)

## fixing the capital of the Netherlands
gsub("The Hague", "Amsterdam", capitals)

## the stringr library has more dedicated string functions, though many are
## duplicates or can easily be derived from base functions.
str_detect(countries, "tia")
str_count(countries, "land")

## these are the same
countries[grepl("land", countries)]
countries[str_detect(countries, "land")]


## TASK: Find all capitals in countries beginning and ending with a vocal and not ending with "land".
## ...

## TASK: Find all countries that contain exactly two words in the title and swap them.
## ...


#================================ More exercises ===============================

## Project Gutenberg offers lots of free classic literature
gutenberg_works(languages = "en")


#=============================== Robinson Crusoe ===============================

## Robinson Crusoe
gutenberg_works(author == "Defoe, Daniel", title == "The Life and Adventures of Robinson Crusoe")
robinson <- gutenberg_download(521, mirror = "http://mirrors.xmission.com/gutenberg/")
robinson <- robinson$text
robinson

## TASK: How many lines in the book mention Friday?
## ...

## TASK: How many lines in the book mention Friday or goats?
## ...

## TASK: On which line does he first mention finding another man's footprint on the beach?
##       What does the paragraph say?
## ...


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
