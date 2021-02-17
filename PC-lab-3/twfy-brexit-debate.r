################################################################################
## Filename: twfy-brexit-debate.r
## Description: 
## Author: Helge Liebert
## Created: So Mär  1 15:41:38 2020
## Last-Updated: Mi. Feb 17 16:45:48 2021
################################################################################

#================================== Libraries ==================================

library("twfy")
library("jsonlite")
library("topicmodels")
library("textclean")
library("wordcloud")
library("slam")
library("tm")
library("data.table")
library("tidytext")
library("stringr")
library("dplyr")
library("ggplot2")
library("ggrepel")
library("uwot")
library("udpipe")
library("lsa")
library("factoextra")
library("word2vec")
library("plotly")
library("fpc")
library("doc2vec")
library("glmnet")
library("caret")


#======================= Get data from TheyWorkForYou API ======================

apikey <- "G3WVqtBtKAbdGVqrd8BKajm8"
set_api_key(apikey)

call <- getDebates(type = "commons", search = "Brexit", num = 1000, page = 1)
info <- call$info
info
pages <- ceiling(info$total_results/info$results_per_page)

## Form of data.frame, header only
brexit.debates <- flatten(as.data.frame(call$rows))[0, ]
str(brexit.debates)

## read all pages of results
for (p in seq(1, pages)) {
  call <- getDebates(type = "commons",  search = "Brexit", num = 1000, page = p)
  call$rows$speaker$office <- NULL
  brexit.debates <- rbind(brexit.debates, flatten(as.data.frame(call$rows)))
}

## save to file
## fwrite(brexit.debates, "Data/brexit-debates.csv")
## saveRDS(brexit.debates, "Data/brexit-debates.rds")

## read from file
## brexit.debates <- readRDS("Data/brexit-debates.rds")

str(brexit.debates)
length(unique(brexit.debates$gid))
length(unique(brexit.debates$hdate))
length(unique(brexit.debates$person_id))
names(brexit.debates)

## copy before transformations
brexit.debates$body.orig <- brexit.debates$body


#=================================== Cleaning ==================================

## brief check
check_text(brexit.debates$body[1:100])
## Encoding(brexit.debates$body) <- "UTF-8"

## some cleaning and harmonizing
## pre-written functions convenient compared to writing all regex on your own
head(brexit.debates$body)
brexit.debates$body <- replace_html(brexit.debates$body)
head(brexit.debates$body)
brexit.debates$body <- replace_non_ascii(brexit.debates$body)
head(brexit.debates$body)
brexit.debates$body <- gsub("&#8212;", " - ", brexit.debates$body)
head(brexit.debates$body, 20)
brexit.debates$body <- gsub("&#[0-9]{3,4};", " ", brexit.debates$body)
## brexit.debates$body <- stringi::stri_trans_general(brexit.debates$body, "latin-ascii")
head(brexit.debates$body, 20)
brexit.debates$body <- replace_names(brexit.debates$body)
## brexit.debates$body <- replace_names(brexit.debates$body, replacement = "NAMEHERE")
head(brexit.debates$body)
brexit.debates$body <- replace_money(brexit.debates$body, replacement = "MONEYHERE")
head(brexit.debates$body)
brexit.debates$body <- replace_date(brexit.debates$body, replacement = "DATEHERE")
head(brexit.debates$body)
brexit.debates$body <- replace_ordinal(brexit.debates$body, num.paste = TRUE)
head(brexit.debates$body)
## brexit.debates$body <- replace_number(brexit.debates$body) ## makes a difference for topics!
brexit.debates$body <- replace_number(brexit.debates$body, remove = TRUE) ## makes a difference for topics!
head(brexit.debates$body)
brexit.debates$body <- add_comma_space(brexit.debates$body)
head(brexit.debates$body)
brexit.debates$body <- replace_contraction(brexit.debates$body)
head(brexit.debates$body)
brexit.debates$body <- replace_white(brexit.debates$body)
head(brexit.debates$body)

## remove blanks
## brexit.debates <- brexit.debates[brexit.debates$body != "", ]


#================================= Topics, both parties =================================

## corpus
## corp <- brexit.debates
corp <- brexit.debates[, c("gid", "body")]
setnames(corp, "gid", "doc_id")
setnames(corp, "body", "text")

## initialize dtm
dtm <- DocumentTermMatrix(
  Corpus(DataframeSource(
    corp
  )),
  control = list(
    language = "english",
    ## weighting = "weightTfIdf",
    weighting = weightTf,
    tolower = TRUE,
    removePunctuation = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    stemming = FALSE,
    wordLengths = c(3, Inf)
  )
)
dtm

## checks
inspect(dtm)
## findFreqTerms(dtm, lowfreq = 10)
findFreqTerms(dtm, lowfreq = 1000)
## dtm <- removeSparseTerms(dtm, sparse=0.90) ## filter some
dtm <- dtm[row_sums(dtm) > 0, ] ## documents can't be empty

## Simple visualization
wordcloud(brexit.debates$body, max.words = 100, random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))

## plot if tf-idf weighting
counts <- sort(colSums(as.matrix(dtm)), decreasing = TRUE)
counts <- data.frame(word = names(counts), freq = counts)
wordcloud(words = counts$word, freq = counts$freq,
          max.words = 100, random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))


## Unsupervised clustering of documents: Topic model
tpm <- LDA(dtm, k = 3, control = list(seed = 100))
topic <- topics(tpm, 1)
freqterms <- terms(tpm, 50)
freqterms

## Plot most frequent terms and associated probabilities by topic
tpmat <- tidy(tpm, matrix = "beta")
topterms <-
    tpmat %>%
    group_by(topic) %>%
    top_n(20, beta) %>%
    ungroup() %>%
    arrange(topic, -beta)
topterms %>%
    mutate(term = reorder(term, beta)) %>%
    ggplot(aes(term, beta, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    coord_flip()

## look at unique terms only per topic
duplicates <- c(freqterms)[duplicated(c(freqterms))]
distinctterms <- lapply(as.list(as.data.frame(freqterms)), function(x) x[!(x %in% duplicates)])
## distinctterms <- as.data.frame(distinctterms)
distinctterms

## differences by party?
table(brexit.debates$speaker.party)


#========================== Conservative Party Topics ==========================

dtm.con <- DocumentTermMatrix(
  Corpus(DataframeSource(
    corp[brexit.debates$speaker.party == "Conservative" ,]
  )),
  control = list(
    language = "english",
    weighting = weightTf,
    tolower = TRUE,
    removePunctuation = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    stemming = FALSE,
    wordLengths = c(3, Inf)
  )
)
dtm.con <- dtm.con[row_sums(dtm.con) > 0, ] ## documents can't be empty

tpm.con <- LDA(dtm.con, k = 3, control = list(seed = 100))
topic.con <- topics(tpm.con, 1)
freqterms.con <- terms(tpm.con, 50)
freqterms.con

## Plot most frequent terms and associated probabilities by topic
tpmat.con <- tidy(tpm.con, matrix = "beta")
topterms.con <-
    tpmat.con %>%
    group_by(topic) %>%
    top_n(20, beta) %>%
    ungroup() %>%
    arrange(topic, -beta)
topterms.con %>%
    mutate(term = reorder(term, beta)) %>%
    ggplot(aes(term, beta, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    coord_flip()

## look at unique terms only per topic
duplicates.con <- c(freqterms.con)[duplicated(c(freqterms.con))]
distinctterms.con <- lapply(as.list(as.data.frame(freqterms.con)), function(x) x[!(x %in% duplicates.con)])
distinctterms.con


#================================ Labour Party Topics ================================

dtm.lab <- DocumentTermMatrix(
  Corpus(DataframeSource(
    corp[brexit.debates$speaker.party == "Labour", ]
  )),
  control = list(
    language = "english",
    weighting = weightTf,
    tolower = TRUE,
    removePunctuation = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    stemming = FALSE,
    wordLengths = c(3, Inf)
  )
)
dtm.lab <- dtm.lab[row_sums(dtm.lab) > 0, ] ## documents can't be empty


tpm.lab <- LDA(dtm.lab, k = 3, control = list(seed = 100))
topic.lab <- topics(tpm.lab, 1)
freqterms.lab <- terms(tpm.lab, 50)
freqterms.lab

## Plot most frequent terms and associated probabilities by topic
tpmat.lab <- tidy(tpm.lab, matrix = "beta")
topterms.lab <-
    tpmat.lab %>%
    group_by(topic) %>%
    top_n(20, beta) %>%
    ungroup() %>%
    arrange(topic, -beta)
topterms.lab %>%
    mutate(term = reorder(term, beta)) %>%
    ggplot(aes(term, beta, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    coord_flip()

## look at unique terms only per topic
duplicates.lab <- c(freqterms.lab)[duplicated(c(freqterms.lab))]
distinctterms.lab <- lapply(as.list(as.data.frame(freqterms.lab)), function(x) x[!(x %in% duplicates.lab)])
distinctterms.lab



#=================================== PCA/LSA ===================================

## since pca works less well for word similarity tasks and interpretations (when
## applied to the term-term matrix), lets apply it to the document-term-matrix.

## we aggregate the document-term matrix by speaker, then use pca/lsa to get a
## reduced dimension that helps assessing document similarity. then we use the
## smaller representation to apply k-means clustering to group politicians.

## check speakers
names(brexit.debates)
length(unique((brexit.debates$speaker.name)))
length(unique((brexit.debates$person_id)))
nrow(unique((brexit.debates[, c("person_id", "speaker.party")])))

## aggregate, in base (more convenient with data.table; or dplyr if you must)
## brexit.speakers <- aggregate(body ~ speaker.name + person_id + speaker.party, data = brexit.debates, paste)
brexit.speakers <- aggregate(body ~ speaker.name + person_id, data = brexit.debates, paste)

## dismiss those who have empty text
brexit.speakers <- brexit.speakers[brexit.speakers$body != "", ]
brexit.speakers <- brexit.speakers[!is.na(brexit.speakers$body), ]

## checks
dim(brexit.speakers)
names(brexit.speakers)
## str(brexit.speakers)
## head(brexit.speakers)

## how many words?
brexit.speakers$wordcount <- str_count(brexit.speakers$body)
qplot(brexit.speakers$wordcount)
qplot(brexit.speakers$wordcount[brexit.speakers$wordcount < 4000], bins = 100)

## corpus base
corp <- brexit.speakers[, c("speaker.name", "body")]
setnames(corp, "speaker.name", "doc_id")
setnames(corp, "body", "text")

## initialize dtm
dtm <- DocumentTermMatrix(
  Corpus(DataframeSource(
    corp
  )),
  control = list(
    language = "english",
    weighting = weightTf,
    tolower = TRUE,
    removePunctuation = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    stemming = FALSE,
    wordLengths = c(3, Inf)
  )
)
dtm


## LSA on the document term matrix
## ls <- lsa(dtm)
ls <- lsa(dtm, 2)
str(ls)
pcs <- as.data.frame(ls$tk)
M <- as.textmatrix(ls)

## kmeans clustering. Try three clusters (three main parties)
km <- kmeans(pcs, centers = 3)
## km
## str(km)
fviz_cluster(km, data = pcs)

## checks
summary(brexit.speakers$wordcount)
brexit.speakers[brexit.speakers$speaker.name == "Jeremy Corbyn", "wordcount"]
brexit.speakers[brexit.speakers$speaker.name == "Valerie Vaz", "wordcount"]
## brexit.speakers[brexit.speakers$speaker.name == "Pete Wishart", "wordcount"]

## filter
outliers <- c("Jeremy Corbyn", "Valerie Vaz")
dtm <- DocumentTermMatrix(
  Corpus(DataframeSource(
    corp[!(corp$doc_id %in% outliers), ]
  )),
  control = list(
    language = "english",
    weighting = weightTf,
    tolower = TRUE,
    removePunctuation = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    stemming = FALSE,
    wordLengths = c(3, Inf)
  )
)
dtm

## repeat lsa/km
ls <- lsa(dtm, 2)
pcs <- as.data.frame(ls$tk)
M <- as.textmatrix(ls)
km <- kmeans(pcs, centers = 3)
fviz_cluster(km, data = pcs)
## plotcluster(dtm, km$cluster)


## filter even more
dtm <- DocumentTermMatrix(
  Corpus(DataframeSource(
    corp[brexit.speakers$wordcount < quantile(brexit.speakers$wordcount, p = 0.95), ]
  )),
  control = list(
    language = "english",
    weighting = weightTf,
    tolower = TRUE,
    removePunctuation = TRUE,
    removeNumbers = TRUE,
    stopwords = TRUE,
    stemming = FALSE,
    wordLengths = c(3, Inf)
  )
)

## repeat lsa/km
ls <- lsa(dtm, 2)
pcs <- as.data.frame(ls$tk)
M <- as.textmatrix(ls)
km <- kmeans(pcs, centers = 3)
fviz_cluster(km, data = pcs)


## How to solve the document length issue?
## sampling, adding document length as a column, ...

## Do the clusters identify party membership?
## ...



#=============================== Word embeddings ===============================

## this data is not ideal to train embeddings, it is too small.
## but it is fast and sufficient for illustration.

## use untransformed or only minimally transformed text as input
text <- brexit.debates$body.orig
text <- replace_html(text)
text <- replace_non_ascii(text)
text <- gsub("&#[0-9]{3,4};", " ", text)
## text <- replace_ordinal(text, num.paste = TRUE)
## text <- replace_number(text, remove = TRUE)
## text <- replace_contraction(text)
## text <- add_comma_space(text)
## text <- replace_white(text)
text <- tolower(text)

## train word2vec to learn embeddings
## vsmodel <- word2vec(x = text, type = "skip-gram", dim = 150, iter = 20)

## save model to file
## write.word2vec(vsmodel, "Data/w2v-brexit.bin")

## read again
vsmodel <- read.word2vec("Data/w2v-brexit.bin")

## all terms
terms <- summary(vsmodel, "vocabulary")

## extract embeddings
embeddings <- as.matrix(vsmodel)
str(embeddings)
dim(embeddings)
head(embeddings)


#=========================== Semantics and similarity ==========================

## some word associations
predict(vsmodel, c("johnson", "corbyn", "bercow", "may", "starmer", "cummings"),
        type = "nearest", top_n = 5)
predict(vsmodel, c("negotiations", "deadline", "vote", "fisheries"),
        type = "nearest", top_n = 5)

## analogy tasks (better with pre-trained embeddings in a different context)
wv <- predict(vsmodel, newdata = c("uk", "continent", "eu"), type = "embedding")
wv <- wv["uk", ] - wv["continent", ] + wv["eu", ]
predict(vsmodel, newdata = wv, type = "nearest", top_n = 5)

## associations: uk without europe
wv <- embeddings["uk", ] - embeddings["europe", ]
predict(vsmodel, newdata = wv, type = "nearest", top_n = 5)

##  associations: brexit with agreement
wv <- embeddings["brexit", ] + embeddings["agreement", ]
predict(vsmodel, newdata = wv, type = "nearest", top_n = 10)

##  associations: brexit without agreement
wv <- embeddings["brexit", ] - embeddings["agreement", ]
predict(vsmodel, newdata = wv, type = "nearest", top_n = 10)



#==================== Project all adjectives in 2 dimensions ===================

## old, pos
## udmodel <- udpipe_download_model(language = "english")
## udmodel <- udpipe_load_model(file = udmodel$file_model)
## corp.pos <- udpipe_annotate(udmodel, text)

## pos-tag the text to identify adjectives, takes quite a while
## corp <- brexit.debates[, c("gid", "body.orig")]
## setnames(corp, "gid", "doc_id")
## setnames(corp, "body.orig", "text")
## corp$text <- text
## corp.pos <- udpipe(corp, "english")
## saveRDS(corp.pos, "Data/brexit-annotated.rds")
corp.pos <- readRDS("Data/brexit-annotated.rds")
head(corp.pos)

## get all adjectives in the corpus
length(unique(corp.pos[, "token"]))
length(unique(corp.pos[corp.pos$upos == "ADJ", "token"]))
adjectives <- unique(corp.pos[corp.pos$upos == "ADJ", "token"])

## get all nouns in the corpus
length(unique(corp.pos[, "token"]))
length(unique(corp.pos[corp.pos$upos == "NOUN", "token"]))
nouns <- unique(corp.pos[corp.pos$upos == "NOUN", "token"])

# visualize 2-dimensional projection of all adjectives in the brexit debate data
# project on 2dim space
viz <- umap(embeddings, n_neighbors = 15, n_threads = 2)
# filter for adjectives
df  <- data.frame(word = rownames(embeddings),
                  xpos = rownames(embeddings),
                  x = viz[, 1], y = viz[, 2],
                  stringsAsFactors = FALSE)
df  <- subset(df, xpos %in% adjectives)
head(df)

## Plot, restrict to first 300 for speed
ggplot(df[1:300, ], aes(x = x, y = y, label = word)) +
  geom_text_repel() + theme_void() +
  labs(title = "word2vec - adjectives in 2D using UMAP")

## Interactive plot
plot_ly(df[1:300, ], x = ~x, y = ~y, type = "scatter", mode = 'text', text = ~word)


#======================= Similar projection of all nouns =======================

embeddings.nouns <- predict(vsmodel, nouns, type = "embedding")
embeddings.nouns <- embeddings.nouns[complete.cases(embeddings.nouns), ]
viz <- umap(embeddings.nouns, n_neighbors = 15, n_threads = 2)
df  <- data.frame(word = rownames(embeddings.nouns),
                  xpos = rownames(embeddings.nouns),
                  x = viz[, 1], y = viz[, 2],
                  stringsAsFactors = FALSE)
plot_ly(df[1:500, ], x = ~x, y = ~y, type = "scatter", mode = 'text', text = ~word)



#========================= Pre-trained embeddings ========================

## Download word2vec, glove or fasttext embeddings
## https://github.com/maxoodf/word2vec
## https://fasttext.cc/docs/en/crawl-vectors.html
## https://nlp.stanford.edu/projects/glove/

## word2vec on English texts corpus, Skip-Gram, Negative Sampling, vector size 500, window 10
model <- read.word2vec(file = "Data/sg_ns_500_10.w2v", normalize = TRUE)
length(summary(model))

## Examples for word similarities, classical analogies and embedding similarities
predict(model, newdata = c("loan", "money"), type = "nearest", top_n = 5)

wv <- predict(model, newdata = c("king", "man", "woman"), type = "embedding")
wv <- wv["king", ] - wv["man", ] + wv["woman", ]
predict(model, newdata = wv, type = "nearest", top_n = 5)

wv <- predict(model, newdata = c("france", "paris", "london"), type = "embedding")
wv <- wv["france", ] - wv["paris", ] + wv["london", ]
predict(model, newdata = wv, type = "nearest", top_n = 5)

wv <- predict(model, newdata = c("physician", "man", "woman"), type = "embedding")
wv <- wv["physician", ] - wv["man", ] + wv["woman", ]
predict(model, newdata = wv, type = "nearest", top_n = 20)

wv <- predict(model, newdata = c("ideology", "person", "racist", "xenophobia"), type = "embedding")
wv <- wv["ideology", ] - wv["person", ] + wv["racist", ]
predict(model, newdata = wv, type = "nearest", top_n = 10)


#==================================== GloVe ====================================

library(text2vec)

## Create iterator over tokens
tokens <- space_tokenizer(text)
str(tokens)

## Create vocabulary. Terms will be unigrams (simple words).
it <- itoken(tokens)
vocab <- create_vocabulary(it)
vocab

## remove infrequent tokens
vocab <- prune_vocabulary(vocab, term_count_min = 5L)

## Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
## use window of 5 for context words to construct term-co-occurence matrix
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
str(tcm)

## inspect: standard is decay weighting with offset position
## (weight = 1 / distance_from_current_word)
head(tcm)

## fit glove
glove <- GlobalVectors$new(rank = 50, x_max = 10)
wvmain <- glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01, n_threads = 8)
dim(wvmain)
tail(wvmain)

## can also retrieve context vectors
wvcontext <- glove$components
tail(wvcontext)
dim(wvcontext)

## could use either of these (typically main),
## or aggregate them by averaging or summing them (suggested in glove paper)
## summing:
wordvectors <- wvmain + t(wvcontext)

## analogy tasks work the same
## (although not well here as the corpus is too small and specific, requires more data)
berlin <- wordvectors["paris", , drop = FALSE] - wordvectors["france", , drop = FALSE] + wordvectors["germany", , drop = FALSE]
cosinesim <- sim2(x = wordvectors, y = berlin, method = "cosine", norm = "l2")
head(sort(cosinesim[,1], decreasing = TRUE), 5)


#========================== Averaged document vectors ==========================

## simple way to get a document representation: just averaging word vectors within a document
## assuming dtm is a document-term-matrix

## isolating common terms
commonterms <- intersect(colnames(dtm), rownames(wordvectors))

## filtering dtm (and normalizing)
## could also re-weight dtm with tf-idf instead of l1 norm
## dtmaveraged <-  as.matrix(dtm)[, common_terms]
dtmaveraged <-  normalize(as.matrix(dtm)[, commonterms], "l1")

## get averaged document vectors ('sentence' vectors)
docvectors <- dtmaveraged %*% wordvectors[commonterms, ]
head(docvectors)

## vector dimensions for multiplication
dim(dtmaveraged)
dim(wordvectors[commonterms, ])
dim(docvectors)

## analogy tasks work just as before, could use this to find e.g. speakers similar to a person
## check which is most similar to first document
cosinesim <- sim2(x = docvectors, y = docvectors[1, , drop = FALSE], method = "cosine", norm = "l2")
head(sort(cosinesim[,1], decreasing = TRUE), 5)


#=================================== Doc2vec ===================================

# this does not return great results, corpus probably too small

## input
corp <- data.frame(doc_id = brexit.debates$gid,
                   text = text,
                   stringsAsFactors = FALSE)

## low dimension, just for illustrations
pv.model <- paragraph2vec(
  x = corp,
  type = "PV-DM",
  dim = 5,
  iter = 3,
  min_count = 5,
  lr = 0.05,
  threads = 1
)

## More realistic settings, careful, this will run for a bit.
## pv.model <- paragraph2vec(
##   x = corp,
##   type = "PV-DBOW",
##   dim = 100,
##   iter = 20,
##   min_count = 5,
##   lr = 0.05,
##   threads = 4
## )
## saveRDS(pv.model, "Data/pv-model.rds")
## pv.model <- readRDS("Data/pv-model.rds")

## Extract the embeddings
word.embeddings <- as.matrix(pv.model, which = "words")
head(word.embeddings)

doc.embeddings <- as.matrix(pv.model, which = "docs")
tail(doc.embeddings)

## Extract the vocabulary
doc.vocab <- summary(pv.model, which = "words")
head(doc.vocab)

## word.vocab <- summary(pv.model, which = "docs")
## head(word.vocab)

# retriev word embeddings (as previously)
predict(pv.model, "brexit", type = "embedding")

# retrieve most similar words to a word (as previously)
predict(pv.model,
  newdata = "brexit",
  type = "nearest",
  which = "word2doc"
)

# retrieve document embeddings
predict(pv.model,
  newdata = c("2021-02-11b.563.0", "2021-02-11b.504.0", "2021-02-11b.468.2"),
  type = "embedding",
  which = "docs"
)

# retrieve most similar documents to a document
predict(pv.model,
  newdata = "2021-02-11b.563.0",
  type = "nearest",
  which = "doc2doc"
)

## find document closest to a sentence
predict(pv.model,
  newdata = list(sent = c("brexit", "will", "not", "disrupt", "trade")),
  type = "nearest",
  which = "sent2doc"
)

## Get embeddings of sentences.
sentences <- list(
  sent1 = c("germany", "and", "france", "dominate", "the", "eu"),
  sent2 = c("brexit", "was", "well", "planned")
)
predict(pv.model, newdata = sentences, type = "embedding")



#===================== feeding document embeddings to a prediction model =====================

docembeddings <- as.data.frame(cbind(speaker.name = rownames(docvectors), as.data.frame(docvectors)))str(docembeddings)
str(docembeddings)
str(brexit.debates)

## aggregate alsoo using the party affiliation
brexit.speakers <- aggregate(body ~ speaker.name + person_id + speaker.party, data = brexit.debates, paste)

## merge embeddings onto speaker data
brexit.speakersplus <- merge(brexit.speakers[, c("speaker.name", "speaker.party")], docembeddings, by = "speaker.name")
brexit.speakersplus <- brexit.speakersplus[complete.cases(brexit.speakersplus), ]
str(brexit.speakersplus)

## inidicators for tory/labour party
brexit.speakersplus$tory <- as.numeric(brexit.speakersplus$speaker.party == "Conservative")
brexit.speakersplus$labour <- as.numeric(brexit.speakersplus$speaker.party == "Labour")

## Partition data in test and training sample
set.seed(100)
testids <- sample(floor(nrow(brexit.speakersplus)/5))

# Train and test data
xtrain <- as.matrix(brexit.speakersplus[-testids, !(names(brexit.speakersplus) %in% c("speaker.name", "speaker.party", "tory", "labour"))])
xtest  <- as.matrix(brexit.speakersplus[ testids, !(names(brexit.speakersplus) %in% c("speaker.name", "speaker.party", "tory", "labour"))])

ytrain <- as.factor(brexit.speakersplus[-testids,  "tory"])
ytest  <- as.factor(brexit.speakersplus[ testids,  "tory"])

dim(xtrain)
length(ytrain)

dim(xtest)
length(ytest)

## Supervised text regression: L1 penalized logistic regression
l1classifier <- cv.glmnet(xtrain, ytrain, alpha = 1, family = "binomial")
l1pred <- as.factor(predict(l1classifier, xtest, s = "lambda.min", type = "class"))
summary(l1pred)

## Performance statistics
round(1-mean(as.numeric(l1pred != ytest)), 2)
confusionMatrix(l1pred, ytest)


## random forest
rfclassifier <- train(
  y = ytrain,
  x = xtrain,
  method = "ranger",
  num.trees = 200,
  tuneGrid = expand.grid(
    mtry = seq(2, 2 * floor(sqrt(ncol(xtrain))), length.out = 10),
    splitrule = "gini",
    min.node.size = c(1,3)
  ),
  trControl = trainControl(
    method = "oob"
  )
)
rfpred <- predict(rfclassifier, xtest)

## Performance statistics
1 - mean(as.numeric(rfpred != ytest))
confusionMatrix(rfpred, ytest)
