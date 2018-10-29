from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.tfidfmodel import TfidfModel
np.random.seed(748)

novels = [emmatized, manslemmatized, northlemmatized, perslemmatized, pridelemmatized, senselemmatized]

# create corpus

dictionary = Dictionary(novels)
corpus = [dictionary.doc2bow(novel) for novel in novels]

# create default dict, sort
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

sorted_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

#top 20 words in corpus

for word_id, word_count in sorted_count[:20]:
    print(dictionary.get(word_id), word_count)

#using tf-idf

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

doc = corpus[5]
pnpweights = tfidf[doc]

sorted_tfidf = sorted(pnpweights, key=lambda w: w[1], reverse=True)

for term_id, weight in sorted_tfidf[:20]:
    print(dictionary.get(term_id), weight)

# LDA analysis - BOW

austen_lda_bow = LdaMulticore(corpus, num_topics = 6, id2word = dictionary,
                                        passes = 2, workers = 2)

for idx, topic in austen_lda_bow.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# LDA - tfidf

austen_lda_tfidf = LdaMulticore(corpus_tfidf, num_topics = 6, id2word = dictionary,
                                passes = 2, workers = 2)

for idx, topic in austen_lda_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))