from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Word2Vec
np.random.seed(748)
from sklearn.manifold import t_sne

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
                                        passes = 10, workers = 2)

for idx, topic in austen_lda_bow.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# LDA - tfidf

austen_lda_tfidf = LdaMulticore(corpus_tfidf, num_topics = 6, id2word = dictionary,
                                passes = 10, workers = 2)

for idx, topic in austen_lda_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic)) #no useful weights


# word2vec to T-SNE
# create sentence corpus via flatten

novelsent = [emsplit_sentences, mansplit_sentences, northsplit_sentences, persplit_sentences, pridesplit_sentences,
             sensplit_sentences]

sentences = [sentence for novel in novelsent for sentence in novel]

austen_w2v = Word2Vec(sentences, size=100, window=5, min_count=10, workers=4, sg=0)

austen_w2v.wv.most_similar("poor")
austen_w2v.similar_by_word("poor")

austen_w2v.save("austen_w2v.model")


# bigram and trigram models
