from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools

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