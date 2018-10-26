import pandas as pd
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

# read in text files
novels = ['emma','mansfieldpark','northanger','persuasion','pridenp','sensensense']
inbooks = []
for novel in novels:
    f = open('./data/' + novel + '.txt', encoding = 'utf-8')
    inbooks.append(f.read())

emma = inbooks[0]
mansfield = inbooks[1]
northanger = inbooks[2]
persuasion = inbooks[3]
pridenp = inbooks[4]
sensensense = inbooks[5]


# EMMA PREPROCESSING - WORDS
# remove 30 lines, punctuation, chapter headers, lemmatize

emma.find('VOLUME')
emma.find('FINIS')
emma = emma[611:883631]

pattern = r"[A-Za-z]+"
emtok = nltk.regexp_tokenize(emma, pattern)
emwords = [w.lower() for w in emtok]

nostops = [w for w in emwords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
emmatized = [wnlemma.lemmatize(w) for w in nostops]
emmatized = [w for w in emmatized if w not in ['chapter']]


# MANSFIELD PARK PREPROCESSING - WORDS

mansfield.find('CHAPTER')
mansfield.find('THE END')
mansfield = mansfield[659:883910]

pattern = r"[A-Za-z]+"
manstok = nltk.regexp_tokenize(mansfield, pattern)
manswords = [w.lower() for w in manstok]

nostops = [w for w in manswords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
manslemmatized = [wnlemma.lemmatize(w) for w in nostops]
manslemmatized = [w for w in manslemmatized if w not in ['chapter']]


# NORTHANGER ABBEY PREPROCESSING - WORDS

northanger.find('CHAPTER')
northanger.rfind('Rambler')
northanger = northanger[1476:433575]

pattern = r"[A-Za-z]+"
northtok = nltk.regexp_tokenize(northanger, pattern)
northwords = [w.lower() for w in northtok]

nostops = [w for w in northwords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
northlemmatized = [wnlemma.lemmatize(w) for w in nostops]
northlemmatized = [w for w in northlemmatized if w not in ['chapter']]


# PERSUASION PREPROCESSING - WORDS

persuasion.find('Chapter 1')
persuasion.rfind('Finis')
persuasion = persuasion[629:467438]

pattern = r"[A-Za-z]+"
perstok = nltk.regexp_tokenize(persuasion, pattern)
perswords = [w.lower() for w in perstok]

nostops = [w for w in perswords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
perslemmatized = [wnlemma.lemmatize(w) for w in nostops]
perslemmatized = [w for w in perslemmatized if w not in ['chapter']]

# PRIDE AND PREJUDICE PREPROCESSING - WORDS

pridenp.find('Chapter 1')
pridenp.rfind('uniting them')
pridenp = pridenp[665:685406]

pattern = r"[A-Za-z]+"
pridetok = nltk.regexp_tokenize(pridenp, pattern)
pridewords = [w.lower() for w in pridetok]

nostops = [w for w in pridewords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
pridelemmatized = [wnlemma.lemmatize(w) for w in nostops]
pridelemmatized = [w for w in pridelemmatized if w not in ['chapter']]


# SENSE AND SENSIBILITY PREPROCESSING - WORDS

sensensense.find('CHAPTER')
sensensense.rfind('THE END')
sensensense = sensensense[698:674328]

pattern = r"[A-Za-z]+"
sensetok = nltk.regexp_tokenize(sensensense, pattern)
sensewords = [w.lower() for w in sensetok]

nostops = [w for w in sensewords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
senselemmatized = [wnlemma.lemmatize(w) for w in nostops]
senselemmatized = [w for w in senselemmatized if w not in ['chapter']]


# create corpus



