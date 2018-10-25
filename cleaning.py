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
novels = ['emma.txt','mansfieldpark.txt','northanger.txt','persuasion.txt','pridenp.txt','sensensense.txt']
inbooks = []
for novel in novels:
    f = open('./data/' + novel, encoding = 'utf-8')
    inbooks.append(f.read())

emma = inbooks[0]
mansfield = inbooks[1]
northanger = inbooks[2]
persuasion = inbooks[3]
pridenp = inbooks[4]
sensensense = inbooks[5]

# EMMA PREPROCESSING
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


# MANSFIELD PARK PREPROCESSING

# NORTHANGER ABBEY PREPROCESSING

# PERSUASION PREPROCESSING

# PRIDE AND PREJUDICE PREPROCESSING

# SENSE AND SENSIBILITY PREPROCESSING



