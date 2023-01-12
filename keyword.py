import sys
from pathlib import Path
import math
import os
import re
import string
from operator import itemgetter
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import pandas as pd
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Read files
corpus = []
path = sys.argv[1]
listFile = os.listdir(path)
for file in listFile:
    txt = Path(f'{path}/{file}').read_text()
    corpus.append(txt)

# Preprocess
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower() # converting all letters to lower case
    corpus[i] = re.sub(r"\d+", "", corpus[i]) # removing numbers
    corpus[i] = "".join([i for i in corpus[i] if i not in string.punctuation]) # removing punctuations
    corpus[i] = corpus[i].split() # tokenization
    corpus[i] = [i for i in corpus[i] if i not in stopwords] # removing stop words
    corpus[i] = [wordnet_lemmatizer.lemmatize(word) for word in corpus[i]] # lemmatization

# TF*IDF
def calculate_idf(corpus):
    words = set()
    for doc in corpus:
        words.update(doc)

    idfDict = dict.fromkeys(words,0)
    for i in range(len(corpus)):
        for word in idfDict:
            if word in corpus[i]:
                idfDict[word] += 1
    for word in idfDict:
        idfDict[word] = math.log(len(corpus) / (1 + idfDict[word]))

    return idfDict

def calculate_tf(corpus):
    corpus_tf_score = []
    for i in range(len(corpus)):
        tfDict = dict.fromkeys(corpus[i], 0)
        for word in tfDict:
            tfDict[word] = math.log(1 + corpus[i].count(word))
        corpus_tf_score.append(tfDict)
    return corpus_tf_score

def process(corpus, top_key):
    corpus_tf_score = calculate_tf(corpus)
    corpus_idf_score = calculate_idf(corpus)
    corpus_top_word = []
    for i in range(len(corpus)):
        tfidf_score = {}
        for word, val in corpus_tf_score[i].items():
            tfidf_score[word] = val * corpus_idf_score[word]
        top_word = dict(sorted(tfidf_score.items(), key = itemgetter(1), reverse = True)[:top_key])
        corpus_top_word.append(list(top_word.keys()))
    
    tmp = {'text file': listFile, 'list of keywords': corpus_top_word}
    df = pd.DataFrame(tmp) 
    df.to_csv(sys.argv[2],index=False)
    
process(corpus,5)