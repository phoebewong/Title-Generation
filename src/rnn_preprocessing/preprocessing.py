# Combine rnn.py and utils_updated.py
#rnn.py
#package for rnn data processing
#and model fitting
#Apr 30, 2019

#import dependencies
import numpy as np
from collections import Counter

def qualify(word):
    '''helper function to select words for tokenization.'''
    #symbols
    symbols = """/?~`!@#$%^&*()_-+=|\{}[];<>"'.,:"""
    #abbreviations
    abb = """e.g.,i.e.,etal.,"""
    #disqualify empty space and words starting with symbol
    if len(word) < 1 or word[0] in symbols:
        return False
    elif len(word) > 2:
        #disqualify abbreviations
        if word in abb:
            return False
        #otherwise count all combinations with length > 2
        else:
            return True
    #if input length is one
    #count only if it is 'a'
    elif len(word) == 1:
        if word in ['a', 'i']:
            return True
    #with input length of 2
    #disqualify those with a symbol as the second character
    elif len(word) == 2:
        if word[1] not in symbols:
            return True
    #otherwise disqualify input
    else:
        return False

class processText:
    '''
    class object for data processing preperation for embedding training.

    Parameters:
    ===========
    1) textVec: list of array-like, vector of text in strings

    Methods:
    ===========
    1) mostCommon: return counter object for word frequencies,
        function called in `padSequence`
    2) getDictionary: update dictionaries of words and tokens,
        function called in `tokenize`
    3) getCorpus: return vector of corpus for embedding training.
        Disqualified words are taken out

    * note that only qualified words are counted and tokenized
        for all methods above.
    '''
    def __init__(self, textVec):

        #initiate class object
        self.textVec = list()
        for vec in textVec:
            #string to list
            vec = [x.strip().split(' ') for x in vec]
            self.textVec.append(vec)

        #prep  dictionaries for update
        self.word2idx = dict()
        self.idx2word = dict()
        self.maxLen = 0
        self.nUnique = 0

    def updateMaxLen(self):
        for vec in self.textVec:
            for txt in vec:
                #get length of sequence
                cntLen = len(txt)
                #update maximum sequence length
                if self.maxLen < cntLen:
                    self.maxLen = cntLen

    def getDictionary(self):

        if len(self.word2idx) != 0:
            print("Dictionary already updated.")

        else:
            #initiate dictionary updates
            #pad with 0
            #end of sequence as 1
            #ignored/disqualified words as 2
            #start tokenization at 3
            pad = 0
            eos = 1
            ign = 2
            start = 3

            self.word2idx['_'] = pad
            self.word2idx['*'] = eos
            self.word2idx['<ign>'] = ign

            for vec in self.textVec:
                for txt in vec:
                    for w in txt:
                        if qualify(w) == True:
                            if w not in self.word2idx.keys():
                                self.word2idx.update({w: start})
                                start += 1

            #update number of unique words in data set
            self.nUnique = start - 3
            #update idx to word dictionary
            self.idx2word = dict((idx,word) for word,idx in self.word2idx.items())

    def tokenize(self):
        #get dictionaries if function hasn't been called
        if len(self.word2idx) == 0:
            self.getDictionary()
        #cache list for tokenization
        tokenizedVec = list()
        for i in range(len(self.textVec)):
            vec = self.textVec[i]
            #cache list for the tokenized vector
            tempVec = list()
            for txt in vec:
                #cache list for sequence
                sVec = list()
                for w in txt:
                    #if word is in dictionary, tokenize
                    if w in self.word2idx:
                        sVec.append(self.word2idx[w])
                    #if word not in dictionary, tag as ignored
                    else:
                        sVec.append(self.word2idx['<ign>'])
                tempVec.append(sVec)
            tokenizedVec.append(tempVec)

        return tokenizedVec

    # def getCorpus(self):
    #     #get dictionaries if function hasn't been called
    #     if len(self.word2idx) == 0:
    #         self.getDictionary()
    #
    #     #cache list all tokenized vector
    #     corpusVec = list()
    #     #for each text vector, title/abstract
    #     for vec in self.textVec:
    #         #cache list for the tokenized vector
    #         tempVec = list()
    #         for txt in vec:
    #             #cache list for sequence
    #             sVec = list()
    #             for w in txt:
    #                 if w in self.word2idx:
    #                     sVec.append(w)
    #             #add end of sequence tag
    #             tempVec.append(sVec)
    #         corpusVec.append(tempVec)
    #     return corpusVec

## Utils_updated.py
#utils_updated.py
#helper functions for data preprocessing
#package update for abstract extraction
#Apr 30, 2019

#import dependencies
import re
import numpy as np

def formatText(x):
    "render space in text and text to lowercase"
    for i in range(len(x)):
        #check for data type
        if type(x[i]) == str:
            try:
                x[i] = x[i].replace('\n', ' ').lower()
            except:
                x[i] = x[i].lower()
    return x

def getAbstract(paper_text, methods = 1):
    "extract abstract from text in two steps"
    #step 1:
    #find 'abstract' in text
    #find the next word/phrase in all cap, wrapped in '\n'
    #extract everything in between as abstract
    if methods == 1:
        try:
            #find abstract
            a1 = re.search('abstract\n', paper_text, re.IGNORECASE)
            paper_text = paper_text[a1.end():]
            #find the next section in all cap
            a2 = re.search(r'\n+[A-Z\s]+\n', paper_text)
            return paper_text[: a2.start()]
        except:
            return np.nan
    #step 2:
    #find abstract in text
    #find next item wrapped between '\n\n' and '\n\n'
    #extract everything in between as abstract
    if methods == 2:
        try:
            a1 = re.search('abstract\n', paper_text, re.IGNORECASE)
            paper_text = paper_text[a1.end():]
            #find the next section in all cap
            a2 = re.search(r'\n\n+.+\n\n', paper_text)
            return paper_text[: a2.start()]
        except:
            return np.nan


def preprocessing(papers, formatCols = ['title', 'abstract','paper_text'], dropnan = False):
    "preliminary data preprocessing for model fitting"
    #avoid modifying original dataset
    papersNew = papers.copy()
    #replace missing values with nan
    papersNew.abstract = papersNew.abstract.apply(lambda x: np.nan if x == 'Abstract Missing' else x)
    #extract missing abstract in two steps
    #steps identified by ad-hoc examination of missing values
    for m in [1, 2]:
        #try searching for abstract in text if value is missing
        papersNew['abstract_new'] = papersNew.paper_text.apply(lambda x: getAbstract(x, methods = m))
        #replace nan in abstract with extracted abstract
        papersNew.loc[papersNew.abstract.isnull(), 'abstract'] = papersNew.abstract_new
        papersNew.drop(['abstract_new'], axis = 1, inplace = True)
    #format columns of interest
    papersNew[formatCols] = papersNew[formatCols].apply(lambda x: formatText(x), axis = 1)
    if dropnan:
        #drop na in abstract
        papersNew = papersNew.dropna(subset = ['abstract'])
        #append abstract and title length to data frame
        papersNew ['aLen'] = papersNew.abstract.apply(lambda x: len(x.split(' ')))
        papersNew ['tLen'] = papersNew.title.apply(lambda x: len(x.split(' ')))
    return papersNew
