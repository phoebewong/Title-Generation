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
