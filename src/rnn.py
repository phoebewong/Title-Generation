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
        if word == 'a':
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
    class object for data processing preperation for RNN model.

    Parameters:
    ===========
    1) textVec: list of array-like, vector of text in strings

    Methods:
    ===========
    1) mostCommon: return counter object for word frequencies,
        function called in `padSequence`
    2) getDictionary: update dictionaries of words and tokens,
        function called in `tokenize`
    3) tokenize: return vector of tokenized sequences with
        the shape of textVec
    4) padSequence: pad tokenized sequences to maximum length

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

    def mostCommon(self):
        #count qualified unique words
        cntWords = Counter()

        for vec in self.textVec:
            for txt in vec:
                #get length of sequence
                cntLen = len(txt)
                for w in txt:
                    if qualify(w) == True:
                        cntWords[w] += 1
                #update maximum sequence length
                if self.maxLen < cntLen:
                    self.maxLen = cntLen

        return cntWords

    def getDictionary(self):

        if len(self.word2idx) != 0:
            print("Dictionary already updated.")

        else:
            #update word and idx dictionaries for tokenization
            #end of sequence tag set to be 0
            eos = 0

            #initiate dictionary update from one
            start = 1
            self.word2idx.update({'eos': 0})
            self.idx2word.update({0: 'eos'})

            for vec in self.textVec:
                for txt in vec:
                    for w in txt:
                        if qualify(w) == True:
                            if w not in self.word2idx.keys():
                                self.word2idx.update({w: start})
                                self.idx2word.update({start: w})
                                start += 1
            #update number of unique words in data set
            self.nUnique = start

            #add padding token to dictionary
            pad = start + 1
            self.word2idx.update({'endpad': pad})
            self.idx2word.update({pad: 'endpad'})

    def tokenize(self):

        #get dictionaries if function hasn't been called
        if len(self.word2idx) == 0:
            self.getDictionary()

        #cache list all tokenized vector
        tokenizedVec = list()
        #for each text vector, title/abstract
        for vec in self.textVec:
            #cache list for the tokenized vector
            tempVec = list()
            for txt in vec:
                #cache list for sequence
                sVec = list()
                for w in txt:
                    try:
                        sVec.append(self.word2idx[w])
                    except:
                        sVec = sVec
                #add end of sequence tag
                sVec.append(self.word2idx['eos'])
                tempVec.append(np.array(sVec))
            tokenizedVec.append(np.array(tempVec))
        return np.array(tokenizedVec)

    def padSequence(self, tVec):
        #call funcion to update maximum sequence length
        if self.maxLen == 0:
            self.mostCommon()
        for vec in tVec:
            for seq in vec:
                while len(seq) < self.maxLen:
                    #pad sequence until maximum sequence length is met
                    seq = seq.append(self.word2idx['endpad'])
        return tVec
