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
