import numpy as np

def getSpace(x):
    "Remove special space character from abstract and paper text"
    for i in range(len(x)):
        try:
            x[i] = x[i].replace('\n', ' ')
        except:
            x[i] = x[i]
    return x

def get_abstract(paper_text):
    "Extract abstract from paper text for observations with missing abstract"
    abst = np.nan
    if 'ABSTRACT ' in paper_text:
        front = paper_text.split("ABSTRACT ", 1)[1]
        abst = front.split("INTRODUCTION", 1)[0]
    elif 'Abstract 'in paper_text:
        front = paper_text.split("Abstract ", 1)[1]
        abst = front.split("1  INTRODUCTION", 1)[0]
    return abst

def preprocessing(papers):
    papers.abstract = papers.abstract.apply(lambda x: np.nan if x == 'Abstract Missing' else x)
    papers[['abstract','paper_text']] = papers[['abstract','paper_text']].apply(lambda x: getSpace(x), axis = 1)
    papers['abstract'] = papers['paper_text'].apply(get_abstract)
    return papers
