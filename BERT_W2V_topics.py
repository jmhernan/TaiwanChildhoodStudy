# explore diffreent distance metrics 
import pandas as pd
import numpy as np
from sklearn import metrics, manifold
from Levenshtein import distance
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import gensim.downloader as gensim_api
import transformers
from gensim.parsing.preprocessing import remove_stopwords

import os
import operator

project_root = '/Users/josehernandez/Documents/eScience/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)

gl_embed = gensim_api.load("glove-wiki-gigaword-300") # create function to load pickle or download 

# explore words for potential topics
text = df_text.apply(tp.clean_text)
text = tp.remove_non_ascii(text)    
len(text)

# what is empty, for Jing to check
text.index("")
# [1218]

text_test = tp.remove_empty(text)
len(text_test)

# remove stop words
text = [remove_stopwords(s) for s in text]
text[0]
# seeing top words
tp.get_top_n_words(text, n=100)

# subesetting for testing 
# text = text[0:100]
len(text)
# Load keywords json
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 'category_keywords.json'))

# BERT 
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
m_bert = transformers.TFBertModel.from_pretrained('bert-base-uncased')

def utils_bert_embedding(txt, tokenizer, bert_model): # handle truncation here 
    idx = tokenizer.encode(txt,truncation=True, max_length=512)
    idx = np.array(idx)[None,:]  
    embedding = bert_model(idx)
    X = np.array(embedding[0][0][1:-1])
    return X

num_words_row = [len(words.split()) for words in text]
max_seq_len = max(num_words_row)
# See how the truncation works ########
test = "how the heck do we do this now if this is being truncated"
idx = tokenizer.encode(test,truncation=True, max_length=5)
#########

mean_vec = [utils_bert_embedding(txt, tokenizer, m_bert).mean(0) for txt in text]    

## create the feature matrix (observations x 768)
X = np.array(mean_vec)
X.shape
X[0]
# Create dict of context 
dict_codes = {key: None for key in cat_keyw.keys()}

for k in cat_keyw.keys():
    dict_codes[k] = tp.get_similar_words(cat_keyw[k],20, gl_embed)

dict_y = {k:utils_bert_embedding(v, tokenizer, m_bert).mean(0) for k,v in dict_codes.items()}
dict_y['FAMILY']
dict_y.keys()

[metrics.pairwise.cosine_similarity(X[0].reshape(1,-1),y.reshape(1,-1)) for y in dict_y.values()]

# We want to iterate the X entry across all the dict_y entries and get a single value for each 

# Create model
similarities = np.array([metrics.pairwise.cosine_similarity(X,[y]).T.tolist()[0] for y in dict_y.values()]).T
test = [metrics.pairwise.cosine_similarity(X,[y]).T.tolist()[0] for y in dict_y.values()] # This returns 100 cosine similarity scores per dict embeddings 7
len(test[0])
type(test)
len(test)
len(similarities)
test = np.array([test[0],test[1],test[2],test[3],test[4],test[5]]).T

from scipy import spatial

test = [spatial.distance.cosine(X[i],y).T for i in range(len(X)) for y in dict_y.values())]

for i,y in zip(range(len(X)),dict_y.values()): # append list with similarity values instead of list comprehension WIP 
    print(i,y)

similarities[1]
test[1]
text[1]
test_1 = [spatial.distance.cosine(X,[y]).T for y in dict_y.values()] # This returns 100 cosine similarity scores per dict embeddings 7

[spatial.distance.cosine(X[1],y) for y in dict_y.values()]

# Word Movers Implementation entries X 6 categories of interest
s1 = text[1]
s2 = text[40]
play = dict_codes["PLAY"]
conflict = dict_codes["CONFLICT"]
store = dict_codes["SHOPPING"]
# using raw entreies 

[gl_embed.wmdistance(s2,y) for y in dict_codes.values()] # WIP figure out own implementation
dict_codes.keys()