# explore diffreent distance metrics 
from numpy.core.fromnumeric import mean
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

from multiprocessing import  Pool

project_root = '/Users/josehernandez/Documents/eScience/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)

# gl_embed = gensim_api.load("glove-wiki-gigaword-300") # create function to load pickle or download 

# explore words for potential topics
text = df_text.apply(tp.clean_text)
text = tp.remove_non_ascii(text)    
len(text)

# what is empty, for Jing to check
text.index("")
# [1218]

text = tp.remove_empty(text)
len(text)

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
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
m_bert = transformers.TFBertModel.from_pretrained('bert-base-uncased')

# See how the truncation works ########
test = "how the heck do we do this now if this is being truncated"
idx = bert_tokenizer.encode(test,truncation=True, max_length=5)

# RoBerta
roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
m_roberta = transformers.TFRobertaModel.from_pretrained('roberta-base')

def utils_embedding(txt, tokenizer, model): # handle truncation here 
    idx = tokenizer.encode(txt,truncation=True, max_length=512)
    idx = np.array(idx)[None,:]  
    embedding = model(idx)
    X = np.array(embedding[0][0][1:-1])
    return X

#########
mean_vec = [utils_embedding(txt=txt, tokenizer=bert_tokenizer, model=m_bert).mean(0) for txt in text]    

# RoBerta 
r_mean_vec = [utils_embedding(txt=txt, tokenizer=roberta_tokenizer, model=m_roberta).mean(0) for txt in text]    

## create the feature matrix (observations x 768)
X = np.array(r_mean_vec) # WIP changed globally
X.shape

###############WIP
# create function to test this everytime
X_sum = np.sum(X)
np.isnan(X_sum)
location = np.argwhere(np.isnan(X)) #check nans from previous implementation 
#######################

dict_y = {k:utils_embedding(v, roberta_tokenizer, m_roberta).mean(0) for k,v in cat_keyw.items()} #WIP
dict_y['FAMILY']
dict_y.keys()

# We want to iterate the X entry across all the dict_y entries and get a single value for each 

# Create model
similarities = np.array([metrics.pairwise.cosine_similarity(X,[y]).T.tolist()[0] for y in dict_y.values()]).T

####WIP to try other distances 
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
############

# Original implementation
labels = list(dict_y.keys())
for i in range(len(similarities)):
    if sum(similarities[i]) == 0:
        similarities[i] = [0]*len(labels)
        similarities[i][np.random.choice(range(len(labels)))] = 1
    similarities[i] = similarities[i] / sum(similarities[i])

# Classify based on Cosine Similarity score
predicted_prob = similarities
predicted = [labels[np.argmax(pred)] for pred in predicted_prob]

text[7] 
predicted[7]
similarities[7]

labels_pred = {labels: predicted_prob[7][idx] for idx, labels in enumerate(labels)}

sorted(labels_pred, key=labels_pred.get, reverse=True)
sorted(predicted_prob[5], reverse=True)
