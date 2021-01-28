import pandas as pd
import numpy as np
from sklearn import metrics, manifold
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import gensim.downloader as gensim_api
import transformers
import pickle

import os

project_root = '/Users/josehernandez/Documents/eScience/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'ChildObservationLemmaDataNoNumNoP_0622.xlsx'))

df_clean = df["Lemmatize"].astype(str)

gl_embed = gensim_api.load("glove-wiki-gigaword-300") # create function to load pickle or download 

# explore words for potential topics
text = df_clean.apply(tp.clean_text)

# Test with smaller sample and implement a better data intake method 
test_set = text[0:100]

text = tp.remove_non_ascii(test_set)    

word_counts = tp.word_count_entry(text)
truncated_text = tp.token_trunc(text, 500)

###
tp.get_top_n_words(truncated_text, n=100)

# Load keywords json
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 'category_keywords.json'))

# Pre-defined vocabulary for behavior codes/categories in observations
# These need to be defined using keywords found in observation data 

dict_codes = {key: None for key in cat_keyw.keys()}

for k in cat_keyw.keys():
    dict_codes[k] = tp.get_similar_words(cat_keyw[k],30, gl_embed)

# plot these clusters
all_words= [w for v in dict_codes.values() for w in v]
X = gl_embed[all_words]

pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
X = pca.fit_transform(X)

dtf = pd.DataFrame()

for k,v in dict_codes.items():
    size = len(dtf) + len(v)
    dtf_group = pd.DataFrame(X[len(dtf):size], columns=["x","y"], index=v)
    dtf_group['cluster'] = k
    dtf = dtf.append(dtf_group)


fig, ax = plt.subplots()
sns.scatterplot(data=dtf, x="x", y="y", hue="cluster", ax=ax)
