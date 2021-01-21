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

# BERT 
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
m_bert = transformers.TFBertModel.from_pretrained('bert-base-uncased')

def utils_bert_embedding(txt, tokenizer, bert_model):
    idx = tokenizer.encode(txt)
    idx = np.array(idx)[None,:]  
    embedding = bert_model(idx)
    X = np.array(embedding[0][0][1:-1])
    return X
    
# 
text_test = tokenizer.encode(truncated_text)
idx = np.array(text_test)[None,:]
embedding = m_bert(idx)

inputs = tokenizer.encode(["Hello, my dog is cute"])
outputs = m_bert(inputs)

mean_vec = [utils_bert_embedding(txt, tokenizer, m_bert).mean(0) for txt in truncated_text]    

## create the feature matrix (observations x 768)
X = np.array(mean_vec)
X.shape
# Create dict of context 
dict_y = {k:utils_bert_embedding(v, tokenizer, m_bert).mean(0) for k,v in dict_codes.items()}

# Create model
similarities = np.array([metrics.pairwise.cosine_similarity(X,[y]).T.tolist()[0] for y in dict_y.values()]).T

labels = list(dict_y.keys())
for i in range(len(similarities)):
    if sum(similarities[i]) == 0:
        similarities[i] = [0]*len(labels)
        similarities[i][np.random.choice(range(len(labels)))] = 1

    similarities[i] = similarities[i] / sum(similarities[i])

# Classify based on Cosine Similarity score
# TO DO Extract the top 3 labels 
predicted_prob = similarities
predicted = [labels[np.argmax(pred)] for pred in predicted_prob]

truncated_text[92]
predicted[92]
similarities[92]