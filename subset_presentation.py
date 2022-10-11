# subset for presentation
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
import json

project_root = '/Users/josehernandez/Documents/eScience/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)

# subset  
subset_index = [0, 3, 17, 22, 45, 60, 72, 78, 186, 320, 352, 385, 564, 892, 1285]

df_obs = df_text.loc[subset_index].reset_index(drop=True)
# gl_embed = gensim_api.load("glove-wiki-gigaword-300") # create function to load pickle or download 

# explore words for potential topics
text = test.apply(tp.clean_text)
text = tp.remove_non_ascii(text)    
len(text)

text = tp.remove_empty(text)
len(text)

# remove stop words
text = [remove_stopwords(s) for s in text]
len(text)

# seeing top words
tp.get_top_n_words(text, n=100)

# Load keywords json
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 'category_keywords.json'))

# Load BERT 
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
m_bert = transformers.TFBertModel.from_pretrained('bert-base-uncased')

def utils_bert_embedding(txt, tokenizer, bert_model): # handle truncation here 
    idx = tokenizer.encode(txt,truncation=True, max_length=512)
    idx = np.array(idx)[None,:]  
    embedding = bert_model(idx)
    X = np.array(embedding[0][0][1:-1])
    return X

mean_vec = [utils_bert_embedding(txt, tokenizer, m_bert).mean(0) for txt in text]    

## create the feature matrix (observations x 768)
X = np.array(mean_vec)
X.shape

# Create dict of context 
dict_codes = {key: None for key in cat_keyw.keys()}

for k in cat_keyw.keys():
    dict_codes[k] = tp.get_similar_words(cat_keyw[k],20, gl_embed)
# Inspect dictionary of keywords
with open('category_keywords_glove.json', 'w') as fp:
    json.dump(dict_codes, fp, indent=4)

dict_y = {k:utils_bert_embedding(v, tokenizer, m_bert).mean(0) for k,v in cat_keyw.items()}

# Create model
similarities = np.array([metrics.pairwise.cosine_similarity(X,[y]).T.tolist()[0] for y in dict_y.values()]).T

labels = list(dict_y.keys())
for i in range(len(similarities)):
    if sum(similarities[i]) == 0:
        similarities[i] = [0]*len(labels)
        similarities[i][np.random.choice(range(len(labels)))] = 1
    similarities[i] = similarities[i] / sum(similarities[i])

# Classify based on Cosine Similarity score
predicted_prob = similarities
predicted = [labels[np.argmax(pred)] for pred in predicted_prob]

test[12]
predicted[12]
similarities[12]

labels_pred = {labels: predicted_prob[12][idx] for idx, labels in enumerate(labels)}

sorted(labels_pred, key=labels_pred.get, reverse=True)
sorted(predicted_prob[12], reverse=True)

sorted_tuples = sorted(labels_pred.items(), key=operator.itemgetter(1), reverse=True)

# putting together a comprehensive look up table in pandas make function WIP
# using all the labels
labels_probability = [0]*len(predicted_prob)

for ind, t in enumerate(predicted_prob):
    labels_pred = {labels: t[idx] for idx, labels in enumerate(labels)}
    prob = labels_pred.items()
    labels_probability[ind] = list(prob)

len(labels_probability)

df_labels_prob = pd.DataFrame(predicted_prob, columns=['FAMILY','SCHOOL','SHOPPING','PLAY','CONFLICT','COOPERATION']).round(4)
df_labels_prob['obs_key'] = df_labels_prob.index
df_labels_prob = df_labels_prob[['obs_key','FAMILY','SCHOOL','SHOPPING','PLAY','CONFLICT','COOPERATION']]

df_labels_prob.style.background_gradient(subset=['FAMILY','SCHOOL','SHOPPING','PLAY','CONFLICT','COOPERATION'])

# graphs 
theme = list(zip(*labels_probability[1]))[0]
score = list(zip(*labels_probability[1]))[1]
x_pos = np.arange(len(theme)) 

import matplotlib.pyplot as plt

plt.bar(x_pos, score,align='center')
plt.xticks(x_pos, theme) 
plt.ylabel('Scaled Cosine Similarity Score')
plt.show()

df_obs[14]
# create columns of themes and text
from collections import defaultdict

d = defaultdict(list)
for a, b in labels_probability:
    d[a].append(b)

validation_df.to_csv(os.path.join(data_path,'validation_subset.csv'))