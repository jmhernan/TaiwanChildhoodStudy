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
import operator

project_root = '/Users/josehernandez/Documents/eScience/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)

gl_embed = gensim_api.load("glove-wiki-gigaword-300") # create function to load pickle or download 

# explore words for potential topics
text = df_text.apply(tp.clean_text)
text = text.str.replace('(\d{2}|d{3})', 'person', regex=True)
# Try to replace the numbers with "person"

# Test with smaller sample and implement a better data intake method 
test_set = text[0:50]
len(test_set)
text = tp.remove_non_ascii(test_set)    

word_counts = tp.word_count_entry(text)
truncated_text = tp.token_trunc(text, 500)

###
tp.get_top_n_words(truncated_text, n=100)

# Load keywords json
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 'category_keywords.json'))

# BERT 
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
m_bert = transformers.TFBertModel.from_pretrained('bert-base-uncased')

def utils_bert_embedding(txt, tokenizer, bert_model):
    idx = tokenizer.encode(txt)
    idx = np.array(idx)[None,:]  
    embedding = bert_model(idx)
    X = np.array(embedding[0][0][1:-1])
    return X
    
mean_vec = [utils_bert_embedding(txt, tokenizer, m_bert).mean(0) for txt in truncated_text]    

## create the feature matrix (observations x 768)
X = np.array(mean_vec)
X.shape

# Create dict of context 
dict_codes = {key: None for key in cat_keyw.keys()}

for k in cat_keyw.keys():
    dict_codes[k] = tp.get_similar_words(cat_keyw[k],30, gl_embed)

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
predicted_prob = similarities
predicted = [labels[np.argmax(pred)] for pred in predicted_prob]

truncated_text[2]
predicted[2]
similarities[2]

labels_pred = {labels: predicted_prob[1][idx] for idx, labels in enumerate(labels)}

sorted(labels_pred, key=labels_pred.get, reverse=True)
sorted(predicted_prob[2], reverse=True)

sorted_tuples = sorted(labels_pred.items(), key=operator.itemgetter(1), reverse=True)

# putting together a comprehensible look up table in pandas make function WIP
# using all the labels
labels_probability = [0]*len(predicted_prob)

for ind, t in enumerate(predicted_prob):
    labels_pred = {labels: t[idx] for idx, labels in enumerate(labels)}
    sorted_tuples = sorted(labels_pred.items(), key=operator.itemgetter(1), reverse=True)
    labels_probability[ind] = list(sorted_tuples)

len(labels_probability)

validation_df = pd.DataFrame(truncated_text, columns=['text'])    
validation_df['reults_ordered'] = labels_probability

validation_df.to_csv(os.path.join(data_path,'validation_test_02032021.csv'))