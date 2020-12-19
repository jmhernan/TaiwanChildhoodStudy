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

sys.path.append(project_root)

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'ChildObservationLemmaDataNoNumNoP_0622.xlsx'))

df_clean = df["Lemmatize"].astype(str)

gl_embed = gensim_api.load("glove-wiki-gigaword-300")

# explore words for potential topics
text = df_clean.apply(tp.clean_text)

# BERT issue with exeeding the word len per observation (512) Check the lengths 
# Issue found missing entry 'nan' on 
def token_trunc(txt, max_length):
    word_counts = [0] * len(txt)
    for index, obs in enumerate(txt):
        word_counts[index] = len(obs.split())
    updated_txt = [l for l in txt if len(l.split()) < max_length]
    return word_counts, updated_txt 

wrd_count, truncated_text = token_trunc(text, 400)

if 'nan' in truncated_text: truncated_text.remove('nan') # Not working! find better fix in data cleaning function!!

###
tp.get_top_n_words(truncated_text, n=100)

# Find potenital key words 
# play, laugh, cry, hit
def get_similar_words(list_words, top, wb_model):
    list_out = list_words
    for w in wb_model.most_similar(list_words, topn=top):
        list_out.append(w[0])
    return list(set(list_out))

# Pre-defined vocabulary for behavior codes/categories in observations
# These need to be defined using keywords found in observation data 

play_sw = get_similar_words(['play','throw','jump'],30, gl_embed)

happy_sw = get_similar_words(['laugh','smile','hold'],30, gl_embed)

aggression_sw = get_similar_words(['cry','hit','yell','push','grab'],30, gl_embed)

# create dict of keywords for coding scheme
dict_codes = {}

dict_codes['PLAY'] = play_sw
dict_codes['HAPPY'] = happy_sw
dict_codes['AGGRESSION'] = aggression_sw

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

ax.legend().texts[0].set_text(None)
ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], yticks=[], yticklabels=[])

for i in range(len(dtf)):
    ax.annotate(dtf.index[i], xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

# BERT 
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
m_bert = transformers.TFBertModel.from_pretrained('bert-base-uncased')

def utils_bert_embedding(txt, tokenizer, bert_model):
    idx = tokenizer.encode(txt)
    idx = np.array(idx)[None,:]  
    embedding = bert_model(idx)
    X = np.array(embedding[0][0][1:-1])
    return X
    
## FIX THIS 
mean_vec = [utils_bert_embedding(txt, tokenizer, m_bert).mean(0) for txt in truncated_text]    

## create the feature matrix (observations x 768)
X = np.array(mean_vec)

np.any(np.isnan(X)) # where are these comming from? 
np.all(np.isfinite(X))

test = list(map(tuple, np.where(np.isnan(X))))

# Create dict of context 
dict_y = {k:utils_bert_embedding(v, tokenizer, m_bert).mean(0) for k,v in dict_codes.items()}

# Create model 
similarities = np.array([metrics.pairwise.cosine_similarity(X,y).T.tolist()[0] for y in dict_y.values()]).T

labels = list(dict_y.keys())
for i in range(len(similarities)):
    if sum(similarities[i]) == 0:
        similarities[i] = [0]*len(labels)
        similarities[i][np.random.choice(range(len(labels)))] = 1

    similarities[i] = similarities[i] / sim(similarities[i])

# Classify based on Cosine Similarity score

