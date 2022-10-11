# pytorch implementation
import numpy as np
import os
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn import metrics, manifold
from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import remove_stopwords

project_root = '/Users/josehernandez/Documents/eScience/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd. read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)

text = df_text.apply(tp.clean_text)
text = tp.remove_non_ascii(text)    
len(text)

# what is empty, for Jing to check
text.index("")

text = tp.remove_empty(text)
len(text)

# remove stop words
text = [remove_stopwords(s) for s in text]
text[0]
# seeing top words
tp.get_top_n_words(text, n=100)

# Load keywords json
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 'category_keywords.json'))

# BERT sentence model specification
model_name = 'sentence-transformers/bert-base-nli-mean-tokens'

# initialize the tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# test inputs 
test_input = text[0:2]

# WIP function
def embedding_mean_vector(txt, tokenizer, model):
    # step 1 tokenize entry
    tokens = tokenizer.encode_plus(txt, max_length=128,
                                            truncation=True, 
                                            padding='max_length', 
                                            return_tensors='pt')
    output = model(**tokens)
    embeddings = output.last_hidden_state

    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    masked_embeddings = embeddings * mask
    # sum along one axis
    summed = torch.sum(masked_embeddings, 1)
    # Then sum the number of values that must be given attention in each 
    # position of the tensor:
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    # calculate mean pooled
    mean_pooled = summed / summed_mask
    # RETURN THIS
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()
    
    return mean_pooled


mean_pooled = embedding_mean_vector(txt=text, tokenizer=bert_tokenizer, model=bert_model)
mean_pooled.shape
type(mean_pooled_2)

# old method 
for sentence in test_input:
    new_tokens = bert_tokenizer.encode_plus(sentence, max_length=128,
                                            truncation=True, 
                                            padding='max_length', 
                                            return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

# list of tensors into single tensor
tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

# pass tokens through the model 
outputs = bert_model(**tokens)
outputs.keys()

# outputs are the last_hidden_state
embeddings = outputs.last_hidden_state
embeddings

# Mean pooling process
attention_mask = tokens['attention_mask']
attention_mask.shape

mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
mask.shape
mask

# Each vector above represents a single token attention mask - 
# each token now has a vector of size 768 representing it's 
# attention_mask status. Then we multiply the two tensors 
# to apply the attention mask:
masked_embeddings = embeddings * mask
masked_embeddings.shape
masked_embeddings

# sum along one axis
summed = torch.sum(masked_embeddings, 1)
summed.shape

# Then sum the number of values that must be given attention in each 
# position of the tensor:
summed_mask = torch.clamp(mask.sum(1), min=1e-9)
summed_mask.shape

# calculate mean pooled
mean_pooled = summed / summed_mask

# calculate similarity 
# convert from PyTorch tensor to numpy array
mean_pooled = mean_pooled.detach().numpy()
type(mean_pooled)
mean_pooled.shape

# WIP: try with function call and list comprehension
mean_pooled_test_e = np.empty((0, 768), float)
mean_pooled_test_e.shape
# WIP:
mean_pooled_test = np.append(mean_pooled_test_e, [embedding_mean_vector(txt=txt, tokenizer=bert_tokenizer, model=bert_model) for txt in test_input], axis=0)
# ValueError: all the input arrays must have same number of dimensions, 
# but the array at index 0 has 2 dimension(s) and the array at index 1 has 3 dimension(s)

type(mean_pooled_test)
mean_pooled_test.shape
mean_pooled_test[0]

# calculate
cosine_similarity(
    [mean_pooled[0]],
    mean_pooled[1:]
)

cosine_similarity(
    [mean_pooled_test[0]],
    mean_pooled_test[1:]
)

mean_pooled_test = np.array([embedding_mean_vector(txt=txt, tokenizer=bert_tokenizer, model=bert_model) for txt in text[0]])