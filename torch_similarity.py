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
test_input = text[0:5]

# tokenize text
tokens = {'input_ids': [], 'attention_mask': []}

###############
# WIP create function to do this for each entry and output the single mean vector
# per entry

tokens_test = bert_tokenizer.encode_plus(text[0], max_length=128,
                                            truncation=True, 
                                            padding='max_length', 
                                            return_tensors='pt')
tokens_test['input_ids']
tokens_test['attention_mask']

outputs_test = bert_model(**tokens_test)
outputs_test.keys()

embeddings_test = outputs_test.last_hidden_state
embeddings_test

attention_mask_test = tokens_test['attention_mask']
attention_mask_test.shape

mask_test = attention_mask_test.unsqueeze(-1).expand(embeddings_test.size()).float()
mask_test.shape
mask_test

masked_embeddings_test = embeddings_test * mask_test
masked_embeddings_test.shape
masked_embeddings_test

# sum along one axis
summed_test = torch.sum(masked_embeddings_test, 1)
summed_test.shape

# Then sum the number of values that must be given attention in each 
# position of the tensor:
summed_mask_test = torch.clamp(mask_test.sum(1), min=1e-9)
summed_mask_test.shape

# calculate mean pooled
mean_pooled_test = summed_test / summed_mask_test

# RETURN THIS
# convert from PyTorch tensor to numpy array
mean_pooled_test = mean_pooled_test.detach().numpy()
###########

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


mean_pooled_2 = embedding_mean_vector(txt=text[0], tokenizer=bert_tokenizer, model=bert_model)
mean_pooled_2.shape
mean_pooled_test.shape
###############

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

# calculate
cosine_similarity(
    [mean_pooled[0]],
    mean_pooled[1:]
)