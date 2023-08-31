# pytorch implementation
import numpy as np
import os
import pandas as pd
import re
import uuid

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn import metrics, manifold
from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import remove_stopwords
from datasets import Dataset
from torch.nn.functional import cosine_similarity

project_root = '/Users/josehernandez/Documents/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd.read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)

clean_text = df_text.apply(tp.clean_text)

entries_df = pd.DataFrame(clean_text)

entries_pt_df = Dataset.from_pandas(entries_df)
######################################################
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 
                                                'category_keywords.json'))

df_themes = pd.DataFrame([(key, ' '.join(value).replace(',', '')) for key, 
                          value in cat_keyw.items()], columns=['theme', 'text'])
######################################################
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1', do_lower_case=True)
model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# gpu
import torch

device = torch.device("cpu")
model.to(device)

#
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

# test embeddings 
embedding = get_embeddings(df_themes['text'][1])
embedding.shape

theme_pt_df = Dataset.from_pandas(df_themes)

embeddings_dataset = theme_pt_df.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

# embeddings_dataset.add_faiss_index(column="embeddings")

######################################################
entries_emb_df = entries_pt_df.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

all_similarity_scores = []

for dataset1_index in range(len(entries_emb_df)):
    dataset1_tensor = torch.tensor(entries_emb_df[dataset1_index]['embeddings'])

    similarity_scores = []

    for dataset2_index in range(len(embeddings_dataset)):
        dataset2_tensor = torch.tensor(embeddings_dataset[dataset2_index]['embeddings'])

        assert dataset1_tensor.dim() == 1 and dataset1_tensor.size() == dataset2_tensor.size()

        similarity_scores.append(cos(dataset1_tensor, dataset2_tensor).item())

    all_similarity_scores.append(similarity_scores)

embeddings_dataset['theme']

df_themes = pd.DataFrame(all_similarity_scores, columns=embeddings_dataset['theme'])
entries_pt_df['text'][0]

# softmax test
test = df_themes.iloc[0].values

import numpy as np

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

# Normalize cosine scores (sum to 1) TODO!!
softmax_values = softmax(test)
print(softmax_values)
print("Sum:", np.sum(softmax_values)) 

# data 
def dataset_to_dataframe(dataset):
    data = {key: [] for key in dataset[0].keys()}
    for item in dataset:
        for key, value in item.items():
            data[key].append(value)
    return pd.DataFrame(data)

pt_df = dataset_to_dataframe(test_df)

result = pd.concat([pt_df, df_themes], axis=1)

# export dataset 
result.to_csv('data/sbert_082023.csv', index=False)
