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

project_root = '/Users/josehernandez/Documents/projects/TaiwanChildhoodStudy/'

import txt_preprocess as tp

data_path = project_root + 'data/'

df = pd.read_excel(os.path.join(data_path,'4ChildObservation_MasterFile_0120_2021.xlsx'))

df_text = df["text"].astype(str)
######################################################
# seperate the sentences
def paragraphs_to_sentences(text):
    # Split text into paragraphs
    paragraphs = text.split("\n")

    # Split each paragraph into sentences and create a nested list
    nested_list = [re.split('(?<=[.!?]) +', p.strip()) for p in paragraphs if p]

    return nested_list

p_s = df_text.apply(paragraphs_to_sentences).apply(lambda x: x[0])
ps_list = p_s.tolist()

# add hash tags to each sentence
# Create a dictionary with paragraph hash IDs
paragraph_dict = {uuid.uuid4().hex: paragraph for paragraph in ps_list}

paragraph_dict.items()
# Apply clean_text to each sentence in each paragraph
paragraph_dict = {paragraph_id: [tp.clean_text(sentence) for sentence in sentences] for paragraph_id, sentences in paragraph_dict.items()}

print(paragraph_dict)

# Convert to a DataFrame
# Create list of tuples
par_data_tuples = [(key, sentence) for key, sentences in paragraph_dict.items() for sentence in sentences]

# Convert list of tuples to DataFrame
par_sentence_df = pd.DataFrame(par_data_tuples, columns=['key', 'sentence'])

# Apply clean_text to each sentence in each paragraph
# df['paragraph'] = df['paragraph'].apply(lambda paragraph: [clean_text(sentence) for sentence in paragraph])

# Load keywords json
cat_keyw = tp.get_metadata_dict(os.path.join(project_root, 
                                                'category_keywords.json'))

df_themes = pd.DataFrame([(key, ' '.join(value).replace(',', '')) for key, 
                          value in cat_keyw.items()], columns=['theme', 'text'])

# add embeddings
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

# test 
embedding = get_embeddings(df_themes['text'][1])
embedding.shape

theme_faiss = Dataset.from_pandas(df_themes)

embeddings_dataset = theme_faiss.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
######
embeddings_dataset.add_faiss_index(column="embeddings")

# test 
sentence = par_sentence_df['sentence'][0]
####
clean_text = df_text.apply(tp.clean_text)
sentence = clean_text[0]
####
sentence_embedding = get_embeddings([sentence]).cpu().detach().numpy()
sentence_embedding.shape

# loop through this and collec the codes and % od the main one
# into a pd dateset....
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", sentence_embedding, k=6
)
####
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f": {row.text}")
    print(f"SCORE: {row.scores}")
    print(f"Code: {row.theme}")
    print("=" * 50)
    print()


####
from torch.nn.functional import cosine_similarity

# assuming 'datasets' is your Dataset object and 'df' is your DataFrame
# and 'tensor' is the column in both where the embeddings are stored

# assuming 'datasets' is your Dataset object and 'df' is your DataFrame
# and 'tensor' is the column in both where the embeddings are stored

# initialize a PyTorch cosine similarity object
clean_text = df_text.apply(tp.clean_text)
test_df = clean_text[0:20]
#test_df[0]
test_df = pd.DataFrame(test_df)

test_df = Dataset.from_pandas(test_df)

test_df = test_df.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

all_similarity_scores = []
# for each tensor in the DataFrame
# assuming 'datasets1' and 'datasets2' are your Dataset objects
# and 'tensor' is the column in both where the embeddings are stored

# initialize a PyTorch cosine similarity object
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

all_similarity_scores = []

# for each tensor in the first Dataset
for dataset1_index in range(len(test_df)):
    dataset1_tensor = torch.tensor(test_df[dataset1_index]['embeddings'])

    similarity_scores = []

    # for each tensor in the second Dataset
    for dataset2_index in range(len(embeddings_dataset)):
        dataset2_tensor = torch.tensor(embeddings_dataset[dataset2_index]['embeddings'])

        # make sure the tensors are 1D and of the same length
        assert dataset1_tensor.dim() == 1 and dataset1_tensor.size() == dataset2_tensor.size()

        # compute the cosine similarity and add to the list
        similarity_scores.append(cos(dataset1_tensor, dataset2_tensor).item())

    # add the list of similarity scores for this tensor to the overall list
    all_similarity_scores.append(similarity_scores)

embeddings_dataset['theme']

df_themes = pd.DataFrame(all_similarity_scores, columns=embeddings_dataset['theme'])
test_df['text'][0]
###################################
test_df_sent = Dataset.from_pandas(par_sentence_df[0:20])

test_df_sent = test_df_sent.map(
    lambda x: {"embeddings": get_embeddings(x["sentence"]).detach().cpu().numpy()[0]}
)

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

all_similarity_scores = []
# for each tensor in the DataFrame
# assuming 'datasets1' and 'datasets2' are your Dataset objects
# and 'tensor' is the column in both where the embeddings are stored

# initialize a PyTorch cosine similarity object
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

all_similarity_scores = []

# for each tensor in the first Dataset
for dataset1_index in range(len(test_df_sent)):
    dataset1_tensor = torch.tensor(test_df_sent[dataset1_index]['embeddings'])

    similarity_scores = []

    # for each tensor in the second Dataset
    for dataset2_index in range(len(embeddings_dataset)):
        dataset2_tensor = torch.tensor(embeddings_dataset[dataset2_index]['embeddings'])

        # make sure the tensors are 1D and of the same length
        assert dataset1_tensor.dim() == 1 and dataset1_tensor.size() == dataset2_tensor.size()

        # compute the cosine similarity and add to the list
        similarity_scores.append(cos(dataset1_tensor, dataset2_tensor).item())

    # add the list of similarity scores for this tensor to the overall list
    all_similarity_scores.append(similarity_scores)



df_themes = pd.DataFrame(all_similarity_scores, columns=embeddings_dataset['theme'])
par_sentence_df['sentence'][0]
test = df_themes.iloc[1].values

import numpy as np

def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

# Example usage:
scores = [0.9, -0.7, 0.8, 0.75, -0.6, 0.85]
softmax_values = softmax(test)
print(softmax_values)
print("Sum:", np.sum(softmax_values))  # This should be very close to 1
