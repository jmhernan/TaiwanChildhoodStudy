# RAG Proof of Concept
from transformers import RagTokenizer, RagSequenceForGeneration

model_name = "facebook/rag-token-base"
tokenizer = RagTokenizer.from_pretrained(model_name)
model = RagSequenceForGeneration.from_pretrained(model_name)

prompt = "What is COVID-19?"

inputs = tokenizer(prompt, return_tensors="pt")

retrieval_output = model.g(inputs)

###
# To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
from transformers import RagRetriever

retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base", dataset="wiki_dpr", index_name="compressed"
)

# To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
from transformers import RagRetriever

dataset = (
    ...
)  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset)

# To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
from transformers import RagRetriever

dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*
index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*
retriever = RagRetriever.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
)

# To load the legacy index built originally for Rag's paper
from transformers import RagRetriever

retriever = RagRetriever.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", index_name="legacy")


from transformers import RagConfig, RagRetriever, RagTokenForGeneration, AutoConfig

# Specify the model names for both the question encoder and generator
question_encoder_model = "facebook/dpr-question_encoder-single-nq-base"
generator_model = "facebook/bart-large"

# Load the configurations for the question encoder and generator
question_encoder_config = AutoConfig.from_pretrained(question_encoder_model)
generator_config = AutoConfig.from_pretrained(generator_model)

# Initialize the RagConfig with both question_encoder and generator configurations
config = RagConfig.from_question_encoder_generator_configs(
    question_encoder_config=question_encoder_config,
    generator_config=generator_config
)

# Initialize the RagRetriever with the given config, dataset, and index
retriever = RagRetriever.from_pretrained(
    question_encoder_model,
    config=config,
    dataset="wiki_dpr",
    index_name="compressed"
)

# Initialize the Rag model with the retriever
rag_model = RagTokenForGeneration.from_pretrained(
    question_encoder_model,
    config=config,
    retriever=retriever
)

