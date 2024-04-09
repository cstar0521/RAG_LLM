import pandas as pd
import tiktoken
from openai import OpenAI

from utils.embeddings_utils import get_embedding

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000 

encoding = tiktoken.get_encoding(embedding_encoding)

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def add_embeddings(df):
    embedded_values = []
    indexes_to_remove = []  

    for i, x in enumerate(df.iloc[:, 1]):
        if len(encoding.encode(x)) <= 8000: 
            embedded_values.append(get_embedding(x))
        else:
            indexes_to_remove.append(i)
    df2 = df.drop(index=indexes_to_remove)
    df2['embedded_values'] = embedded_values
    return df2