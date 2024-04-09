import pandas as pd
import tiktoken
from openai import OpenAI
from typing import List
from openai import OpenAI

client = OpenAI(max_retries=5)

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000 

encoding = tiktoken.get_encoding(embedding_encoding)

client = OpenAI()

def reduce_df(df):
    '''
    This method ensures that the request sent to OpenAI API will not exceed it's limit.
    '''
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = [len(encoding.encode(x)) for x in df["Text"]]
    #df["n_tokens"] = df.combine.apply(lambda x: len(encoding.encode(x))) 
    #print("\n\n\n",max(df["n_tokens"]),"\n\n\n")
    df = df[df.n_tokens <= max_tokens]
    return df

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

'''
def get_embeddings(
    list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
  #  dodac dokumentacje tu
   # This method is based on an external repository :  https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
   # dodac dokumentacje tu
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]

def add_embeddings(df):
    embedded_values = []
    indexes_to_remove = []  

    for i, x in enumerate(df.iloc[:, 1]):
        if len(encoding.encode(x)) <= 8000: 
            embedded_values.append(get_embeddings(x))
        else:
            indexes_to_remove.append(i)
    df2 = df.drop(index=indexes_to_remove)
    df2['embedded_values'] = embedded_values
    return df2

'''