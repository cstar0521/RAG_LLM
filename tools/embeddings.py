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
    try:
        encoding = tiktoken.get_encoding(embedding_encoding)
        df["n_tokens"] = [len(encoding.encode(x)) for x in df["Text"]]
        #df["n_tokens"] = df.combine.apply(lambda x: len(encoding.encode(x))) 
        #print("\n\n\n",max(df["n_tokens"]),"\n\n\n")
        df = df[df.n_tokens <= max_tokens]
        return df
    except Exception as e:
        print(f"An error occurred while reducing DataFrame: {str(e)}")
        return None

def get_embedding(text, model="text-embedding-3-small"):
   '''
   This method create embedding for a single string, using OPENAI API.
   '''
   try:
       text = text.replace("\n", " ")
       return client.embeddings.create(input=[text], model=model).data[0].embedding
   except Exception as e:
       print(f"An error occurred while creating embedding: {str(e)}")
       return None