from prompts import instruction_str, prompt_template, context
from typing import List
import note_engine
from embeddings import get_embedding, reduce_df
import getpass
import os
import pandas as pd
import numpy as np
import tiktoken
from dotenv import find_dotenv, load_dotenv
from llama_index.core.query_engine import PandasQueryEngine as pqe
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from astrapy.db import AstraDB

#os.environ["OPENAI_API_KEY"] = getpass.getpass()

#loading env, data and creating pandas query engine that consumes information based on prompts.py
dotenv_path=find_dotenv()
load_dotenv(dotenv_path)

# Initialize the client
db = AstraDB(
  token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
  api_endpoint=os.getenv("ASTRA_ENDPOINT"))

#loading data
if os.path.isfile('data/embedded_data.csv'):
    print("The file 'embedded_data.csv' already exists in this directory.")
    response = input("Do you want to load data once again and apply embedding anyways? (Y/N) - default N ").strip().upper()

    if response == 'Y':
        articles_path = os.path.join("data", "medium.csv")
        articles_df = pd.read_csv(articles_path)
        articles_df = reduce_df(articles_df)
        print("Getting embeddings ...")
        articles_df['embedded_values'] = articles_df['Text'].apply(get_embedding)
        print("Saving csv ...")
        articles_df.to_csv('embedded_data.csv', index=False)
    else:
        articles_path = os.path.join("data", "embedded_data.csv")
        articles_df = pd.read_csv(articles_path)

#myEmbedding = OpenAIEmbeddings
#articles_df['ada_embedding'] = articles_df.ada_embedding.apply(eval).apply(np.array)
#articles_df['embedded_values'] = articles_df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

#articles_df['embedded_values'] = get_embeddings(articles_df['Text'].tolist())
#articles_df = reduce_df(articles_df)
#print("Getting embeddings ...")
#articles_df['embedded_values'] = articles_df['Text'].apply(get_embedding)
#print("Saving csv ...")
#articles_df.to_csv('embedded_data.csv', index=False)

#prompts
 
articles_query_engine = pqe(df=articles_df, verbose=True, instruction_str=instruction_str)
articles_query_engine.update_prompts({"pandas_prompt": prompt_template})

articles_metadata = ToolMetadata(
            name="articles_data",
            description=(
                "this gives information about facts from Medium articles."
                "Use a detailed plain text question as input to the tool."
                         ),
        )

query_engine_tools = [
   #note_engine,
    QueryEngineTool(
        query_engine=articles_query_engine,
        metadata=articles_metadata,
    ),
]

#llm init
llm = OpenAI(model="gpt-3.5-turbo-1106")

#chatting with llm
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)