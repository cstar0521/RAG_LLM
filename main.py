from tools.prompts import instruction_str, prompt_template, context
from tools.embeddings import get_embedding, reduce_df
from dotenv import load_dotenv
import pandas as pd
import os
from llama_index.core.query_engine import PandasQueryEngine as pqe
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

def load_articles_df():
    articles_path = os.path.join("data", "embedded_data.csv")
    if os.path.isfile(articles_path):
        print("The file 'embedded_data.csv' already exists in this directory.")
        response = input("Do you want to load data once again and apply embedding anyways? (Y/N) - default N ").strip().upper()

        if response == 'Y':
            articles_df = pd.read_csv(os.path.join("data", "medium.csv"))
            articles_df = reduce_df(articles_df)
            print("Getting embeddings ...")
            articles_df['embedded_values'] = articles_df['Text'].apply(get_embedding)
            print("Saving csv ...")
            articles_df.to_csv(articles_path, index=False)
        else:
            articles_df = pd.read_csv(articles_path)
    else:
        print("The file 'embedded_data.csv' does not exist in this directory.")
        articles_df = None
    return articles_df

def query_agent(articles_df):
    if articles_df is not None:
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
            QueryEngineTool(
                query_engine=articles_query_engine,
                metadata=articles_metadata,
            ),
        ]

        llm = OpenAI(model="gpt-3.5-turbo-1106")

        agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True, context=context)

        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            result = agent.query(prompt)
            print(result)
    else:
        print("No data to query.")

if __name__ == "__main__":
    articles_df = load_articles_df()
    query_agent(articles_df)