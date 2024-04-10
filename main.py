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
embedded_articles_path = os.path.join("data", "embedded_data.csv")
original_articles_path = os.path.join("data", "medium.csv")

def load_articles_df():
    if os.path.isfile(embedded_articles_path):
        print("The file 'embedded_data.csv' already exists in this directory.")
        response = input("Do you want to load data once again and apply embedding anyway? (Y/N) - default N ").strip().upper()
        if response != 'Y':
            articles_df = pd.read_csv(embedded_articles_path)
        else:
            try:
                articles_df = pd.read_csv(original_articles_path)
                articles_df = modify_articles_df(articles_df)
            except Exception as e:
                print(f"An error occurred while loading 'medium.csv': {str(e)}")
                articles_df = None
    else:
            try:
                articles_df = pd.read_csv(original_articles_path)
                articles_df = modify_articles_df(articles_df)
            except Exception as e:
                print(f"An error occurred while loading 'medium.csv': {str(e)}")
                articles_df = None

    return articles_df

#TODO: zastosowac dry zeby kod sie nie powielal powyzej, zmienic kolejnosc metod jesli konieczne

def modify_articles_df(articles_df):
    '''
    # Zmiana: Dodano blok try-except
    The goal of this method is to create vector embeddings from the provided dataframe.
    The embeddings are created only for the rows that do not exceed OpenAI API limits.
    '''
    try:
        articles_df = reduce_df(articles_df)
        if articles_df is None:
            print("An error occurred while processing the DataFrame. Please upload the file again.")
            return None
        
        print("Getting embeddings ...")
        articles_df['embedded_values'] = articles_df['Text'].apply(get_embedding)
        if articles_df['embedded_values'].isnull().values.any():
            print("An error occurred while creating embeddings. Please upload the file again.")
            return None

        print("Saving csv ...")
        articles_df.to_csv(embedded_articles_path, index=False)
    except Exception as e:
        print(f"An error occurred while modifying the DataFrame: {str(e)}")
        return None
    
    return articles_df


def query_agent(articles_df):
    '''
    The goal of this method is to create an PandasQueryEngine agent that will allow the user to chat with LLM chatbot.
    '''
    if articles_df is not None:
        try:
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
        except Exception as e:
            print(f"An error occurred while running the query agent: {str(e)}")
    else:
        print("No data to query.")

if __name__ == "__main__":
    articles_df = load_articles_df()
    query_agent(articles_df)