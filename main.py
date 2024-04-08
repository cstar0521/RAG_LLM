from prompts import instruction_str, prompt_template
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.query_engine import PandasQueryEngine as pqe


#loading env, data and creating pandas query engine that consumes information based on prompts.py
load_dotenv()

articles_path = os.path.join("data", "medium.csv")
articles_df = pd.read_csv(articles_path)
 
articles_query_engine = pqe(df=articles_df, verbose=True, instruction_str=instruction_str)
articles_query_engine.update_prompts({"pandas_prompt": prompt_template})

print("\n",articles_query_engine.query("in 3 words what is Web Scraping"))