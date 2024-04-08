from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine as pqe

load_dotenv()

articles_path = os.path.join("data", "medium.csv")
articles_df = pd.read_csv(articles_path)
 
articles_query_engine = pqe(df=articles_df, verbose=True)
