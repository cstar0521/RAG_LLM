from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

articles_path = os.path.join("data", "medium.csv")
articles_df = pd.read_csv(articles_path)
 
print(articles_df.head())
