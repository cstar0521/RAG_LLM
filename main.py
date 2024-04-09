from prompts import instruction_str, prompt_template, context
import note_engine
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.query_engine import PandasQueryEngine as pqe
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

#loading env, data and creating pandas query engine that consumes information based on prompts.py

load_dotenv()

articles_path = os.path.join("data", "medium.csv")
articles_df = pd.read_csv(articles_path)
 
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
#chatting with llm
llm = OpenAI(model="gpt-3.5-turbo-1106")
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)