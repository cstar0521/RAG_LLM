from typing import List
from openai import OpenAI

client = OpenAI(max_retries=5)

def get_embeddings(
    list_of_text: List[str], model="text-embedding-3-small", **kwargs
) -> List[List[float]]:
    '''
    This method is based on an external repository :  https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
    '''
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]
