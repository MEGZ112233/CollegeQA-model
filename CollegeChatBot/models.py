from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from config import get_api_key

def initialize_models(model_name, api_len=10):
    embedding_model = HuggingFaceEmbeddings(
        model_name="Bo8dady/finetuned4-College-embeddings",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    models = [ChatGoogleGenerativeAI(model=model_name, api_key=get_api_key(i)) for i in range(api_len)]
    return embedding_model, models