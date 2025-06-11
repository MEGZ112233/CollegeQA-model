from fastapi import FastAPI
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel
from config import setup_environment
from models import initialize_models
from utils import upload_chunks
from faiss_handler import manage_faiss_index
from chatbot import get_answer
import os
import uvicorn

app = FastAPI()

# Global variables
retriever = None
models = None
memory = ConversationBufferWindowMemory(k=4, memory_key="chat_history", return_messages=True)
faiss_indexes = {}

# Define the request model
class QuestionInput(BaseModel):
    question: str


@app.on_event("startup")
async def startup_event():
    global retriever, models , faiss_indexes
    print("🔧 Initializing environment and models...")
    setup_environment()
    embeddings, models = initialize_models("gemini-2.0-flash")
    data_sources_config = {
        "abstracts": {"file": "abstracts.csv", "index_folder": "faiss_abstracts"},
        "emails": {"file": "emails.csv", "index_folder": "faiss_emails"},
        "finals": {"file": "finals.csv", "index_folder": "faiss_finals"},
        "rules": {"file": "rules.csv", "index_folder": "faiss_rules"},
    }
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for db_name, config in data_sources_config.items():
        data_path = os.path.join(base_dir, "data", config["file"])
        index_folder_path = os.path.join(base_dir, config["index_folder"])
        chunks = upload_chunks(data_path)
        faiss_index = manage_faiss_index(chunks, embeddings, index_folder_path)
        faiss_indexes[db_name] = faiss_index


    chunk_path = os.path.join(base_dir, "merged.csv")
    chunks = upload_chunks(chunk_path)

    faiss_index = manage_faiss_index(chunks, embeddings, "faiss_index")
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    print("✅ Startup complete.")


@app.get("/ModularChatBot")
async def chatbot_api(payload: QuestionInput):
    global retriever, models , memory
    question = payload.question
    print("Received:", question)
    answer = await get_answer(question, models, faiss_indexes , memory)
    memory.save_context({"question": question}, {"answer": answer})
    print(memory)
    return {"answer": answer}


if __name__ == "__main__":
    # Run using uvicorn from this file directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
