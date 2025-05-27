from fastapi import FastAPI
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

# Define the request model
class QuestionInput(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global retriever, models
    print("🔧 Initializing environment and models...")
    setup_environment()
    embeddings, models = initialize_models("gemini-2.5-flash-preview-04-17")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    chunk_path = os.path.join(base_dir, "merged.csv")
    chunks = upload_chunks(chunk_path)
    
    faiss_index = manage_faiss_index(chunks, embeddings, "faiss_index")
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    print("✅ Startup complete.")

@app.post("/ModularChatBot")
async def chatbot_api(payload: QuestionInput):
    global retriever, models
    question = payload.question
    print("Received:", question)
    answer = await get_answer(question, retriever, models)
    return {"answer": answer}

if __name__ == "__main__":
    # Run using uvicorn from this file directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
