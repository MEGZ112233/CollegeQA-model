from fastapi import FastAPI ,HTTPException
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel
from config import setup_environment
from models import initialize_models
from utils import upload_chunks
from faiss_handler import manage_faiss_index
from chatbot import get_answer
from typing import Optional
import os
import uvicorn
import uuid
from langchain_community.chat_message_histories import RedisChatMessageHistory


app = FastAPI()

# Global variables
retriever = None
models = None
REDIS_URL  =None
faiss_indexes = {}

# Define the request model
class QuestionInput(BaseModel):
    question: str
    session_id : Optional[str] = None

class SessionHistoryInput(BaseModel):
    session_id: str


@app.on_event("startup")
async def startup_event():
    global retriever, models , faiss_indexes ,  REDIS_URL
    print("🔧 Initializing environment and models...")
    setup_environment()
    REDIS_URL = os.getenv("REDIS_URL")
    embeddings, models = initialize_models("gemini-2.5-flash-preview-05-20")
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
    try:
        from redis import Redis
        r = Redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Successfully connected to Redis.")
    except Exception as e:
        print(f"❌ Could not connect to Redis at {REDIS_URL}: {e}")
        exit(1)

    faiss_index = manage_faiss_index(chunks, embeddings, "faiss_index")
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    print("✅ Startup complete.")


@app.post("/ModularChatBot")
async def chatbot_api(payload: QuestionInput):
    global retriever, models
    question = payload.question
    session_id = payload.session_id

    if not session_id:
        session_id = str(uuid.uuid4())
        print(f" NEW SESSION ID: {session_id}")

    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url = os.getenv("REDIS_URL")
    )

    current_session_memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True
    )
    print("Received:", question)
    answer = await get_answer(question, models, faiss_indexes , current_session_memory)
    current_session_memory.save_context({"question": question}, {"answer": answer})
    print(current_session_memory)
    return {"session_id": session_id, "answer": answer}


@app.post("/get_chat_history")
async def get_chat_history_api(payload: SessionHistoryInput):
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try :
        if not isinstance(session_id,str) :
            session_id = str(session_id)
        message_history = RedisChatMessageHistory(
            session_id=session_id,
            url = os.getenv("REDIS_URL")
        )
        messages = message_history.messages
        formatted_history = []
        for message in messages:
            formatted_history.append({"type" : message.type , "message" : message.content})
        return {"session_id": session_id, "chat_history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error while retrieving chat history:{str(e)}")

    
if __name__ == "__main__":
    # Run using uvicorn from this file directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
