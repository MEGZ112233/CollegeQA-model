from fastapi import FastAPI, HTTPException, Depends
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
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from sqlalchemy.ext.asyncio import AsyncSession
import schema
from database import engine, AsyncSessionLocal
from sqlalchemy import select


app = FastAPI()

# Global variables
retriever = None
models = None
REDIS_URL = None
faiss_indexes = {}


# Define the request model
class QuestionInput(BaseModel):
    question: str
    user_id: int
    session_id: Optional[str] = None
    title: Optional[str] = None


class SessionHistoryInput(BaseModel):
    session_id: str

class UserSessionsInput(BaseModel):
    user_id: int

class SessionInfo(BaseModel) :
    session_id: str
    title: str

class UserSessionsResponse(BaseModel) :
    user_id: int
    session_info: list[SessionInfo]

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@app.on_event("startup")
async def startup_event():
    global retriever, models, faiss_indexes
    print("🔧 Initializing environment and models...")
    setup_environment()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(schema.Base.metadata.create_all)
    except Exception as e:
        print(f"❌ Could not connect to postgres : {e}")
        exit(1)
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

    faiss_index = manage_faiss_index(chunks, embeddings, "faiss_index")
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    print("✅ Startup complete.")


@app.post("/ModularChatBot")
async def chatbot_api(payload: QuestionInput, db: AsyncSession = Depends(get_db)):
    global retriever, models
    question = payload.question
    session_id = payload.session_id
    user_id = payload.user_id
    title = payload.title

    if not session_id:
        if not title:
            raise HTTPException(status_code=400, detail="title  is required to create a session.")
        session_id = str(uuid.uuid4())
        print(f" NEW SESSION ID: {session_id}")
        db_user_session = schema.UserSession(user_id=user_id, session_id=session_id, title=title)
        db.add(db_user_session)
        await db.commit()
        await db.refresh(db_user_session)
        print(f"✅ Successfully saved session mapping for user '{user_id}' using ORM.")

    message_history = PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=os.getenv("POSTGRES_URL"),
        table_name="chat_history"
    )

    current_session_memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True
    )
    print("Received:", question)
    answer = await get_answer(question, models, faiss_indexes, current_session_memory)
    current_session_memory.save_context({"question": question}, {"answer": answer})
    print(current_session_memory)
    return {"session_id": session_id, "answer": answer}


@app.post("/get_chat_history")
async def get_chat_history_api(payload: SessionHistoryInput):
    session_id = payload.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        if not isinstance(session_id, str):
            session_id = str(session_id)
        message_history = PostgresChatMessageHistory(
            session_id=session_id,
            connection_string=os.getenv("POSTGRES_URL"),
            table_name="chat_history"
        )
        messages = message_history.messages
        formatted_history = []
        for message in messages:
            formatted_history.append({"type": message.type, "message": message.content})
        return {"session_id": session_id, "chat_history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error while retrieving chat history:{str(e)}")

@app.post("/get_user_sessions" , response_model = UserSessionsResponse)
async def get_user_sessions_api(payload: UserSessionsInput ,  db: AsyncSession = Depends(get_db)):
    user_id = payload.user_id
    query = (
        select(schema.UserSession)
        .where(schema.UserSession.user_id == user_id)
        .order_by(schema.UserSession.created_at.desc())
    )
    result = await db.execute(query)
    user_sessions = result.scalars().all()
    formatted_sessions = [] 
    for session in user_sessions:
        formatted_sessions.append(SessionInfo(title = session.title , session_id=session.session_id))
    return UserSessionsResponse(user_id=user_id, session_info=formatted_sessions)
if __name__ == "__main__":
    # Run using uvicorn from this file directly
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
