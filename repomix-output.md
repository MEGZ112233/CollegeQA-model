This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: CollegeChatBot
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
CollegeChatBot/
  app.py
  chains_and_prompts.py
  chatbot.py
  config.py
  database.py
  faiss_handler.py
  models.py
  schema.py
  splitData.py
  utils.py
```

# Files

## File: CollegeChatBot/config.py
````python
import os
from dotenv import load_dotenv

def setup_environment():
    load_dotenv()

def get_api_key(index):
    return os.getenv(f"G{index}")
````

## File: CollegeChatBot/database.py
````python
import os
from sqlalchemy.ext.asyncio import create_async_engine , async_sessionmaker
from  dotenv import load_dotenv


load_dotenv()

POSTGRES_URL = os.getenv("SQL_POSTGRES_URL")

engine = create_async_engine(POSTGRES_URL)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
````

## File: CollegeChatBot/faiss_handler.py
````python
import os
from langchain.vectorstores import FAISS

def manage_faiss_index(chunks, embeddings, index_folder):
    index_file = os.path.join(index_folder, "index.faiss")
    if os.path.exists(index_file):
        try:
            return FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass
    index = FAISS.from_texts(chunks, embeddings)
    index.save_local(index_folder)
    return index
````

## File: CollegeChatBot/models.py
````python
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
````

## File: CollegeChatBot/schema.py
````python
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    session_id = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
````

## File: CollegeChatBot/splitData.py
````python
import pandas as pd
import os
# Load your file
base_dir = os.path.dirname(os.path.dirname(__file__))
chunk_path =   os.path.join(base_dir, "merged.csv")
df = pd.read_csv(chunk_path)

startAbstract = 0
endAbstract = 230
df.iloc[startAbstract:endAbstract].to_csv(f'abstracts.csv', index=False)
startEmail = 230
endEmail = 410
df.iloc[startEmail:endEmail].to_csv(f'emails.csv', index=False)
startRules = 410
endRules = 438
df.iloc[startRules:endRules].to_csv(f'rules.csv', index=False)
startFinal = 439
endFinals = 498
df.iloc[startFinal:endFinals].to_csv(f'finals.csv', index=False)
````

## File: CollegeChatBot/utils.py
````python
import csv
import json

from langsmith import expect
from numpy.f2py.auxfuncs import throw_error


def upload_chunks(data_path):
    chunks = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0]:
                chunks.append(row[0])
    return chunks
def parse_list(response) :
    response = response.strip()
    if response.startswith("```python"):
        response = response.replace("```python", "")
    elif response.startswith("```"):
        response = response.replace("```", "")
    response = response.strip('\n')
    try:
      jsonList  = json.loads(response)
      if isinstance(jsonList, list) and len(jsonList) == 4 and all(isinstance(x, int) for x in jsonList):
            return jsonList
      else :
          raise Exception(f"Invalid response from API: {response}")


    except json.JSONDecodeError as e:
        parse_error_message = f"JSON decoding error: {e}. Output was not valid JSON. and the output was {response}"
        print(parse_error_message)
        return [0,2,2,2]

    except Exception as e:
        parse_error_message = f"An unexpected error occurred during parsing: {e} and the output was {response}"
        print(parse_error_message)
        return [0,2,2,2]
````

## File: CollegeChatBot/chains_and_prompts.py
````python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.schema import StrOutputParser
from operator import itemgetter


class Templates:
    """Define and return prompt templates for question answering and refinement."""
    qa_template = ChatPromptTemplate.from_messages([
        ("system", """ 
            You will be given a question and context, and you must understand the context and the question to answer in natural language.
           If the conversation history is available, use it to understand follow-up questions.  
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "question: {question}\n context: {context}"),
    ])

    refine_template = ChatPromptTemplate.from_messages([
        ("system",
         "Rephrase the student's question for clarity while preserving its meaning and if there something mispealing correct it. Make sure the output is in English and only one question."),
        ("human", "Question to refine: {question}")
    ])

    cluster_template = ChatPromptTemplate.from_messages([
        ("system", """ You are an intelligent query analysis assistant for a college information system. Your primary task is to determine the optimal number of documents to retrieve from each of four distinct data sources to best answer a student's question.
        The four data sources are, in this specific order:
        1. Abstracts: Contains summaries and details of student graduation projects and research.
        2. Emails: Contains contact information (email addresses, departments) for college doctors and staff.
        3. Finals: Contains links to past final exam papers for various courses.
        4. Rules: Contains official college regulations, academic rules, course prerequisites, and program structures.
        You will be given:
        1. The User's Question.
        2. the chat history . 
        3. A Sample Snippet from each of the four data sources to help you understand the type of information each contains.
        Your goal is to output a Python list of 4 integers. Each integer represents the estimated number of relevant documents that should be retrieved from the corresponding data source to answer the question. The order in the list MUST match the order of the data sources listed above: [number_for_Abstracts, number_for_Emails, number_for_Finals, number_for_Rules]
        Guidelines for determining the number of documents:
        * Relevance is Key: If a data source is highly relevant and likely to contain the core answer, suggest 1-3 documents. For broader queries requiring exploration (e.g., "find projects about AI"), you might suggest up to 10 from 'Abstracts'.
        * Specificity:
            * 
            * If the question asks for a single piece of specific information (e.g., "email of Dr. Smith"), 1 or 2  document from the 'Emails' source is likely sufficient.
            * If the question asks about multiple specific items (e.g., "final exams for CS101 for 2020, 2021, and 2022"), you might suggest 1-3 documents from 'Finals', assuming each year/course might be a separate document or closely related ones.
        * Irrelevance: If a data source is clearly irrelevant to the question, use 0 for that source.
        * Conciseness: Prefer fewer documents if the answer is likely to be found in a small number of highly relevant chunks. Avoid requesting many documents unnecessarily.
        * No Information: If you believe none of the data sources can answer the question, you can output [0, 0, 0, 0].
        Example Scenarios:
        * Question: "What is Dr. Eman Hamdi's email address?"
            * 
            * Expected Output Reasoning: This is a direct request for an email. 'Emails' is the only relevant source.
            * Expected Output: [0, 1, 0, 0]
        * Question: "Can you show me some graduation project abstracts on machine learning?"
            * 
            * Expected Output Reasoning: This is about project abstracts. 'Abstracts' is primary. A few examples would be good.
            * Expected Output: [3, 0, 0, 0] (Suggesting 3 abstracts)
        * Question: "Where can I find the final exam for Algorithms from 2021 and what are the prerequisites for the advanced algorithms course?"
            * 
            * Expected Output Reasoning: Needs 'Finals' for the exam link and 'Rules' for course prerequisites.
            * Expected Output: [0, 0, 1, 1] (or [0, 0, 1, 2] if prerequisites are complex and might span multiple rule chunks)
        * Question: "What are the college's rules regarding plagiarism and what's the email for the head of the CS department?"
            * 
            * Expected Output Reasoning: Needs 'Rules' for plagiarism and 'Emails' for the department head.
            * Expected Output: [0, 1, 0, 1] 
        """),
        MessagesPlaceholder(variable_name="chat_history")
        ,
        ("human", """
                Now, analyze the following:
        User's Question:
              {question}
            
        Data Source Samples:
        1. Sample from 'Abstracts' Data Source:
              **Faculty Information**
        
        Faculty of Computer Science and Information
        Scientific Computing Department
        
        **Abstract**
        
        During the last few decades, with the rise of Youtube, Amazon, Netflix and many other such web services, recommender systems have taken more and more place in our lives. From e-commerce (suggest to buyers articles that could interest them) to online advertisement (suggest to users the right contents, matching their preferences), recommender systems are today unavoidable in our daily online journeys.
        
        **Recommender Systems as Algorithms**
        
        In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries).
            
        2. Sample from 'Emails' Data Source:
              Dr. Eman Hamdi is part of the Unknown department and can be reached at emanhamdi@cis.asu.edu.eg.
        3. Sample from 'Finals' Data Source:
              The final exam for Algorithms Analysis & Design course, offered by the scientific computing department, from 2021, is available at the following link: [https://drive.google.com/file/d/1mT21jrptv4w2IdFd5G-oVxI_6lR_SrS9/view
        The final exam for Algorithms Analysis & Design course, offered by the scientific computing department, from 2020, is available at the following link: [https://drive.google.com/file/d/1W_X01ASI0yyo6fCG4SwZYlvoE-Sh_C0g/view?usp=drive_link
        4. Sample from 'Rules' Data Source:
              Topic: Scientific Computing Fourth Level Courses
        Summary: Outlines the courses, credit hours, and prerequisites for the seventh and eighth semesters of the fourth level Scientific Computing program.
        Chunk: ""Fourth Level Courses For Scientific Computing (SC4) 
        Seventh Semester  
        • SCO422: Computational Geometry (3 Credit Hours) – Prerequisite: SCO311 (Computer 
        Graphics) 
        • SCO411: Neural Networks & Deep Learning (3 Credit Hours) – Prerequisite: BSC225 (Linear 
        Algebra) 
        • SCO421: Computer Vision (3 Credit Hours) – Prerequisite: CIS243 (Artificial Intelligence) 
        • Total Credit Hours: 18 
        Eighth Semester 
        • CSY410: Computer and Network Security (3 Credit Hours) – Prerequisite: CIS365
            
        Your Output (Python list of 4 integers only):
                """)
    ])


def get_chains(models, idx, memory):
    """Build LangChain pipelines for question refinement, retrieval, and answering."""
    chain_question_refine = Templates.refine_template | models[idx] | StrOutputParser()
    chain_answer_question = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
            )
            | Templates.qa_template | models[idx] | StrOutputParser()
    )
    chain_cluster_question = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")
            )
            | Templates.cluster_template | models[idx] | StrOutputParser()
    )
    return chain_question_refine, chain_answer_question, chain_cluster_question
````

## File: CollegeChatBot/chatbot.py
````python
from langchain.memory import ConversationBufferWindowMemory

from chains_and_prompts import get_chains
from utils import *
from langchain_core.runnables import RunnableLambda, RunnableParallel
from chains_and_prompts import Templates
import asyncio

api_len = 10


async def get_answer(question, models, faiss_indexes , memory):
    idx = 0

    try:
        idx += 1
        idx %= api_len

        _, answer_chain, cluster = get_chains(models, idx, memory)
        output = await cluster.ainvoke({"question": question})
        output = parse_list(output)
        print(output)
        db_names = ["abstracts", "emails", "finals", "rules"]
        retrieved_docs = []
        retrieved_tasks = []
        for i, counter in enumerate(output):
            if counter > 0:
                faiss_index = faiss_indexes.get(db_names[i])

                if faiss_index:
                    dynamic_retriever = faiss_index.as_retriever(search_kwargs={"k": counter})
                    retrieved_tasks.append(dynamic_retriever.ainvoke(question))
                else:
                    print(f"  Warning: FAISS index for '{db_names[i]}' not found in global 'faiss_indexes'.")

        retrieved_docs_list = await asyncio.gather(*retrieved_tasks)
        for docs_from_db in retrieved_docs_list:
            for doc in docs_from_db:
                retrieved_docs.append(doc.page_content)
        context = "\n\n".join(retrieved_docs)
        try:
            idx += 1
            idx %= len(models)
            answer = await answer_chain.ainvoke({"question": question, "context": context})
            return answer
        except Exception as e:
            print(f"Error during final answer generation (LLM index {idx}): {e}")
            # Try next LLM or return an error message
            return "I apologize, but I encountered an error while generating the answer. Please try again later."




    except Exception as e:
        print(f"Error at index {idx}: {e}")
````

## File: CollegeChatBot/app.py
````python
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
````
