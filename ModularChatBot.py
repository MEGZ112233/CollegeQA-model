import os
from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda
from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
import csv
app = Flask(__name__)
api_len = 10 
def setup_environment():
    """Set up the environment by loading API keys."""
    load_dotenv()
def initialize_models(model_name):
    """Initialize embeddings and language model."""
    EMBEDDING_MODEL_NAME = "Bo8dady/finetuned4-College-embeddings"  

    embeddings = HuggingFaceEmbeddings(
                 model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},  
    encode_kwargs={'normalize_embeddings': True}
)

   
    models = []
    for i in range(0,api_len) :
        models.append(ChatGoogleGenerativeAI(model=model_name, api_key= os.getenv(f"G{i}")))
    return embeddings, models    
def upload_chunks(data_path):
    chunks =[]
    with open(data_path , 'r' , encoding ='utf-8') as file:
        reader =  csv.reader(file)
        for i, row in enumerate(reader):
                if row: 
                    if len(row) > 0:
                        chunks.append(row[0])
    return chunks
def manage_faiss_index(chunks, embeddings, index_folder):
    """Load or create a FAISS index based on the PDF's modification time."""
    index_file = os.path.join(index_folder, "index.faiss")
    if os.path.exists(index_file):
        try:
            faiss_index = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing FAISS index.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Creating a new one.")
            faiss_index = FAISS.from_texts(chunks, embeddings)
            faiss_index.save_local(index_folder)
    else:
        print("FAISS index not found. Creating a new one.")
        faiss_index = FAISS.from_texts(chunks, embeddings)
        faiss_index.save_local(index_folder) 
    return faiss_index
def setup_retriever(faiss_index , num_retrievs=10):
    """Set up the retriever using the FAISS index."""
    return faiss_index.as_retriever(search_kwargs={"k": num_retrievs})

def initialize_prompt_templates():
    """Define and return prompt templates for question answering and refinement."""
    qa_template = ChatPromptTemplate.from_messages([
        ("system", """Answer the question based on the context below and make the answer very organized and have markup . 
           Context: {context}
           Question: {question}
        """),
        ("human", "question: {question}\n context: {context}"),
    ])

    refine_template = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the student's question for clarity while preserving its meaning and if there something mispealing corect it. and make sure that the outpute is in english and  only one question? "),
        ("human", "Question to refine: {question}")
    ])

    return qa_template, refine_template
def get_chains(retriever  , qa_template, refine_template , model , idx):
    chain_question_refine = refine_template | model[idx] | StrOutputParser()
    chain_retrieve_docs = retriever | RunnableLambda(lambda x: "\n\n".join([doc.page_content for doc in x]))
    chain_answer_question = qa_template | model[idx] | StrOutputParser()
    return chain_question_refine, chain_retrieve_docs, chain_answer_question
def get_answer(question):
    """Generate an answer to a question using the defined chains."""
    idx =  0 
    while True :
        idx+=1 
        idx%= api_len
        try:
            chain_question_refine, chain_retrieve_docs, chain_answer_question = get_chains(retriever, qa_template, refine_template, models, idx)
            refined_question = chain_question_refine.invoke({"question": question})
            context = chain_retrieve_docs.invoke(refined_question)
            print(refined_question)
            answer = chain_answer_question.invoke({"question": refined_question, "context": context})  
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None
@app.route('/ModularChatBot', methods=['POST'])
def chatbot():
    # Get the question from the JSON payload
    data = request.get_json()
    question = data.get('question', '')
    print("Question received:", question)
    # Process the question (this is where your logic goes)
    answer = get_answer(question)
    # Return the answer as JSON
    return jsonify({'answer': answer})         
if __name__ == "__main__":
    data_path = "chunks.csv"
    index_folder = "faiss_index"
    setup_environment() ; 
    embeddings, models = initialize_models("gemini-2.0-flash")
    chunks = upload_chunks(data_path)
    faiss_index = manage_faiss_index(chunks, embeddings, index_folder)
    qa_template, refine_template = initialize_prompt_templates()
    retriever = setup_retriever(faiss_index)
    app.run(host='0.0.0.0', port=3000)


