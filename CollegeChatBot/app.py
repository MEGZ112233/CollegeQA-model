from flask import Flask, request, jsonify
from config import setup_environment
from models import initialize_models
from utils import upload_chunks
from faiss_handler import manage_faiss_index
from chains_and_prompts import initialize_prompt_templates
from chatbot import get_answer
import os
app = Flask(__name__)

@app.route('/ModularChatBot', methods=['POST'])
def chatbot_api():
    data = request.get_json()
    question = data.get('question', '')
    print("Received:", question)
    answer = get_answer(question, retriever, qa_template, refine_template, summarize_template, models)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    setup_environment()
    embeddings, models = initialize_models("gemini-2.0-flash-thinking-exp-01-21")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    chunk_path = os.path.join(base_dir, "merged.csv")
    chunks = upload_chunks(chunk_path)
    faiss_index = manage_faiss_index(chunks, embeddings, "faiss_index")
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    qa_template, refine_template ,  summarize_template= initialize_prompt_templates()
    app.run(host='0.0.0.0', port=3000)
