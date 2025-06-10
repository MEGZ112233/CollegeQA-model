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