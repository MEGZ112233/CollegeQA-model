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
