from chains_and_prompts import get_chains
from langchain_core.runnables import RunnableLambda , RunnableParallel

api_len = 10

def get_answer(question, retriever, qa_template, refine_template , summarize_template, models):
    idx = 0
    
    try:
            idx += 1
            idx %= api_len
            refine, answer_chain , distributed_chain = get_chains(qa_template, refine_template  ,summarize_template  , models, idx)
            refined_question = refine.invoke({"question": question})
        
            docs =  retriever.invoke(refined_question)
            
            distributed_output = distributed_chain.invoke({"docs": docs, "question": refined_question})
            print(distributed_output)
            context  = "\n\n".join(f"{key} : {value}" for key ,value  in distributed_output.items())
            
            return answer_chain.invoke({"question": refined_question, "context": context})

    except Exception as e:
            print(f"Error at index {idx}: {e}")
            idx = (idx + 1) % api_len