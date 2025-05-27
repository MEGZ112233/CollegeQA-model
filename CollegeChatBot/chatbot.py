from chains_and_prompts import get_chains
from langchain_core.runnables import RunnableLambda , RunnableParallel
from chains_and_prompts import Templates
import asyncio 

api_len = 10

async def get_answer(question, retriever ,  models):
    idx = 0
    
    try:
            idx += 1
            idx %= api_len
            refine, answer_chain =  get_chains(models, idx)
            refined_question = await refine.ainvoke({"question": question})
            docs =  await retriever.ainvoke(refined_question)
            async def run_all_chains():
                tasks = [Templates.get_single( model , refined_question, d ) for model, d in zip(models , docs)]
                return await asyncio.gather(*tasks)
            
            distributed_output = await run_all_chains()


            context = "\n\n".join(f"{i}: {value}" for i, value in enumerate(distributed_output))

            return await answer_chain.ainvoke({"question": refined_question, "context": context})
    except Exception as e:
            print(f"Error at index {idx}: {e}")
            idx = (idx + 1) % api_len