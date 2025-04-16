from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda , RunnableParallel
from langchain.schema import StrOutputParser
from operator import itemgetter
def initialize_prompt_templates():
    """Define and return prompt templates for question answering and refinement."""
    qa_template = ChatPromptTemplate.from_messages([
        ("system", """ 
           you will be given  question and context and you must understand the context and the question and answer based on the context in natural language.  
        """),
        ("human", "question: {question}\n context: {context}"),
    ])

    refine_template = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the student's question for clarity while preserving its meaning and if there something mispealing correct it. Make sure the output is in English and only one question."),
        ("human", "Question to refine: {question}")
    ])

    summarize_template = ChatPromptTemplate.from_messages([
        """You are a helpful assistant. Given the following document and question, provide  only the main information the may be helpful to answer the question.
            Document:
                {document}

            Question:
                {question}

            important information:
            """
    ])
    return qa_template, refine_template ,summarize_template
def get_single(summarize_doc   , model ): 
     return {
         "document": itemgetter("document") , 
         "question": itemgetter("question")
     } |  summarize_doc | model | StrOutputParser()

def get_chains(qa_template, refine_template, summarize_template , models, idx):
    """Build LangChain pipelines for question refinement, retrieval, and answering."""
    chain_question_refine = refine_template | models[idx] | StrOutputParser()
    chain_answer_question = qa_template | models[idx] | StrOutputParser()
    def prepare_inputs(i):
        return RunnableLambda(lambda x: {
            "document": x["docs"][i].page_content,
            "question": x["question"]
        }) 
    distributed_chain = RunnableParallel({
        f"doc_{i}": prepare_inputs(i) |get_single(summarize_template, models[i])
        for i in range(5)
    })

    return chain_question_refine , chain_answer_question , distributed_chain