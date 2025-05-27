from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda , RunnableParallel
from langchain.schema import StrOutputParser
from operator import itemgetter
class Templates : 
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

    async def get_single(model, question, doc):
        """Get a single output from the model using the document and question."""
        # Step 1: Apply the prompt template
        input_dict = {
            "document": doc.page_content,
            "question": question
        }
        
        # Step 2: Apply the prompt template
        prompt = Templates.summarize_template
        formatted_prompt = await prompt.ainvoke(input_dict)
        
        # Step 3: Run the model with the formatted prompt
        model_output = await model.ainvoke(formatted_prompt)
        
        # Step 4: Parse the output to string
        output_parser = StrOutputParser()
        final_output = await output_parser.ainvoke(model_output)
        
        return final_output

def get_chains(models, idx):
    """Build LangChain pipelines for question refinement, retrieval, and answering."""
    chain_question_refine = Templates.refine_template | models[idx] | StrOutputParser()
    chain_answer_question = Templates.qa_template | models[idx] | StrOutputParser()

    return chain_question_refine , chain_answer_question 